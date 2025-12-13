"""
SAM-3D-Objects Processor
Wrapper around the SAM-3D-Objects inference pipeline
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
import yaml
from pathlib import Path


class SAMProcessor:
    """
    Process images to create 3D Gaussian Splat models using SAM-3D-Objects.
    
    This class wraps the SAM-3D-Objects inference pipeline and adds
    caption-based mask generation capabilities.
    """
    
    def __init__(self, config_path: str = "sam_config.yaml"):
        """
        Initialize SAMProcessor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self._setup_paths()
        self._load_inference_pipeline()
        self._load_sam_model()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_paths(self):
        """Setup paths to SAM-3D-Objects code."""
        # Add notebook directory to path for imports
        sam3d_path = Path("sam-3d-objects")
        if not sam3d_path.exists():
            raise FileNotFoundError(
                "SAM-3D-Objects repository not found. "
                "Please clone it: git clone https://github.com/facebookresearch/sam-3d-objects.git"
            )
        
        notebook_path = sam3d_path / "notebook"
        if str(notebook_path) not in sys.path:
            sys.path.append(str(notebook_path))
    
    def _load_inference_pipeline(self):
        """Load the SAM-3D-Objects inference pipeline."""
        try:
            from inference import Inference
            
            tag = self.config['checkpoint']['tag']
            config_path = self.config['checkpoint']['path']
            compile_model = self.config['inference']['compile']
            
            print(f"Loading SAM-3D inference pipeline from {config_path}...")
            self.inference = Inference(config_path, compile=compile_model)
            print("Inference pipeline loaded successfully!")
            
        except ImportError as e:
            raise ImportError(
                f"Could not import inference module. Make sure SAM-3D-Objects is properly installed.\n"
                f"Error: {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load inference pipeline: {str(e)}")
    
    def _load_sam_model(self):
        """Load SAM model for mask generation."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            model_type = self.config['sam']['model_type']
            checkpoint = self.config['sam']['checkpoint']
            
            if not os.path.exists(checkpoint):
                print(f"Warning: SAM checkpoint not found at {checkpoint}")
                print("Mask generation from captions will not be available.")
                self.sam_mask_generator = None
                return
            
            print(f"Loading SAM model ({model_type})...")
            device = "cuda" if self.config['device']['use_cuda'] and torch.cuda.is_available() else "cpu"
            
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            
            self.sam_mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=self.config['sam']['points_per_side'],
                pred_iou_thresh=self.config['sam']['pred_iou_thresh'],
                stability_score_thresh=self.config['sam']['stability_score_thresh'],
            )
            
            print("SAM model loaded successfully!")
            
        except ImportError:
            print("Warning: segment_anything not installed. Caption-based masking unavailable.")
            self.sam_mask_generator = None
        except Exception as e:
            print(f"Warning: Could not load SAM model: {str(e)}")
            self.sam_mask_generator = None
    
    def _get_mask(
        self,
        image: Image.Image,
        caption: str,
        threshold: float
    ) -> np.ndarray:
        """
        Extract mask from image using text caption.
        
        Args:
            image: Input PIL Image
            caption: Text description of object to segment
            threshold: Confidence threshold for mask generation
            
        Returns:
            Binary mask as numpy array (H, W)
        """
        if self.sam_mask_generator is None:
            raise RuntimeError(
                "SAM model not loaded. Cannot generate masks from captions."
            )
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Generate all masks
        print(f"Generating masks for caption: '{caption}'")
        masks = self.sam_mask_generator.generate(image_np)
        
        # Sort by confidence and size
        masks = sorted(
            masks,
            key=lambda x: (x['predicted_iou'], x['area']),
            reverse=True
        )
        
        # Filter by threshold
        valid_masks = [m for m in masks if m['predicted_iou'] >= threshold]
        
        if not valid_masks:
            raise ValueError(
                f"No masks found with confidence >= {threshold}. "
                f"Try lowering the threshold."
            )
        
        # For now, return the best mask
        # TODO: Add CLIP-based filtering to match caption
        best_mask = valid_masks[0]['segmentation']
        
        print(f"Selected mask with IoU: {valid_masks[0]['predicted_iou']:.3f}, "
              f"Area: {valid_masks[0]['area']} pixels")
        
        return best_mask.astype(np.uint8)
    
    def _create_3d_model(
        self,
        image: Image.Image,
        mask: np.ndarray
    ) -> str:
        """
        Create 3D Gaussian Splat model from image and mask.
        
        Args:
            image: Input PIL Image
            mask: Binary mask array (H, W) with values 0 or 1
            
        Returns:
            Path to generated PLY file
        """
        try:
            # Ensure mask is the right format
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # Run inference
            seed = self.config['inference']['seed']
            print(f"Running SAM-3D inference (seed={seed})...")
            
            output = self.inference(image, mask, seed=seed)
            
            # Save Gaussian splat
            output_path = self.config['output']['save_path']
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Saving Gaussian splat to {output_path}...")
            output["gs"].save_ply(output_path)
            
            print(f"3D model saved successfully!")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"3D model creation failed: {str(e)}")
    
    def model_from_caption(
        self,
        image: Union[Image.Image, str],
        caption: str,
        threshold: float = 0.5
    ) -> str:
        """
        Generate 3D Gaussian Splat model from image and text caption.
        
        This is the main API method that:
        1. Loads the image (if path provided)
        2. Generates a mask using SAM based on the caption
        3. Creates a 3D Gaussian Splat model
        4. Saves it as a PLY file
        
        Args:
            image: PIL Image or path to image file
            caption: Text description of object to segment
            threshold: Confidence threshold for mask generation (0-1)
            
        Returns:
            Path to generated PLY file
            
        Example:
            >>> processor = SAMProcessor("config.yaml")
            >>> ply_path = processor.model_from_caption(
            ...     "photo.jpg",
            ...     "a red car",
            ...     threshold=0.5
            ... )
        """
        # Load image if path provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = Image.open(image).convert('RGB')
            print(f"Loaded image from {image}")
        
        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image or path string")
        
        print(f"\nProcessing image with caption: '{caption}'")
        print(f"Threshold: {threshold}")
        
        # Step 1: Extract mask from caption
        print("\n[1/2] Extracting mask...")
        mask = self._get_mask(image, caption, threshold)
        
        # Step 2: Create 3D model
        print("\n[2/2] Creating 3D model...")
        model_path = self._create_3d_model(image, mask)
        
        print(f"\n✓ Complete! Model saved to: {model_path}")
        return model_path
    
    def model_from_mask(
        self,
        image: Union[Image.Image, str],
        mask: Union[np.ndarray, str]
    ) -> str:
        """
        Generate 3D model from image and pre-computed mask.
        
        Args:
            image: PIL Image or path to image file
            mask: Binary mask array or path to mask file
            
        Returns:
            Path to generated PLY file
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Load mask
        if isinstance(mask, str):
            mask = np.array(Image.open(mask))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Take first channel if RGB
        
        # Ensure binary
        mask = (mask > 0).astype(np.uint8)
        
        print("\nCreating 3D model from provided mask...")
        model_path = self._create_3d_model(image, mask)
        
        print(f"\n✓ Complete! Model saved to: {model_path}")
        return model_path


# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = SAMProcessor("config.yaml")
    
    # Method 1: From caption (automatic mask generation)
    model_path = processor.model_from_caption(
        image="path/to/image.png",
        caption="a chair",
        threshold=0.5
    )
    
    # Method 2: From pre-computed mask (like original demo.py)
    # model_path = processor.model_from_mask(
    #     image="path/to/image.png",
    #     mask="path/to/mask.png"
    # )
    
    print(f"Generated model: {model_path}")
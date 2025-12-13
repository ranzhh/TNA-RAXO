from PIL import Image as PILImage
import torch
import numpy as np
import os
import sys
import yaml
from pathlib import Path
from typing import Union, List, Optional


class SAMProcessor:
    """
    Process images to create 3D Gaussian Splat models using SAM-3D-Objects.
    
    This class wraps the SAM-3D-Objects inference pipeline and adds
    caption-based mask generation capabilities.
    
    Uses lazy loading to manage GPU memory on limited-memory GPUs.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize SAMProcessor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self._setup_paths()
        
        # Lazy-loaded models
        self.inference = None
        self.sam_mask_generator = None
        self.sam_model = None
        self.clip_model = None
        self.clip_processor = None
        self.clip_device = None
        
        # Device configuration
        self.device = "cuda" if self.config['device']['use_cuda'] and torch.cuda.is_available() else "cpu"
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.
        
        Tries the provided path first; if not found, searches upward from
        the script's directory and current working directory for a matching
        filename (e.g., "config.yaml").
        """
        cfg_path = Path(config_path)
        
        # 1) Direct path
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                return yaml.safe_load(f)
        
        # 2) Search in CWD
        cwd_candidate = Path.cwd() / cfg_path.name
        if cwd_candidate.exists():
            with open(cwd_candidate, 'r') as f:
                return yaml.safe_load(f)
        
        # 3) Search upwards from this file's directory
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            candidate = parent / cfg_path.name
            if candidate.exists():
                with open(candidate, 'r') as f:
                    return yaml.safe_load(f)
        
        # 4) Optional env var override
        env_path = os.environ.get('SAM3D_CONFIG')
        if env_path and Path(env_path).exists():
            with open(env_path, 'r') as f:
                return yaml.safe_load(f)
        
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Tried CWD and parent directories of {here}. "
            f"You can also set SAM3D_CONFIG to an absolute path."
        )
    
    def _setup_paths(self):
        """Setup paths to SAM-3D-Objects code.
        
        Resolves the repo path from multiple candidates:
        - Provided in config under `paths.sam3d_repo`
        - Environment variable `SAM3D_REPO`
        - Common locations relative to CWD and home
        - Workspace attachment at /home/disi/sam-3d-objects
        """
        candidates = []
        # From config
        try:
            cfg_repo = self.config.get('paths', {}).get('sam3d_repo')
            if cfg_repo:
                candidates.append(Path(cfg_repo))
        except Exception:
            pass
        # From env
        env_repo = os.environ.get('SAM3D_REPO')
        if env_repo:
            candidates.append(Path(env_repo))
        # Common local paths
        candidates.extend([
            Path('./sam-3d-objects'),
            Path('../sam-3d-objects'),
            Path.home() / 'sam-3d-objects',
            Path('/home/disi/sam-3d-objects'),
        ])
        
        sam3d_path = None
        for p in candidates:
            if p.exists() and (p / 'notebook' / 'inference.py').exists():
                sam3d_path = p
                break
        
        if sam3d_path is None:
            raise FileNotFoundError(
                "SAM-3D-Objects repository not found. Checked: "
                + ", ".join(str(x) for x in candidates)
                + ". Set `paths.sam3d_repo` in config.yaml or SAM3D_REPO env var."
            )
        
        notebook_path = sam3d_path / 'notebook'
        if str(notebook_path) not in sys.path:
            sys.path.append(str(notebook_path))
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    def _unload_inference_pipeline(self):
        """Unload inference pipeline from GPU to free memory."""
        if self.inference is not None:
            # Move models to CPU to free GPU memory
            try:
                self.inference.pipeline.to('cpu')
            except Exception:
                pass
            self._clear_gpu_memory()
    
    def _reload_inference_pipeline(self):
        """Reload inference pipeline to GPU."""
        if self.inference is not None:
            try:
                self.inference.pipeline.to(self.device)
            except Exception:
                pass
    
    def _unload_sam_clip(self):
        """Unload SAM and CLIP models from GPU to free memory."""
        if self.sam_model is not None:
            self.sam_model.to('cpu')
        if self.clip_model is not None:
            self.clip_model.to('cpu')
        self._clear_gpu_memory()
    
    def _ensure_inference_loaded(self):
        """Ensure inference pipeline is loaded (lazy loading)."""
        if self.inference is not None:
            return
        self._load_inference_pipeline()
    
    def _ensure_sam_loaded(self):
        """Ensure SAM model is loaded (lazy loading)."""
        if self.sam_mask_generator is not None:
            return
        self._load_sam_model()
    
    def _ensure_clip_loaded(self):
        """Ensure CLIP model is loaded (lazy loading)."""
        if self.clip_model is not None:
            return
        self._load_clip_model()
    
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
            
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=self.device)
            self.sam_model = sam  # Keep reference for memory management
            
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
    
    def _load_clip_model(self):
        """Load CLIP model for text-image matching using Hugging Face transformers."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            clip_config = self.config.get('clip', {})
            model_name = clip_config.get('model', 'openai/clip-vit-base-patch32')
            
            print(f"Loading CLIP model ({model_name})...")
            
            self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_device = self.device
            
            print("CLIP model loaded successfully!")
            
        except ImportError:
            print("Warning: transformers not installed. Install with: pip install transformers")
            print("Caption-based mask selection will fall back to IoU-based ranking.")
            self.clip_model = None
            self.clip_processor = None
            self.clip_device = None
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {str(e)}")
            self.clip_model = None
            self.clip_processor = None
            self.clip_device = None
    
    def _score_masks_with_clip(
        self,
        image: PILImage.Image,
        masks: List[dict],
        caption: str,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Score mask candidates against caption using CLIP.
        
        Args:
            image: Original PIL Image
            masks: List of mask dictionaries from SAM
            caption: Text description to match
            top_k: Number of top masks to evaluate (for efficiency)
            
        Returns:
            List of (mask_dict, clip_score) tuples sorted by score descending
        """
        if self.clip_model is None:
            # Fallback: return masks with zero CLIP scores
            return [(m, 0.0) for m in masks[:top_k]]
        
        image_np = np.array(image)
        scored_masks = []
        
        # Only evaluate top-k masks for efficiency
        masks_to_score = masks[:top_k]
        
        for mask_dict in masks_to_score:
            segmentation = mask_dict['segmentation']
            
            # Extract masked region with bounding box crop
            bbox = mask_dict.get('bbox')  # [x, y, w, h]
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                # Add padding for context
                pad = int(max(w, h) * 0.1)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image_np.shape[1], x + w + pad)
                y2 = min(image_np.shape[0], y + h + pad)
                
                # Crop and apply mask
                cropped = image_np[y1:y2, x1:x2].copy()
                mask_crop = segmentation[y1:y2, x1:x2]
                
                # Apply mask (keep object, white background)
                masked_region = cropped.copy()
                masked_region[~mask_crop] = 255  # White background
            else:
                # Fallback: use full image with mask applied
                masked_region = image_np.copy()
                masked_region[~segmentation] = 255
            
            # Convert to PIL for CLIP processor
            masked_pil = PILImage.fromarray(masked_region)
            
            # Process image and text together
            inputs = self.clip_processor(
                text=[caption],
                images=masked_pil,
                return_tensors="pt",
                padding=True
            ).to(self.clip_device)
            
            # Compute CLIP similarity
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                # logits_per_image has shape [1, 1] for single image-text pair
                similarity = outputs.logits_per_image.item() / 100.0  # Normalize
            
            scored_masks.append((mask_dict, similarity))
        
        # Sort by CLIP score descending
        scored_masks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_masks
    
    def _get_mask(
        self,
        image: PILImage.Image,
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
        # Ensure SAM is loaded (lazy loading)
        self._ensure_sam_loaded()
        
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
        
        # Use CLIP-based filtering to match caption (if enabled)
        clip_config = self.config.get('clip', {})
        use_clip = clip_config.get('enabled', True)
        
        if use_clip:
            # Lazy load CLIP
            self._ensure_clip_loaded()
        top_k = clip_config.get('top_k', 10)
        
        if use_clip and self.clip_model is not None:
            print(f"Scoring top {min(top_k, len(valid_masks))} masks with CLIP...")
            scored_masks = self._score_masks_with_clip(
                image, valid_masks, caption, top_k=top_k
            )
            best_mask_dict, clip_score = scored_masks[0]
            best_mask = best_mask_dict['segmentation']
            
            print(f"Selected mask with CLIP score: {clip_score:.3f}, "
                  f"IoU: {best_mask_dict['predicted_iou']:.3f}, "
                  f"Area: {best_mask_dict['area']} pixels")
        else:
            # Fallback to IoU-based selection
            best_mask_dict = valid_masks[0]
            best_mask = best_mask_dict['segmentation']
            
            print(f"Selected mask with IoU: {best_mask_dict['predicted_iou']:.3f}, "
                  f"Area: {best_mask_dict['area']} pixels")
        
        return best_mask.astype(np.uint8)
    
    def _create_3d_model(
        self,
        image: PILImage.Image,
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
            # Ensure inference is loaded (lazy loading)
            self._ensure_inference_loaded()
            
            # Convert PIL image to numpy array if needed
            if isinstance(image, PILImage.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Ensure mask is the right format: 2D array (H, W) with binary values
            # The inference.merge_mask_to_rgba function expects mask to be 2D
            # It will convert to 0-255 and add the channel dimension
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Remove channel dimension if present
            # Ensure binary (0 or 1) - the inference code multiplies by 255
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            
            # Run inference
            seed = self.config['inference']['seed']
            print(f"Running SAM-3D inference (seed={seed})...")
            
            # Merge mask to RGBA format as the inference code expects
            rgba_image = self.inference.merge_mask_to_rgba(image_np, mask)
            
            # Call the pipeline directly with decode_formats=["gaussian"] 
            # to skip mesh decoding (saves significant GPU memory)
            output = self.inference._pipeline.run(
                rgba_image,
                None,  # mask already merged
                seed,
                stage1_only=False,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=True,
                use_vertex_color=True,
                decode_formats=["gaussian"],  # Skip mesh to save GPU memory
            )
            
            # Save Gaussian splat
            output_path = self.config['output']['save_path']
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Saving Gaussian splat to {output_path}...")
            # Output structure: "gaussian" is a list, first element is the GaussianModel
            # Alternatively, the postprocessed output has "gs" key
            gs_output = output.get("gs") or output["gaussian"][0]
            gs_output.save_ply(output_path)
            
            print(f"3D model saved successfully!")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"3D model creation failed: {str(e)}")
    
    def model_from_caption(
        self,
        image: Union[PILImage.Image, str],
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
            image = PILImage.open(image).convert('RGB')
            print(f"Loaded image from {image}")
        
        # Ensure PIL Image
        if not isinstance(image, PILImage.Image):
            raise TypeError("image must be a PIL Image or path string")
        
        print(f"\nProcessing image with caption: '{caption}'")
        print(f"Threshold: {threshold}")
        
        # Step 1: Extract mask from caption (uses SAM and optionally CLIP)
        print("\n[1/2] Extracting mask...")
        mask = self._get_mask(image, caption, threshold)
        
        # Free GPU memory before loading inference pipeline
        print("\n    Freeing GPU memory for 3D model generation...")
        self._unload_sam_clip()
        
        # Step 2: Create 3D model (uses SAM-3D inference)
        print("\n[2/2] Creating 3D Gaussian Splat model...")
        model_3d = self._create_3d_model(image, mask)
        return model_3d


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate 3D Gaussian Splat from image using SAM-3D-Objects"
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "-c", "--caption",
        type=str,
        required=True,
        help="Text description of object to segment"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output PLY file path (default: outputs/<image_name>.ply)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Mask confidence threshold (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test CLIP loading, don't run full pipeline"
    )
    
    args = parser.parse_args()
    
    print("Running SAM-3D Processor...")
    print("=" * 50)
    
    if args.test_only:
        # Quick test to verify CLIP can be loaded
        try:
            print("\nTesting CLIP model loading...")
            from transformers import CLIPProcessor, CLIPModel
            
            model_name = "openai/clip-vit-base-patch32"
            print(f"Loading {model_name}...")
            
            clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            clip_processor = CLIPProcessor.from_pretrained(model_name)
            
            print("✓ CLIP model loaded successfully!")
            
            # Quick inference test
            test_text = ["a photo of a cat", "a photo of a dog"]
            inputs = clip_processor(text=test_text, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                text_features = clip_model.get_text_features(**inputs)
            
            print(f"✓ Text encoding works! Shape: {text_features.shape}")
            print("\nCLIP integration test passed!")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run full pipeline
        try:
            # Initialize processor
            processor = SAMProcessor(args.config)
            
            # Set output path if provided
            if args.output:
                processor.config['output']['save_path'] = args.output
            else:
                # Default output based on input image name
                image_name = Path(args.image).stem
                output_path = f"outputs/{image_name}.ply"
                processor.config['output']['save_path'] = output_path
            
            # Run the pipeline
            ply_path = processor.model_from_caption(
                image=args.image,
                caption=args.caption,
                threshold=args.threshold
            )
            
            print("\n" + "=" * 50)
            print(f"✓ Success! 3D model saved to: {ply_path}")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

"""
SAM3D Processor - Generate 3D models from images using SAM segmentation.

This module provides the SAMProcessor class that:
1. Segments objects in images using SAM + CLIP
2. Creates 3D mesh models from the segmented regions

Dependencies:
    - segment-anything: SAM model for mask generation
    - transformers: CLIP for text-image matching
    - open3d: 3D mesh generation (fallback mode)
    - torch, numpy, PIL: Core processing

Usage:
    processor = SAMProcessor("config.yaml")
    output_path = processor.model_from_caption("image.jpg", "a red car")
"""

from PIL import Image as PILImage
import torch
import numpy as np
import os
import sys
import yaml
from pathlib import Path
from typing import Union, List, Optional


# =============================================================================
# SAMProcessor Class
# =============================================================================

class SAMProcessor:
    """
    Process images to create 3D models using SAM-based segmentation.
    
    Uses lazy loading to manage GPU memory efficiently.
    Falls back to Open3D-based mesh generation when full SAM-3D is unavailable.
    """
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize SAMProcessor.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_paths()
        
        # Lazy-loaded models (None until needed)
        self.inference = None
        self.sam_mask_generator = None
        self.sam_model = None
        self.clip_model = None
        self.clip_processor = None
        self.clip_device = None
        
        # Device configuration
        use_cuda = self.config.get('device', {}).get('use_cuda', True)
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        candidates = [
            Path(config_path),
            Path(__file__).parent / config_path,
            Path(__file__).parent.parent / config_path,
            Path.cwd() / config_path,
        ]
        
        for path in candidates:
            if path.exists():
                with open(path) as f:
                    return yaml.safe_load(f)
        
        # Return default config if not found
        print(f"Warning: Config not found, using defaults")
        return self._default_config()
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'device': {'use_cuda': True},
            'sam': {
                'model_type': 'vit_b',
                'checkpoint': 'checkpoints/sam_vit_b.pth',
                'points_per_side': 32,
                'pred_iou_thresh': 0.88,
                'stability_score_thresh': 0.95,
            },
            'clip': {
                'enabled': True,
                'model': 'openai/clip-vit-base-patch32',
                'top_k': 5,
            },
            'inference': {'seed': 42, 'compile': False},
            'checkpoint': {'tag': 'default', 'path': ''},
            'output': {'save_path': 'output.ply'},
        }
    
    def _setup_paths(self):
        """Setup paths to SAM-3D-Objects code if available."""
        # Try to find SAM-3D-Objects repo
        candidates = [
            self.config.get('paths', {}).get('sam3d_repo'),
            os.environ.get('SAM3D_REPO'),
            Path.cwd().parent / "sam-3d-objects",
            Path.home() / "sam-3d-objects",
            Path(__file__).parent.parent.parent.parent / "sam-3d-objects",
        ]
        
        for path in candidates:
            if path and Path(path).exists():
                notebook_path = Path(path) / "notebook"
                if notebook_path.exists() and str(notebook_path) not in sys.path:
                    sys.path.insert(0, str(notebook_path))
                
                # Set environment variables
                os.environ.setdefault("LIDRA_SKIP_INIT", "true")
                if "CONDA_PREFIX" not in os.environ:
                    os.environ["CONDA_PREFIX"] = sys.prefix
                break
    
    # -------------------------------------------------------------------------
    # GPU Memory Management
    # -------------------------------------------------------------------------
    
    def _clear_gpu_memory(self):
        """Clear CUDA cache to free GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _unload_sam_clip(self):
        """Unload SAM and CLIP models to free GPU memory."""
        self.sam_mask_generator = None
        self.sam_model = None
        self.clip_model = None
        self.clip_processor = None
        self._clear_gpu_memory()
    
    def _unload_inference_pipeline(self):
        """Unload inference pipeline to free GPU memory."""
        if self.inference is not None:
            self.inference._pipeline = None
            self.inference = None
        self._clear_gpu_memory()
    
    # -------------------------------------------------------------------------
    # Lazy Model Loading
    # -------------------------------------------------------------------------
    
    def _ensure_sam_loaded(self):
        """Ensure SAM model is loaded."""
        if self.sam_mask_generator is not None:
            return
        self._load_sam_model()
    
    def _ensure_clip_loaded(self):
        """Ensure CLIP model is loaded."""
        if self.clip_model is not None:
            return
        self._load_clip_model()
    
    def _ensure_inference_loaded(self):
        """Ensure SAM-3D-Objects inference pipeline is loaded."""
        if self.inference is not None:
            return
        self._load_inference_pipeline()
    
    def _load_sam_model(self):
        """Load SAM model for mask generation."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            model_type = self.config['sam']['model_type']
            checkpoint = self.config['sam']['checkpoint']
            
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
            
            print(f"Loading SAM model ({model_type})...")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=self.device)
            self.sam_model = sam
            
            self.sam_mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=self.config['sam'].get('points_per_side', 32),
                pred_iou_thresh=self.config['sam'].get('pred_iou_thresh', 0.88),
                stability_score_thresh=self.config['sam'].get('stability_score_thresh', 0.95),
            )
            print("SAM model loaded successfully!")
            
        except ImportError as e:
            raise ImportError(f"segment_anything not installed: {e}")
    
    def _load_clip_model(self):
        """Load CLIP model for text-image matching."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model_name = self.config['clip'].get('model', 'openai/clip-vit-base-patch32')
            print(f"Loading CLIP model ({model_name})...")
            
            self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            
            # Use CPU for CLIP to save GPU memory for SAM
            self.clip_device = "cpu"
            self.clip_model.to(self.clip_device)
            self.clip_model.eval()
            
            print("CLIP model loaded successfully!")
            
        except ImportError as e:
            raise ImportError(f"transformers not installed: {e}")
    
    def _load_inference_pipeline(self):
        """Load SAM-3D-Objects inference pipeline (requires kaolin)."""
        try:
            from inference import Inference
            
            config_path = self.config['checkpoint']['path']
            print(f"Loading SAM-3D inference pipeline from {config_path}...")
            
            self.inference = Inference(config_path, compile=False)
            print("Inference pipeline loaded successfully!")
            
        except ImportError as e:
            raise ImportError(
                f"SAM-3D-Objects not available. Using fallback mode.\nError: {e}"
            )
    
    # -------------------------------------------------------------------------
    # Mask Generation (SAM + CLIP)
    # -------------------------------------------------------------------------
    
    def _get_mask(self, image: PILImage.Image, caption: str, threshold: float) -> np.ndarray:
        """
        Generate mask for object matching caption.
        
        Args:
            image: Input PIL Image
            caption: Text description of object to segment
            threshold: Confidence threshold (0-1)
            
        Returns:
            Binary mask as numpy array (H, W)
        """
        self._ensure_sam_loaded()
        
        # Generate candidate masks
        image_np = np.array(image)
        print(f"Generating masks for caption: '{caption}'")
        masks = self.sam_mask_generator.generate(image_np)
        
        if not masks:
            raise ValueError("SAM generated no masks")
        
        # Score masks with CLIP if enabled
        if self.config['clip'].get('enabled', True):
            scored_masks = self._score_masks_with_clip(image, masks, caption)
            best_mask_dict = scored_masks[0][0]  # Highest scoring
        else:
            # Use largest mask by default
            best_mask_dict = max(masks, key=lambda m: m['area'])
        
        # Extract and return binary mask
        best_mask = best_mask_dict['segmentation']
        if best_mask.dtype == bool:
            best_mask = best_mask.astype(np.uint8)
        
        return best_mask
    
    def _score_masks_with_clip(
        self, 
        image: PILImage.Image, 
        masks: List[dict], 
        caption: str
    ) -> List[tuple]:
        """
        Score masks against caption using CLIP.
        
        Returns:
            List of (mask_dict, score) tuples, sorted by score descending
        """
        self._ensure_clip_loaded()
        
        top_k = self.config['clip'].get('top_k', 5)
        image_np = np.array(image)
        
        # Pre-filter to top candidates by IoU/area
        sorted_masks = sorted(masks, key=lambda m: m['predicted_iou'] * m['area'], reverse=True)
        candidates = sorted_masks[:min(top_k, len(sorted_masks))]
        
        print(f"Scoring top {len(candidates)} masks with CLIP...")
        
        scored = []
        for mask_dict in candidates:
            # Crop image to masked region
            mask = mask_dict['segmentation']
            bbox = mask_dict['bbox']  # [x, y, w, h]
            x, y, w, h = [int(v) for v in bbox]
            
            # Ensure valid crop region
            h_img, w_img = image_np.shape[:2]
            x2, y2 = min(x + w, w_img), min(y + h, h_img)
            if x2 <= x or y2 <= y:
                continue
            
            # Create masked crop
            cropped = image_np[y:y2, x:x2].copy()
            mask_crop = mask[y:y2, x:x2]
            
            # Apply mask (set background to gray)
            cropped[~mask_crop] = 128
            crop_image = PILImage.fromarray(cropped)
            
            # Score with CLIP
            inputs = self.clip_processor(
                text=[caption],
                images=crop_image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.clip_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                score = outputs.logits_per_image.item()
            
            scored.append((mask_dict, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if scored:
            best = scored[0]
            print(f"Selected mask with CLIP score: {best[1]:.3f}, "
                  f"IoU: {best[0]['predicted_iou']:.3f}, Area: {best[0]['area']} pixels")
        
        return scored
    
    # -------------------------------------------------------------------------
    # 3D Model Generation
    # -------------------------------------------------------------------------
    
    def _create_3d_model(
        self, 
        image: PILImage.Image, 
        mask: np.ndarray, 
        output_format: str = "mesh"
    ) -> str:
        """
        Create 3D model from image and mask.
        
        Tries full SAM-3D-Objects pipeline first, falls back to Open3D.
        
        Args:
            image: Input PIL Image
            mask: Binary mask (H, W)
            output_format: "mesh" or "gaussian"
            
        Returns:
            Path to generated PLY file
        """
        try:
            self._ensure_inference_loaded()
            return self._create_3d_model_full(image, mask, output_format)
        except (ImportError, RuntimeError) as e:
            print(f"\n    Warning: Full SAM-3D not available: {str(e)[:80]}")
            print("    Using simplified Open3D mesh generation...")
            return self._create_3d_model_simple(image, mask)
    
    def _create_3d_model_simple(self, image: PILImage.Image, mask: np.ndarray) -> str:
        """
        Create 2.5D mesh from image and mask using Open3D.
        
        This is a fallback when SAM-3D-Objects is not available.
        Creates a dome-shaped mesh from the masked region.
        """
        import open3d as o3d
        
        image_np = np.array(image)
        h, w = mask.shape[:2]
        
        # Ensure 2D binary mask
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)
        
        # Get masked pixel coordinates
        ys, xs = np.where(mask > 0)
        
        if len(xs) == 0:
            print("Warning: Empty mask")
            points = np.array([[0, 0, 0]])
            colors = np.array([[0.5, 0.5, 0.5]])
        else:
            # Get colors from image
            colors = image_np[ys, xs].astype(np.float64) / 255.0
            
            # Normalize coordinates to [-1, 1]
            xs_norm = (xs - w/2) / (w/2)
            ys_norm = -(ys - h/2) / (h/2)
            
            # Create dome-shaped depth
            cx, cy = xs.mean(), ys.mean()
            dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
            max_dist = max(dist.max(), 1)
            zs = 0.3 * (1 - (dist / max_dist) ** 2)
            
            points = np.stack([xs_norm, ys_norm, zs], axis=1)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Downsample if too many points
        if len(points) > 50000:
            pcd = pcd.voxel_down_sample(voxel_size=0.02)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 2]))
        
        # Output path
        output_path = self.config['output']['save_path']
        output_path = str(Path(output_path).with_suffix('.ply'))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try Poisson reconstruction
        try:
            if len(pcd.points) > 100:
                print("    Running Poisson surface reconstruction...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8
                )
                
                # Remove low-density vertices
                densities = np.asarray(densities)
                vertices_to_remove = densities < np.quantile(densities, 0.05)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
                # Crop to original bounds
                bbox = pcd.get_axis_aligned_bounding_box()
                mesh = mesh.crop(bbox)
                
                print(f"    Created mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
                o3d.io.write_triangle_mesh(output_path, mesh)
            else:
                o3d.io.write_point_cloud(output_path, pcd)
        except Exception as e:
            print(f"    Mesh failed: {e}, saving point cloud...")
            o3d.io.write_point_cloud(output_path, pcd)
        
        print(f"    3D model saved to: {output_path}")
        return output_path
    
    def _create_3d_model_full(
        self, 
        image: PILImage.Image, 
        mask: np.ndarray, 
        output_format: str = "mesh"
    ) -> str:
        """
        Create 3D model using full SAM-3D-Objects inference.
        
        Requires kaolin and full SAM-3D-Objects installation.
        
        Args:
            output_format: "mesh" for direct mesh generation, "gaussian" for Gaussian Splat
        """
        # Prepare image and mask
        image_np = np.array(image) if isinstance(image, PILImage.Image) else image
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = (mask > 127 if mask.max() > 1 else mask).astype(np.uint8)
        
        # Run inference
        seed = self.config['inference']['seed']
        
        rgba_image = self.inference.merge_mask_to_rgba(image_np, mask)
        
        # Use gaussian decoder - mesh decoder requires too much VRAM on V100
        # Native mesh decoder needs >16GB VRAM, so we use gaussian + Poisson conversion
        if output_format == "mesh":
            print(f"Running SAM-3D inference (seed={seed})...")
            # Use gaussian only (mesh decoder OOMs on 16GB GPU)
            decode_formats = ["gaussian"]
            with_mesh_postprocess = False
        else:
            print(f"Running SAM-3D inference with Gaussian decoder (seed={seed})...")
            decode_formats = ["gaussian"]
            with_mesh_postprocess = False
        
        output = self.inference._pipeline.run(
            rgba_image, None, seed,
            stage1_only=False,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            decode_formats=decode_formats,
        )
        
        # Get output path
        output_path = self.config['output']['save_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get gaussian splat output
        gs_output = output.get("gs") or output["gaussian"][0]
        
        if output_format == "mesh":
            # Convert Gaussian Splat to triangle mesh via Poisson reconstruction
            ply_path = str(Path(output_path).with_suffix('.ply'))
            self._gaussian_to_mesh(gs_output, ply_path)
            output_path = ply_path
        else:
            # Save Gaussian splat directly
            ply_path = str(Path(output_path).with_suffix('.ply'))
            gs_output.save_ply(ply_path)
            output_path = ply_path
        
        print(f"3D model saved to: {output_path}")
        return output_path
    
    def _gaussian_to_mesh(self, gs_output, output_path: str):
        """Convert Gaussian splat output to triangle mesh."""
        import open3d as o3d
        
        points = gs_output.get_xyz.detach().cpu().numpy()
        sh_features = gs_output._features_dc.detach().cpu().numpy()
        colors = (sh_features.squeeze() * 0.5 + 0.5).clip(0, 1)
        
        if colors.ndim == 1:
            colors = np.tile(colors[:, None], (1, 3))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        print("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        o3d.io.write_triangle_mesh(output_path, mesh)
    
    # -------------------------------------------------------------------------
    # Main Public API
    # -------------------------------------------------------------------------
    
    def model_from_caption(
        self,
        image: Union[PILImage.Image, str],
        caption: str,
        threshold: float = 0.5,
        output_format: str = "mesh"
    ) -> str:
        """
        Generate 3D model from image and text caption.
        
        This is the main API method that:
        1. Loads the image
        2. Generates a mask using SAM + CLIP
        3. Creates a 3D model
        
        Args:
            image: PIL Image or path to image file
            caption: Text description of object to segment
            threshold: Confidence threshold (0-1)
            output_format: "mesh" or "gaussian"
            
        Returns:
            Path to generated PLY file
            
        Example:
            >>> processor = SAMProcessor()
            >>> path = processor.model_from_caption("photo.jpg", "a red car")
        """
        # Load image if path
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = PILImage.open(image).convert('RGB')
            print(f"Loaded image from {image}")
        
        if not isinstance(image, PILImage.Image):
            raise TypeError("image must be PIL Image or path string")
        
        print(f"\nProcessing image with caption: '{caption}'")
        print(f"Threshold: {threshold}, Format: {output_format}")
        
        # Step 1: Extract mask
        print("\n[1/2] Extracting mask...")
        mask = self._get_mask(image, caption, threshold)
        
        # Free GPU memory before 3D generation
        print("\n    Freeing GPU memory...")
        self._unload_sam_clip()
        
        # Step 2: Create 3D model
        print(f"\n[2/2] Creating 3D {output_format} model...")
        return self._create_3d_model(image, mask, output_format)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for SAM3D Processor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate 3D model from image using SAM segmentation"
    )
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-c", "--caption", required=True, help="Object description")
    parser.add_argument("-o", "--output", default=None, help="Output PLY path")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Mask threshold (0-1)")
    parser.add_argument("-f", "--format", default="mesh", choices=["mesh", "gaussian"])
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    
    args = parser.parse_args()
    
    print("Running SAM-3D Processor...")
    print("=" * 50)
    
    try:
        processor = SAMProcessor(args.config)
        
        # Set output path
        if args.output:
            processor.config['output']['save_path'] = args.output
        else:
            image_name = Path(args.image).stem
            processor.config['output']['save_path'] = f"output/{image_name}.ply"
        
        output_path = processor.model_from_caption(
            image=args.image,
            caption=args.caption,
            threshold=args.threshold,
            output_format=args.format
        )
        
        print("\n" + "=" * 50)
        print(f"✓ Success! 3D model saved to: {output_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
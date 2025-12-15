"""
SAM3D + Blender Pipeline Processor

Processes web-retrieved images through:
1. SAM3D: Create 3D models from images
2. Blender: Render multiple views
3. Mask generation: Extract masks from non-black pixels
4. Create annotations for style transfer
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image


# Path to SAM-3D-Objects venv Python (has kaolin + pytorch3d properly installed)
SAM3D_PYTHON = "/home/disi/sam-3d-objects/.venv/bin/python"


def process_sam3d(image_path: str, category: str, output_path: str, threshold: float = 0.5) -> bool:
    """Run SAM3D to create a 3D model from an image.
    
    Uses SAM-3D-Objects venv Python which has all required dependencies (kaolin, pytorch3d).
    """
    cmd = [
        SAM3D_PYTHON, "raxo2/processors/sam3d.py",
        "-i", image_path,
        "-c", category,
        "-o", output_path,
        "-t", str(threshold)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ✗ SAM3D failed: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ✗ SAM3D timeout")
        return False
    except Exception as e:
        print(f"  ✗ SAM3D error: {e}")
        return False


def process_blender(ply_path: str, output_dir: str, n_views: int = 8) -> bool:
    """Run Blender to render multiple views of a 3D model."""
    cmd = [
        "blender", "--background", "--python", "raxo2/processors/blender.py",
        "--", "--input", ply_path, "--output-dir", output_dir, "--views", str(n_views)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ✗ Blender failed: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ✗ Blender timeout")
        return False
    except Exception as e:
        print(f"  ✗ Blender error: {e}")
        return False


def create_mask_from_render(render_path: str, mask_path: str, threshold: int = 10) -> tuple:
    """Create a binary mask from a render (non-black pixels)."""
    img = Image.open(render_path).convert("RGB")
    img_array = np.array(img)
    
    # Mask: any pixel that is not black (threshold to handle slight noise)
    mask = np.any(img_array > threshold, axis=2).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(mask_path)
    
    return img.width, img.height


def mask_to_rle(mask_path: str) -> dict:
    """Convert a mask image to RLE format."""
    mask = np.array(Image.open(mask_path).convert('L'))
    mask_binary = (mask > 127).astype(np.uint8)
    
    rle = mask_util.encode(np.asfortranarray(mask_binary))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def run_pipeline(
    annotations_file: str,
    images_dir: str,
    sam3d_out: str,
    blender_out: str,
    blender_masks_out: str,
    n_views: int = 8
) -> list:
    """Run the full SAM3D + Blender pipeline on all images."""
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    print(f"Found {len(images)} images to process")
    
    all_renders = []
    
    for idx, img_info in enumerate(images):
        image_name = img_info['image_name']
        category = img_info.get('super_category', img_info.get('category', 'unknown'))
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Skipping {image_name}: file not found")
            continue
        
        print(f"\n[{idx+1}/{len(images)}] Processing: {image_name} (category: {category})")
        
        # Step 1: SAM3D
        base_name = Path(image_name).stem
        ply_output = os.path.join(sam3d_out, f"{base_name}.ply")
        
        print(f"  → Running SAM3D...")
        if not process_sam3d(image_path, category, ply_output):
            continue
        
        if not os.path.exists(ply_output):
            print(f"  ✗ PLY file not created")
            continue
        print(f"  ✓ SAM3D output: {ply_output}")
        
        # Step 2: Blender
        render_dir = os.path.join(blender_out, base_name)
        os.makedirs(render_dir, exist_ok=True)
        
        print(f"  → Running Blender ({n_views} views)...")
        if not process_blender(ply_output, render_dir, n_views):
            continue
        print(f"  ✓ Blender renders saved to: {render_dir}")
        
        # Step 3: Create masks
        mask_dir = os.path.join(blender_masks_out, base_name)
        os.makedirs(mask_dir, exist_ok=True)
        
        print(f"  → Creating masks from renders...")
        render_files = sorted(Path(render_dir).glob("view_*.png"))
        
        for render_file in render_files:
            try:
                mask_path = os.path.join(mask_dir, render_file.name)
                width, height = create_mask_from_render(str(render_file), mask_path)
                
                all_renders.append({
                    'image_name': f"{base_name}/{render_file.name}",
                    'category': category,
                    'super_category': category,
                    'mask_path': f"{base_name}/{render_file.name}",
                    'original_image': image_name,
                    'bbox_xyxy': [0, 0, width, height]
                })
            except Exception as e:
                print(f"  ✗ Error creating mask for {render_file.name}: {e}")
        
        print(f"  ✓ Created {len(render_files)} masks")
    
    return all_renders


def create_annotations_with_masks(renders: list, blender_masks_out: str, output_file: str):
    """Create annotations with RLE masks for style transfer."""
    images_with_masks = []
    
    for img_info in renders:
        mask_path = os.path.join(blender_masks_out, img_info['mask_path'])
        if not os.path.exists(mask_path):
            continue
        
        img_info['mask'] = mask_to_rle(mask_path)
        images_with_masks.append(img_info)
    
    with open(output_file, 'w') as f:
        json.dump(images_with_masks, f, indent=2)
    
    print(f"Created {len(images_with_masks)} annotations with masks")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='SAM3D + Blender Pipeline')
    parser.add_argument('--annotations', required=True, help='Path to annotations.json from web retrieval')
    parser.add_argument('--images_dir', required=True, help='Path to images directory')
    parser.add_argument('--sam3d_out', required=True, help='Output directory for SAM3D PLY files')
    parser.add_argument('--blender_out', required=True, help='Output directory for Blender renders')
    parser.add_argument('--blender_masks_out', required=True, help='Output directory for masks')
    parser.add_argument('--output_annotations', required=True, help='Output path for annotations with masks')
    parser.add_argument('--n_views', type=int, default=8, help='Number of Blender views to render')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.sam3d_out, exist_ok=True)
    os.makedirs(args.blender_out, exist_ok=True)
    os.makedirs(args.blender_masks_out, exist_ok=True)
    
    print("==============================================")
    print("Running SAM3D + Blender Pipeline")
    print("==============================================")
    
    # Run pipeline
    all_renders = run_pipeline(
        args.annotations,
        args.images_dir,
        args.sam3d_out,
        args.blender_out,
        args.blender_masks_out,
        args.n_views
    )
    
    # Save intermediate annotations
    intermediate_file = os.path.join(args.blender_out, 'annotations.json')
    with open(intermediate_file, 'w') as f:
        json.dump({'images': all_renders, 'source': 'sam3d_blender_pipeline'}, f, indent=2)
    
    # Create annotations with masks
    create_annotations_with_masks(all_renders, args.blender_masks_out, args.output_annotations)
    
    print("\n============================================")
    print(f"Pipeline complete!")
    print(f"Generated {len(all_renders)} renders with masks")
    print("============================================")


if __name__ == "__main__":
    main()

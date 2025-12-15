"""
GroundingDINO inference on COCO-format dataset.
Generates initial object detection results that can be used with SAM2 for mask generation.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import torch

# GroundingDINO imports
from groundingdino.util.inference import load_model, load_image, predict

# Configuration
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_grounding_dino():
    """Load GroundingDINO model from HuggingFace."""
    from huggingface_hub import hf_hub_download
    
    # Download config and weights
    config_path = hf_hub_download(
        repo_id='ShilongLiu/GroundingDINO',
        filename='GroundingDINO_SwinB.cfg.py'
    )
    weights_path = hf_hub_download(
        repo_id='ShilongLiu/GroundingDINO',
        filename='groundingdino_swinb_cogcoor.pth'
    )
    
    print(f"Loading GroundingDINO model on {DEVICE}...")
    model = load_model(config_path, weights_path, device=DEVICE)
    print("Model loaded successfully.")
    return model


def run_inference(model, image_path, text_prompt, box_threshold, text_threshold):
    """Run GroundingDINO inference on a single image."""
    try:
        image_source, image_tensor = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=DEVICE
        )
        
        h, w = image_source.shape[:2]
        
        # Convert from normalized cxcywh to pixel xywh (COCO format)
        results = []
        for i, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
            cx, cy, bw, bh = box.tolist()
            # Convert to COCO format: x, y, width, height (top-left corner)
            x = (cx - bw/2) * w
            y = (cy - bh/2) * h
            width = bw * w
            height = bh * h
            
            results.append({
                'bbox': [x, y, width, height],
                'score': float(score),
                'phrase': phrase.strip().lower()
            })
        
        return results
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


def main(args):
    # Load COCO ground truth to get categories and images
    print(f"Loading annotations from {args.gt}...")
    with open(args.gt, 'r') as f:
        coco_data = json.load(f)
    
    # Extract categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_names = list(categories.values())
    
    # Build text prompt (all categories separated by dots)
    text_prompt = " . ".join(category_names) + " ."
    print(f"Categories: {category_names}")
    print(f"Text prompt: {text_prompt}")
    
    # Create category name to id mapping
    name_to_id = {cat['name'].lower(): cat['id'] for cat in coco_data['categories']}
    
    # Load model
    model = load_grounding_dino()
    
    # Process images
    images = coco_data['images']
    print(f"Found {len(images)} images to process")
    
    all_detections = []
    detection_id = 1
    
    for img_info in tqdm(images, desc="Running GroundingDINO"):
        image_path = os.path.join(args.image_path, img_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        results = run_inference(
            model, 
            image_path, 
            text_prompt, 
            args.box_threshold, 
            args.text_threshold
        )
        
        for det in results:
            # Match phrase to category
            phrase = det['phrase']
            matched_cat_id = None
            
            # Try exact match first
            if phrase in name_to_id:
                matched_cat_id = name_to_id[phrase]
            else:
                # Try partial match
                for cat_name, cat_id in name_to_id.items():
                    if phrase in cat_name or cat_name in phrase:
                        matched_cat_id = cat_id
                        break
            
            if matched_cat_id is None:
                # Default to first category if no match
                matched_cat_id = list(categories.keys())[0]
            
            all_detections.append({
                'id': detection_id,
                'image_id': img_info['id'],
                'category_id': matched_cat_id,
                'bbox': det['bbox'],
                'score': det['score'],
                'area': det['bbox'][2] * det['bbox'][3]
            })
            detection_id += 1
    
    print(f"Generated {len(all_detections)} detections")
    
    # Save in COCO Results format (list of detections)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_detections, f)
    
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GroundingDINO on COCO dataset')
    parser.add_argument('--gt', required=True, help='Path to COCO ground truth annotations')
    parser.add_argument('--image_path', required=True, help='Path to images directory')
    parser.add_argument('--out', required=True, help='Output file path for detections')
    parser.add_argument('--box_threshold', type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument('--text_threshold', type=float, default=DEFAULT_TEXT_THRESHOLD)
    
    args = parser.parse_args()
    main(args)

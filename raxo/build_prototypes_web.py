"""
Build prototypes from web-retrieved images with masks.
This version handles the list format from google_image_retrievalv2.py 
(different from the COCO format used by build_prototypes_with_masks_prop.py)
"""

from tqdm import tqdm
import torch
import torchvision.transforms as T
import argparse
import json
import os
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from torch.utils.data import Dataset, DataLoader

DINO_DIMENSION = 768
DINO_PATCH_SIZE = 14
device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser(description='Build prototypes from web-retrieved images with masks')
parser.add_argument('--gt', required=True, help='Path to the annotations_with_masks.json file (list format)')
parser.add_argument('--image_path', required=True, help='Path to the directory containing the style-transferred images')
parser.add_argument('--out', required=True, help='Path where prototypes will be saved (.pt file)')


class WebImagesDataset(Dataset):
    """Dataset for web-retrieved images with masks (list format)."""
    
    def __init__(self, annotations, image_path, transform):
        self.annotations = annotations
        self.image_path = image_path
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        anot = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.image_path, anot['image_name'])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get category (use super_category if available, else category)
        category = anot.get('super_category', anot.get('category', 'unknown'))
        
        # Decode mask and resize to match DINOv2 token grid (16x16 for 224x224 input with patch 14)
        mask = mask_util.decode(anot['mask'])
        mask_resized = self._resize_mask(mask)
        
        if self.transform:
            image = self.transform(image)
            
        return image, {'category': category, 'image_name': anot['image_name']}, mask_resized
    
    def _resize_mask(self, mask):
        """Resize mask to 16x16 (matching DINOv2 token grid) and flatten to 256 tokens."""
        # Resize mask to 16x16 using PIL
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((16, 16), Image.NEAREST)
        mask_array = np.array(mask_resized) / 255.0
        
        # Flatten to 256 and apply threshold (0.3 means at least 30% of patch is object)
        mask_flat = mask_array.flatten()
        mask_binary = (mask_flat > 0.3).astype(np.float32)
        
        return torch.tensor(mask_binary)


def build_model():
    """Build DINOv2 model and transforms."""
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    return transform, model


def custom_collate_fn(batch):
    """Custom collate function to handle mixed types."""
    images = torch.stack([item[0] for item in batch], dim=0)
    annotations = [item[1] for item in batch]
    masks = torch.stack([item[2] for item in batch], dim=0)
    return images, annotations, masks


def compute_prototypes(model, dataloader):
    """Compute prototypes for each category."""
    category_features = {}
    
    for images, annotations, masks in tqdm(dataloader, desc="Computing prototypes"):
        images = images.to(device)
        
        with torch.no_grad():
            # Get intermediate features from DINOv2
            outputs = model.get_intermediate_layers(images)
            features = outputs[0].cpu()  # Shape: [batch, 256, 768]
        
        # Apply mask to get object features
        masks_expanded = masks.unsqueeze(-1)  # Shape: [batch, 256, 1]
        masked_features = features * masks_expanded
        
        # Compute mean feature per image (only over masked tokens)
        sum_masked = masked_features.sum(dim=1)  # Shape: [batch, 768]
        count_masked = masks.sum(dim=1, keepdim=True).clamp(min=1)  # Shape: [batch, 1]
        mean_features = sum_masked / count_masked  # Shape: [batch, 768]
        
        # Group by category
        for feat, anot in zip(mean_features, annotations):
            category = anot['category']
            if category not in category_features:
                category_features[category] = []
            category_features[category].append(feat)
    
    # Build final prototypes
    prototypes = {}
    for category, features in category_features.items():
        if not features:
            continue
            
        # Add individual prototypes
        for i, feat in enumerate(features, start=1):
            key = f"{category}_{i}"
            prototypes[key] = {
                "prototype": feat,
                "supercategory": category
            }
        
        # Add mean prototype
        mean_proto = torch.stack(features).mean(dim=0)
        prototypes[category] = {
            "prototype": mean_proto,
            "supercategory": category
        }
        
        print(f"Category '{category}': {len(features)} samples")
    
    return prototypes


def main(args):
    """Main function to build prototypes from web-retrieved images."""
    
    # Load annotations (list format)
    print(f"Loading annotations from {args.gt}")
    with open(args.gt, 'r') as f:
        annotations = json.load(f)
    
    # Handle both list format and dict format with 'images' key
    if isinstance(annotations, dict) and 'images' in annotations:
        annotations = annotations['images']
    
    print(f"Found {len(annotations)} images")
    
    # Build model
    transform, model = build_model()
    
    # Create dataset and dataloader
    dataset = WebImagesDataset(annotations, args.image_path, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=custom_collate_fn
    )
    
    # Compute prototypes
    prototypes = compute_prototypes(model, dataloader)
    
    # Add a negative prototype (mean of low-confidence features could be added here)
    # For now, just add a placeholder
    prototypes['negative'] = {
        "prototype": torch.zeros(DINO_DIMENSION),
        "supercategory": "negative"
    }
    
    # Save prototypes
    print(f"Saving {len(prototypes)} prototypes to {args.out}")
    torch.save(prototypes, args.out)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

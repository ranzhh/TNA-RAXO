"""
Final version of the method to build the colour database: 
1) given all the categories names, extract the materials (super-categories) using Gemini
2) take the colour of each supercategory removing 
3) save the colour per_cat
"""

import json
import logging
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import os
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gemini configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL = "gemini-2.5-flash"

PROMPT = """
You are a computer expert specializing in material classification. Your task is to analyze a given list of objects, determine their primary material composition, and group them accordingly.

Instructions:
Identify the main materials present among the objects (e.g., metal, organic, inorganic, plastic, ceramic, etc.).
Assign each object to the most appropriate material category. Each object should belong to only one category based on its primary composition.
Return the results in JSON format, where the keys are material categories, and the values are lists of objects belonging to those categories.
Example:
Input:
Objects: gun, bat, pressure_vessel, beer_glass, fur_coat, lemon

Expected Output (JSON):

  "metal": ['gun', 'bat'],
  "inorganic": ['pressure_vessel', 'beer_glass'],
  "organic": ['fur_coat', 'lemon']

Now, classify the following list of objects: {}. Return only the json format without any markdown formatting or code blocks.
"""

# Dataset names for processing
DATASET_NAMES = ("PIXray", "pidray", "CLCXray", "HiXray", "DvXray", "xray_fsod", "eds", "COMPASS-XP")


parser = argparse.ArgumentParser(description='Build colour database from GT in coco format using Gemini')
parser.add_argument('--cats_gt', required=True, help='Path to the ground truth file in COCO format')
parser.add_argument('--out', required=True, help='Output directory for colour files')
parser.add_argument('--support_set', required=True, help='Path to the support set JSON file with masks')
parser.add_argument('--images_path', required=True, help='Path to the support set images directory')
parser.add_argument('--materials_cache', default=None, help='Path to cache file for materials classification (optional)')


def call_gemini_for_materials(cats: list) -> dict:
    """Call the Gemini API to classify objects by material."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please add it to your .env file.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    custom_prompt = PROMPT.format(cats)
    
    logger.info(f"Calling Gemini API to classify {len(cats)} categories by material...")
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=custom_prompt
        )
        
        # Parse the response text to extract JSON
        generated_text = response.text.strip()
        # Clean up any markdown formatting
        cleaned_string = generated_text.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(cleaned_string)
        
        logger.info(f"Material classification successful. Found {len(parsed_json)} material groups:")
        for material, objects in parsed_json.items():
            logger.info(f"  - {material}: {objects}")
                
        return parsed_json
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        logger.error(f"Raw response: {generated_text}")
        raise
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise


def clustering(d_name: str, support: COCO, material: tuple, images_path: str) -> np.ndarray:
    """
    Calculate the mean color for a material category by averaging colors from all images
    of its constituent object categories (excluding the specified dataset).
    """
    mean_colors = []
    
    for category in support.loadCats(support.getCatIds()):
        if category['name'] not in material[1]:
            continue
        category_id = category['id']
        
        img_ids = support.getImgIds(catIds=[category_id])
        img_info_list = support.loadImgs(img_ids)
        # Filter out images from the current dataset
        imgs_filtered = [img for img in img_info_list if d_name not in img['file_name']]

        # Process each selected image
        for img_info in imgs_filtered:
            img_path = os.path.join(images_path, img_info['file_name'])
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue

            # Load the image using PIL
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            # Get the annotations (masks) for the selected image
            ann_ids = support.getAnnIds(imgIds=[img_info['id']], catIds=[category_id])
            anns = support.loadAnns(ann_ids)

            for ann in anns:
                # Get the segmentation mask of the object
                if 'mask' not in ann:
                    continue
                mask = mask_util.decode(ann['mask'])

                # Skip if no valid mask (empty object)
                if np.sum(mask) == 0:
                    continue

                # Crop the object from the image using the mask
                cropped_object = image_np * np.expand_dims(mask, axis=-1)

                # Calculate the mean color for the cropped object
                mean_color = np.mean(cropped_object[mask > 0], axis=0)
                mean_colors.append(mean_color)
        
    # If we collected any colors, calculate the average mean color
    if mean_colors:
        avg_mean_color = np.mean(mean_colors, axis=0)
        return avg_mean_color.astype(int)
    else:
        logger.warning(f"No colors found for material: {material[0]}")
        return np.array([128, 128, 128])  # Default gray


def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    
    # Load categories from ground truth file
    with open(args.cats_gt, 'r') as f:
        file = json.load(f)
    
    # Determine materials cache path
    materials_cache_path = args.materials_cache or os.path.join(args.out, 'colour_knowledge_material_clustering.json')
    
    # 1) Obtain or load the material-based supercategories
    if not os.path.exists(materials_cache_path):
        cats = [x['name'] for x in file['categories']]
        materials = call_gemini_for_materials(cats)
        
        # Save the results to the cache file
        with open(materials_cache_path, 'w') as out_file:
            json.dump(materials, out_file, indent=4)
        logger.info(f"Materials classification saved to {materials_cache_path}")
    else:
        logger.info(f"Loading cached materials from {materials_cache_path}")
        with open(materials_cache_path, 'r') as f:
            materials = json.load(f)
    
    # 2) Obtain the colours from the material excluding the prohibited dataset (one each time)
    for d_name in tqdm(DATASET_NAMES, desc="Processing datasets"):
        # Open support set file
        support = COCO(args.support_set)
        
        # Colour of each material
        mat_colour = {}
        color_dict_per_cat = {}
        
        for mat in materials.items():
            # Get colours 
            try:
                color_mat = clustering(d_name, support, mat, args.images_path)
                color_mat = color_mat.tolist()
            except Exception as e:
                logger.warning(f"Error processing material {mat[0]} for dataset {d_name}: {e}")
                color_mat = [128, 128, 128]  # Default gray
                
            mat_colour[mat[0]] = color_mat

            # Finally save the colour per category
            for cat in mat[1]:
                color_dict_per_cat[cat] = color_mat
                
        # Save results for this dataset
        save_name = os.path.join(args.out, f"{d_name}_colour.json")
        with open(save_name, 'w') as out_file:
            json.dump(mat_colour, out_file, indent=4)
        logger.info(f"Material colours saved to {save_name}")
        
        save_name = os.path.join(args.out, f"{d_name}_colour_per_cat.json")
        with open(save_name, 'w') as out_file:
            json.dump(color_dict_per_cat, out_file, indent=4)
        logger.info(f"Category colours saved to {save_name}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

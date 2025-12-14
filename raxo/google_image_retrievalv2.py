"""
Thanks to: https://github.com/cvlab-columbia/DoubleRight/blob/master/google_image_search_url.py
Modified to use GroundingDINO standalone (rf-groundingdino) instead of mmdet
"""

##############################################################
import argparse
import os
import json
import logging
from dotenv import load_dotenv
from google_images_search import GoogleImagesSearch
##############################################################
from PIL import Image
import torch

# GroundingDINO imports (standalone version)
from groundingdino.util.inference import load_model, load_image, predict

import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
##############################################################

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')
CX = os.getenv('GOOGLE_CX')
INIT_IMAGES_DOWNLOAD_DIR = 'imgs/'
INIT_IMAGES_JSON = 'annotations.json'
FINAL_IMAGES_ = 'imgs_final/'
prompt = "A photo of a "
##############################################################


##############################################################
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SCORE_THS = 0.5
BOX_THS = 0.35  # Box threshold for GroundingDINO
TEXT_THS = 0.25  # Text threshold for GroundingDINO
LIMIT = 60  # Limit of tries per categories. Number of images that will be searched from internet

# Default model paths for GroundingDINO - dynamically find from installed package
import groundingdino
_GD_PATH = os.path.dirname(groundingdino.__file__)
DEFAULT_CONFIG = os.path.join(_GD_PATH, "config", "GroundingDINO_SwinB_cfg.py")
DEFAULT_WEIGHTS = "IDEA-Research/grounding-dino-base"  # Will be downloaded from HuggingFace


parser = argparse.ArgumentParser(description='Web image search of given categories')
parser.add_argument('--cats', required=True, help='Path the .json file where categories are stores')
parser.add_argument('--n', required=True, help='Number of examples images per class')
parser.add_argument('--model_config', required=False, default=DEFAULT_CONFIG, help='Path to the GroundingDINO config file')
parser.add_argument('--model_weights', required=False, default=DEFAULT_WEIGHTS, help='Path to the GroundingDINO checkpoint file')
parser.add_argument('--out', required=True, help='Path to the directory where results will be saved')
parser.add_argument('--cats_from_gt', required=True, help='Path to the directory where GT test info is located. COCO format')
parser.add_argument('--box_threshold', type=float, default=BOX_THS, help='Box threshold for GroundingDINO')
parser.add_argument('--text_threshold', type=float, default=TEXT_THS, help='Text threshold for GroundingDINO')


def build_inferencer(args):
    """Load GroundingDINO model. Downloads weights from HuggingFace if needed."""
    from huggingface_hub import hf_hub_download
    
    # Check if weights path is a HuggingFace repo ID or local file
    weights_path = args.model_weights
    if not os.path.exists(weights_path):
        logger.info("Downloading GroundingDINO weights from HuggingFace...")
        weights_path = hf_hub_download(
            repo_id='ShilongLiu/GroundingDINO',
            filename='groundingdino_swinb_cogcoor.pth'
        )
        logger.info(f"Weights downloaded to: {weights_path}")
    
    logger.info(f"Loading GroundingDINO model from {weights_path}")
    model = load_model(args.model_config, weights_path, device=DEVICE)
    logger.info(f"Model loaded on {DEVICE}")
    return model


def run_inference(model, image_path, text_prompt, box_threshold, text_threshold):
    """
    Run GroundingDINO inference on a single image.
    Returns: boxes (xyxy format), logits, phrases
    """
    # Load and preprocess image
    image_source, image_tensor = load_image(image_path)
    
    # Run prediction
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE
    )
    
    # Convert boxes from normalized cxcywh to xyxy pixel coordinates
    h, w = image_source.shape[:2]
    boxes_xyxy = []
    for box in boxes:
        cx, cy, bw, bh = box.tolist()
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        boxes_xyxy.append([x1, y1, x2, y2])
    
    return boxes_xyxy, logits.tolist(), phrases


# This main keep the top1 detection for retrieved image, if its confidence is bigger than the threshold
def main2(args):
    # 1) Build model inferencer
    model = build_inferencer(args)
    os.makedirs(args.out, exist_ok=True)
    
    # 2) Obtain categories for wich images from internet will be retrieved
    # Support the LLM augmentation, and the classic behaviour
    if os.path.exists(args.cats):
        data_cats = json.load(open(args.cats))
        if 'category_list_real' in data_cats.keys():
            real_cats = data_cats['category_list_real']
            find_cats = data_cats['category_list_find']
            mapper = {find:real for find,real in zip(find_cats, real_cats)}
            logger.info(f"Category mapper: {mapper}")
        else:
            real_cats = data_cats['categories']
            find_cats = data_cats['categories']
            mapper = {find:real for find,real in zip(find_cats, real_cats)}
            logger.info(f"Category mapper: {mapper}")
    else:
        from pycocotools.coco import COCO
        coco = COCO(args.cats_from_gt)
        # Get category IDs and names
        categories = coco.loadCats(coco.getCatIds())
        real_cats = [cat["name"] for cat in categories]
        find_cats = [cat["name"] for cat in categories]
        mapper = {find:real for find,real in zip(find_cats, real_cats)}
        logger.info(f"Category mapper: {mapper}")


    
    final_json = {"images":[]}
    try:
        img_id = 0
        for category in find_cats:
            # 3) For each category obtain the web images
            query = prompt + f'{category}'
            logger.info(f'Searching for: {query}')
            _search_params = {
                    'q': query,
                    'num': LIMIT,
                    'searchType': 'image',
                    'dateRestrict': 'y7',  # Results from the last 6 years
                    'safe': '',
                    'fileType': 'jpg|png',
                    'imgType': 'photo',
                    'imgSize': '',
                    'imgDominantColor': '',
                    'rights': '',
                    'imgColorType': "",
                    'lr': 'lang_en',
                }

            # Init Google Image Search every iteration to avoid repetition error
            gis = GoogleImagesSearch(API_KEY, CX)
            try:
                gis.search(search_params=_search_params)
            except Exception as Error:
                logger.error(f'Error during query: {Error}')
                logger.error(f'Query error with: {query}')
                continue

            # 4) While the number of good images in the category is not args.n (or still images)
            n_images_cat_i = 0
            for image in gis.results():
                try:
                    if n_images_cat_i >= int(args.n):
                        break

                    path = os.path.join(args.out, INIT_IMAGES_DOWNLOAD_DIR)
                    os.makedirs(path, exist_ok=True)
                    image.download(path)
                    
                    original_name = os.path.basename(image.path)
                    _, file_extension = os.path.splitext(original_name)

                    new_img_name = f"cat_{mapper[category]}_{img_id}{file_extension}"
                    img_id += 1
                    new_image_path = image.path.replace(original_name, new_img_name)
                    os.rename(image.path, new_image_path)

                    # Run GroundingDINO inference
                    text_prompt = f"{category}"
                    
                    try:
                        boxes, scores, phrases = run_inference(
                            model, 
                            new_image_path, 
                            text_prompt,
                            args.box_threshold,
                            args.text_threshold
                        )
                    except Exception as e:
                        logger.warning(f"Inference error for {new_image_path}: {e}")
                        continue
                    
                    # Check if we have any detections
                    if len(boxes) == 0 or len(scores) == 0:
                        continue
                        
                    # Get best detection (highest score)
                    best_idx = scores.index(max(scores))
                    bbox_final = boxes[best_idx]
                    score = scores[best_idx]
                    
                    if score > SCORE_THS:
                        # Keep it
                        try:
                            img_aux = Image.open(new_image_path)
                        except Exception:
                            continue
                        
                        # Append annotations
                        final_json['images'].append({
                            "image_name": new_img_name, 
                            "url": image.url, 
                            "category": category, 
                            "super_category": mapper[category], 
                            "bbox_xyxy": bbox_final,
                            "score": score
                        })
                        n_images_cat_i += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing image for category {category}: {e}")
                    continue


            logger.info(f"Category {category} of super_category {mapper[category]} ends with {n_images_cat_i} objects !!")   

        # Save final json
        with open(os.path.join(args.out, INIT_IMAGES_JSON), 'w') as f:
            json.dump(final_json, f, indent=2)
        logger.info(f"Saved annotations to {os.path.join(args.out, INIT_IMAGES_JSON)}")
    
    except Exception as e:
        logger.error(f"Critical error during processing: {e}")
        with open(os.path.join(args.out, INIT_IMAGES_JSON), 'w') as f:
            json.dump(final_json, f, indent=2)


if __name__ == "__main__":
    args = parser.parse_args()
    main2(args)

import os

import dotenv
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

dotenv.load_dotenv()


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return "cuda"
    elif torch.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return "mps"
    else:
        return "cpu"


def build_sam_model():
    SAM_PATH = os.getenv("SAM_PATH")

    if not SAM_PATH:
        raise ValueError("SAM_PATH environment variable is not set.")

    sam2_checkpoint = f"{SAM_PATH}/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = f"/{SAM_PATH}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=get_device())
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return sam2_predictor

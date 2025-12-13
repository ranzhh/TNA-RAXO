import argparse
import copy
import json
import os

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
from PIL import Image
from pycocotools.coco import COCO
from supervision.draw.color import ColorPalette
from tqdm import tqdm
from utils import build_sam_model, get_device

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

DINO_DIMENSION = 768
device = get_device()


parser = argparse.ArgumentParser(description="Build prototypes from GT in coco format")
parser.add_argument(
    "--gt", required=True, help="Path to the ground truth file in COCO format. Needed to load imgs"
)
parser.add_argument("--res", required=True, help="Path to the results file in COCO-res format")
parser.add_argument(
    "--image_path",
    required=True,
    help="Path to the directory containing the images referenced in gt",
)
parser.add_argument(
    "--limit",
    type=int,
    help="Limit the number of images to process. Default None (all images)",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=None,
    help="Only process top K highest confidence detections per image. Default None (all detections)",
)
parser.add_argument(
    "--score_threshold",
    type=float,
    default=None,
    help="Only process detections with score >= threshold. Default None (no filtering)",
)


def save_coco(coco_dt, filtradas_anns, save_name):
    save_dict = {}
    for k in coco_dt.dataset.keys():
        if k != "annotations":
            save_dict[k] = coco_dt.dataset[k]

    # Add anotations
    save_dict["annotations"] = filtradas_anns

    print(save_name)
    with open(save_name, "w") as fp:
        s = json.dumps(save_dict, indent=4, sort_keys=True)
        fp.write(s)


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def plot(img_path, boxes_xyxy, masks, imag_id_name):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3d = cv2.merge([img_gray, img_gray, img_gray])
    detections = sv.Detections(
        xyxy=boxes_xyxy,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
    )

    box_annotator = sv.BoxAnnotator(
        color=ColorPalette.from_hex(CUSTOM_COLOR_MAP), color_lookup=sv.ColorLookup.INDEX
    )
    annotated_frame = box_annotator.annotate(scene=gray_3d.copy(), detections=detections)

    mask_annotator = sv.MaskAnnotator(
        color=ColorPalette.from_hex(CUSTOM_COLOR_MAP), color_lookup=sv.ColorLookup.INDEX
    )
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join("./borrar_sam_res", f"sam2_{imag_id_name}.jpg"), annotated_frame)


def main(args, plot_result=False):
    # 1) Create SAM
    sam2_predictor = build_sam_model()

    # 2) OPEN DATASET
    gt = COCO(args.gt)
    dataset = gt.loadRes(args.res)

    imgs = dataset.loadImgs(dataset.getImgIds()[: args.limit])

    # Take into accoutn each image can have more than 1 annotation:
    annotations_with_masks = []
    imag_id_name = 0
    for img in tqdm(imgs):
        # Open image
        img_path = os.path.join(args.image_path, img["file_name"])
        image = Image.open(img_path)
        sam2_predictor.set_image(np.array(image.convert("RGB")))

        # Load annotations of this image:
        annotations = dataset.loadAnns(dataset.getAnnIds(imgIds=img["id"]))

        # Skip if no annotations
        if len(annotations) == 0:
            continue

        # Filter annotations by score threshold if specified
        if args.score_threshold is not None:
            annotations = [a for a in annotations if a.get("score", 1.0) >= args.score_threshold]

        # Limit to top-k if specified (assumes annotations are already sorted by score descending)
        if args.top_k is not None:
            annotations = annotations[: args.top_k]

        # Skip if no annotations after filtering
        if len(annotations) == 0:
            continue

        # Collect all boxes for this image (batch processing)
        boxes_xywh = np.array([anot["bbox"] for anot in annotations])
        boxes_xyxy = np.column_stack(
            [
                boxes_xywh[:, 0],
                boxes_xywh[:, 1],
                boxes_xywh[:, 0] + boxes_xywh[:, 2],
                boxes_xywh[:, 1] + boxes_xywh[:, 3],
            ]
        )

        # Single batched prediction for all boxes in this image
        masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if plot_result:
            plot(img_path, boxes_xyxy, masks, imag_id_name)
            imag_id_name += 1

        # Now save results in COCO format. We need to keep the same anot_id than the box to be
        # able to load the mask in the next step.
        # Iterate through annotations and their corresponding masks
        for anot, mask in zip(annotations, masks):
            new_anot = copy.deepcopy(anot)
            new_anot["mask"] = single_mask_to_rle(mask)
            annotations_with_masks.append(new_anot)

    # Save json file with unique name to avoid overwriting existing files
    output_name = args.res.replace(".json", "_with_masks_batched.json")
    save_coco(dataset, annotations_with_masks, output_name)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args, plot_result=False)

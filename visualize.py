#!/usr/bin/env python3
"""
Interactive COCO visualization tool for GT + detection JSONs.

Example:
python3 visualize.py --dataset CLCXray --detector VLDet --score-thr 0.25
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    print(
        "This script requires: opencv-python, numpy, matplotlib. Install with pip.",
        file=sys.stderr,
    )
    raise

# optional pycocotools for mask decoding (if available)
try:
    from pycocotools import mask as maskUtils  # type: ignore

    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_mappings(coco):
    # coco is parsed JSON (dict with keys 'images','annotations','categories')
    images = {img["id"]: img for img in coco.get("images", [])}
    images_by_name = {img["file_name"]: img for img in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_image[a["image_id"]].append(a)
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    return images, images_by_name, anns_by_image, cats


def load_detections(dt_json):
    # dt_json is list of detection dicts or a dict that contains list
    if isinstance(dt_json, dict) and "annotations" in dt_json:
        dets = dt_json["annotations"]
    elif isinstance(dt_json, list):
        dets = dt_json
    else:
        # many COCO detection outputs are list of dicts
        dets = dt_json
    dets_by_image = defaultdict(list)
    for d in dets:
        dets_by_image[d["image_id"]].append(d)
    return dets_by_image


def draw_boxes(img, boxes, labels=None, color=(0, 255, 0), thickness=2, alpha_fill=0.15):
    # boxes: list of [x,y,w,h]
    overlay = img.copy()
    for i, b in enumerate(boxes):
        x, y, w, h = map(int, b)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha_fill, img, 1 - alpha_fill, 0, img)
    for i, b in enumerate(boxes):
        x, y, w, h = map(int, b)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        if labels:
            txt = labels[i]
            cv2.putText(
                img, txt, (x, max(y - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )


def draw_mask(img, seg, color=(0, 0, 255), alpha=0.45):
    # seg can be polygon list or RLE; require pycocotools for RLE
    h, w = img.shape[:2]
    mask = None
    if isinstance(seg, dict) and HAS_PYCOCO:
        # RLE
        mask = maskUtils.decode(seg)
    elif isinstance(seg, list):
        # polygon(s)
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    if mask is None:
        return
    colored = np.zeros_like(img, dtype=np.uint8)
    colored[:] = color
    mask3 = np.repeat(mask[:, :, None].astype(bool), 3, axis=2)
    img[mask3] = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)[mask3]


def find_annotation_file(dataset_path):
    """Find the annotation JSON file in the dataset annotations directory."""
    ann_dir = dataset_path / "annotations"
    if not ann_dir.exists():
        return None

    # Look for full_test.json first
    full_test = ann_dir / "full_test.json"
    if full_test.exists():
        return full_test

    # Otherwise look for other test files
    patterns = ["test.json", "*.json"]
    for pattern in patterns:
        files = list(ann_dir.glob(pattern))
        if files:
            return files[0]

    return None


def visualize_image(
    img_entry,
    anns_by_image,
    dets_by_image,
    cats,
    images_dir,
    score_thr,
    show_masks,
    current_idx,
    total_images,
):
    """Visualize a single image with GT and detection boxes."""
    file_name = img_entry["file_name"]
    image_path = images_dir / file_name

    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read image: {image_path}", file=sys.stderr)
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # collect GT boxes and labels
    gt_anns = anns_by_image.get(img_entry["id"], [])
    gt_boxes = []
    gt_labels = []
    for a in gt_anns:
        if "bbox" in a:
            gt_boxes.append(a["bbox"])
            gt_labels.append(cats.get(a.get("category_id"), str(a.get("category_id"))))

    # collect detections for this image
    dets = dets_by_image.get(img_entry["id"], [])
    dt_boxes = []
    dt_labels = []
    dt_masks = []
    for d in dets:
        if d.get("score", 1.0) < score_thr:
            continue
        if "bbox" in d:
            dt_boxes.append(d["bbox"])
            label = (
                f"{cats.get(d.get('category_id'), d.get('category_id'))} {d.get('score', 0):.2f}"
            )
            dt_labels.append(label)
        if show_masks and "segmentation" in d:
            dt_masks.append(d["segmentation"])

    vis = img.copy()
    # GT: green filled then box
    if gt_boxes:
        draw_boxes(
            vis, gt_boxes, labels=gt_labels, color=(0, 200, 0), thickness=2, alpha_fill=0.15
        )
    # Detections: red fills and outlines
    if dt_boxes:
        draw_boxes(
            vis, dt_boxes, labels=dt_labels, color=(220, 20, 20), thickness=2, alpha_fill=0.12
        )
    # Masks (render on top)
    if show_masks and HAS_PYCOCO:
        # draw GT masks if available in GT
        for a in gt_anns:
            if "segmentation" in a:
                draw_mask(vis, a["segmentation"], color=(0, 200, 0), alpha=0.35)
        # draw detection masks
        for d in dets:
            if d.get("score", 0) >= score_thr and "segmentation" in d:
                draw_mask(vis, d["segmentation"], color=(220, 20, 20), alpha=0.35)
    elif show_masks and not HAS_PYCOCO:
        print(
            "pycocotools not installed; polygons will render but RLE masks won't. Install pycocotools to render RLE.",
            file=sys.stderr,
        )

    return vis, file_name, len(gt_boxes), len(dt_boxes)


def main():
    p = argparse.ArgumentParser(description="Interactive visualization of COCO detections")
    p.add_argument("--dataset", required=True, help="Dataset name (e.g., CLCXray, DvXray, pidray)")
    p.add_argument("--detector", required=True, help="Detector name (e.g., VLDet, CoDet, detic)")
    p.add_argument("--score-thr", type=float, default=0.0, help="Score threshold for detections")
    p.add_argument(
        "--suffix",
        default="",
        help="Optional suffix for detection file (e.g., 'with_masks' to load coco_results.bbox_{dataset}_with_masks_{suffix}.json)",
    )
    p.add_argument(
        "--show-masks", action="store_true", help="Attempt to render segmentation masks if present"
    )
    args = p.parse_args()

    # Construct paths based on dataset and detector
    script_dir = Path(__file__).parent
    dataset_path = script_dir / ".." / "data" / "datasets" / args.dataset
    dataset_path = dataset_path.resolve()

    # Find annotation file
    annotation_file = find_annotation_file(dataset_path)
    if annotation_file is None:
        print(f"Could not find annotation file in {dataset_path / 'annotations'}", file=sys.stderr)
        sys.exit(1)

    images_dir = dataset_path / "test"
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # Construct detection results path with optional suffix
    if args.suffix:
        results_filename = f"coco_results.bbox_{args.dataset}_{args.suffix}.json"
    else:
        results_filename = f"coco_results.bbox_{args.dataset}.json"

    results_path = (
        script_dir / ".." / "results" / "initial_detections" / args.detector / results_filename
    )
    results_path = results_path.resolve()

    if not results_path.exists():
        print(f"Detection results not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading annotations from: {annotation_file}")
    print(f"Loading detections from: {results_path}")
    print(f"Images directory: {images_dir}")

    # Load data
    coco = load_json(annotation_file)
    images, images_by_name, anns_by_image, cats = build_mappings(coco)

    dt_json = load_json(results_path)
    dets_by_image = load_detections(dt_json)

    # Print diagnostic info
    total_dets = sum(len(dets) for dets in dets_by_image.values())
    print(f"Total detections loaded: {total_dets}")
    print(f"Images with detections: {len(dets_by_image)}")
    if dets_by_image:
        sample_img_id = next(iter(dets_by_image))
        sample_dets = dets_by_image[sample_img_id]
        print(f"Sample: Image {sample_img_id} has {len(sample_dets)} detections")
        if sample_dets:
            print(f"  First detection score: {sample_dets[0].get('score', 'N/A')}")

    # Get list of all image IDs
    image_ids = sorted(images.keys())
    if not image_ids:
        print("No images found in annotation file", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_ids)} images")
    print("Controls: Click or press any key to go to next image, 'q' to quit")

    # Interactive visualization
    current_idx = 0
    fig = plt.figure(figsize=(12, 12))

    def on_key_or_click(event):
        nonlocal current_idx
        # Handle keyboard events
        if hasattr(event, "key"):
            if event.key == "q":
                plt.close(fig)
                return
            elif event.key in ["right", "n", " "]:
                current_idx = (current_idx + 1) % len(image_ids)
            elif event.key in ["left", "p"]:
                current_idx = (current_idx - 1) % len(image_ids)
            else:
                # Any other key goes to next
                current_idx = (current_idx + 1) % len(image_ids)
        # Handle mouse click events
        elif hasattr(event, "button"):
            current_idx = (current_idx + 1) % len(image_ids)

        show_current_image()

    def show_current_image():
        nonlocal current_idx
        img_id = image_ids[current_idx]
        img_entry = images[img_id]

        result = visualize_image(
            img_entry,
            anns_by_image,
            dets_by_image,
            cats,
            images_dir,
            args.score_thr,
            args.show_masks,
            current_idx,
            len(image_ids),
        )

        if result is None:
            # Skip to next image if current one fails
            current_idx = (current_idx + 1) % len(image_ids)
            show_current_image()
            return

        vis, file_name, n_gt, n_dt = result

        plt.clf()
        plt.imshow(vis)
        plt.axis("off")

        # Build title with command arguments
        args_str = f"--dataset {args.dataset} --detector {args.detector}"
        if args.suffix:
            args_str += f" --suffix {args.suffix}"
        if args.score_thr > 0:
            args_str += f" --score-thr {args.score_thr}"

        title = (
            f"[{current_idx + 1}/{len(image_ids)}] {file_name}  (GT:{n_gt} Det:{n_dt})\n{args_str}"
        )
        plt.title(title, fontsize=10)
        plt.tight_layout()
        fig.canvas.draw()

    # Connect event handlers
    fig.canvas.mpl_connect("key_press_event", on_key_or_click)
    fig.canvas.mpl_connect("button_press_event", on_key_or_click)

    # Show first image
    show_current_image()
    plt.show()


if __name__ == "__main__":
    main()

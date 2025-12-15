RUN_NAME="cena"

########################################################
#### CHANGE THIS VARIABLES
########################################################
RESULTS_PATH="/results/"
DATASET_PATH="/data/datasets/"
SUPPORT_PATH="/data/datasets/support_full/"
datasets=("CLCXray") # "pidray" "CLCXray" "DET-COMPASS" "HiXray" "DvXray")
detectors=("groundingDINO") # "detic" "CoDet" "VLDet")

########################################################
#### ABLATION STUDY FLAGS
########################################################
# Enable/disable components for ablation experiments
USE_LLM_QUERIES=false    # Google: use LLM (Gemini) to generate diverse queries
QUERY_MODE="direct"      # Query generation mode: "direct" (LLM generates full queries) or "compositional" (LLM generates attribute lists)
N_QUERIES=2             # Number of queries to generate with LLM (per category)
USE_BLENDER=false        # Blender: multi-view 3D rendering
USE_SAM3D=false          # SAM3D: 3D model from mask
IMAGES_PER_CATEGORY=30   # Number of images to retrieve from Google per category
########################################################
RUN_RESULTS_PATH="${RESULTS_PATH}${RUN_NAME}/"

#############################################################
#1) BUILD PROTOTYPES FOR EACH DATASET
# Note that images are retrieved from the internet using the Google API. 
# You must configure it or use our precomputed visual descriptors.
#############################################################


for dataset in "${datasets[@]}"
do
    echo "Building web-powered visual descriptors in dataset: $dataset"

    # # 1) GENERATE HYPONYMYS   (not in the ICCV version)
    # python3 raxo/build_hrchy.py \
    #     --categories  ${DATASET_PATH}${dataset}/annotations/full_test.json \
    #     --method llm \
    #     --out ${RESULTS_PATH}prototypes_web_2/${detector}/${dataset}/ \
    #     --n_hyponyms 3

    # 1.5) GENERATE COLOUR KNOWLEDGE (skip if dataset-specific file already exists)
    # if [ ! -f "${RUN_RESULTS_PATH}colours/${dataset}_colour_per_cat.json" ]; then
    #     echo "Generating colour knowledge database for ${dataset}..."
    #     uv run python raxo/obtain_cat_colours_v3.py \
    #         --cats_gt ${DATASET_PATH}${dataset}/annotations/full_test.json \
    #         --out ${RUN_RESULTS_PATH}colours/ \
    #         --support_set ${SUPPORT_PATH}full_30_support_set_with_masks.json \
    #         --images_path ${SUPPORT_PATH}images
    # else
    #     echo "Colour knowledge file already exists: ${RUN_RESULTS_PATH}colours/${dataset}_colour_per_cat.json"
    # fi

    # # 2) WEB RETRIEVAL (using GroundingDINO standalone)
    # # Build LLM flag if enabled
    # LLM_FLAG=""
    # if [ "$USE_LLM_QUERIES" = true ]; then
    #     LLM_FLAG="--use_llm_queries --query_mode ${QUERY_MODE} --n_queries ${N_QUERIES}"
    #     echo "Using LLM-generated queries (Gemini) with mode: ${QUERY_MODE}, n_queries: ${N_QUERIES}"
    # fi
    
    # CUDA_VISIBLE_DEVICES=0 uv run python raxo/google_image_retrievalv2.py \
    #     --cats ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/categories.json \
    #     --cats_from_gt ${DATASET_PATH}${dataset}/annotations/full_test.json \
    #     --out ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/ \
    #     --box_threshold 0.35 \
    #     --text_threshold 0.25 \
    #     --n ${IMAGES_PER_CATEGORY} \
    #     ${LLM_FLAG}
    
    # # 2.5) SAM3D → BLENDER → MASKS PIPELINE (when enabled)
    # # ALREADY COMPLETED - Commenting out to restart from style transfer
    # if [ "$USE_SAM3D" = true ] && [ "$USE_BLENDER" = true ]; then
    #     echo "=============================================="
    #     echo "Running SAM3D + Blender pipeline for dataset: $dataset"
    #     echo "=============================================="
        
    #     # Define output directories
    #     SAM3D_OUT="${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/sam3d_out/"
    #     BLENDER_OUT="${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/blender_renders/"
    #     BLENDER_MASKS_OUT="${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/blender_masks/"
    #     ANNOTATIONS_FILE="${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/annotations.json"
    #     IMAGES_DIR="${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/imgs"
    #     OUTPUT_ANNOTATIONS="${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/blender_annotations_with_masks.json"
        
    #     # Run the SAM3D + Blender pipeline
    #     CUDA_VISIBLE_DEVICES=0 uv run python raxo/sam3d_blender_pipeline.py \
    #         --annotations "$ANNOTATIONS_FILE" \
    #         --images_dir "$IMAGES_DIR" \
    #         --sam3d_out "$SAM3D_OUT" \
    #         --blender_out "$BLENDER_OUT" \
    #         --blender_masks_out "$BLENDER_MASKS_OUT" \
    #         --output_annotations "$OUTPUT_ANNOTATIONS" \
    #         --n_views 8
        
    #     echo "SAM3D + Blender pipeline complete for dataset: $dataset"
    # fi
    # # echo "Skipping SAM3D + Blender pipeline (already completed)"
    
    # # 3) OBTAIN MASKS (skip if using SAM3D+Blender pipeline)
    # if [ "$USE_SAM3D" = true ] && [ "$USE_BLENDER" = true ]; then
    #     echo "Skipping standard mask generation (using Blender masks instead)"
    # else
    #     CUDA_VISIBLE_DEVICES=0 uv run python raxo/obtain_masks.py \
    #         --gt ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/annotations.json \
    #         --image_path ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/imgs \
    #         --branch unknown
    # fi
    
    # # 4) STYLE TRANSFER v2
    # if [ "$USE_SAM3D" = true ] && [ "$USE_BLENDER" = true ]; then
    #     echo "Running style transfer on Blender renders..."
    #     uv run python raxo/style_transfer_v2.py \
    #         --gt ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/blender_annotations_with_masks.json \
    #         --image_path ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/blender_renders \
    #         --out ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2 \
    #         --colours ${RUN_RESULTS_PATH}colours/${dataset}_colour_per_cat.json
    # else
    #     uv run python raxo/style_transfer_v2.py \
    #         --gt ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/annotations_with_masks.json \
    #         --image_path ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/imgs \
    #         --colours ${RUN_RESULTS_PATH}colours/${dataset}_colour_per_cat.json \
    #         --out ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2
    # fi

    # 5) BUILD the prototypes (using web-specific script for list format)
    # if [ "$USE_SAM3D" = true ] && [ "$USE_BLENDER" = true ]; then
    #     CUDA_VISIBLE_DEVICES=0 uv run python raxo/build_prototypes_ours.py \
    #         --gt ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/blender_annotations_with_masks.json \
    #         --image_path ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2 \
    #         --out ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}_prot_${RUN_NAME}.pt
    # else
    #     CUDA_VISIBLE_DEVICES=0 uv run python raxo/build_prototypes_ours.py \
    #         --gt ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/images/annotations_with_masks.json \
    #         --image_path ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2 \
    #         --out ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}_prot_${RUN_NAME}.pt
    # fi
    
done


#############################################################
# 2) CLASSIFICATION
#############################################################
for detector in "${detectors[@]}"
do
    # Loop through each dataset
    for dataset in "${datasets[@]}"
    do
        echo "Obtaining results with ${RUN_NAME} in dataset: $dataset"

        # Nota para pixray: models/ICCV25_experimentation/prototypes_web/PIXray_prot_sam2_style2_v2.pt
        
        CUDA_VISIBLE_DEVICES=0 uv run python raxo/main_with_masks_prop.py \
            --json_res ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks.json \
            --image_path ${DATASET_PATH}${dataset}/test \
            --prototypes ${RUN_RESULTS_PATH}prototypes_web_2/${dataset}_prot_${RUN_NAME}.pt \
            --nms 0.8 \
            --name ${RUN_NAME} \
            --branch known \
            --out ${RUN_RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}.json

        uv run python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt ${RUN_RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}.json


        uv run python raxo/uncertanty_estimation_fixed.py \
            --dets ${RUN_RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}.json

        uv run python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt ${RUN_RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_uncertanty.json

        
        # Compare web branch vs database branch (requires database branch results)
        uv run python raxo/final_cocoapi.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt_database ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_batched_nms_0.8_our_method_disi_uncertanty.json \
            --cocoDt_web ${RUN_RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_uncertanty.json
        
    done
done



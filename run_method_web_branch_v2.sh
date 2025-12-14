
# UNKNOWN BRANCH with masks

########################################################
#### CHANGE THIS VARIABLES
########################################################
RESULTS_PATH="/results/"
DATASET_PATH="/data/datasets/"
SUPPORT_PATH="/data/datasets/support_full/"
COLOUR_OUTPUT="${RESULTS_PATH}colour_output/"
datasets=("CLCXray") # "pidray" "CLCXray" "DET-COMPASS" "HiXray" "DvXray")
detectors=("groundingDINO") # "detic" "CoDet" "VLDet")

########################################################
#### ABLATION STUDY FLAGS
########################################################
# Enable/disable components for ablation experiments
USE_LLM_QUERIES=false    # Google: use LLM (Gemini) to generate diverse queries
USE_BLENDER=false        # Blender: multi-view 3D rendering (placeholder - not implemented)
USE_SAM3D=false          # SAM3D: 3D model from mask (placeholder - not implemented)
########################################################


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

    # 1.5) GENERATE COLOUR KNOWLEDGE (skip if already exists)
    if [ ! -f "${COLOUR_OUTPUT}colour_knowledge_material_clustering.json" ]; then
        echo "Generating colour knowledge database..."
        uv run python raxo/obtain_cat_colours_v3.py \
            --cats_gt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --out ${COLOUR_OUTPUT} \
            --support_set ${SUPPORT_PATH}full_30_support_set_with_masks.json \
            --images_path ${SUPPORT_PATH}images
    fi

    # 2) WEB RETRIEVAL (using GroundingDINO standalone)
    # Build LLM flag if enabled
    LLM_FLAG=""
    if [ "$USE_LLM_QUERIES" = true ]; then
        LLM_FLAG="--use_llm_queries"
        echo "Using LLM-generated queries (Gemini)"
    fi
    
    CUDA_VISIBLE_DEVICES=0 uv run python raxo/google_image_retrievalv2.py \
        --cats ${RESULTS_PATH}prototypes_web_2/${dataset}/categories.json \
        --n 30 \
        --out ${RESULTS_PATH}prototypes_web_2/${dataset}/images/ \
        --cats_from_gt ${DATASET_PATH}${dataset}/annotations/full_test.json \
        --box_threshold 0.35 \
        --text_threshold 0.25 \
        ${LLM_FLAG}
    
    # 3) OBTAIN MASKS
    CUDA_VISIBLE_DEVICES=0 uv run python raxo/obtain_masks.py \
        --gt ${RESULTS_PATH}prototypes_web_2/${dataset}/images/annotations.json \
        --image_path ${RESULTS_PATH}prototypes_web_2/${dataset}/images/imgs \
        --branch unknown
    
    # 4) STYLE TRANSFER v2
    uv run python raxo/style_transfer_v2.py \
        --gt ${RESULTS_PATH}prototypes_web_2/${dataset}/images/annotations_with_masks.json \
        --image_path ${RESULTS_PATH}prototypes_web_2/${dataset}/images/imgs \
        --out ${RESULTS_PATH}prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2 \
        --colours ${COLOUR_OUTPUT}${dataset}_colour_per_cat.json

    # 5) BUILD the prototypes (using web-specific script for list format)
    CUDA_VISIBLE_DEVICES=0 uv run python raxo/build_prototypes_web.py \
        --gt ${RESULTS_PATH}prototypes_web_2/${dataset}/images/annotations_with_masks.json \
        --image_path ${RESULTS_PATH}prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2 \
        --out ${RESULTS_PATH}prototypes_web_2/${dataset}_prot_sam2_style2_v2.pt
done


#############################################################
# 2) CLASSIFICATION
#############################################################
for detector in "${detectors[@]}"
do
    # Loop through each dataset
    for dataset in "${datasets[@]}"
    do
        echo "Obtaining results with web-branch in dataset: $dataset"

        # Nota para pixray: models/ICCV25_experimentation/prototypes_web/PIXray_prot_sam2_style2_v2.pt
        
        CUDA_VISIBLE_DEVICES=0 uv run python raxo/main_with_masks_prop.py \
            --json_res ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks.json \
            --image_path ${DATASET_PATH}${dataset}/test \
            --prototypes ${RESULTS_PATH}prototypes_web_2/${dataset}_prot_sam2_style2_v2.pt \
            --nms 0.8 \
            --name web_branch_v2 \
            --branch known

        uv run python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2.json


        uv run python raxo/uncertanty_estimation_fixed.py \
            --dets ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2.json

        uv run python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2.json

        
        # Compare web branch vs database branch (requires database branch results)
        # Uses the most recent database branch results: _disi_uncertanty.json
        uv run python raxo/final_cocoapi.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt_database ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_disi_uncertanty.json \
            --cocoDt_web ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2_uncertanty.json
        
    done
done



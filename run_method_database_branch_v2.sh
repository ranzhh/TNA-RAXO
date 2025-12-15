# KNOWN BRANCH with masks

########################################################
#### CHANGE THIS VARIABLES
########################################################
RESULTS_PATH="/results"
DATASET_PATH="/data/datasets"
datasets=( "CLCXray") # "pidray", "DvXray") # "DET-COMPASS" "HiXray" "CLCXray" "PiXray")
detectors=("groundingDINO") # "detic" "CoDet" "VLDet")
########################################################

# echo "----OBTAINING BASELINE RESULTS----"

# for detector in "${detectors[@]}"
# do
#     for dataset in "${datasets[@]}"
#     do
#     echo "Baseline results in dataset $dataset" with detector $detector
#     python3 raxo/cocoapi_2.py \
#         --cocoGt ${DATASET_PATH}/${dataset}/annotations/full_test.json \
#         --cocoDt ${RESULTS_PATH}/initial_detections/${detector}/coco_results.bbox_${dataset}.json
#     done
# done
# echo
# echo

#############################################################
# 2) TO BUILD VISUAL DESCRIPTORS (only needed if you do not use the precomputed ones)
#############################################################

# echo "----BUILDING IN-HOUSE VISUAL DESCRIPTORS----"
# CUDA_VISIBLE_DEVICES=3 python raxo/obtain_masks.py \
#      --gt ${DATASET_PATH}support_full/full_30_support_set.json \
#      --image_path ${DATASET_PATH}support_full/images \

# CUDA_VISIBLE_DEVICES=3 python raxo/build_prototypes_with_masks_prop.py \
#      --gt ${DATASET_PATH}support_full/full_30_support_set_with_masks.json \
#      --image_path ${DATASET_PATH}support_full/images \
#      --out ${RESULTS_PATH}visual_descriptors/known_prototypes_sam2.pt



#############################################################
# 3) CLASSIFICATION
#############################################################



for detector in "${detectors[@]}"
do
    for dataset in "${datasets[@]}"
    do

        echo "Obtaining results with database-branch in dataset: $dataset"

        # 1) # Extract masks. See extract_masks_inference.sh and extract_masks_inference_run.sh
        CUDA_VISIBLE_DEVICES=0 uv run python raxo/obtain_masks_res_inference.py \
            --gt ${DATASET_PATH}/${dataset}/annotations/full_test.json \
            --res ${RESULTS_PATH}/initial_detections/${detector}/coco_results.bbox_${dataset}.json \
            --image_path ${DATASET_PATH}/${dataset}/test/
            # --limit 1


        # 2) clasification
        CUDA_VISIBLE_DEVICES=0 uv run python raxo/main_with_masks_prop.py \
            --json_res ${RESULTS_PATH}/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_batched.json \
            --image_path ${DATASET_PATH}/${dataset}/test \
            --prototypes ${RESULTS_PATH}/visual_descriptors/known_prototypes_sam2.pt \
            --nms 0.8 \
            --name disi \
            --branch known

        uv run python raxo/uncertanty_estimation_fixed.py \
            --dets ${RESULTS_PATH}/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_batched_nms_0.8_our_method_disi.json

        # Theese are the 100/0 results (using only the in-house samples)
        uv run python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}/${dataset}/annotations/full_test.json \
            --cocoDt ${RESULTS_PATH}/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_batched_nms_0.8_our_method_disi.json

    done
done
echo
echo
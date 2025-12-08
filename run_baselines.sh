# KNOWN BRANCH with masks

########################################################
#### CHANGE THIS VARIABLES
########################################################
RESULTS_PATH="../results"
DATASET_PATH="../data/datasets"
datasets=("CLCXray" "pidray" "DvXray") # "PiXray" "DET-COMPASS" "HiXray")
detectors=("VLDet")
########################################################

# echo "----OBTAINING BASELINE RESULTS----"

for detector in "${detectors[@]}"
do
    for dataset in "${datasets[@]}"
    do
    echo "Baseline results in dataset $dataset" with detector $detector
    python3 raxo/cocoapi_2.py \
        --cocoGt ${DATASET_PATH}/${dataset}/annotations/full_test.json \
        --cocoDt ${RESULTS_PATH}/initial_detections/${detector}/coco_results.bbox_${dataset}.json
    done
done
echo
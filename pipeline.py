import argparse
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

DEFAULT_RESULTS_PATH = "../results/"
DEFAULT_DATASET_PATH = "../data/datasets/"

class RAXODatasets(Enum):
    CLCXRAY = "clcxray"
    DETCOMPASS = "detcompass"
    DVXRAY = "dvxray"
    HIXRAY = "hixray"
    PIDRAY = "pidray"
    PIXRAY = "pixray"

class RAXOBackbones(Enum):
    GROUNDINGDINO = "groundingdino"
    CODET = "codet"
    DETIC = "detic"
    VLDET = "vldet"
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAXO experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[dataset.value for dataset in RAXODatasets],
        required=True,
        help="Dataset to use for the experiment."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        choices=[backbone.value for backbone in RAXOBackbones],
        required=True,
        help="Backbone model to use."
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help="Path to store the results."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the datasets."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Running RAXO with dataset: {args.dataset} and backbone: {args.backbone}")
    
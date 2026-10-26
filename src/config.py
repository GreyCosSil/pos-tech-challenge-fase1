from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME
# DATASET_PATH = DATA_DIR / "raw" / DATASET_FILENAME

MODEL_PATH = MODELS_DIR / "telco_churn_best_model.joblib"
EXPERIMENT_PATH = MODELS_DIR / "telco_churn_experiment.json"
COMPARISON_PATH = MODELS_DIR / "telco_churn_model_comparison.csv"

TARGET_COLUMN = "Churn"
POSITIVE_CLASS = "Yes"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
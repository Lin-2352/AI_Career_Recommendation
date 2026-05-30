from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_ARTIFACT_PATH = MODELS_DIR / "career_model.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
MASTER_DATA_PATH = PROCESSED_DATA_DIR / "master_career_data.csv"

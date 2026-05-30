from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_recommender.data_pipeline import build_master_dataset
from career_recommender.modeling import save_training_outputs, train_career_model
from career_recommender.paths import METRICS_PATH, MODEL_ARTIFACT_PATH


def main() -> None:
    master = build_master_dataset()
    artifact, metrics = train_career_model(master)
    save_training_outputs(artifact, metrics)
    print(f"Saved model artifact to {MODEL_ARTIFACT_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Accuracy: {metrics['metadata']['accuracy']}")


if __name__ == "__main__":
    main()

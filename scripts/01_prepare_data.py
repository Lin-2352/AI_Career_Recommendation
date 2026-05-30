from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_recommender.data_pipeline import build_master_dataset
from career_recommender.paths import MASTER_DATA_PATH


def main() -> None:
    master = build_master_dataset()
    print(f"Prepared {len(master)} records at {MASTER_DATA_PATH}")
    print(f"Career labels: {master['target_career'].nunique()}")
    print("Sources:")
    print(master["source"].value_counts().to_string())


if __name__ == "__main__":
    main()

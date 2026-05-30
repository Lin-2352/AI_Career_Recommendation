# Architecture

## Overview

The system is a local-first machine learning application with a Streamlit user interface. It prepares career profile data from two CSV datasets, trains a text classification model, stores a deployable artifact, and serves ranked recommendations from the saved model.

## Layers

| Layer | Files | Responsibility |
| --- | --- | --- |
| Raw data | `data/raw/` | Stores the two source CSV datasets used for training. |
| Data preparation | `career_recommender/data_pipeline.py`, `scripts/01_prepare_data.py` | Validates source columns, aligns schemas, normalizes profile fields, and creates the master training table. |
| Model training | `career_recommender/modeling.py`, `scripts/02_train_model.py` | Builds the TF-IDF and Random Forest pipeline, evaluates holdout performance, and writes artifacts. |
| Inference helpers | `career_recommender/recommendation.py` | Builds user profile text, ranks predictions, and generates skill suggestions. |
| User interface | `app.py` | Captures profile inputs and displays ranked career matches, confidence, signals, and skill recommendations. |
| Documentation | `README.md`, `docs/` | Explains setup, workflow, architecture, and user operation. |

## Data Flow

1. `data/raw/career_guidance.csv` and `data/raw/career_recommendation.csv` are loaded.
2. Dataset-specific fields are mapped into a shared schema.
3. Education, skills, interests, and age band are combined into `profile_text`.
4. `profile_text` becomes the model input and `target_career` becomes the label.
5. The trained pipeline is saved to `models/career_model.joblib`.
6. Streamlit loads the artifact and ranks career matches using `predict_proba`.

## Model Design

The model uses `TfidfVectorizer` for text features and `LogisticRegression` for classification. This pairing performs well on sparse text profiles and produces ranked class probabilities for the UI. The pipeline is saved as one artifact so deployment does not need to rebuild preprocessing logic at runtime. Evaluation tracks top-1 accuracy plus top-3 and top-5 match rates because the product presents ranked recommendations.

## Artifact Strategy

Generated model and metrics files are intentionally small and versioned:

- `models/career_model.joblib`
- `reports/model_metrics.json`

Generated processed CSV files are ignored because they can be rebuilt from the raw datasets.

## Security and Repository Hygiene

The repository ignores virtual environments, IDE folders, local secrets, Kaggle credentials, cache folders, temporary outputs, logs, and generated processed data. Streamlit secrets must be stored locally or in the hosting provider, not in Git.

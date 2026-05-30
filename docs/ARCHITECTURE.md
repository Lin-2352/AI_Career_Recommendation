# Architecture

## Overview

The system is a local-first machine learning application with a Streamlit user interface. It prepares career profile data from active CSV datasets, trains a text classification model, stores a deployable artifact, and serves ranked recommendations from the saved model.

## Layers

| Layer | Files | Responsibility |
| --- | --- | --- |
| Raw data | `data/raw/` | Stores active source CSV datasets used for training. |
| Data preparation | `backend/career_recommender/data_pipeline.py`, `scripts/01_prepare_data.py` | Validates source columns, aligns schemas, normalizes profile fields, and creates the master training table. |
| Model training | `backend/career_recommender/modeling.py`, `scripts/02_train_model.py` | Builds the TF-IDF and Multinomial Naive Bayes pipeline, evaluates holdout performance, and writes artifacts. |
| Inference helpers | `backend/career_recommender/recommendation.py` | Builds profile text, applies model probabilities, reranks matches with profile alignment, and generates skill suggestions. |
| User interface | `frontend/streamlit_app.py`, `frontend/ui_helpers.py`, `app.py` | Captures profile inputs and displays ranked career matches, fit score, model probability, signals, and skill recommendations. |
| Tests | `tests/` | Verifies data integrity, validation behavior, recommendation probes, confidence labeling, and Streamlit startup. |
| Documentation | `README.md`, `docs/` | Explains setup, workflow, architecture, and user operation. |

## Data Flow

1. `data/raw/career_guidance.csv`, `data/raw/career_recommendation.csv`, and `data/raw/student_scores_sanitized.csv` are loaded.
2. Dataset-specific fields are mapped into a shared schema.
3. Education, skills, interests, and age band are combined into `profile_text`.
4. `profile_text` becomes the model input and `target_career` becomes the label.
5. The trained pipeline is saved to `models/career_model.joblib`.
6. Streamlit loads the artifact and ranks career matches using model probabilities plus profile-alignment scoring.

## Model Design

The model uses `TfidfVectorizer` for text features and `MultinomialNB` for classification. This pairing is effective for non-negative sparse text vectors and produces ranked class probabilities for the UI. The UI then reranks candidates with a profile-alignment score so explicit user skills can correct weak model-only ordering. Evaluation tracks top-1 accuracy plus top-3 and top-5 match rates because the product presents ranked recommendations.

## Frontend and Backend Split

`backend/` owns data, training, artifact paths, and recommendation logic. `frontend/` owns Streamlit rendering and UI validation helpers. The root `app.py` remains a thin deployment launcher so `streamlit run app.py` and Streamlit Community Cloud stay simple.

## Artifact Strategy

Generated model and metrics files are intentionally small and versioned:

- `models/career_model.joblib`
- `reports/model_metrics.json`

Generated processed CSV files are ignored because they can be rebuilt from the raw datasets.

## Security and Repository Hygiene

The repository ignores virtual environments, IDE folders, local secrets, Kaggle credentials, cache folders, temporary outputs, logs, and generated processed data. Streamlit secrets must be stored locally or in the hosting provider, not in Git.

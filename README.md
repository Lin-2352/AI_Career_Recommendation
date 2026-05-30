# AI Career Recommendation

AI Career Recommendation is a Streamlit and scikit-learn application that recommends career paths from a user's education, skills, and interests. It combines two career datasets, trains a TF-IDF and Logistic Regression model, and serves ranked recommendations through a polished local web app.

## Features

- Unified data preparation pipeline for both source datasets.
- TF-IDF text vectorization over education, skills, interests, and age band.
- Logistic Regression classifier with holdout evaluation.
- Ranked top career matches with confidence values.
- Top-3 and top-5 match-rate metrics for ranked recommendation quality.
- Skill suggestions based on the highest-ranked career.
- Streamlit interface with model metadata and training summary.
- Git-ready project hygiene with focused `.gitignore` rules.

## Project Structure

```text
AI_Career_Recommendation/
├── app.py
├── career_recommender/
│   ├── data_pipeline.py
│   ├── modeling.py
│   ├── paths.py
│   └── recommendation.py
├── data/
│   └── raw/
│       ├── career_guidance.csv
│       └── career_recommendation.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT_WORKFLOW.md
│   └── USER_GUIDE.md
├── models/
│   └── career_model.joblib
├── reports/
│   └── model_metrics.json
├── scripts/
│   ├── 01_prepare_data.py
│   └── 02_train_model.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Setup

Activate the existing environment:

```powershell
.\career_env\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Prepare Data

The repository expects:

- `data/raw/career_guidance.csv`
- `data/raw/career_recommendation.csv`

Build the processed training table:

```powershell
python scripts/01_prepare_data.py
```

## Train the Model

```powershell
python scripts/02_train_model.py
```

Training writes:

- `models/career_model.joblib`
- `reports/model_metrics.json`

## Run Locally

```powershell
streamlit run app.py
```

Then open the URL shown by Streamlit, usually `http://localhost:8501`.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)

## Git Workflow

The first stable release lives on `main`. Future work should be developed on named branches created from the current parent branch, then pushed separately before merging.

Recommended branch examples:

- `feature/model-comparison`
- `feature/user-profile-export`
- `fix/retraining-validation`
- `docs/streamlit-deployment`

## Security

Do not commit virtual environments, IDE folders, `.env` files, Streamlit secrets, Kaggle credentials, logs, cache folders, or private user data. The `.gitignore` file is configured for those cases.

## Limitations

The recommendation is based on the available training labels and text features. It should be used as a career exploration aid, not as a final career decision or hiring assessment.

# AI Career Recommendation

AI Career Recommendation is a Streamlit and scikit-learn application that recommends career paths from a user's education, skills, and interests. It combines career datasets, trains a TF-IDF and Multinomial Naive Bayes model, and serves ranked recommendations through a polished local web app.

## Features

- Backend package for data preparation, modeling, ranking, and recommendation helpers.
- Frontend package for the Streamlit interface and UI validation helpers.
- TF-IDF text vectorization over education, skills, interests, and age band.
- Multinomial Naive Bayes classifier with holdout evaluation.
- Profile-aware reranking that blends model probability with explicit skill alignment.
- Ranked top career matches with fit score, model probability, and profile alignment.
- Top-3 and top-5 match-rate metrics for ranked recommendation quality.
- Skill suggestions based on the highest-ranked career.
- Automated tests for data integrity, validation, recommendation probes, and Streamlit startup.
- Git-ready project hygiene with focused `.gitignore` rules.

## Project Structure

```text
AI_Career_Recommendation/
|-- app.py
|-- backend/
|   `-- career_recommender/
|       |-- data_pipeline.py
|       |-- modeling.py
|       |-- paths.py
|       `-- recommendation.py
|-- frontend/
|   |-- streamlit_app.py
|   `-- ui_helpers.py
|-- data/
|   `-- raw/
|       |-- career_guidance.csv
|       |-- career_recommendation.csv
|       `-- student_scores_sanitized.csv
|-- docs/
|-- models/
|-- reports/
|-- scripts/
|-- tests/
|-- requirements.txt
`-- .streamlit/
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
- `data/raw/student_scores_sanitized.csv`

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

## Test

```powershell
python -m unittest discover -s tests -v
```

## Run Locally

```powershell
streamlit run app.py
```

Then open the URL shown by Streamlit, usually `http://localhost:8501`.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)
- [Datasets](docs/DATASETS.md)

## Git Workflow

The first stable release lives on `main`. Future work should be developed on named branches created from the current parent branch, then pushed separately before merging.

Recommended branch examples:

- `feature/model-comparison`
- `feature/user-profile-export`
- `fix/retraining-validation`
- `docs/streamlit-deployment`

## Security

Do not commit virtual environments, IDE folders, `.env` files, Streamlit secrets, Kaggle credentials, logs, cache folders, or private user data. The `.gitignore` file excludes the original student-score file because it contains names and email addresses.

## Limitations

The recommendation is based on the available training labels and text features. It should be used as a career exploration aid, not as a final career decision or hiring assessment.

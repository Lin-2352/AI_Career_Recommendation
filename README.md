# AI Career Recommendation

AI Career Recommendation is a local-first Streamlit SaaS-style application for career matching, student placement planning, professional pivot analysis, ATS resume optimization, interview preparation, and Kimi-powered dataset-grounded chat.

The app uses a trained scikit-learn TF-IDF and Multinomial Naive Bayes model for career probabilities, deterministic analytics for ATS and interview checks, and an API firewall that rotates OpenAI-compatible keys before quota exhaustion.

## Features

- Five-tab Streamlit workspace: Career Match, Student Hub, Pivot Dashboard, Resume ATS, and Interview & Chat.
- Top-3 career recommendations from `predict_proba()` with model confidence, market-adjusted confidence, and profile alignment.
- Skills gap analyzer showing skills already present and critical missing skills for the target role.
- Student placement probability, salary tier forecasting, degree-to-occupation mapping, and study habit viability scoring.
- Professional pivot scoring, flight-risk detection, and lateral transition mapping through cosine similarity.
- PDF, DOCX, TXT, and image resume parsing with ATS keyword scoring, formatting checks, weak verb detection, impact-gap detection, summaries, and PDF resume export.
- Interview question generation, STAR response scoring, filler-word analysis, recorded audio capture, and optional OpenAI transcription.
- Kimi `kimi-k2.6` chatbot wired through the standard `openai` Python client and grounded in local raw datasets.
- Thread-safe API key rotation with 80 percent soft caps, mocked tests, rotating logs, Docker, and Compose support.

## Project Structure

```text
AI_Career_Recommendation/
|-- app.py
|-- backend/
|   |-- chatbot/
|   |   `-- rag_engine.py
|   |-- core/
|   |   |-- api_manager.py
|   |   |-- document_parser.py
|   |   `-- logger.py
|   `-- career_recommender/
|       |-- data_pipeline.py
|       |-- interview_prep.py
|       |-- modeling.py
|       |-- paths.py
|       |-- recommendation.py
|       `-- resume_analyzer.py
|-- frontend/
|   |-- assets/custom_style.css
|   |-- streamlit_app.py
|   `-- ui_helpers.py
|-- tests/
|-- docs/
|-- Dockerfile
|-- docker-compose.yml
|-- run_client.bat
|-- run_client.sh
`-- requirements.txt
```

## Setup

```powershell
.\career_env\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set keys only when you want external AI features:

```text
MOONSHOT_KEYS=["your_kimi_key"]
OPENAI_KEYS=["your_openai_key_for_audio_transcription"]
```

The core app, model recommendations, ATS checks, and tests work without live API calls.

## Run Locally

```powershell
streamlit run app.py
```

Open the local Streamlit URL, usually `http://localhost:8501`.

## Docker

```powershell
.\run_client.bat
```

Unix:

```bash
./run_client.sh
```

Both scripts check Docker, build the image, and run Compose on port `8501`.

## Data And Training

Tracked training inputs are:

- `data/raw/career_guidance.csv`
- `data/raw/career_recommendation.csv`
- `data/raw/student_scores_sanitized.csv`

Additional raw CSVs can be placed in `data/raw/` for local RAG and analysis, but new raw CSVs are ignored by Git by default to avoid committing large or sensitive datasets.

```powershell
python scripts/01_prepare_data.py
python scripts/02_train_model.py
```

Training writes `models/career_model.joblib` and `reports/model_metrics.json`.

## Test

```powershell
python -m pytest -q
```

The API suite uses mocked OpenAI-compatible clients and never triggers live endpoints.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)
- [Datasets](docs/DATASETS.md)

## Repository Hygiene

The repository ignores virtual environments, `.env` files, Streamlit secrets, logs, caches, generated processed data, and newly downloaded raw CSVs. Keep private datasets and API keys local.

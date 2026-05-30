# Architecture

## Overview

The system is a local-first Streamlit application with a Python backend. It combines deterministic analytics, a trained career classifier, document parsing, dataset-grounded retrieval, and optional OpenAI-compatible LLM calls behind a quota-protecting API manager.

## Layers

| Layer | Files | Responsibility |
| --- | --- | --- |
| Core services | `backend/core/` | API key pooling, token accounting, rotating logs, PDF/DOCX/TXT/image parsing. |
| Career intelligence | `backend/career_recommender/` | Data preparation, model training, probabilistic ranking, student forecasting, pivot scoring, resume analysis, and interview preparation. |
| Chatbot and RAG | `backend/chatbot/rag_engine.py` | Loads local CSV datasets, builds a vector index, retrieves context, and calls Kimi through the standard OpenAI client path. |
| Frontend | `frontend/streamlit_app.py`, `frontend/assets/custom_style.css` | Five-tab user workspace, validation, session state, charts, uploads, chat, audio capture, and ATS export. |
| Tests | `tests/` | Unit and smoke tests for API rotation, parsing, recommendation behavior, ATS math, STAR checks, and Streamlit boot. |
| Deployment | `Dockerfile`, `docker-compose.yml`, `run_client.*` | Reproducible Streamlit container on port `8501`. |

## Data Flow

1. Raw CSVs in `data/raw/` are validated and normalized by `data_pipeline.py`.
2. Education, age band, skills, and interests become `profile_text`.
3. `modeling.py` trains TF-IDF plus Multinomial Naive Bayes and stores the artifact.
4. `recommendation.py` calls `predict_proba()`, reranks with profile alignment, and surfaces top matches.
5. Resume uploads go through `document_parser.py` and `resume_analyzer.py`.
6. Chat questions use `DatasetRAGEngine` to retrieve local dataset context before the Kimi request.
7. All external AI calls use `APIManager`, which estimates tokens, rotates at 80 percent soft cap, and retires keys on quota or rate failures.

## API Firewall

`backend/core/api_manager.py` loads key pools from `.env`, supports JSON arrays such as `MOONSHOT_KEYS=["key1","key2"]`, and protects providers with:

- `threading.Lock()` around key state changes.
- Token estimates using `tiktoken` with a deterministic fallback if tokenizer initialization stalls.
- 80 percent soft-cap rotation before a request is sent.
- Error interception for quota and rate failures, including `402`, `429`, and `insufficient_quota`.
- Critical logging when a key is retired or rotated.
- Mocked tests that do not call live endpoints.

## UI Architecture

The frontend is intentionally thin. It collects validated inputs, stores multi-step state in `st.session_state`, and delegates all scoring, parsing, ranking, and API work to backend modules. The root `app.py` remains a launcher so `streamlit run app.py` is stable across local and container environments.

## Dataset Strategy

The three curated CSV files required for training are tracked. Additional downloaded raw datasets are local-only by default and can still be used by RAG because the engine scans `data/raw/*.csv` at runtime.

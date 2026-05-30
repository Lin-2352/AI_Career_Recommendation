# User Guide

## Running The App

```powershell
.\career_env\Scripts\activate
streamlit run app.py
```

For Docker:

```powershell
.\run_client.bat
```

Open `http://localhost:8501`.

## Career Match

1. Enter age and education.
2. Select hard skills, soft skills, and interests.
3. Add market context with city development index and company size.
4. Select **Generate top matches**.

The result shows the top three careers, raw model confidence, market-adjusted confidence, profile alignment, skills you already have, and critical missing skills.

## Student Hub

Use the placement forecaster with GPA, college tier, and competencies such as DSA, web development, SQL, and internship experience. The degree mapping section reads local raw datasets and plots where matching fields of study lead. The study habit tool estimates viability from weekly study hours, subject scores, and available baseline correlations.

## Pivot Dashboard

The flight-risk calculator estimates pivot urgency from satisfaction, work-life balance, and years of experience. The lateral transition table compares current skills against career skill maps and ranks adjacent moves by cosine similarity.

## Resume ATS

Upload PDF, DOCX, TXT, or supported image resumes. PDFs use `pdfplumber`, Word files use `python-docx`, text files are parsed locally, and image resumes require a configured Kimi key.

The analyzer provides:

- ATS match score using found keywords divided by total required keywords.
- Keywords found and keyword gaps.
- Formatting warnings.
- Weak action verb suggestions.
- Bullets missing measurable impact.
- Optional role-specific summary generation.
- ATS-friendly PDF resume export.

## Interview And Chat

Generate interview questions from the selected role and skills. Paste a STAR answer to check whether Situation, Task, Action, and Result are present and whether the Result includes a metric.

The audio sandbox records a mock answer through Streamlit. OpenAI transcription is optional and requires `OPENAI_KEYS`; filler-word analysis can run on either API transcript text or pasted transcript text.

The chatbot uses local dataset retrieval and Kimi when `MOONSHOT_KEYS` is configured. Resume files can be attached in the chat input flow.

## Environment Keys

Copy `.env.example` to `.env` and set only the providers you plan to use.

```text
MOONSHOT_KEYS=["your_kimi_key"]
OPENAI_KEYS=["your_openai_key"]
```

Leave keys empty for offline model, student, pivot, ATS, and interview heuristic features.

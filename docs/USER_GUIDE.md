# User Guide

## Running the App

From the project root:

```powershell
.\career_env\Scripts\activate
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Creating a Recommendation

1. Choose your age.
2. Select the closest education level or field.
3. Add skill starters or type your own skills.
4. Add interest starters or type your own interests.
5. Select **Generate recommendation**.

The app returns the best career match, ranked alternatives, confidence values, detected profile signals, and suggested skills to strengthen.

## Interpreting Confidence

Confidence is the model's probability score for a class compared with other trained labels. It is useful for ranking options, but it is not a guarantee of job fit, hiring success, or academic placement.

## Retraining

After replacing or editing the raw CSV files, retrain the model:

```powershell
python scripts/02_train_model.py
```

Then restart Streamlit so the new artifact is loaded.

## Expected Data Files

The training scripts expect:

- `data/raw/career_guidance.csv`
- `data/raw/career_recommendation.csv`

Both files are public career recommendation datasets and should remain free of private user records.

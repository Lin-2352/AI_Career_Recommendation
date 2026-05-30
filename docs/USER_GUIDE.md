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

Required fields are marked with a red star. Optional profile fields can improve the match when they are relevant, but they are not required. The app returns the best career match, ranked alternatives, fit score, model probability, detected profile signals, and suggested skills to strengthen.

## Interpreting Confidence

Fit score is the top recommendation's share among the displayed ranked matches after combining model probability with profile alignment. Model probability is the classifier's raw probability for that career label. Both values are useful for comparison, but neither is a guarantee of job fit, hiring success, or academic placement.

Weak or unfamiliar inputs may still produce a ranked list, but the app labels those cases as exploratory when the profile has low alignment with known career skill patterns.

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
- `data/raw/student_scores_sanitized.csv`

Training files should remain free of private user records. The original student-score file is sanitized before use and should not be committed.

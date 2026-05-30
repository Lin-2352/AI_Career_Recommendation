import re

import pandas as pd

EDUCATION_PLACEHOLDER = "Select education or field"


def split_terms(value: str) -> list[str]:
    return [term.strip() for term in re.split(r"[,;|/\n]+", value or "") if term.strip()]


def valid_text(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", value or ""))


def validate_inputs(education: str, skills: str, interests: str) -> list[str]:
    errors = []
    if education == EDUCATION_PLACEHOLDER:
        errors.append("Education or field is required.")
    if len(split_terms(skills)) < 2 or not valid_text(skills):
        errors.append("Add at least two meaningful skills.")
    if len(split_terms(interests)) < 1 or not valid_text(interests):
        errors.append("Add at least one meaningful professional interest.")
    return errors


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def confidence_band(fit_share: float, model_probability: float, profile_alignment: float) -> str:
    if profile_alignment < 0.1:
        return "Exploratory match"
    if fit_share >= 0.38 and (model_probability >= 0.12 or profile_alignment >= 0.4):
        return "Strong recommendation"
    if fit_share >= 0.24 and (model_probability >= 0.06 or profile_alignment >= 0.25):
        return "Reliable ranked match"
    return "Exploratory match"


def source_table(metrics: dict) -> pd.DataFrame:
    distribution = metrics.get("source_distribution", {})
    return pd.DataFrame(distribution.items(), columns=["Source", "Records"]).sort_values("Records", ascending=False)

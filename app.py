from html import escape
import json
import re

import pandas as pd
import streamlit as st

from career_recommender.modeling import load_artifact, predict_top_careers
from career_recommender.paths import METRICS_PATH, MODEL_ARTIFACT_PATH
from career_recommender.recommendation import (
    EDUCATION_OPTIONS,
    INTEREST_STARTERS,
    SKILL_STARTERS,
    extract_profile_signals,
    profile_from_inputs,
    skill_suggestions,
)

st.set_page_config(
    page_title="AI Career Recommendation",
    page_icon=":material/work:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1220px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .page-header {
        border-bottom: 1px solid #D6DAE2;
        padding: 0.25rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: #2563EB;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .headline {
        color: #0F172A;
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.12;
        margin-bottom: 0.45rem;
    }
    .subhead {
        color: #475569;
        max-width: 760px;
        font-size: 1rem;
    }
    .field-label {
        align-items: center;
        color: #111827;
        display: flex;
        font-size: 0.92rem;
        font-weight: 650;
        gap: 0.35rem;
        margin: 0.25rem 0 0.28rem 0;
    }
    .required-star {
        color: #DC2626;
        font-weight: 800;
    }
    .optional-pill {
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        color: #64748B;
        font-size: 0.68rem;
        font-weight: 650;
        padding: 0.03rem 0.42rem;
        text-transform: uppercase;
    }
    .result-panel {
        background: #FFFFFF;
        border: 1px solid #D8DEE9;
        border-radius: 8px;
        padding: 1rem 1.1rem;
    }
    .result-kicker {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }
    .result-career {
        color: #0F172A;
        font-size: 1.9rem;
        font-weight: 780;
        line-height: 1.16;
    }
    .confidence-band {
        color: #2563EB;
        font-size: 0.96rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .signal-chip {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        color: #1F2937;
        display: inline-block;
        font-size: 0.84rem;
        margin: 0.16rem;
        padding: 0.22rem 0.58rem;
    }
    .small-note {
        color: #64748B;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifact() -> dict:
    if not MODEL_ARTIFACT_PATH.exists():
        st.error("Model artifact is missing. Run `python scripts/02_train_model.py` first.")
        st.stop()
    return load_artifact()


@st.cache_data
def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def render_label(text: str, required: bool) -> None:
    marker = '<span class="required-star">*</span>' if required else '<span class="optional-pill">Optional</span>'
    st.markdown(f'<div class="field-label">{escape(text)} {marker}</div>', unsafe_allow_html=True)


def split_terms(value: str) -> list[str]:
    return [term.strip() for term in re.split(r"[,;|\n]+", value) if term.strip()]


def valid_text(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", value))


def validate_inputs(education: str, skills: str, interests: str) -> list[str]:
    errors = []
    if education == "Select education or field":
        errors.append("Education or field is required.")
    if len(split_terms(skills)) < 2 or not valid_text(skills):
        errors.append("Add at least two meaningful skills.")
    if len(split_terms(interests)) < 1 or not valid_text(interests):
        errors.append("Add at least one meaningful professional interest.")
    return errors


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def confidence_band(best_probability: float, second_probability: float | None, top_five_rate: float | None) -> str:
    margin = best_probability - second_probability if second_probability is not None else best_probability
    if best_probability >= 0.35 or margin >= 0.12:
        return "Strong recommendation"
    if best_probability >= 0.16 or margin >= 0.05 or (top_five_rate is not None and top_five_rate >= 0.85):
        return "Reliable ranked match"
    return "Exploratory match"


def source_table(metrics: dict) -> pd.DataFrame:
    distribution = metrics.get("source_distribution", {})
    return pd.DataFrame(distribution.items(), columns=["Source", "Records"]).sort_values("Records", ascending=False)


artifact = get_artifact()
metrics = get_metrics()
metadata = artifact.get("metadata", {})
top_five_rate = metadata.get("top_5_accuracy")

with st.sidebar:
    st.header("Model Status")
    st.caption(metadata.get("estimator", "Classifier"))
    st.metric("Training records", metadata.get("training_records", "N/A"))
    st.metric("Career labels", metadata.get("target_count", "N/A"))
    if top_five_rate is not None:
        st.metric("Top-5 match rate", percent(top_five_rate))
    if metadata.get("accuracy") is not None:
        st.metric("Top-1 accuracy", percent(metadata["accuracy"]))
    st.caption(f"Trained UTC: {metadata.get('trained_at_utc', 'N/A')}")
    st.divider()
    st.caption("Predictions are career exploration guidance and should be reviewed with human judgment.")

st.markdown(
    """
    <div class="page-header">
        <div class="eyebrow">Career intelligence workspace</div>
        <div class="headline">AI Career Recommendation</div>
        <div class="subhead">Build a profile from education, skills, and interests, then review ranked career matches from the trained model.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

form_col, output_col = st.columns([1, 1], gap="large")

with form_col:
    with st.container(border=True):
        st.subheader("Candidate Profile")
        with st.form("career_profile_form"):
            render_label("Age", True)
            age = st.slider("Age", min_value=16, max_value=70, value=22, label_visibility="collapsed")

            render_label("Education or field", True)
            education = st.selectbox(
                "Education or field",
                ["Select education or field", *EDUCATION_OPTIONS],
                index=0,
                label_visibility="collapsed",
            )

            render_label("Skills", True)
            starter_skills = st.multiselect(
                "Skill starters",
                SKILL_STARTERS,
                placeholder="Select known skills",
                label_visibility="collapsed",
            )
            custom_skills = st.text_area(
                "Additional skills",
                placeholder="Python, SQL, machine learning, dashboarding",
                max_chars=500,
                height=92,
                label_visibility="collapsed",
            )

            render_label("Professional interests", True)
            starter_interests = st.multiselect(
                "Interest starters",
                INTEREST_STARTERS,
                placeholder="Select interest areas",
                label_visibility="collapsed",
            )
            custom_interests = st.text_area(
                "Additional interests",
                placeholder="AI products, healthcare analytics, software systems",
                max_chars=420,
                height=92,
                label_visibility="collapsed",
            )

            optional_left, optional_right = st.columns(2)
            with optional_left:
                render_label("Experience level", False)
                experience_level = st.selectbox(
                    "Experience level",
                    ["Not specified", "Student", "Internship", "Entry level", "Mid level", "Senior"],
                    label_visibility="collapsed",
                )
            with optional_right:
                render_label("Preferred track", False)
                preferred_track = st.selectbox(
                    "Preferred track",
                    ["Not specified", "Technical", "Research", "Business", "Design", "Management"],
                    label_visibility="collapsed",
                )

            render_label("Preferred industries", False)
            industries = st.multiselect(
                "Preferred industries",
                ["Technology", "Healthcare", "Finance", "Education", "Design", "Marketing", "Research", "Government"],
                placeholder="Select relevant industries",
                label_visibility="collapsed",
            )

            submitted = st.form_submit_button("Generate recommendation", type="primary", use_container_width=True)

skills = ", ".join([*starter_skills, custom_skills]).strip(", ")
interests = ", ".join([*starter_interests, custom_interests]).strip(", ")
extra_context = " ".join(
    value
    for value in [
        experience_level if experience_level != "Not specified" else "",
        preferred_track if preferred_track != "Not specified" else "",
        ", ".join(industries),
    ]
    if value
)
validation_errors = validate_inputs(education, skills, interests) if submitted else []

with output_col:
    with st.container(border=True):
        st.subheader("Recommendation")
        if submitted and validation_errors:
            st.error("Please resolve the required fields before generating a recommendation.")
            for error in validation_errors:
                st.write(f"- {error}")
        elif submitted:
            profile_text = profile_from_inputs(age, education, skills, interests, extra_context)
            recommendations = predict_top_careers(artifact, profile_text, top_n=5)
            best = recommendations[0]
            second = recommendations[1]["confidence"] if len(recommendations) > 1 else None
            displayed_total = sum(item["confidence"] for item in recommendations) or 1
            fit_share = best["confidence"] / displayed_total
            band = confidence_band(best["confidence"], second, top_five_rate)

            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-kicker">Best ranked match</div>
                    <div class="result-career">{escape(best["career"])}</div>
                    <div class="confidence-band">{escape(band)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_left, metric_mid, metric_right = st.columns(3)
            metric_left.metric("Fit score", percent(fit_share))
            metric_mid.metric("Model probability", percent(best["confidence"]))
            metric_right.metric("Top-5 validation", percent(top_five_rate) if top_five_rate is not None else "N/A")
            st.progress(min(100, int(round(fit_share * 100))))

            ranking = pd.DataFrame(
                {
                    "Career": [item["career"] for item in recommendations],
                    "Fit share": [(item["confidence"] / displayed_total) * 100 for item in recommendations],
                    "Model probability": [item["confidence"] for item in recommendations],
                }
            )
            st.bar_chart(ranking.set_index("Career")["Fit share"], color="#2563EB")
            st.dataframe(
                ranking,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Fit share": st.column_config.ProgressColumn("Fit share", format="%.1f%%", min_value=0, max_value=100),
                    "Model probability": st.column_config.NumberColumn("Model probability", format="%.2%"),
                },
            )

            suggestions = skill_suggestions(best["career"], skills)
            if suggestions:
                st.markdown("**Next skills to strengthen**")
                st.write(", ".join(suggestions))

            signals = extract_profile_signals(skills, interests)
            if signals:
                st.markdown("**Profile signals detected**")
                chips = "".join(f'<span class="signal-chip">{escape(signal)}</span>' for signal in signals)
                st.markdown(chips, unsafe_allow_html=True)

            with st.expander("Model input"):
                st.code(profile_text)
        else:
            st.info("Required fields are marked with a red star.")
            if top_five_rate is not None:
                st.markdown(
                    f'<div class="small-note">Current model top-5 validation rate: <strong>{percent(top_five_rate)}</strong></div>',
                    unsafe_allow_html=True,
                )

if metrics:
    with st.expander("Training Summary"):
        summary_left, summary_right = st.columns(2)
        with summary_left:
            st.write(f"Records: {metadata.get('record_count', 'N/A')}")
            st.write(f"Labels: {metadata.get('target_count', 'N/A')}")
            st.write(f"Estimator: {metadata.get('estimator', 'N/A')}")
        with summary_right:
            st.write(f"Top-1 accuracy: {metadata.get('accuracy', 'N/A')}")
            st.write(f"Top-3 match rate: {metadata.get('top_3_accuracy', 'N/A')}")
            st.write(f"Top-5 match rate: {metadata.get('top_5_accuracy', 'N/A')}")
        if metrics.get("source_distribution"):
            st.dataframe(source_table(metrics), hide_index=True, use_container_width=True)

from html import escape
import json

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
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .result-panel {
        border: 1px solid #D8DEE9;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #FFFFFF;
    }
    .result-title {
        color: #4B5563;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }
    .result-career {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .signal-chip {
        display: inline-block;
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        padding: 0.24rem 0.62rem;
        margin: 0.15rem;
        background: #F8FAFC;
        color: #1F2937;
        font-size: 0.86rem;
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


artifact = get_artifact()
metrics = get_metrics()
metadata = artifact.get("metadata", {})

with st.sidebar:
    st.header("Model")
    st.caption(metadata.get("estimator", "Classifier"))
    st.metric("Training records", metadata.get("training_records", "N/A"))
    st.metric("Career labels", metadata.get("target_count", "N/A"))
    top_5_accuracy = metadata.get("top_5_accuracy")
    if top_5_accuracy is not None:
        st.metric("Top-5 match rate", f"{top_5_accuracy:.1%}")
    accuracy = metadata.get("accuracy")
    if accuracy is not None:
        st.metric("Top-1 accuracy", f"{accuracy:.1%}")
    st.caption(f"Trained: {metadata.get('trained_at_utc', 'N/A')}")
    st.divider()
    st.caption("Predictions are guidance signals, not hiring or admission decisions.")

st.title("AI Career Recommendation")
st.write("Enter a profile and generate ranked career matches from the trained recommendation model.")

form_col, output_col = st.columns([1.05, 0.95], gap="large")

with form_col:
    with st.form("career_profile_form"):
        st.subheader("Profile")
        age = st.slider("Age", min_value=16, max_value=70, value=22)
        education = st.selectbox("Education or field", EDUCATION_OPTIONS, index=2)
        starter_skills = st.multiselect("Skill starters", SKILL_STARTERS)
        custom_skills = st.text_area(
            "Skills",
            placeholder="Example: Python, SQL, statistics, model evaluation",
            height=110,
        )
        starter_interests = st.multiselect("Interest starters", INTEREST_STARTERS)
        custom_interests = st.text_area(
            "Interests",
            placeholder="Example: data science, AI products, healthcare analytics",
            height=110,
        )
        submitted = st.form_submit_button("Generate recommendation", type="primary", use_container_width=True)

skills = ", ".join([*starter_skills, custom_skills]).strip(", ")
interests = ", ".join([*starter_interests, custom_interests]).strip(", ")

with output_col:
    st.subheader("Recommendation")
    if submitted:
        if not skills or not interests:
            st.warning("Add at least one skill and one interest.")
        else:
            profile_text = profile_from_inputs(age, education, skills, interests)
            recommendations = predict_top_careers(artifact, profile_text, top_n=5)
            best = recommendations[0]
            confidence_percent = best["confidence"] * 100
            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-title">Best match</div>
                    <div class="result-career">{escape(best["career"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(100, int(round(confidence_percent))))
            st.caption(f"Confidence: {confidence_percent:.2f}%")

            ranking = pd.DataFrame(
                {
                    "Career": [item["career"] for item in recommendations],
                    "Confidence": [round(item["confidence"] * 100, 2) for item in recommendations],
                }
            )
            st.bar_chart(ranking.set_index("Career"), y="Confidence", color="#2563EB")
            st.dataframe(ranking, hide_index=True, use_container_width=True)

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
        st.info("Complete the profile form to see ranked career matches.")

if metrics:
    with st.expander("Training summary"):
        labels = metrics.get("label_distribution", {})
        st.write(f"Records: {metadata.get('record_count', 'N/A')}")
        st.write(f"Labels: {metadata.get('target_count', 'N/A')}")
        st.write(f"Top-1 accuracy: {metadata.get('accuracy', 'N/A')}")
        st.write(f"Top-3 match rate: {metadata.get('top_3_accuracy', 'N/A')}")
        st.write(f"Top-5 match rate: {metadata.get('top_5_accuracy', 'N/A')}")
        st.dataframe(
            pd.DataFrame(labels.items(), columns=["Career", "Records"]).sort_values("Records", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

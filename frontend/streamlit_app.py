from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.career_recommender.interview_prep import (
    analyze_filler_words,
    evaluate_star_response,
    generate_question_bank,
)
from backend.career_recommender.modeling import load_artifact
from backend.career_recommender.paths import METRICS_PATH, MODEL_ARTIFACT_PATH, RAW_DATA_DIR
from backend.career_recommender.recommendation import (
    CAREER_SKILL_MAP,
    EDUCATION_OPTIONS,
    INTEREST_STARTERS,
    SKILL_STARTERS,
    adjust_confidence_for_market,
    degree_to_reality_distribution,
    extract_profile_signals,
    flight_risk_calculator,
    lateral_transition_mapping,
    placement_probability_forecast,
    profile_from_inputs,
    rank_profile_recommendations,
    skills_gap_analysis,
    split_skill_terms,
    stable_percent,
    study_habit_viability,
)
from backend.career_recommender.resume_analyzer import (
    analyze_resume_text,
    build_resume_pdf,
    generate_resume_summaries,
)
from backend.chatbot.rag_engine import DatasetRAGEngine, KimiRAGChatbot
from backend.core.api_manager import APIKeyPoolExhaustedError, APIManager
from backend.core.document_parser import ParsedDocument, extract_text
from frontend.ui_helpers import EDUCATION_PLACEHOLDER, confidence_band, percent, source_table, validate_inputs

STYLE_PATH = Path(__file__).parent / "assets" / "custom_style.css"
SOFT_SKILL_OPTIONS = [
    "Communication",
    "Leadership",
    "Problem Solving",
    "Collaboration",
    "Stakeholder Management",
    "Time Management",
    "Adaptability",
    "Analytical Thinking",
]


@st.cache_resource
def get_artifact() -> dict[str, Any]:
    """Load the trained career model artifact."""
    if not MODEL_ARTIFACT_PATH.exists():
        st.error("Model artifact is missing. Run `python scripts/02_train_model.py` first.")
        st.stop()
    return load_artifact()


@st.cache_data
def get_metrics() -> dict[str, Any]:
    """Load training metrics for sidebar diagnostics."""
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def get_raw_tables() -> dict[str, pd.DataFrame]:
    """Load local raw CSV datasets for student and RAG dashboards."""
    tables: dict[str, pd.DataFrame] = {}
    if not RAW_DATA_DIR.exists():
        return tables
    for path in sorted(RAW_DATA_DIR.glob("*.csv")):
        try:
            tables[path.name] = pd.read_csv(path)
        except Exception:
            tables[path.name] = pd.DataFrame()
    return tables


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> DatasetRAGEngine:
    """Build and cache the local dataset RAG index."""
    engine = DatasetRAGEngine()
    engine.build_index()
    return engine


def load_css() -> None:
    """Apply the Streamlit SaaS styling overrides."""
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_label(text: str, required: bool) -> None:
    """Render a compact required or optional field label."""
    marker = '<span class="required-star">*</span>' if required else '<span class="optional-pill">Optional</span>'
    st.markdown(f'<div class="field-label">{escape(text)} {marker}</div>', unsafe_allow_html=True)


def chip_list(values: Sequence[str], css_class: str) -> str:
    """Render values as HTML chips."""
    if not values:
        return '<span class="small-note">None detected</span>'
    return "".join(f'<span class="chip {css_class}">{escape(value)}</span>' for value in values)


def split_keywords(value: str) -> list[str]:
    """Split comma, newline, or semicolon separated keywords."""
    return split_skill_terms(value)


def maybe_api_manager() -> APIManager | None:
    """Create an API manager only when an API action is requested."""
    try:
        return APIManager()
    except Exception:
        return None


def render_sidebar(metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    """Render model and dataset status in the sidebar."""
    with st.sidebar:
        st.header("System Status")
        st.metric("Training records", metadata.get("training_records", "N/A"))
        st.metric("Career labels", metadata.get("target_count", "N/A"))
        top_five_rate = metadata.get("top_5_accuracy")
        if top_five_rate is not None:
            st.metric("Top-5 match rate", percent(float(top_five_rate)))
        if metadata.get("accuracy") is not None:
            st.metric("Top-1 accuracy", percent(float(metadata["accuracy"])))
        st.caption(f"Trained UTC: {metadata.get('trained_at_utc', 'N/A')}")
        if metrics.get("source_distribution"):
            st.divider()
            st.caption("Dataset mix")
            st.dataframe(source_table(dict(metrics)), hide_index=True, width="stretch")


def render_header() -> None:
    """Render application heading."""
    st.markdown(
        """
        <div class="app-shell">
            <div class="eyebrow">Career intelligence workspace</div>
            <div class="headline">AI Career Recommendation</div>
            <div class="subhead">A structured workspace for career matching, student planning, professional pivots, ATS resume optimization, and interview preparation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_career_match_tab(artifact: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    """Render probabilistic career matching and skills gap analysis."""
    form_col, result_col = st.columns([0.92, 1.08], gap="large")
    with form_col:
        with st.container(border=True):
            st.subheader("Dynamic Profile Builder")
            with st.form("career_match_form"):
                render_label("Age", True)
                age = st.slider("Age", min_value=16, max_value=70, value=22, label_visibility="collapsed")
                render_label("Education level or field", True)
                education = st.selectbox("Education", [EDUCATION_PLACEHOLDER, *EDUCATION_OPTIONS], label_visibility="collapsed")
                render_label("Hard skills", True)
                hard_skills = st.multiselect("Hard skills", SKILL_STARTERS, placeholder="Select technical skills", label_visibility="collapsed")
                extra_hard_skills = st.text_area("Additional hard skills", height=84, max_chars=500, label_visibility="collapsed")
                render_label("Soft skills", False)
                soft_skills = st.multiselect("Soft skills", SOFT_SKILL_OPTIONS, placeholder="Select soft skills", label_visibility="collapsed")
                render_label("Professional interests", True)
                interests = st.multiselect("Interests", INTEREST_STARTERS, placeholder="Select interest areas", label_visibility="collapsed")
                extra_interests = st.text_area("Additional interests", height=84, max_chars=420, label_visibility="collapsed")
                render_label("Market context", False)
                market_left, market_right = st.columns(2)
                city_index = market_left.slider("City development index", 0.0, 1.0, 0.72, 0.01)
                company_size = market_right.selectbox("Company size", ["Startup", "Small", "Mid-size", "Enterprise", "Large Enterprise"])
                submitted = st.form_submit_button("Generate top matches", type="primary", width="stretch")
    skills = ", ".join([*hard_skills, *soft_skills, extra_hard_skills]).strip(", ")
    interest_text = ", ".join([*interests, extra_interests]).strip(", ")
    validation_errors = validate_inputs(education, skills, interest_text) if submitted else []
    with result_col:
        with st.container(border=True):
            st.subheader("Top 3 Match List")
            if submitted and validation_errors:
                st.error("Please resolve the required fields before generating recommendations.")
                for error in validation_errors:
                    st.write(f"- {error}")
            elif submitted:
                profile_text = profile_from_inputs(age, education, skills, interest_text, company_size)
                recommendations = rank_profile_recommendations(dict(artifact), profile_text, skills, interest_text, company_size, top_n=3)
                table = pd.DataFrame(
                    [
                        {
                            "Career": item["career"],
                            "Model Confidence": item["confidence"],
                            "Market Adjusted Confidence": adjust_confidence_for_market(item["confidence"], city_index, company_size),
                            "Profile Alignment": item["profile_alignment"],
                        }
                        for item in recommendations
                    ]
                )
                best = recommendations[0]
                band = confidence_band(best["fit_score"] / max(sum(item["fit_score"] for item in recommendations), 1), best["confidence"], best["profile_alignment"])
                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="result-kicker">Best ranked match</div>
                        <div class="result-career">{escape(best["career"])}</div>
                        <div class="small-note">{escape(band)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Model Confidence": st.column_config.ProgressColumn("Model Confidence", format="%.2f", min_value=0, max_value=1),
                        "Market Adjusted Confidence": st.column_config.ProgressColumn("Market Adjusted Confidence", format="%.2f", min_value=0, max_value=1),
                        "Profile Alignment": st.column_config.ProgressColumn("Profile Alignment", format="%.2f", min_value=0, max_value=1),
                    },
                )
                target_requirements = CAREER_SKILL_MAP.get(best["career"], [])
                gap = skills_gap_analysis(skills, target_requirements)
                st.markdown("**Skills You Have**")
                st.markdown(chip_list(gap["skills_you_have"], "chip-green"), unsafe_allow_html=True)
                st.markdown("**Critical Skills Missing**")
                st.markdown(chip_list(gap["critical_skills_missing"], "chip-red"), unsafe_allow_html=True)
                signals = extract_profile_signals(skills, interest_text)
                if signals:
                    st.markdown("**Profile signals**")
                    st.markdown(chip_list(signals, "chip-neutral"), unsafe_allow_html=True)
            else:
                top_five_rate = metadata.get("top_5_accuracy")
                st.info("Required fields are marked with a red star.")
                if top_five_rate is not None:
                    st.caption(f"Current model top-5 validation rate: {percent(float(top_five_rate))}")


def render_student_hub_tab(tables: Mapping[str, pd.DataFrame]) -> None:
    """Render academic placement, degree mapping, and study habit tools."""
    forecast_col, mapping_col = st.columns([0.92, 1.08], gap="large")
    with forecast_col:
        with st.container(border=True):
            st.subheader("Placement Probability Forecaster")
            gpa = st.slider("GPA", 0.0, 10.0, 7.4, 0.1)
            college_tier = st.selectbox("College tier", [1, 2, 3, 4], format_func=lambda value: f"Tier {value}")
            comp_left, comp_right = st.columns(2)
            competencies = {
                "DSA": comp_left.checkbox("DSA", value=True),
                "Web Development": comp_left.checkbox("Web Development", value=True),
                "SQL": comp_right.checkbox("SQL", value=True),
                "Internship": comp_right.checkbox("Internship", value=False),
            }
            placement_baseline = tables.get("student_placement_salary_elite_v2.csv", pd.DataFrame())
            forecast = placement_probability_forecast(gpa, int(college_tier), competencies, placement_baseline)
            st.metric("Placement probability", stable_percent(forecast["placement_probability"]))
            st.metric("Expected salary tier", forecast["expected_salary_tier"])
            if forecast["baseline_salary_lpa"]:
                st.caption(f"Nearest dataset salary baseline: {forecast['baseline_salary_lpa']:.1f} LPA")
        with st.container(border=True):
            st.subheader("Study Habit Optimizer")
            weekly_hours = st.slider("Weekly self-study hours", 0.0, 50.0, 18.0, 1.0)
            score_columns = st.columns(3)
            scores = {
                "math": score_columns[0].slider("Math", 0, 100, 78),
                "english": score_columns[1].slider("English", 0, 100, 76),
                "science": score_columns[2].slider("Science", 0, 100, 80),
            }
            baseline = tables.get("student_scores_sanitized.csv", pd.DataFrame())
            viability = study_habit_viability(weekly_hours, scores, baseline)
            st.metric("Viability score", stable_percent(viability["viability_score"]))
            st.caption(f"Study-score correlation baseline: {viability['study_score_correlation']:.2f}")
    with mapping_col:
        with st.container(border=True):
            st.subheader("Degree-to-Reality Mapping")
            available = {name: table for name, table in tables.items() if not table.empty}
            if not available:
                st.info("No local raw datasets are available for degree mapping.")
                return
            dataset_name = st.selectbox("Dataset", list(available))
            dataframe = available[dataset_name]
            candidate_columns = [column for column in dataframe.columns if "field" in column.lower() or "education" in column.lower()]
            if candidate_columns:
                field_column = st.selectbox("Field column", candidate_columns)
                field_options = sorted(str(value) for value in dataframe[field_column].dropna().unique())[:100]
                field = st.selectbox("Field of study", field_options) if field_options else ""
                distribution = degree_to_reality_distribution(dataframe, field)
                if distribution.empty:
                    st.info("This dataset does not expose a compatible occupation column.")
                else:
                    fig = px.bar(distribution, x="occupation", y="count", text="count")
                    fig.update_layout(xaxis_title="", yaxis_title="Records", height=440)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No field or education column was found in the selected dataset.")


def render_pivot_tab(tables: Mapping[str, pd.DataFrame]) -> None:
    """Render professional pivot risk and lateral transition mapping."""
    risk_col, transition_col = st.columns([0.9, 1.1], gap="large")
    with risk_col:
        with st.container(border=True):
            st.subheader("Flight Risk Calculator")
            satisfaction = st.slider("Job satisfaction", 1, 10, 5)
            balance = st.slider("Work-life balance", 1, 10, 5)
            years = st.slider("Years of experience", 0.0, 25.0, 3.0, 0.5)
            career_change_baseline = tables.get("career_change_prediction_dataset.csv", pd.DataFrame())
            risk = flight_risk_calculator(satisfaction, balance, years, career_change_baseline)
            st.metric("Time to pivot", stable_percent(risk["time_to_pivot_percent"]))
            st.write(risk["recommendation"])
    with transition_col:
        with st.container(border=True):
            st.subheader("Lateral Transition Mapping")
            selected_skills = st.multiselect("Current job skills", SKILL_STARTERS, placeholder="Select skills you use now")
            custom_skills = st.text_area("Additional current skills", height=88)
            transitions = lateral_transition_mapping([*selected_skills, *split_keywords(custom_skills)], top_n=5)
            if transitions:
                st.dataframe(
                    pd.DataFrame(transitions),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f"),
                        "estimated_salary_floor_ratio": st.column_config.ProgressColumn("Salary floor", min_value=0, max_value=1, format="%.2f"),
                    },
                )
            else:
                st.info("Add current skills to calculate adjacent career moves.")


def render_resume_tab() -> None:
    """Render ATS resume parsing, scoring, summary generation, and PDF export."""
    upload_col, output_col = st.columns([0.9, 1.1], gap="large")
    with upload_col:
        with st.container(border=True):
            st.subheader("ATS Resume Optimization Suite")
            uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"])
            target_role = st.text_input("Target role", "Data Analyst")
            keyword_text = st.text_area("Required keywords", "Python, SQL, dashboarding, stakeholder communication, data analysis", height=92)
            analyze_clicked = st.button("Analyze resume", type="primary", width="stretch")
            if analyze_clicked and uploaded is not None:
                try:
                    manager = maybe_api_manager() if Path(uploaded.name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else None
                    parsed = extract_text(uploaded, uploaded.name, manager)
                    st.session_state["resume_parsed"] = parsed
                    st.session_state["resume_keywords"] = split_keywords(keyword_text)
                    st.session_state["target_role"] = target_role
                except Exception as exc:
                    st.error(f"Resume parsing failed: {exc}")
            elif analyze_clicked:
                st.error("Upload a resume file before analysis.")
        with st.container(border=True):
            st.subheader("Professional Resume Builder")
            with st.form("resume_builder_form"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone")
                location = st.text_input("Location")
                linkedin = st.text_input("LinkedIn")
                summary = st.text_area("Summary", height=80)
                skills = st.text_area("Skills", height=80)
                experience = st.text_area("Experience", height=140)
                projects = st.text_area("Projects", height=100)
                education = st.text_area("Education", height=80)
                build_clicked = st.form_submit_button("Build ATS PDF", width="stretch")
            if build_clicked:
                pdf_bytes = build_resume_pdf(
                    {
                        "name": name,
                        "email": email,
                        "phone": phone,
                        "location": location,
                        "linkedin": linkedin,
                        "summary": summary,
                        "skills": skills,
                        "experience": experience,
                        "projects": projects,
                        "education": education,
                    }
                )
                st.download_button("Download resume PDF", pdf_bytes, "ats_resume.pdf", "application/pdf", width="stretch")
    with output_col:
        parsed_doc = st.session_state.get("resume_parsed")
        keywords = st.session_state.get("resume_keywords", split_keywords(keyword_text))
        if isinstance(parsed_doc, ParsedDocument):
            analysis = analyze_resume_text(parsed_doc.text, keywords, parsed_doc.warnings)
            ats = analysis["ats"]
            st.metric("ATS match score", f"{ats['score']:.1f}%")
            st.progress(min(100, int(round(ats["score"]))))
            left, right = st.columns(2)
            with left:
                st.markdown("**Keywords Found**")
                st.markdown(chip_list(ats["found_keywords"], "chip-green"), unsafe_allow_html=True)
            with right:
                st.markdown("**Keyword Gaps**")
                st.markdown(chip_list(ats["missing_keywords"], "chip-red"), unsafe_allow_html=True)
            st.markdown("**Formatting Sanity Check**")
            for warning in analysis["formatting_warnings"]:
                st.warning(warning)
            st.markdown("**Action Verb Enhancer**")
            if analysis["action_verb_findings"]:
                st.dataframe(pd.DataFrame(analysis["action_verb_findings"]), hide_index=True, width="stretch")
            else:
                st.success("No weak action verbs detected.")
            st.markdown("**Quantifiable Impact Checker**")
            if analysis["impact_gaps"]:
                st.dataframe(pd.DataFrame(analysis["impact_gaps"]), hide_index=True, width="stretch")
            else:
                st.success("All substantial bullets include metric evidence.")
            if st.button("Generate role-specific summaries", width="stretch"):
                manager = maybe_api_manager()
                summaries = generate_resume_summaries(parsed_doc.text, st.session_state.get("target_role", target_role), manager)
                st.session_state["resume_summaries"] = summaries
            for summary in st.session_state.get("resume_summaries", []):
                st.write(summary)
        else:
            st.info("Upload and analyze a resume to view ATS diagnostics.")


def render_interview_chat_tab() -> None:
    """Render interview question generation, STAR coaching, audio sandbox, and RAG chat."""
    interview_col, chat_col = st.columns([0.95, 1.05], gap="large")
    with interview_col:
        with st.container(border=True):
            st.subheader("Advanced Interview Preparation")
            target_role = st.text_input("Interview target role", "Data Analyst")
            skill_values = st.multiselect("Explicit skills", SKILL_STARTERS, default=["Python", "SQL"], placeholder="Select skills")
            if st.button("Generate question bank", type="primary", width="stretch"):
                st.session_state["question_bank"] = generate_question_bank(target_role, skill_values)
            for section, questions in st.session_state.get("question_bank", {}).items():
                st.markdown(f"**{section.title()}**")
                for question in questions:
                    st.write(f"- {question}")
        with st.container(border=True):
            st.subheader("STAR Story Builder")
            story = st.text_area("Interview answer", height=150)
            if st.button("Evaluate STAR response", width="stretch"):
                st.session_state["star_result"] = evaluate_star_response(story)
            result = st.session_state.get("star_result")
            if result:
                st.metric("STAR score", f"{result['score']:.1f}%")
                st.write(result["components"])
                if result["metric_warning"]:
                    st.warning(result["metric_warning"])
        with st.container(border=True):
            st.subheader("Mock Interview Audio Sandbox")
            audio = st.audio_input("Record a mock answer")
            transcript = st.text_area("Transcript for filler analysis", height=92)
            audio_left, audio_right = st.columns(2)
            if audio_left.button("Transcribe audio", width="stretch", disabled=audio is None):
                manager = maybe_api_manager()
                if manager is None:
                    st.error("Transcription provider is not configured.")
                else:
                    try:
                        st.session_state["audio_transcript"] = manager.audio_transcription(audio.getvalue(), "interview.wav")
                    except Exception as exc:
                        st.error(f"Audio transcription failed: {exc}")
            if st.session_state.get("audio_transcript"):
                st.text_area("API transcript", st.session_state["audio_transcript"], height=92)
            if audio_right.button("Analyze fillers", width="stretch"):
                active_transcript = st.session_state.get("audio_transcript") or transcript
                st.session_state["filler_analysis"] = analyze_filler_words(active_transcript)
            if st.session_state.get("filler_analysis"):
                st.write(st.session_state["filler_analysis"])
    with chat_col:
        with st.container(border=True):
            st.subheader("Kimi Dataset-Grounded Chatbot")
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            payload = st.chat_input(
                "Ask about career paths or upload a resume",
                accept_file=True,
                file_type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
            )
            if payload:
                user_text, files = unpack_chat_payload(payload)
                resume_context = ""
                for file in files:
                    try:
                        parsed = extract_text(file, file.name, maybe_api_manager())
                        resume_context += f"\nUploaded file {file.name}:\n{parsed.text[:4000]}"
                    except Exception as exc:
                        resume_context += f"\nUploaded file {file.name} could not be parsed: {exc}"
                if user_text or resume_context:
                    st.session_state["messages"].append({"role": "user", "content": user_text or "Uploaded resume for review."})
                    try:
                        chatbot = KimiRAGChatbot(api_manager=APIManager(), rag_engine=get_rag_engine())
                        answer = chatbot.answer(user_text or "Review the uploaded resume.", st.session_state["messages"], resume_context)
                    except APIKeyPoolExhaustedError as exc:
                        answer = f"API keys are unavailable or exhausted: {exc}"
                    except Exception as exc:
                        answer = f"Chatbot request failed: {exc}"
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.rerun()


def unpack_chat_payload(payload: Any) -> tuple[str, list[Any]]:
    """Return message text and uploaded files from Streamlit chat input payloads."""
    if isinstance(payload, str):
        return payload, []
    text = getattr(payload, "text", "")
    files = list(getattr(payload, "files", []) or [])
    if isinstance(payload, Mapping):
        text = str(payload.get("text", ""))
        files = list(payload.get("files", []) or [])
    return str(text), files


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="AI Career Recommendation",
        page_icon=":material/work:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()
    artifact = get_artifact()
    metrics = get_metrics()
    metadata = artifact.get("metadata", {})
    render_sidebar(metadata, metrics)
    render_header()
    tabs = st.tabs(["Career Match", "Student Hub", "Pivot Dashboard", "Resume ATS", "Interview & Chat"])
    with tabs[0]:
        render_career_match_tab(artifact, metadata)
    with tabs[1]:
        render_student_hub_tab(get_raw_tables())
    with tabs[2]:
        render_pivot_tab(get_raw_tables())
    with tabs[3]:
        render_resume_tab()
    with tabs[4]:
        render_interview_chat_tab()


if __name__ == "__main__":
    main()

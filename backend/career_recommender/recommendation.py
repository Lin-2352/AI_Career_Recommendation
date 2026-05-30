from collections import Counter
import math
import re
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.career_recommender.data_pipeline import age_band

EDUCATION_OPTIONS = [
    "Secondary school",
    "High School / Foundation",
    "Diploma",
    "Bachelor's",
    "Master's",
    "PhD",
    "Engineering",
    "Business",
    "Science",
    "Arts",
    "Computer Science",
    "Information Technology",
    "Design",
]

SKILL_STARTERS = [
    "Python",
    "SQL",
    "Data Analysis",
    "Machine Learning",
    "Cloud Computing",
    "Project Management",
    "Financial Modeling",
    "Digital Marketing",
    "UX Research",
    "Graphic Design",
    "Java",
    "React",
    "Cybersecurity",
    "Statistics",
    "System Design",
    "Database Management",
    "Networking",
    "Quality Assurance",
    "Technical Writing",
    "Research Methods",
]

INTEREST_STARTERS = [
    "Artificial Intelligence",
    "Data Science",
    "Software Development",
    "Healthcare",
    "Finance",
    "Business Strategy",
    "Design",
    "Marketing",
    "Research",
    "Entrepreneurship",
    "Teaching",
    "Product Development",
    "Cloud Infrastructure",
    "Security",
    "Game Development",
]

CAREER_SKILL_MAP = {
    "Data Scientist": ["Python", "SQL", "Statistics", "Machine Learning", "Model Evaluation"],
    "Data Engineer": ["SQL", "Python", "Data Pipelines", "Databases", "Cloud Computing"],
    "Data Analyst": ["SQL", "Excel", "Python", "Dashboarding", "Business Metrics"],
    "Machine Learning Engineer": ["Python", "Machine Learning", "Model Deployment", "MLOps", "Cloud Computing"],
    "Artificial Intelligence Engineer": ["Python", "Machine Learning", "Deep Learning", "MLOps", "AI System Design"],
    "AI Researcher": ["Python", "Deep Learning", "Research Methods", "Mathematics", "Experiment Design"],
    "AI Specialist": ["Python", "Prompt Engineering", "Machine Learning", "AI Product Evaluation", "Data Ethics"],
    "NLP Engineer": ["Python", "Text Processing", "Transformers", "Model Evaluation", "APIs"],
    "Deep Learning Engineer": ["Python", "Neural Networks", "GPU Workflows", "TensorFlow or PyTorch", "MLOps"],
    "Software Engineer": ["Data Structures", "System Design", "Git", "Testing", "Cloud Computing"],
    "Software Developer": ["Programming Fundamentals", "Git", "APIs", "Testing", "Databases"],
    "Front-end Developer": ["HTML", "CSS", "JavaScript", "React", "Accessibility"],
    "Front End Developer": ["HTML", "CSS", "JavaScript", "React", "Accessibility"],
    "Backend Developer": ["APIs", "Databases", "Authentication", "System Design", "Testing"],
    "Back End Developer": ["APIs", "Databases", "Authentication", "System Design", "Testing"],
    "Full Stack Developer": ["React", "APIs", "Databases", "Cloud Deployment", "Testing"],
    "Cloud Engineer": ["Cloud Platforms", "Linux", "Networking", "Infrastructure as Code", "Monitoring"],
    "Cloud Architect": ["Cloud Platforms", "System Design", "Networking", "Security", "Cost Optimization"],
    "DevOps Engineer": ["CI/CD", "Linux", "Docker", "Cloud Platforms", "Monitoring"],
    "Cybersecurity Analyst": ["Networking", "Security Monitoring", "Threat Analysis", "Linux", "Incident Response"],
    "Cybersecurity Specialist": ["Network Security", "Risk Assessment", "Vulnerability Testing", "Cloud Security", "Incident Response"],
    "Financial Analyst": ["Excel", "Financial Modeling", "Accounting", "Data Visualization", "Business Communication"],
    "Business Analyst": ["Requirements Analysis", "SQL", "Process Mapping", "Stakeholder Communication", "Dashboards"],
    "Project Manager": ["Planning", "Risk Management", "Agile Delivery", "Stakeholder Communication", "Reporting"],
    "Digital Marketer": ["SEO", "Campaign Analytics", "Content Strategy", "A/B Testing", "Marketing Automation"],
    "Marketing Manager": ["Market Research", "Campaign Planning", "Analytics", "Brand Strategy", "Budgeting"],
    "Graphic Designer": ["Visual Design", "Typography", "Color Theory", "Figma", "Portfolio Building"],
    "UX Designer": ["User Research", "Wireframing", "Prototyping", "Usability Testing", "Figma"],
    "UX Researcher": ["Interviewing", "Survey Design", "Usability Testing", "Research Synthesis", "Product Thinking"],
    "Content Strategist": ["Editorial Planning", "SEO", "Audience Research", "Analytics", "Brand Voice"],
    "Research Scientist": ["Research Methods", "Statistics", "Experiment Design", "Technical Writing", "Python"],
    "Research Analyst": ["Data Analysis", "Research Methods", "Statistics", "Reporting", "Domain Knowledge"],
    "Biostatistician": ["Statistics", "R or Python", "Clinical Data", "Study Design", "Regulatory Awareness"],
    "Embedded Systems Engineer": ["C", "Microcontrollers", "Electronics", "Debugging", "Real-time Systems"],
    "Automation Engineer": ["Scripting", "Testing", "Process Automation", "APIs", "CI/CD"],
    "Mobile Developer": ["Mobile UI", "APIs", "Testing", "App Store Delivery", "Performance"],
    "Mobile App Developer": ["Mobile UI", "APIs", "Testing", "App Store Delivery", "Performance"],
    "Game Developer": ["Programming", "Game Engines", "Math", "Graphics", "Testing"],
    "Quality Assurance Engineer": ["Test Planning", "Automation", "Bug Reporting", "APIs", "CI/CD"],
    "Software Tester": ["Test Cases", "Automation", "Bug Reporting", "Regression Testing", "Quality Metrics"],
    "Database Administrator": ["SQL", "Backup Recovery", "Performance Tuning", "Security", "Monitoring"],
    "Network Engineer": ["Networking", "Routing", "Switching", "Security", "Troubleshooting"],
    "Network Administrator": ["Networking", "Monitoring", "User Support", "Security", "Troubleshooting"],
    "Computer Network Architect": ["Network Design", "Cloud Networking", "Security", "Capacity Planning", "Documentation"],
    "Computer Programmer": ["Programming Fundamentals", "Data Structures", "Debugging", "Version Control", "Testing"],
    "Computer Systems Analyst": ["Requirements Analysis", "Systems Design", "SQL", "Documentation", "Stakeholder Communication"],
    "Systems Analyst": ["Requirements Analysis", "Systems Design", "SQL", "Documentation", "Stakeholder Communication"],
    "IT Project Manager": ["Agile Delivery", "Planning", "Risk Management", "Stakeholder Communication", "Reporting"],
    "IT Consultant": ["Business Analysis", "Systems Design", "Communication", "Cloud Platforms", "Documentation"],
    "IT Support Specialist": ["Troubleshooting", "Networking", "Operating Systems", "Customer Support", "Documentation"],
    "Technical Support Engineer": ["Troubleshooting", "Networking", "Operating Systems", "Customer Support", "Documentation"],
    "Technical Writer": ["Technical Writing", "Information Architecture", "Editing", "APIs", "Documentation Tools"],
    "Computer Science Teacher": ["Computer Science Fundamentals", "Lesson Planning", "Communication", "Assessment", "Mentoring"],
    "Business Intelligence Analyst": ["SQL", "Data Modeling", "Dashboards", "Business Metrics", "Communication"],
    "UI/UX Designer": ["User Research", "Wireframing", "Prototyping", "Usability Testing", "Figma"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "APIs", "Accessibility"],
    "Blockchain Developer": ["Distributed Systems", "Smart Contracts", "Cryptography", "Security", "Testing"],
    "Computer Hardware Engineer": ["Digital Logic", "Electronics", "Computer Architecture", "Testing", "Documentation"],
    "Computer and Information Research Scientist": ["Research Methods", "Algorithms", "Mathematics", "Technical Writing", "Python"],
    "Computer and Information Systems Manager": ["Team Leadership", "Systems Strategy", "Budgeting", "Security", "Stakeholder Communication"],
    "Computer Systems Manager": ["Team Leadership", "Systems Strategy", "Budgeting", "Security", "Stakeholder Communication"],
    "Lawyer": ["Critical Reading", "Research", "Writing", "Argumentation", "Ethics"],
    "Doctor": ["Biology", "Chemistry", "Patient Communication", "Research", "Clinical Reasoning"],
    "Teacher": ["Communication", "Lesson Planning", "Assessment", "Subject Expertise", "Mentoring"],
    "Accountant": ["Accounting", "Excel", "Financial Reporting", "Attention to Detail", "Compliance"],
    "Banker": ["Finance", "Risk Analysis", "Customer Communication", "Excel", "Market Awareness"],
    "Business Owner": ["Market Research", "Finance", "Sales", "Operations", "Leadership"],
    "Writer": ["Writing", "Editing", "Research", "Audience Analysis", "Portfolio Building"],
    "Scientist": ["Research Methods", "Statistics", "Experiment Design", "Technical Writing", "Data Analysis"],
    "Artist": ["Portfolio Building", "Visual Design", "Creative Direction", "Presentation", "Digital Tools"],
    "Designer": ["Visual Design", "User Research", "Figma", "Typography", "Portfolio Building"],
    "Government Officer": ["Public Policy", "Communication", "Administration", "Analytical Reasoning", "Ethics"],
    "Construction Engineer": ["Engineering Drawing", "Project Planning", "Safety", "Materials", "Site Coordination"],
    "Real Estate Developer": ["Market Analysis", "Finance", "Negotiation", "Project Planning", "Regulatory Awareness"],
    "Stock Investor": ["Financial Analysis", "Risk Management", "Market Research", "Portfolio Strategy", "Data Analysis"],
    "Social Network Studies": ["Research Methods", "Communication", "Data Analysis", "Psychology", "Media Studies"],
    "Tech": ["Programming Fundamentals", "Databases", "Cloud Computing", "Git", "Problem Solving"],
    "Business": ["Market Analysis", "Operations", "Communication", "Financial Literacy", "Strategy"],
    "Finance": ["Accounting", "Financial Modeling", "Excel", "Risk Analysis", "Data Visualization"],
    "Design": ["Visual Design", "User Research", "Portfolio Building", "Figma", "Presentation"],
    "Healthcare": ["Healthcare Systems", "Data Privacy", "Patient Experience", "Research Methods", "Analytics"],
}

SIGNAL_KEYWORDS = [
    "Python",
    "SQL",
    "Java",
    "React",
    "Machine Learning",
    "Data Analysis",
    "Cloud",
    "Cybersecurity",
    "Finance",
    "Marketing",
    "Design",
    "Research",
    "Healthcare",
    "Project Management",
    "AI",
    "UX",
    "Networking",
    "Testing",
    "Writing",
    "Teaching",
    "Game",
]

BROAD_CAREER_LABELS = {"Tech", "Business", "Finance", "Design", "Healthcare"}
GENERIC_TITLE_WORDS = {
    "and",
    "engineer",
    "developer",
    "analyst",
    "specialist",
    "manager",
    "officer",
    "administrator",
    "professional",
}
TOKEN_ALIASES = {
    "financial": {"finance", "financial"},
    "finance": {"finance", "financial"},
    "ai": {"ai", "artificial", "intelligence"},
    "ux": {"ux", "user", "experience"},
    "frontend": {"front", "frontend"},
    "backend": {"back", "backend"},
}


def profile_from_inputs(age: int, education: str, skills: str, interests: str, extra_context: str = "") -> str:
    values = [age_band(age), education, skills, interests, extra_context]
    return " ".join(str(value).strip() for value in values if str(value).strip())


def normalize_blob(value: str) -> str:
    return value.lower().replace("-", " ").replace("_", " ")


def tokens_from_text(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#]+", normalize_blob(value)))


def token_present(token: str, tokens: set[str]) -> bool:
    aliases = TOKEN_ALIASES.get(token, {token})
    return bool(tokens.intersection(aliases))


def phrase_present(phrase: str, blob: str, tokens: set[str]) -> bool:
    normalized = normalize_blob(phrase)
    words = [word for word in tokens_from_text(normalized) if len(word) > 1]
    if normalized in blob:
        return True
    return bool(words) and all(token_present(word, tokens) for word in words)


def career_alignment_score(career: str, profile_text: str) -> float:
    blob = normalize_blob(profile_text)
    tokens = tokens_from_text(profile_text)
    catalog = CAREER_SKILL_MAP.get(career, [])
    catalog_score = 0.0
    if catalog:
        matches = sum(1 for skill in catalog if phrase_present(skill, blob, tokens))
        catalog_score = matches / len(catalog)
    title_words = [
        word
        for word in tokens_from_text(career)
        if len(word) > 2 and word not in GENERIC_TITLE_WORDS
    ]
    title_score = 0.0
    if title_words:
        title_score = sum(1 for word in title_words if token_present(word, tokens)) / len(title_words)
    return max(catalog_score, title_score * 0.35)


def rank_profile_recommendations(
    artifact: dict,
    profile_text: str,
    skills: str,
    interests: str,
    extra_context: str = "",
    top_n: int = 5,
) -> list[dict]:
    if top_n <= 0:
        return []
    model = artifact["model"]
    probabilities = model.predict_proba([profile_text])[0]
    classes = model.named_steps["classifier"].classes_
    alignment_text = " ".join([profile_text, skills, interests, extra_context])
    ranked = []
    for career, probability in zip(classes, probabilities, strict=True):
        career_name = str(career)
        model_probability = float(probability)
        alignment = career_alignment_score(career_name, alignment_text)
        fit_score = (model_probability * 0.68) + (alignment * 0.32)
        if career_name in BROAD_CAREER_LABELS:
            fit_score *= 0.88
        ranked.append(
            {
                "career": career_name,
                "confidence": model_probability,
                "profile_alignment": alignment,
                "fit_score": fit_score,
            }
        )
    return sorted(ranked, key=lambda item: item["fit_score"], reverse=True)[:top_n]


def skill_suggestions(career: str, current_skills: str, limit: int = 5) -> list[str]:
    catalog = CAREER_SKILL_MAP.get(career)
    if catalog is None:
        catalog = next((skills for key, skills in CAREER_SKILL_MAP.items() if key.lower() in career.lower()), [])
    current = normalize_blob(current_skills)
    missing = [skill for skill in catalog if normalize_blob(skill) not in current]
    return missing[:limit]


def extract_profile_signals(skills: str, interests: str, limit: int = 8) -> list[str]:
    blob = normalize_blob(f"{skills} {interests}")
    signals = []
    for keyword in SIGNAL_KEYWORDS:
        if normalize_blob(keyword) in blob and keyword not in signals:
            signals.append(keyword)
    return signals[:limit]


def split_skill_terms(values: str | Sequence[str]) -> list[str]:
    """Normalize skill text or selected values into deduplicated display terms."""
    if isinstance(values, str):
        raw_terms = re.split(r"[,;|/\n]+", values)
    else:
        raw_terms = [str(value) for value in values]
    output: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        cleaned = re.sub(r"\s+", " ", str(term).strip())
        key = normalize_blob(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def skills_gap_analysis(user_skills: str | Sequence[str], target_requirements: str | Sequence[str]) -> dict[str, list[str]]:
    """Compare user skills with target requirements and return present and missing skills."""
    user_terms = split_skill_terms(user_skills)
    requirement_terms = split_skill_terms(target_requirements)
    user_lookup = {normalize_blob(term): term for term in user_terms}
    have: list[str] = []
    missing: list[str] = []
    for requirement in requirement_terms:
        normalized = normalize_blob(requirement)
        matched = next((display for key, display in user_lookup.items() if normalized in key or key in normalized), "")
        if matched:
            have.append(requirement)
        else:
            missing.append(requirement)
    return {"skills_you_have": have, "critical_skills_missing": missing}


def placement_probability_forecast(
    gpa: float,
    college_tier: int,
    competencies: Mapping[str, bool | int | float],
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Forecast placement probability and salary tier from academic and competency signals."""
    bounded_gpa = min(max(float(gpa), 0.0), 10.0)
    bounded_tier = min(max(int(college_tier), 1), 4)
    competency_scores = [float(value) for value in competencies.values()]
    competency_strength = sum(1.0 for value in competency_scores if value > 0) / max(len(competency_scores), 1)
    gpa_score = bounded_gpa / 10.0
    tier_bonus = {1: 0.18, 2: 0.11, 3: 0.05, 4: 0.0}[bounded_tier]
    probability = (0.52 * gpa_score) + (0.32 * competency_strength) + tier_bonus
    probability = min(max(probability, 0.05), 0.98)
    baseline_salary = 0.0
    if dataset is not None and {"cgpa", "college_tier", "placed"}.issubset(dataset.columns):
        numeric = dataset.copy()
        numeric["cgpa"] = pd.to_numeric(numeric["cgpa"], errors="coerce")
        numeric["college_tier"] = pd.to_numeric(numeric["college_tier"], errors="coerce")
        numeric["placed"] = pd.to_numeric(numeric["placed"], errors="coerce")
        nearby = numeric[
            (numeric["college_tier"] == bounded_tier)
            & (numeric["cgpa"].between(max(0.0, bounded_gpa - 1.0), min(10.0, bounded_gpa + 1.0)))
        ]
        if nearby.empty:
            nearby = numeric.dropna(subset=["placed"])
        if not nearby.empty:
            baseline_probability = float(nearby["placed"].mean())
            probability = min(max((0.62 * probability) + (0.38 * baseline_probability), 0.05), 0.98)
            if "salary_lpa" in nearby.columns:
                salaries = pd.to_numeric(nearby["salary_lpa"], errors="coerce").dropna()
                baseline_salary = float(salaries.median()) if not salaries.empty else 0.0
    if probability >= 0.82:
        salary_tier = "Premium"
    elif probability >= 0.64:
        salary_tier = "Competitive"
    elif probability >= 0.45:
        salary_tier = "Developing"
    else:
        salary_tier = "Foundation"
    return {
        "placement_probability": probability,
        "expected_salary_tier": salary_tier,
        "gpa_score": gpa_score,
        "competency_strength": competency_strength,
        "baseline_salary_lpa": baseline_salary,
    }


def study_habit_viability(
    weekly_self_study_hours: float,
    subject_scores: Mapping[str, float],
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate a study viability score using correlation when dataset baselines exist."""
    scores = [float(score) for score in subject_scores.values() if pd.notna(score)]
    average_score = sum(scores) / max(len(scores), 1)
    correlation = 0.0
    if dataset is not None and {"weekly_self_study_hours"}.issubset(dataset.columns):
        score_columns = [column for column in dataset.columns if column.endswith("_score")]
        if score_columns:
            baseline = dataset[["weekly_self_study_hours", *score_columns]].apply(pd.to_numeric, errors="coerce")
            baseline["average_score"] = baseline[score_columns].mean(axis=1)
            correlation_value = baseline["weekly_self_study_hours"].corr(baseline["average_score"])
            correlation = 0.0 if pd.isna(correlation_value) else float(correlation_value)
    hours_score = min(max(float(weekly_self_study_hours), 0.0), 40.0) / 40.0
    academic_score = min(max(average_score, 0.0), 100.0) / 100.0
    viability = min(max((0.44 * hours_score) + (0.46 * academic_score) + (0.10 * max(correlation, 0.0)), 0.0), 1.0)
    return {
        "viability_score": viability,
        "average_subject_score": average_score,
        "study_score": hours_score,
        "study_score_correlation": correlation,
    }


def flight_risk_calculator(
    job_satisfaction: int,
    work_life_balance: int,
    years_experience: float,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Estimate professional pivot urgency from satisfaction, balance, and tenure signals."""
    satisfaction_risk = 1.0 - (min(max(job_satisfaction, 1), 10) / 10.0)
    balance_risk = 1.0 - (min(max(work_life_balance, 1), 10) / 10.0)
    tenure_factor = min(max(float(years_experience), 0.0), 15.0) / 15.0
    risk = min(max((0.42 * satisfaction_risk) + (0.36 * balance_risk) + (0.22 * tenure_factor), 0.0), 1.0)
    if dataset is not None and {"Job Satisfaction", "Work-Life Balance", "Years of Experience", "Career Change Interest"}.issubset(dataset.columns):
        numeric = dataset.copy()
        numeric["Job Satisfaction"] = pd.to_numeric(numeric["Job Satisfaction"], errors="coerce")
        numeric["Work-Life Balance"] = pd.to_numeric(numeric["Work-Life Balance"], errors="coerce")
        numeric["Years of Experience"] = pd.to_numeric(numeric["Years of Experience"], errors="coerce")
        numeric["Career Change Interest"] = pd.to_numeric(numeric["Career Change Interest"], errors="coerce")
        nearby = numeric[
            (numeric["Job Satisfaction"].between(job_satisfaction - 1, job_satisfaction + 1))
            & (numeric["Work-Life Balance"].between(work_life_balance - 1, work_life_balance + 1))
            & (numeric["Years of Experience"].between(max(0.0, years_experience - 2.0), years_experience + 2.0))
        ].dropna(subset=["Career Change Interest"])
        if nearby.empty:
            nearby = numeric.dropna(subset=["Career Change Interest"])
        if not nearby.empty:
            baseline_risk = float(nearby["Career Change Interest"].clip(lower=0, upper=1).mean())
            risk = min(max((0.68 * risk) + (0.32 * baseline_risk), 0.0), 1.0)
    if risk >= 0.72:
        recommendation = "Plan an active pivot within 3 months"
    elif risk >= 0.48:
        recommendation = "Prepare a controlled transition within 6 to 9 months"
    else:
        recommendation = "Optimize current role before a major pivot"
    return {"time_to_pivot_percent": risk, "recommendation": recommendation}


def lateral_transition_mapping(
    current_skills: str | Sequence[str],
    top_n: int = 5,
    salary_floor_ratio: float = 0.85,
) -> list[dict[str, Any]]:
    """Rank adjacent careers by cosine similarity against the user's current skills."""
    user_text = " ".join(split_skill_terms(current_skills))
    if not user_text.strip() or top_n <= 0:
        return []
    careers = [career for career in CAREER_SKILL_MAP if career not in BROAD_CAREER_LABELS]
    corpus = [user_text, *[" ".join(CAREER_SKILL_MAP[career]) for career in careers]]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    ranked = sorted(zip(careers, similarities, strict=True), key=lambda item: item[1], reverse=True)
    output: list[dict[str, Any]] = []
    for career, similarity in ranked[:top_n]:
        salary_cut_risk = max(0.0, (1.0 - float(similarity)) * 0.35)
        output.append(
            {
                "career": career,
                "similarity": float(similarity),
                "estimated_salary_floor_ratio": max(salary_floor_ratio, 1.0 - salary_cut_risk),
                "shared_skills": skills_gap_analysis(current_skills, CAREER_SKILL_MAP[career])["skills_you_have"],
                "missing_skills": skills_gap_analysis(current_skills, CAREER_SKILL_MAP[career])["critical_skills_missing"],
            }
        )
    return output


def adjust_confidence_for_market(confidence: float, city_development_index: float, company_size: str) -> float:
    """Adjust model confidence by market maturity and company-size stability."""
    city_index = min(max(float(city_development_index), 0.0), 1.0)
    company_factor = {
        "startup": 0.94,
        "small": 0.97,
        "mid-size": 1.00,
        "enterprise": 1.04,
        "large enterprise": 1.05,
    }.get(normalize_blob(company_size), 1.0)
    market_factor = 0.88 + (0.24 * city_index)
    return min(max(float(confidence) * market_factor * company_factor, 0.0), 0.99)


def degree_to_reality_distribution(dataframe: pd.DataFrame, field_of_study: str) -> pd.DataFrame:
    """Return occupation distribution for a selected field of study."""
    if dataframe.empty or not field_of_study:
        return pd.DataFrame(columns=["occupation", "count", "share"])
    field_columns = [column for column in dataframe.columns if normalize_blob(column) in {"field of study", "field_of_study", "education"}]
    occupation_columns = [
        column
        for column in dataframe.columns
        if normalize_blob(column)
        in {"occupation", "recommended career path", "recommended_career_path", "recommended career", "recommended_career"}
    ]
    if not field_columns or not occupation_columns:
        return pd.DataFrame(columns=["occupation", "count", "share"])
    field_column = field_columns[0]
    occupation_column = occupation_columns[0]
    filtered = dataframe[dataframe[field_column].astype(str).str.contains(field_of_study, case=False, na=False)]
    counts = Counter(filtered[occupation_column].dropna().astype(str))
    total = sum(counts.values()) or 1
    rows = [
        {"occupation": occupation, "count": count, "share": count / total}
        for occupation, count in counts.most_common(12)
    ]
    return pd.DataFrame(rows, columns=["occupation", "count", "share"])


def stable_percent(value: float) -> str:
    """Format a numeric ratio as a human-readable percentage."""
    if not math.isfinite(float(value)):
        return "0.0%"
    return f"{float(value) * 100:.1f}%"

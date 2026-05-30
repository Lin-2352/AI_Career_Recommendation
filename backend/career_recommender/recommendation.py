import re

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

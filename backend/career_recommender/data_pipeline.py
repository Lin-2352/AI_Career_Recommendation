from pathlib import Path

import pandas as pd

from backend.career_recommender.paths import MASTER_DATA_PATH, RAW_DATA_DIR

GUIDANCE_FILE = "career_guidance.csv"
RECOMMENDATION_FILE = "career_recommendation.csv"
CS_STUDENT_FILE = "computer_science_student_career_datasetMar62024.csv"
STUDENT_SCORES_FILE = "student_scores_sanitized.csv"

GUIDANCE_COLUMNS = {
    "Age",
    "Field_of_Study",
    "Year_of_Study",
    "GPA",
    "Relevant_Coursework",
    "Employment_Type",
    "Entrepreneurial_Experience",
    "Startup_Participation",
    "Career_Interests",
    "Entrepreneurial_Aspirations",
    "Top_Recommended_Industries",
    "Recommended_Career_Path",
}

RECOMMENDATION_COLUMNS = {
    "Age",
    "Education",
    "Skills",
    "Interests",
    "Recommended_Career",
}

CS_STUDENT_COLUMNS = {
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "C#",
    "PHP",
    "Ruby",
    "Swift",
    "Go",
    "Rust",
    "Others",
    "Software_Development_Experience",
    "Database_Management",
    "Networking_Skills",
    "Web_Development_Experience",
    "Communication_Skills",
    "Problem_Solving_Abilities",
    "Teamwork_Collaboration",
    "Time_Management",
    "Adaptability",
    "GPA",
    "Coursework_Completion_Status",
    "Academic_Achievements",
    "Personal_Interests",
    "Internship_Experience",
    "Certifications_Training",
    "Leadership_Experience",
    "Career_Goals",
}

STUDENT_SCORE_COLUMNS = {
    "part_time_job",
    "absence_days",
    "extracurricular_activities",
    "weekly_self_study_hours",
    "career_aspiration",
    "math_score",
    "history_score",
    "physics_score",
    "chemistry_score",
    "biology_score",
    "english_score",
    "geography_score",
}

LANGUAGE_COLUMNS = ["Python", "Java", "C++", "JavaScript", "C#", "PHP", "Ruby", "Swift", "Go", "Rust", "Others"]
COMPETENCY_COLUMNS = [
    "Software_Development_Experience",
    "Database_Management",
    "Networking_Skills",
    "Web_Development_Experience",
    "Communication_Skills",
    "Problem_Solving_Abilities",
    "Teamwork_Collaboration",
    "Time_Management",
    "Adaptability",
]
SUBJECT_COLUMNS = [
    "math_score",
    "history_score",
    "physics_score",
    "chemistry_score",
    "biology_score",
    "english_score",
    "geography_score",
]


def clean_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.replace(r"[_;|]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def age_band(value: object) -> str:
    if pd.isna(value):
        return "Age unknown"
    age = int(float(value))
    if age <= 23:
        return "Early career"
    if age <= 34:
        return "Growing professional"
    if age <= 50:
        return "Experienced professional"
    return "Senior professional"


def gpa_band(value: object) -> str:
    if pd.isna(value):
        return "GPA unknown"
    score = float(value)
    if score >= 3.5:
        return "Strong GPA"
    if score >= 3.0:
        return "Competitive GPA"
    if score >= 2.5:
        return "Developing GPA"
    return "Foundation GPA"


def rating_band(name: str, value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    label = name.replace("_", " ")
    if pd.isna(number):
        return ""
    if number >= 8:
        return f"Advanced {label}"
    if number >= 5:
        return f"Intermediate {label}"
    if number >= 2:
        return f"Foundation {label}"
    return ""


def subject_band(name: str, value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    label = name.replace("_score", "").replace("_", " ")
    if pd.isna(number):
        return ""
    if number >= 85:
        return f"Strong {label}"
    if number >= 70:
        return f"Solid {label}"
    if number >= 55:
        return f"Developing {label}"
    return f"Foundation {label}"


def study_band(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "Study hours unknown"
    if number >= 35:
        return "High self study hours"
    if number >= 20:
        return "Consistent self study hours"
    if number >= 10:
        return "Moderate self study hours"
    return "Limited self study hours"


def yes_no_label(value: object, positive: str, negative: str) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return negative
    return positive if number >= 1 else negative


def join_text_columns(columns: list[pd.Series]) -> pd.Series:
    table = pd.concat(columns, axis=1).fillna("")
    return (
        table.apply(lambda row: " ".join(str(part).strip() for part in row if str(part).strip()), axis=1)
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def validate_columns(dataframe: pd.DataFrame, required_columns: set[str], dataset_name: str) -> None:
    missing = sorted(required_columns.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def load_raw_datasets(raw_dir: Path = RAW_DATA_DIR, include_experimental: bool = False) -> dict[str, pd.DataFrame]:
    guidance_path = raw_dir / GUIDANCE_FILE
    recommendation_path = raw_dir / RECOMMENDATION_FILE
    if not guidance_path.exists():
        raise FileNotFoundError(f"Missing dataset: {guidance_path}")
    if not recommendation_path.exists():
        raise FileNotFoundError(f"Missing dataset: {recommendation_path}")
    guidance = pd.read_csv(guidance_path)
    recommendation = pd.read_csv(recommendation_path)
    validate_columns(guidance, GUIDANCE_COLUMNS, GUIDANCE_FILE)
    validate_columns(recommendation, RECOMMENDATION_COLUMNS, RECOMMENDATION_FILE)
    datasets = {GUIDANCE_FILE: guidance, RECOMMENDATION_FILE: recommendation}

    cs_path = raw_dir / CS_STUDENT_FILE
    if include_experimental and cs_path.exists():
        cs_students = pd.read_csv(cs_path)
        validate_columns(cs_students, CS_STUDENT_COLUMNS, CS_STUDENT_FILE)
        datasets[CS_STUDENT_FILE] = cs_students

    student_scores_path = raw_dir / STUDENT_SCORES_FILE
    if student_scores_path.exists():
        student_scores = pd.read_csv(student_scores_path)
        validate_columns(student_scores, STUDENT_SCORE_COLUMNS, STUDENT_SCORES_FILE)
        datasets[STUDENT_SCORES_FILE] = student_scores

    return datasets


def prepare_guidance_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["age"] = pd.to_numeric(dataframe["Age"], errors="coerce")
    output["source"] = "career_guidance"
    output["age_band"] = output["age"].map(age_band)

    field = clean_series(dataframe["Field_of_Study"])
    year = clean_series(dataframe["Year_of_Study"]).map(lambda value: f"Year {value}" if value else "")
    gpa = dataframe["GPA"].map(gpa_band)
    coursework = dataframe["Relevant_Coursework"].map(
        lambda value: yes_no_label(value, "Relevant coursework completed", "Foundational coursework")
    )
    employment = clean_series(dataframe["Employment_Type"])
    entrepreneurship = dataframe["Entrepreneurial_Experience"].map(
        lambda value: yes_no_label(value, "Entrepreneurship experience", "No entrepreneurship experience")
    )
    startup = dataframe["Startup_Participation"].map(
        lambda value: yes_no_label(value, "Startup participation", "No startup participation")
    )
    interests = clean_series(dataframe["Career_Interests"])
    aspirations = clean_series(dataframe["Entrepreneurial_Aspirations"])
    industries = clean_series(dataframe["Top_Recommended_Industries"])

    output["education_level"] = join_text_columns([field, year, gpa])
    output["technical_skills"] = join_text_columns([field, coursework, employment, entrepreneurship, startup])
    output["professional_interests"] = join_text_columns([interests, aspirations, industries])
    output["target_career"] = clean_series(dataframe["Recommended_Career_Path"])
    output["profile_text"] = join_text_columns(
        [output["age_band"], output["education_level"], output["technical_skills"], output["professional_interests"]]
    )
    return output


def prepare_recommendation_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["age"] = pd.to_numeric(dataframe["Age"], errors="coerce")
    output["source"] = "career_recommendation"
    output["age_band"] = output["age"].map(age_band)
    output["education_level"] = clean_series(dataframe["Education"])
    output["technical_skills"] = clean_series(dataframe["Skills"])
    output["professional_interests"] = clean_series(dataframe["Interests"])
    output["target_career"] = clean_series(dataframe["Recommended_Career"])
    output["profile_text"] = join_text_columns(
        [output["age_band"], output["education_level"], output["technical_skills"], output["professional_interests"]]
    )
    return output


def prepare_computer_science_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["age"] = pd.Series(pd.NA, index=dataframe.index, dtype="Int64")
    output["source"] = "computer_science_student_careers"
    output["age_band"] = "Student profile"
    output["education_level"] = join_text_columns(
        [
            pd.Series("Computer Science", index=dataframe.index),
            dataframe["GPA"].map(gpa_band),
            clean_series(dataframe["Coursework_Completion_Status"]),
            clean_series(dataframe["Academic_Achievements"]).map(lambda value: f"{value} academic achievements"),
        ]
    )
    output["technical_skills"] = dataframe.apply(
        lambda row: " ".join(
            value
            for value in [rating_band(column, row[column]) for column in [*LANGUAGE_COLUMNS, *COMPETENCY_COLUMNS]]
            if value
        ),
        axis=1,
    )
    output["professional_interests"] = join_text_columns(
        [
            clean_series(dataframe["Personal_Interests"]),
            clean_series(dataframe["Internship_Experience"]).map(lambda value: f"Internship experience {value}"),
            clean_series(dataframe["Certifications_Training"]).map(lambda value: f"Certifications training {value}"),
            clean_series(dataframe["Leadership_Experience"]).map(lambda value: f"Leadership experience {value}"),
        ]
    )
    output["target_career"] = clean_series(dataframe["Career_Goals"])
    output["profile_text"] = join_text_columns(
        [output["age_band"], output["education_level"], output["technical_skills"], output["professional_interests"]]
    )
    return output


def prepare_student_scores_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["age"] = pd.Series(pd.NA, index=dataframe.index, dtype="Int64")
    output["source"] = "student_scores"
    output["age_band"] = "School profile"
    output["education_level"] = join_text_columns(
        [
            pd.Series("Secondary school academic profile", index=dataframe.index),
            dataframe["weekly_self_study_hours"].map(study_band),
            dataframe["absence_days"].map(lambda value: f"{value} absence days"),
        ]
    )
    output["technical_skills"] = dataframe.apply(
        lambda row: " ".join(value for value in [subject_band(column, row[column]) for column in SUBJECT_COLUMNS] if value),
        axis=1,
    )
    output["professional_interests"] = join_text_columns(
        [
            dataframe["part_time_job"].map(
                lambda value: yes_no_label(value, "Part time work exposure", "No part time work exposure")
            ),
            dataframe["extracurricular_activities"].map(
                lambda value: yes_no_label(value, "Extracurricular activities", "No extracurricular activities")
            ),
        ]
    )
    output["target_career"] = clean_series(dataframe["career_aspiration"])
    output["profile_text"] = join_text_columns(
        [output["age_band"], output["education_level"], output["technical_skills"], output["professional_interests"]]
    )
    return output


def prepare_master_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [
        prepare_guidance_dataset(datasets[GUIDANCE_FILE]),
        prepare_recommendation_dataset(datasets[RECOMMENDATION_FILE]),
    ]
    if CS_STUDENT_FILE in datasets:
        frames.append(prepare_computer_science_dataset(datasets[CS_STUDENT_FILE]))
    if STUDENT_SCORES_FILE in datasets:
        frames.append(prepare_student_scores_dataset(datasets[STUDENT_SCORES_FILE]))

    master = pd.concat(frames, ignore_index=True)
    required = ["education_level", "technical_skills", "professional_interests", "target_career", "profile_text"]
    master = master.replace("", pd.NA).dropna(subset=required)
    master["age"] = pd.to_numeric(master["age"], errors="coerce").astype("Int64")
    master = master.drop_duplicates(subset=["profile_text", "target_career"]).reset_index(drop=True)
    return master


def build_master_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = MASTER_DATA_PATH,
    include_experimental: bool = False,
) -> pd.DataFrame:
    datasets = load_raw_datasets(raw_dir, include_experimental=include_experimental)
    master = prepare_master_dataset(datasets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    return master

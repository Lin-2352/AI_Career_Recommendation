from pathlib import Path

import pandas as pd

from career_recommender.paths import MASTER_DATA_PATH, RAW_DATA_DIR

GUIDANCE_FILE = "career_guidance.csv"
RECOMMENDATION_FILE = "career_recommendation.csv"

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


def load_raw_datasets(raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return guidance, recommendation


def prepare_guidance_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["source"] = "career_guidance"
    output["age"] = pd.to_numeric(dataframe["Age"], errors="coerce")
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
    output["source"] = "career_recommendation"
    output["age"] = pd.to_numeric(dataframe["Age"], errors="coerce")
    output["age_band"] = output["age"].map(age_band)
    output["education_level"] = clean_series(dataframe["Education"])
    output["technical_skills"] = clean_series(dataframe["Skills"])
    output["professional_interests"] = clean_series(dataframe["Interests"])
    output["target_career"] = clean_series(dataframe["Recommended_Career"])
    output["profile_text"] = join_text_columns(
        [output["age_band"], output["education_level"], output["technical_skills"], output["professional_interests"]]
    )
    return output


def prepare_master_dataset(guidance: pd.DataFrame, recommendation: pd.DataFrame) -> pd.DataFrame:
    master = pd.concat(
        [prepare_guidance_dataset(guidance), prepare_recommendation_dataset(recommendation)],
        ignore_index=True,
    )
    required = ["age", "education_level", "technical_skills", "professional_interests", "target_career", "profile_text"]
    master = master.replace("", pd.NA).dropna(subset=required)
    master["age"] = master["age"].astype(int)
    master = master.drop_duplicates(subset=["profile_text", "target_career"]).reset_index(drop=True)
    return master


def build_master_dataset(raw_dir: Path = RAW_DATA_DIR, output_path: Path = MASTER_DATA_PATH) -> pd.DataFrame:
    guidance, recommendation = load_raw_datasets(raw_dir)
    master = prepare_master_dataset(guidance, recommendation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    return master

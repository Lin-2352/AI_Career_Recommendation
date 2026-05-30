from __future__ import annotations

from backend.career_recommender.interview_prep import (
    analyze_filler_words,
    evaluate_star_response,
    generate_question_bank,
)


def test_star_missing_metric_flags_result() -> None:
    """STAR evaluation flags result statements that lack a numerical metric."""
    result = evaluate_star_response(
        "During my internship, the team had slow reporting. "
        "I needed to improve the dashboard. "
        "I built a cleaner SQL model and coordinated reviews. "
        "The result improved stakeholder confidence."
    )

    assert result["components"]["result"] is True
    assert result["result_has_metric"] is False
    assert result["metric_warning"] == "Result needs a numerical metric."


def test_star_with_metric_scores_cleanly() -> None:
    """STAR evaluation accepts quantified result evidence."""
    result = evaluate_star_response(
        "Situation: reporting was delayed. "
        "Task: I needed to reduce turnaround time. "
        "Action: I implemented automated SQL checks. "
        "Result: turnaround improved by 35%."
    )

    assert result["missing_components"] == []
    assert result["result_has_metric"] is True
    assert result["score"] == 100.0


def test_question_bank_fallback_has_required_sections() -> None:
    """Question-bank fallback returns complete interview sections without API calls."""
    bank = generate_question_bank("Data Analyst", ["SQL", "Python"])

    assert set(bank) == {"technical", "behavioral", "situational"}
    assert all(len(questions) == 5 for questions in bank.values())


def test_filler_word_analysis_counts_phrases() -> None:
    """Transcript analysis counts common filler words and rates them against word count."""
    analysis = analyze_filler_words("Um I actually think, you know, this is basically the right approach.")

    assert analysis["counts"]["um"] == 1
    assert analysis["counts"]["you know"] == 1
    assert analysis["total_fillers"] == 4
    assert analysis["filler_rate"] > 0

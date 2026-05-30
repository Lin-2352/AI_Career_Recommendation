from __future__ import annotations

from backend.career_recommender.resume_analyzer import (
    analyze_resume_text,
    calculate_ats_match_score,
    find_quantifiable_impact_gaps,
    flag_weak_action_verbs,
)


def test_ats_keyword_math_is_exact() -> None:
    """ATS score uses found keywords divided by total required keywords."""
    result = calculate_ats_match_score(
        "Built Python APIs and SQL dashboards for machine learning reporting.",
        ["Python", "SQL", "Machine Learning", "Docker"],
    )

    assert result["score"] == 75.0
    assert result["found_count"] == 3
    assert result["total_required_keywords"] == 4
    assert result["missing_keywords"] == ["docker"]


def test_weak_action_verbs_are_flagged() -> None:
    """Weak resume verbs produce stronger replacement suggestions."""
    findings = flag_weak_action_verbs("Helped build dashboards and was responsible for weekly reporting.")
    weak_phrases = {item["weak_phrase"] for item in findings}
    assert {"helped", "responsible for"}.issubset(weak_phrases)


def test_quantifiable_impact_checker_flags_bullets_without_metrics() -> None:
    """Resume bullets missing numbers are returned as impact gaps."""
    gaps = find_quantifiable_impact_gaps(
        "- Improved onboarding conversion by 18%\n"
        "- Built internal dashboard for customer success teams"
    )

    assert len(gaps) == 1
    assert "dashboard" in gaps[0]["bullet"].lower()


def test_full_resume_analysis_combines_core_sections() -> None:
    """Resume analysis returns ATS, verb, impact, and formatting sections."""
    analysis = analyze_resume_text(
        "Did dashboard migration for finance teams\nReduced manual reporting by 40%",
        ["dashboard", "finance", "python"],
    )

    assert analysis["ats"]["score"] == 66.66666666666666
    assert analysis["action_verb_findings"]
    assert analysis["impact_gaps"]
    assert analysis["formatting_warnings"]

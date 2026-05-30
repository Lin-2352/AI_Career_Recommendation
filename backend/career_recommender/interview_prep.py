"""Interview preparation prompts, STAR evaluation, and transcript analysis."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from backend.core.api_manager import APIManager

FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "actually", "literally", "sort of", "kind of"]


def build_question_prompt(target_role: str, skills: Sequence[str], seniority: str = "entry level") -> str:
    """Build an LLM prompt for role-specific technical, behavioral, and situational questions."""
    skill_text = ", ".join(skill for skill in skills if str(skill).strip())
    return (
        "Generate interview questions as strict JSON with keys technical, behavioral, and situational. "
        "Each key must contain five concise questions tailored to the candidate.\n"
        f"Target role: {target_role}\nSeniority: {seniority}\nSkills: {skill_text}"
    )


def generate_question_bank(
    target_role: str,
    skills: Sequence[str],
    seniority: str = "entry level",
    api_manager: APIManager | None = None,
) -> dict[str, list[str]]:
    """Generate an interview question bank through an LLM with deterministic fallback."""
    if api_manager is None:
        skill_text = ", ".join(skill for skill in skills if str(skill).strip()) or "the candidate's stated skills"
        return {
            "technical": [
                f"How would you apply {skill_text} to solve a realistic {target_role} problem?",
                "Explain a project where you improved reliability, performance, or maintainability.",
                "How do you validate that a technical solution is working as intended?",
                "Describe how you debug an unfamiliar production issue.",
                "What tradeoffs would you consider before choosing a tool or framework?",
            ],
            "behavioral": [
                "Tell me about a time you took ownership of an ambiguous problem.",
                "Describe a conflict with a teammate and how you resolved it.",
                "Give an example of feedback you received and how it changed your work.",
                "Tell me about a deadline risk and how you communicated it.",
                "Describe a time you learned a skill quickly for a project.",
            ],
            "situational": [
                "What would you do if requirements changed two days before delivery?",
                "How would you prioritize three urgent tasks from different stakeholders?",
                "How would you handle discovering a mistake after deployment?",
                "What would you do if your proposed approach was challenged by senior peers?",
                "How would you ramp up on a domain you have never worked in before?",
            ],
        }
    prompt = build_question_prompt(target_role, skills, seniority)
    response = api_manager.chat_completion(
        messages=[
            {"role": "system", "content": "You generate practical interview questions and respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1100,
        temperature=0.25,
    )
    parsed = parse_question_bank(response)
    return parsed if parsed else generate_question_bank(target_role, skills, seniority, None)


def parse_question_bank(response: str) -> dict[str, list[str]]:
    """Parse a JSON question bank and discard malformed sections."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return {}
    output: dict[str, list[str]] = {}
    for key in ["technical", "behavioral", "situational"]:
        values = payload.get(key, [])
        if isinstance(values, list):
            output[key] = [str(item).strip() for item in values if str(item).strip()][:5]
    return output if all(output.get(key) for key in ["technical", "behavioral", "situational"]) else {}


def evaluate_star_response(response_text: str) -> dict[str, Any]:
    """Evaluate whether an interview response contains STAR components and measurable result evidence."""
    text = response_text.strip()
    lowered = text.lower()
    components = {
        "situation": has_component(lowered, ["situation", "context", "when", "while", "at my", "during"]),
        "task": has_component(lowered, ["task", "goal", "needed to", "responsible for", "objective"]),
        "action": has_component(lowered, ["action", "i built", "i led", "i created", "i implemented", "i analyzed", "i coordinated"]),
        "result": has_component(lowered, ["result", "outcome", "improved", "reduced", "increased", "delivered", "achieved"]),
    }
    missing = [name for name, present in components.items() if not present]
    result_has_metric = bool(re.search(r"(\d+(?:\.\d+)?\s?%|\$\s?\d+|₹\s?\d+|\b\d+\b|\b\d+x\b)", text))
    score = (sum(1 for present in components.values() if present) / 4.0) * 100.0
    if components["result"] and not result_has_metric:
        score = max(0.0, score - 15.0)
    return {
        "components": components,
        "missing_components": missing,
        "result_has_metric": result_has_metric,
        "metric_warning": "" if result_has_metric else "Result needs a numerical metric.",
        "score": score,
    }


def has_component(text: str, markers: Sequence[str]) -> bool:
    """Return whether any STAR marker appears in the response text."""
    return any(marker in text for marker in markers)


def analyze_filler_words(transcript: str) -> dict[str, Any]:
    """Count filler words in an interview transcript."""
    lowered = transcript.lower()
    counts: dict[str, int] = {}
    for filler in FILLER_WORDS:
        pattern = rf"(?<![a-z]){re.escape(filler)}(?![a-z])"
        count = len(re.findall(pattern, lowered))
        if count:
            counts[filler] = count
    total = sum(counts.values())
    word_count = len(re.findall(r"\b\w+\b", transcript))
    filler_rate = total / max(word_count, 1)
    return {"counts": counts, "total_fillers": total, "word_count": word_count, "filler_rate": filler_rate}

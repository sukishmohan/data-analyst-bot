"""
insight_generator.py — Insight Generation & Reflection Agent (Ollama / Llama3)

Two responsibilities:
  1. Generate plain-English business insights from raw analytical results.
  2. Reflection loop: evaluate if the answer is complete and sufficient.

Uses local Llama3 model via Ollama — no paid APIs.
"""

import json
from typing import Any

from utils import call_ollama, call_ollama_json, get_logger, truncate

log = get_logger("insight_generator")


# ──────────────────────────────────────────────
# 1. Insight Generation
# ──────────────────────────────────────────────

_INSIGHT_SYSTEM = """\
You are a senior business analyst. Given the raw analysis output
and the original query, write a concise insight summary.

Rules:
- Write 3-6 bullet points of key findings.
- Use simple business language, no code or jargon.
- Include specific numbers, percentages, and comparisons.
- Highlight surprises, risks, or opportunities.
- End with a one-sentence executive recommendation.
- Format with bullet points using "-" character.
"""


def generate_insights(
    query: str,
    result_text: str,
    plan_title: str,
) -> str:
    """Generate human-friendly insights from the analysis result."""
    user_msg = (
        f"Original Question: {query}\n\n"
        f"Analysis Title: {plan_title}\n\n"
        f"Raw Result:\n{truncate(result_text, 2500)}\n\n"
        "Write the insight summary now."
    )

    insights = call_ollama(prompt=user_msg, system=_INSIGHT_SYSTEM, temperature=0.4)
    log.info("Generated insights (%d chars).", len(insights))
    return insights


# ──────────────────────────────────────────────
# 2. Reflection / Completeness Check
# ──────────────────────────────────────────────

_REFLECTION_SYSTEM = """\
You are a quality-assurance reviewer for data analysis.

Given the original query, the analysis plan, and the result,
decide whether the answer is COMPLETE and CORRECT.

Return JSON:
{
  "is_complete": true or false,
  "confidence": 0.0 to 1.0,
  "issues": ["issue1", "issue2"],
  "suggestion": "what additional analysis to run, or empty string"
}

Guidelines:
- Mark incomplete if the result is empty, has obvious errors,
  or clearly doesn't answer the question.
- Mark incomplete if important context is missing.
- confidence < 0.5 means definitely re-run.
- Be generous: a reasonable partial answer is still "complete".
"""


def reflect(
    query: str,
    plan: dict[str, Any],
    result_text: str,
) -> dict[str, Any]:
    """Evaluate whether the analysis result adequately answers the query."""
    user_msg = (
        f"Query: {query}\n\n"
        f"Plan: {json.dumps(plan.get('steps', []), indent=2)}\n\n"
        f"Result:\n{truncate(result_text, 2000)}\n\n"
        "Evaluate completeness. Return JSON only."
    )

    result = call_ollama_json(prompt=user_msg, system=_REFLECTION_SYSTEM, temperature=0.15)
    result.setdefault("is_complete", True)
    result.setdefault("confidence", 0.8)
    result.setdefault("issues", [])
    result.setdefault("suggestion", "")

    log.info("Reflection: complete=%s  confidence=%.2f",
             result["is_complete"], result["confidence"])
    if result["issues"]:
        log.info("  Issues: %s", result["issues"])

    return result

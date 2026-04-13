"""
planner.py — Planning Agent (Ollama / Llama3)

Takes the structured intent from the Query Parser and generates a
step-by-step analytical execution plan. Uses local Llama3 model.
"""

import json
from typing import Any

from utils import call_ollama_json, get_logger

log = get_logger("planner")


_SYSTEM_PROMPT = """\
You are an expert data-analysis planner.

Given a structured query intent and dataset schema, produce a JSON execution plan:
{{
  "title": "<concise title for this analysis>",
  "steps": [
    {{
      "step_number": 1,
      "action": "<description of what to do>",
      "type": "<data_prep | aggregation | filter | sort | compute | visualize | forecast>"
    }}
  ],
  "result_type": "<table | chart | number | text>",
  "chart_config": {{
    "type": "<line | bar | horizontal_bar | pie | histogram | scatter | heatmap | none>",
    "x": "<column for x-axis>",
    "y": "<column for y-axis>",
    "title": "<chart title>",
    "xlabel": "<x-axis label>",
    "ylabel": "<y-axis label>",
    "color_by": null
  }},
  "needs_forecast": false,
  "explanation_hint": "<what kind of business insight to highlight>"
}}

RULES:
- Steps must be sequential and complete.
- Always start with data preparation (date parsing, type conversion) if needed.
- Include a visualization step at the end if chart_type is not "none".
- If the query mentions "loss", "negative", or "loss-making", add a filter for profit < 0.
- Use exact column names from the schema.
- Return ONLY valid JSON.

DATASET SCHEMA:
{schema}

NUMERIC COLUMNS: {numeric_cols}
DATE COLUMNS: {date_cols}
CATEGORICAL COLUMNS: {cat_cols}
"""


def generate_plan(intent: dict[str, Any], profile: dict) -> dict[str, Any]:
    """Generate a step-by-step execution plan from the parsed intent."""
    schema_lines = [f"  - {c} ({profile['dtypes'][c]})"
                    for c in profile["columns"]]
    schema_str = "\n".join(schema_lines)

    system = _SYSTEM_PROMPT.format(
        schema=schema_str,
        numeric_cols=profile.get("numeric_columns", []),
        date_cols=profile.get("date_columns", []),
        cat_cols=profile.get("categorical_columns", []),
    )

    user_msg = (
        f"Query intent:\n{json.dumps(intent, indent=2)}\n\n"
        "Generate the execution plan."
    )

    plan = call_ollama_json(prompt=user_msg, system=system, temperature=0.15)

    # Defaults
    plan.setdefault("title", intent.get("description", "Analysis"))
    plan.setdefault("steps", [])
    plan.setdefault("result_type", "table")
    plan.setdefault("chart_config", {"type": "none"})
    plan.setdefault("needs_forecast", False)
    plan.setdefault("explanation_hint", "")

    log.info("Plan (%d steps): %s", len(plan["steps"]), plan["title"])
    for s in plan["steps"]:
        log.info("  Step %s [%s]: %s",
                 s.get("step_number", "?"), s.get("type", "?"), s.get("action", ""))

    return plan

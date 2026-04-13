"""
query_parser.py — Query Understanding Agent (Ollama / Llama3)

Accepts a natural-language user query and produces a structured
intent object describing:
  - metrics   (sales, profit, quantity ...)
  - dimensions (date, region, category ...)
  - filters, chart_type, query_type, time_granularity

Uses local Ollama Llama3 model — no paid APIs.
"""

import json
from typing import Any

from utils import call_ollama_json, get_logger

log = get_logger("query_parser")


_SYSTEM_PROMPT = """\
You are a data-analytics query parser. You receive a natural-language question about a dataset.

Return ONLY a JSON object with these keys:
{{
  "metrics": ["<numeric column names to compute>"],
  "aggregation": "<sum | mean | count | max | min | median>",
  "dimensions": ["<column names for grouping or x-axis>"],
  "filters": [
    {{"column": "<col>", "operator": "<== | != | > | < | >= | <= | in | contains>", "value": "<val>"}}
  ],
  "sort": {{"by": "<column>", "ascending": true or false}},
  "limit": null,
  "query_type": "<trend | comparison | ranking | distribution | correlation | forecast | general>",
  "chart_type": "<line | bar | horizontal_bar | pie | histogram | scatter | none>",
  "time_granularity": "<daily | weekly | monthly | quarterly | yearly | null>",
  "description": "<one-line restatement of the user intent>"
}}

RULES:
- Only use column names from the dataset schema below.
- For trend queries with dates, put date column in "dimensions" and set time_granularity.
- Default aggregation is "sum" for monetary metrics, "count" for quantity.
- Return ONLY valid JSON.

DATASET COLUMNS & TYPES:
{schema}

SAMPLE VALUES:
{samples}
"""


def parse_query(query: str, profile: dict) -> dict[str, Any]:
    """
    Parse a natural-language query into a structured intent dictionary.
    """
    # Build schema description
    schema_lines = []
    for col in profile["columns"]:
        dtype = profile["dtypes"][col]
        schema_lines.append(f"  - {col} ({dtype})")
    schema_str = "\n".join(schema_lines)

    # Sample values
    sample_lines = []
    for col, vals in profile.get("sample_values", {}).items():
        sample_lines.append(f"  - {col}: {vals[:8]}")
    sample_str = "\n".join(sample_lines) if sample_lines else "(none)"

    system = _SYSTEM_PROMPT.format(schema=schema_str, samples=sample_str)

    intent = call_ollama_json(prompt=query, system=system, temperature=0.1)

    # Validate & patch defaults
    intent.setdefault("metrics", [])
    intent.setdefault("aggregation", "sum")
    intent.setdefault("dimensions", [])
    intent.setdefault("filters", [])
    intent.setdefault("sort", None)
    intent.setdefault("limit", None)
    intent.setdefault("query_type", "general")
    intent.setdefault("chart_type", "bar")
    intent.setdefault("time_granularity", None)
    intent.setdefault("description", query)

    log.info("Parsed intent: %s", json.dumps(intent, indent=2))
    return intent

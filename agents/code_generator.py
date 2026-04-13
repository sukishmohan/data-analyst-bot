"""
code_generator.py — Code Generation Agent (Ollama / Llama3)

Generates clean, executable Python/Pandas code from the execution plan.
Uses local Llama3 model via Ollama. Includes regeneration for auto-fix.
"""

import json
import re
from typing import Any

from utils import call_ollama, get_logger

log = get_logger("code_generator")


_SYSTEM_PROMPT = """\
You are a Python code generator for data analysis.

You will receive an execution plan and a dataset schema.

Generate a SINGLE block of valid Python code that:
- Assumes `df` is an already-loaded pandas DataFrame (do NOT load CSV).
- Assumes `import pandas as pd`, `import numpy as np`, `import matplotlib`, `import matplotlib.pyplot as plt` are already done.
- Stores the analytical result in a variable called `result` (DataFrame, Series, or scalar).
- If a chart is needed, create a matplotlib figure:
  - Use `fig, ax = plt.subplots(figsize=(12, 6))`.
  - Add title, axis labels, and value annotations.
  - Use `plt.tight_layout()`.
  - Do NOT call `plt.show()`.
- Print the result with `print(result)` or `print(result.to_string())` for DataFrames.

SAFETY RULES (MANDATORY):
- Do NOT import os, sys, subprocess, shutil, or pathlib.
- Do NOT read or write files.
- Do NOT make network requests.
- Do NOT use exec() or eval().
- Only use: pandas, numpy, matplotlib, datetime, math, statistics.

CODE QUALITY:
- Add brief comments for each logical block.
- Handle edge cases (empty results, division by zero).
- Use .copy() when modifying the original DataFrame.
- For percentage calculations, round to 2 decimal places.
- For date operations, use pd.to_datetime() with errors='coerce'.
- CRITICAL: When resampling dates in Pandas >= 2.2, use 'ME' (month-end) instead of 'M', 'YE' instead of 'Y'.
- CRITICAL: When applying aggregations (e.g. .sum(), .mean()) on grouped data, ALWAYS select specific numeric columns first (e.g. `df.groupby('A')['B'].sum()`) or use `numeric_only=True` to prevent accidental string concatenation!

Return ONLY the Python code. No markdown fences, no explanations, no extra text.
"""


def generate_code(plan: dict[str, Any], profile: dict) -> str:
    """Generate executable Pandas code from the plan and dataset profile."""
    schema_lines = [f"  - {c} ({profile['dtypes'][c]})"
                    for c in profile["columns"]]
    schema_str = "\n".join(schema_lines)

    sample_info = ""
    for col, vals in profile.get("sample_values", {}).items():
        sample_info += f"  {col}: {vals[:6]}\n"

    user_msg = (
        f"EXECUTION PLAN:\n{json.dumps(plan, indent=2)}\n\n"
        f"DATASET SCHEMA:\n{schema_str}\n\n"
        f"SAMPLE VALUES:\n{sample_info}\n"
        "Generate the Python code now. Return ONLY code, nothing else."
    )

    code = call_ollama(prompt=user_msg, system=_SYSTEM_PROMPT, temperature=0.15)
    code = _strip_fences(code)

    log.info("Generated %d lines of code.", code.count("\n") + 1)
    return code


def regenerate_code(
    plan: dict[str, Any],
    profile: dict,
    previous_code: str,
    error_msg: str,
) -> str:
    """
    Regenerate code after a failed execution attempt.
    Sends the previous code AND the error message back to the LLM.
    """
    schema_lines = [f"  - {c} ({profile['dtypes'][c]})"
                    for c in profile["columns"]]
    schema_str = "\n".join(schema_lines)

    user_msg = (
        f"The following code FAILED with an error.\n\n"
        f"PLAN:\n{json.dumps(plan, indent=2)}\n\n"
        f"DATASET SCHEMA:\n{schema_str}\n\n"
        f"PREVIOUS CODE:\n{previous_code}\n\n"
        f"ERROR:\n{error_msg}\n\n"
        "Fix the code and return ONLY valid Python. "
        "Make sure to handle edge cases and column name mismatches."
    )

    code = call_ollama(prompt=user_msg, system=_SYSTEM_PROMPT, temperature=0.2)
    code = _strip_fences(code)

    log.info("Regenerated code (%d lines) after error.", code.count("\n") + 1)
    return code


def _strip_fences(code: str) -> str:
    """Remove markdown code fences (```python ... ```) if present."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence
        code = "\n".join(lines)
    return code.strip()

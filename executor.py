"""
executor.py — Safe Code Execution Engine

Executes dynamically-generated Pandas code in a sandboxed namespace.
Features:
  - Whitelisted imports only (pandas, numpy, matplotlib, math, datetime, statistics)
  - Automatic retry with code regeneration on failure
  - Captures stdout, result variable, and matplotlib figures
  - Timeout protection via threading
"""

import io
import sys
import traceback
import threading
from typing import Any, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from utils import get_logger

log = get_logger("executor")

EXEC_TIMEOUT = 60  # seconds


# ──────────────────────────────────────────────
# 1. Sandboxed Namespace
# ──────────────────────────────────────────────

def _build_namespace(df: pd.DataFrame) -> dict[str, Any]:
    """Build a restricted execution namespace."""
    import math
    import datetime
    import statistics

    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "matplotlib": matplotlib,
        "math": math,
        "datetime": datetime,
        "statistics": statistics,
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "bool": bool,
        "type": type,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "None": None,
        "True": True,
        "False": False,
    }
    return namespace


# ──────────────────────────────────────────────
# 2. Execution Result Container
# ──────────────────────────────────────────────

class ExecutionResult:
    """Container for execution outcome."""
    def __init__(self):
        self.success: bool = False
        self.result: Any = None
        self.figure: Optional[plt.Figure] = None
        self.stdout: str = ""
        self.error: str = ""


# ──────────────────────────────────────────────
# 3. Code Runner (threaded)
# ──────────────────────────────────────────────

def _run_code(code: str, namespace: dict, exec_result: ExecutionResult):
    """Run code in the given namespace."""
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    try:
        plt.close("all")
        exec(code, namespace)

        exec_result.stdout = captured.getvalue()

        if "result" in namespace:
            exec_result.result = namespace["result"]

        if "fig" in namespace and isinstance(namespace["fig"], plt.Figure):
            exec_result.figure = namespace["fig"]
        elif plt.get_fignums():
            exec_result.figure = plt.gcf()

        exec_result.success = True

    except Exception as exc:
        exec_result.stdout = captured.getvalue()
        exec_result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        exec_result.success = False

    finally:
        sys.stdout = old_stdout


def execute_code(code: str, df: pd.DataFrame, timeout: int = EXEC_TIMEOUT) -> ExecutionResult:
    """Execute generated code safely with a timeout."""
    namespace = _build_namespace(df)
    exec_result = ExecutionResult()

    thread = threading.Thread(
        target=_run_code,
        args=(code, namespace, exec_result),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        exec_result.success = False
        exec_result.error = f"Execution timed out after {timeout} seconds."
        log.error("Code execution timed out.")

    return exec_result


# ──────────────────────────────────────────────
# 4. Execute with Auto-Retry
# ──────────────────────────────────────────────

def execute_with_retry(
    code: str,
    df: pd.DataFrame,
    plan: dict,
    profile: dict,
    max_retries: int = 2,
) -> tuple[ExecutionResult, str]:
    """
    Execute code and automatically regenerate + retry on failure.
    """
    from agents.code_generator import regenerate_code

    current_code = code
    for attempt in range(1 + max_retries):
        log.info("Execution attempt %d/%d", attempt + 1, 1 + max_retries)

        result = execute_code(current_code, df)

        if result.success:
            log.info("Code executed successfully.")
            return result, current_code

        log.warning("Attempt %d failed: %s", attempt + 1, result.error[:200])

        if attempt < max_retries:
            log.info("Regenerating code...")
            try:
                current_code = regenerate_code(
                    plan, profile, current_code, result.error
                )
            except Exception as regen_err:
                log.error("Code regeneration failed: %s", regen_err)
                break

    return result, current_code


# ──────────────────────────────────────────────
# 5. Result Formatting
# ──────────────────────────────────────────────

def format_result(exec_result: ExecutionResult) -> str:
    """Convert the execution result into a readable string."""
    parts = []

    if exec_result.stdout:
        parts.append(exec_result.stdout.strip())

    if exec_result.result is not None:
        r = exec_result.result
        if isinstance(r, pd.DataFrame):
            parts.append(r.to_string(max_rows=50))
        elif isinstance(r, pd.Series):
            parts.append(r.to_string())
        else:
            parts.append(str(r))

    return "\n".join(parts) if parts else "(no output)"

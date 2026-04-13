"""
main.py — AI Data Analyst Agent — Main Orchestrator (Ollama Edition)

Coordinates all agent modules in a structured pipeline:

  User Query -> Query Parser -> Planner -> Code Generator ->
  Executor (+ Retry) -> Reflection -> Insights -> Visualization -> Report

Runs fully local using Ollama + Llama3. No paid APIs.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from utils import get_logger, safe_print, check_ollama_available, check_model_available, OLLAMA_MODEL
from data_loader import load_and_prepare, load_csv, profile_dataset, auto_clean
from agents.query_parser import parse_query
from agents.planner import generate_plan
from agents.code_generator import generate_code
from agents.insight_generator import generate_insights, reflect
from executor import execute_with_retry, format_result
from visualization import save_figure, generate_fallback_chart
from report_exporter import export_text_report, export_pdf_report
from forecaster import forecast_series

log = get_logger("main")

# Global state
_df: Optional[pd.DataFrame] = None
_profile: Optional[dict] = None
MAX_REFLECTION_RETRIES = 1


# ──────────────────────────────────────────────
# 1. Initialisation
# ──────────────────────────────────────────────

def initialise(csv_path: str | Path = None) -> tuple[pd.DataFrame, dict]:
    """Load and prepare the dataset. Caches globally."""
    global _df, _profile

    if _df is not None and _profile is not None:
        return _df, _profile

    if csv_path is None:
        project = Path(__file__).resolve().parent
        candidates = list(project.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError("No CSV file found in project directory.")
        csv_path = candidates[0]
        log.info("Auto-detected dataset: %s", csv_path.name)

    _df, _profile = load_and_prepare(csv_path)
    return _df, _profile


# ──────────────────────────────────────────────
# 2. Core Analysis Pipeline
# ──────────────────────────────────────────────

def run_analysis(query: str, export_report: bool = False) -> dict[str, Any]:
    """Run the full agent pipeline for a single query."""
    df, profile = initialise()

    safe_print("\n" + "=" * 60)
    safe_print(f"  QUERY: {query}")
    safe_print("=" * 60)

    # Step 1: Parse query
    safe_print("\n[Step 1] Understanding your question...")
    intent = parse_query(query, profile)
    safe_print(f"   Intent: {intent.get('description', query)}")
    safe_print(f"   Type: {intent.get('query_type')}  |  Chart: {intent.get('chart_type')}")

    # Step 2: Generate plan
    safe_print("\n[Step 2] Creating execution plan...")
    plan = generate_plan(intent, profile)
    safe_print(f"   Title: {plan['title']}")
    for s in plan.get("steps", []):
        safe_print(f"   {s.get('step_number', '?')}. [{s.get('type', '')}] {s.get('action', '')}")

    # Step 3: Handle forecast queries
    if plan.get("needs_forecast") or intent.get("query_type") == "forecast":
        return _run_forecast(query, intent, plan, profile, df, export_report)

    # Step 4: Generate code
    safe_print("\n[Step 3] Generating analysis code...")
    code = generate_code(plan, profile)
    safe_print(f"   Generated {code.count(chr(10)) + 1} lines of code.")

    # Step 5: Execute with retry
    safe_print("\n[Step 4] Executing analysis...")
    exec_result, final_code = execute_with_retry(code, df, plan, profile, max_retries=2)

    if not exec_result.success:
        safe_print(f"\n[ERROR] Execution failed after retries: {exec_result.error[:300]}")
        return {
            "query": query, "intent": intent, "plan": plan,
            "code": final_code, "result_text": "",
            "insights": "Analysis could not be completed due to an execution error.",
            "chart_path": None, "reflection": None, "report_paths": [],
            "error": exec_result.error,
        }

    result_text = format_result(exec_result)

    # Step 6: Reflection loop
    safe_print("\n[Step 5] Validating results...")
    reflection = reflect(query, plan, result_text)
    retries = 0

    while (not reflection.get("is_complete", True)
           and reflection.get("confidence", 1.0) < 0.5
           and retries < MAX_REFLECTION_RETRIES):
        retries += 1
        safe_print(f"   Result incomplete (confidence={reflection['confidence']:.0%}). "
                   f"Re-running... (attempt {retries})")
        suggestion = reflection.get("suggestion", "")
        if suggestion:
            refined_intent = parse_query(suggestion, profile)
            plan = generate_plan(refined_intent, profile)
            code = generate_code(plan, profile)
            exec_result, final_code = execute_with_retry(code, df, plan, profile, max_retries=1)
            if exec_result.success:
                result_text = format_result(exec_result)
            reflection = reflect(query, plan, result_text)

    safe_print(f"   Confidence: {reflection.get('confidence', 0):.0%}")

    # Step 7: Visualization
    safe_print("\n[Step 6] Generating visualization...")
    chart_path = None
    fig = exec_result.figure

    if fig is None:
        fig = generate_fallback_chart(
            exec_result.result,
            plan.get("chart_config", {}),
            plan.get("title", "Analysis"),
        )

    if fig is not None:
        chart_path = save_figure(fig, name=plan.get("title", "chart"))
        safe_print(f"   Chart saved: {chart_path}")
    else:
        safe_print("   No chart generated (numeric/text result).")

    # Step 8: Generate insights
    safe_print("\n[Step 7] Generating business insights...")
    insights = generate_insights(query, result_text, plan.get("title", ""))
    safe_print("\n" + "-" * 50)
    safe_print("INSIGHTS:")
    safe_print("-" * 50)
    safe_print(insights)

    # Step 9: Export reports
    report_paths = []
    if export_report:
        safe_print("\n[Step 8] Exporting reports...")
        txt_path = export_text_report(
            query, plan["title"], plan.get("steps", []),
            result_text, insights,
        )
        report_paths.append(txt_path)
        safe_print(f"   Text report: {txt_path}")

        pdf_path = export_pdf_report(
            query, plan["title"], plan.get("steps", []),
            result_text, insights, chart_path,
        )
        report_paths.append(pdf_path)
        safe_print(f"   PDF report:  {pdf_path}")

    safe_print("\n" + "=" * 60)
    safe_print("  ANALYSIS COMPLETE")
    safe_print("=" * 60)

    return {
        "query": query, "intent": intent, "plan": plan,
        "code": final_code, "result_text": result_text,
        "insights": insights, "chart_path": chart_path,
        "reflection": reflection, "report_paths": report_paths,
    }


# ──────────────────────────────────────────────
# 3. Forecast Handler
# ──────────────────────────────────────────────

def _run_forecast(query, intent, plan, profile, df, export_report):
    """Handle forecast-type queries."""
    safe_print("\n[Forecast] Running forecast pipeline...")

    date_cols = profile.get("date_columns", [])
    date_col = date_cols[0] if date_cols else "Order Date"

    metrics = intent.get("metrics", ["Sales"])
    value_col = metrics[0] if metrics else "Sales"

    forecast_df, fig = forecast_series(df, date_col, value_col, periods=6, freq="MS")

    result_text = forecast_df.to_string()
    chart_path = save_figure(fig, name=f"{value_col}_forecast")
    safe_print(f"   Forecast chart saved: {chart_path}")

    insights = generate_insights(query, result_text, f"{value_col} Forecast")
    safe_print("\n" + "-" * 50)
    safe_print("INSIGHTS:")
    safe_print("-" * 50)
    safe_print(insights)

    report_paths = []
    if export_report:
        txt_path = export_text_report(
            query, f"{value_col} Forecast", plan.get("steps", []),
            result_text, insights,
        )
        report_paths.append(txt_path)

    safe_print("\n" + "=" * 60)
    safe_print("  FORECAST COMPLETE")
    safe_print("=" * 60)

    return {
        "query": query, "intent": intent, "plan": plan,
        "code": "(forecast module)", "result_text": result_text,
        "insights": insights, "chart_path": chart_path,
        "reflection": {"is_complete": True, "confidence": 0.9},
        "report_paths": report_paths,
    }


# ──────────────────────────────────────────────
# 4. Interactive CLI
# ──────────────────────────────────────────────

def interactive_cli():
    """Run the agent in interactive terminal mode."""
    banner = """
============================================================
                                                              
   AI DATA ANALYST AGENT  v2.0  (Local Ollama Edition)       
   ------------------------------------------------         
   Powered by Llama3 via Ollama (fully local, free)          
                                                              
   Ask me anything about your Superstore data!               
                                                              
   Sample queries:                                           
     - Show monthly sales trend                              
     - Which category has highest profit?                    
     - Find loss-making sub-categories                       
     - Compare regions by revenue                            
     - Forecast sales for next 6 months                      
                                                              
   Commands:                                                 
     export  -- Toggle report export (PDF + text)            
     profile -- Show dataset profile                         
     quit    -- Exit                                         
                                                              
============================================================
"""
    safe_print(banner)

    # Check Ollama availability
    if not check_ollama_available():
        safe_print("[ERROR] Ollama is not running!")
        safe_print("   Start it with: ollama serve")
        safe_print("   Then pull the model: ollama pull llama3")
        sys.exit(1)

    if not check_model_available():
        safe_print(f"[WARNING] Model '{OLLAMA_MODEL}' not found in Ollama.")
        safe_print(f"   Pull it with: ollama pull {OLLAMA_MODEL}")
        safe_print("   Attempting to continue anyway...\n")

    # Load dataset
    try:
        initialise()
    except Exception as exc:
        safe_print(f"\n[ERROR] Failed to load dataset: {exc}")
        sys.exit(1)

    export_on = False

    while True:
        try:
            query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("\nGoodbye!")
            break

        if not query:
            continue

        cmd = query.lower()
        if cmd in ("quit", "exit", "q"):
            safe_print("Goodbye!")
            break
        elif cmd == "export":
            export_on = not export_on
            safe_print(f"   Report export: {'ON' if export_on else 'OFF'}")
            continue
        elif cmd == "profile":
            from data_loader import print_dataset_report
            print_dataset_report(_profile)
            continue

        try:
            run_analysis(query, export_report=export_on)
        except Exception as exc:
            log.error("Analysis failed: %s", exc, exc_info=True)
            safe_print(f"\n[ERROR] {exc}")
            safe_print("   Please try rephrasing your question.")


if __name__ == "__main__":
    interactive_cli()

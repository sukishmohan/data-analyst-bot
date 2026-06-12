"""
report_exporter.py — Export analysis reports to PDF and text.

Generates a structured report containing:
  • Analysis title and timestamp
  • Original query
  • Execution plan summary
  • Result table (truncated)
  • Chart image (embedded in PDF)
  • Business insights
"""

import datetime
from pathlib import Path
from typing import Optional

from utils import get_logger, REPORTS_DIR, safe_filename

log = get_logger("report_exporter")


# ──────────────────────────────────────────────
# 1. Text Report
# ──────────────────────────────────────────────

def export_text_report(
    query: str,
    plan_title: str,
    plan_steps: list[dict],
    result_text: str,
    insights: str,
    filename: Optional[str] = None,
) -> Path:
    """Export analysis as a plain-text report."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = filename or safe_filename(plan_title) + "_report.txt"
    path = REPORTS_DIR / fname

    lines = [
        "=" * 70,
        f"  ANALYSIS REPORT — {plan_title}",
        "=" * 70,
        f"Generated: {ts}",
        "",
        "QUERY:",
        f"  {query}",
        "",
        "EXECUTION PLAN:",
    ]
    for s in plan_steps:
        lines.append(f"  {s.get('step_number', '?')}. [{s.get('type', '')}] {s.get('action', '')}")

    lines += [
        "",
        "RESULT:",
        result_text[:5000],
        "",
        "INSIGHTS:",
        insights,
        "",
        "=" * 70,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Text report saved: %s", path)
    return path


# ──────────────────────────────────────────────
# 2. PDF Report
# ──────────────────────────────────────────────

def export_pdf_report(
    query: str,
    plan_title: str,
    plan_steps: list[dict],
    result_text: str,
    insights: str,
    chart_path: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Export analysis as a PDF report.
    Requires the fpdf2 library.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        log.warning("fpdf2 not installed — falling back to text report.")
        return export_text_report(query, plan_title, plan_steps,
                                  result_text, insights, filename)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = filename or safe_filename(plan_title) + "_report.pdf"
    path = REPORTS_DIR / fname

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, plan_title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {ts}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Query
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Query", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, query)
    pdf.ln(5)

    # Plan
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Execution Plan", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for s in plan_steps:
        step_text = f"{s.get('step_number', '?')}. [{s.get('type', '')}] {s.get('action', '')}"
        pdf.multi_cell(0, 5, step_text)
    pdf.ln(5)

    # Chart
    if chart_path and chart_path.exists():
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Visualization", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        try:
            pdf.image(str(chart_path), w=170)
        except Exception as exc:
            log.warning("Could not embed chart image: %s", exc)
        pdf.ln(5)

    # Result
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Result", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 8)
    # Truncate for PDF
    result_trunc = result_text[:3000]
    for line in result_trunc.split("\n"):
        pdf.multi_cell(0, 4, line)
    pdf.ln(5)

    # Insights
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Business Insights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    # Strip markdown bold markers for PDF
    clean_insights = insights.replace("**", "").replace("*", "- ")
    pdf.multi_cell(0, 6, clean_insights)

    pdf.output(str(path))
    log.info("PDF report saved: %s", path)
    return path

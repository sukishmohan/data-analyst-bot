"""
server.py — FastAPI backend for AI Data Analyst Agent
"""

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from utils import get_logger, check_llm_available, CHARTS_DIR
from data_loader import load_and_prepare
from main import run_analysis, initialise

log = get_logger("server")

app = FastAPI(title="AI Data Analyst API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store analysis results by ID for chart serving
_results: dict[str, dict] = {}


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str


class AnalyzeResponse(BaseModel):
    id: str
    title: str
    query: str
    steps: list[dict]
    result_text: str
    insights: str
    chart_url: str | None
    code: str
    confidence: float
    error: str | None


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

@app.on_event("startup")
def startup():
    if not check_llm_available():
        log.warning("NVIDIA_API_KEY is not set. Set it as an environment variable.")
    try:
        initialise()
        log.info("Dataset loaded successfully on startup.")
    except Exception as e:
        log.warning("Could not auto-load dataset on startup: %s", e)


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_configured": check_llm_available(),
    }


@app.get("/api/profile")
def get_profile():
    try:
        _, profile = initialise()
        return {
            "shape": profile["shape"],
            "columns": [
                {
                    "name": c,
                    "type": profile["dtypes"].get(c, "unknown"),
                    "missing": profile.get("missing", {}).get(c, 0),
                }
                for c in profile["columns"]
            ],
            "date_columns": profile.get("date_columns", []),
            "categorical_columns": profile.get("categorical_columns", []),
            "numeric_columns": profile.get("numeric_columns", []),
            "sample_values": profile.get("sample_values", {}),
            "numeric_stats": profile.get("numeric_stats", {}),
            "duplicate_rows": profile.get("duplicate_rows", 0),
            "total_missing": profile.get("total_missing", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not check_llm_available():
        raise HTTPException(
            status_code=503,
            detail="NVIDIA_API_KEY is not configured. Set it as an environment variable.",
        )

    try:
        result = run_analysis(req.query)
    except Exception as e:
        log.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    analysis_id = uuid.uuid4().hex[:12]
    _results[analysis_id] = result

    chart_path = result.get("chart_path")
    chart_url = f"/api/charts/{analysis_id}" if chart_path else None
    reflection = result.get("reflection") or {}

    return AnalyzeResponse(
        id=analysis_id,
        title=result.get("plan", {}).get("title", "Analysis"),
        query=result["query"],
        steps=result.get("plan", {}).get("steps", []),
        result_text=result.get("result_text", ""),
        insights=result.get("insights", ""),
        chart_url=chart_url,
        code=result.get("code", ""),
        confidence=reflection.get("confidence", 0.0),
        error=result.get("error"),
    )


@app.get("/api/charts/{analysis_id}")
def serve_chart(analysis_id: str):
    result = _results.get(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    chart_path = result.get("chart_path")
    if not chart_path or not Path(chart_path).exists():
        raise HTTPException(status_code=404, detail="Chart not found.")

    return FileResponse(str(chart_path), media_type="image/png")


@app.get("/api/sample-queries")
def sample_queries():
    return [
        "Show monthly sales trend",
        "Which category has highest profit?",
        "Find loss-making sub-categories",
        "Compare regions by revenue",
        "Forecast sales for next 6 months",
        "Top 10 customers by sales",
        "Profit margin by category",
        "Quarterly sales growth rate",
    ]

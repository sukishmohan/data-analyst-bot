"""
utils.py — Shared utilities, configuration, and NVIDIA NIM API client.

Centralises NVIDIA NIM API calls, logging, and common helper functions
used across every module.
"""

import os
import sys
import time
import json
import re
import logging
import requests
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent

# User-provided NVIDIA API key
def get_api_key():
    return os.getenv("NVIDIA_API_KEY", "")

NVIDIA_API_KEY = get_api_key()


LLM_MODEL = "meta/llama-3.1-8b-instruct" 

# Output directories
OUTPUT_DIR: Path = _PROJECT_ROOT / "outputs"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"

for _d in (OUTPUT_DIR, CHARTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 2. Logging
# ──────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)-22s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("utils")


# ──────────────────────────────────────────────
# 3. NVIDIA NIM API Client
# ──────────────────────────────────────────────

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

def check_llm_available() -> bool:
    """Check if NVIDIA API key exists."""
    return True if NVIDIA_API_KEY else False


def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """
    Send a chat completion request to NVIDIA NIM and return the response text.
    """
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 429 and attempt < max_retries - 1:
                err_text = exc.response.text
                wait_match = re.search(r"try again in (\d+\.?\d*)s", err_text)
                wait_time = float(wait_match.group(1)) + 1.0 if wait_match else 20.0
                log.warning(f"Rate Limit (429). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            log.error("NVIDIA NIM API call failed: %s", exc)
            raise RuntimeError(f"NVIDIA NIM API Error: {exc}")
        except Exception as exc:
            log.error("NVIDIA NIM API call failed: %s", exc)
            raise RuntimeError(f"NVIDIA NIM API Error: {exc}")


def call_llm_json(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
) -> dict:
    """
    Call NVIDIA NIM requesting JSON output.
    """
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system + "\n\nCRITICAL: Return ONLY a valid JSON object."},
            {"role": "user", "content": prompt}
        ]
    }

    max_retries = 3
    raw = ""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            break
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 429 and attempt < max_retries - 1:
                err_text = exc.response.text
                wait_match = re.search(r"try again in (\d+\.?\d*)s", err_text)
                wait_time = float(wait_match.group(1)) + 1.0 if wait_match else 20.0
                log.warning(f"Rate Limit (429). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"NVIDIA NIM API Error: {exc}")
        except Exception as exc:
            raise RuntimeError(f"NVIDIA NIM API Error: {exc}")

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Last resort
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from NVIDIA NIM response:\n{raw[:500]}")


# ──────────────────────────────────────────────
# 4. Miscellaneous Helpers
# ──────────────────────────────────────────────

def truncate(text: str, max_len: int = 3000) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 20] + "\n... [truncated] ..."


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)[:120]


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))

"""
llm.py — The one place the pipeline calls Claude.

Every LLM step (business screening, topic classification, feature-gap naming, incident extraction) goes through
`parse_cached`, which:
    - returns a cached answer if one exists (data/processed/llm_cache/), so re-runs are free
    - otherwise calls Claude with structured output validated against a pydantic schema,
      with server-side fallbacks on and low effort (these are short classification tasks)
    - logs token usage and cost for every real API call to data/processed/llm_usage.jsonl,
      which the dashboard's Tool Telemetry tab reads

Settings come from .env (next to the .exe or in the project root; see .env.example):
    LLM_MODEL                default claude-opus-5-5
    LLM_MONTHLY_BUDGET_USD   monthly LLM budget in USD, default 1000
    LLM_BUDGET_WARN_SHARE    share of the budget at which the dashboard turns red, default 0.8
A missing or unreadable number falls back to its default, with a warning; BUDGET_SOURCE records
which happened so the dashboard can say where the budget came from.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from src.load import HOME_DIR, PROCESSED_DIR, PROJECT_ROOT

load_dotenv(HOME_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

def _env_number(name: str, default: float, low: float, high: float) -> tuple[float, str]:
    """(value, source) for a numeric .env setting; source is ".env" or "default"."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default, "default"
    try:
        value = float(raw)
        if low <= value <= high:
            return value, ".env"
    except ValueError:
        pass
    logging.getLogger(__name__).warning("%s=%r is not a number between %g and %g; using %g", name, raw, low, high, default)
    return default, "default"


LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
MONTHLY_BUDGET_USD, _budget_src = _env_number("LLM_MONTHLY_BUDGET_USD", 1000.0, 0.0, 1e9)
BUDGET_WARN_SHARE, _warn_src = _env_number("LLM_BUDGET_WARN_SHARE", 0.8, 0.0, 1.0)
BUDGET_SOURCE = {"LLM_MONTHLY_BUDGET_USD": _budget_src, "LLM_BUDGET_WARN_SHARE": _warn_src}

CACHE_DIR = PROCESSED_DIR / "llm_cache"
USAGE_LOG = PROCESSED_DIR / "llm_usage.jsonl"

# USD per million tokens: (input, output, cache read). Cache writes bill at 1.25x input.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
UNKNOWN_MODEL_PRICE = PRICES["claude-opus-5-5"]   # conservative default for an unlisted model

log = logging.getLogger(__name__)
_log_lock = threading.Lock()
_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_read: int = 0, cache_write: int = 0) -> float:
    p_in, p_out, p_read = PRICES.get(model, UNKNOWN_MODEL_PRICE)
    return (input_tokens * p_in + output_tokens * p_out + cache_read * p_read
            + cache_write * p_in * 1.25) / 1e6


def log_usage(step: str, response) -> dict:
    """Append one API call's usage to the usage log and return the record."""
    u = response.usage
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": step,
        "model": getattr(response, "model", LLM_MODEL),
        "input_tokens": int(u.input_tokens or 0),
        "output_tokens": int(u.output_tokens or 0),
        "cache_read_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        "request_id": getattr(response, "_request_id", None),
    }
    rec["cost_usd"] = round(cost_usd(rec["model"], rec["input_tokens"], rec["output_tokens"],
                                     rec["cache_read_tokens"], rec["cache_write_tokens"]), 6)
    with _log_lock:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def load_usage(path=USAGE_LOG) -> pd.DataFrame:
    """Every logged API call (empty frame if nothing has been logged yet)."""
    cols = ["ts", "step", "model", "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "request_id", "cost_usd"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def uncosted_cache_entries(cache_dir=CACHE_DIR) -> int:
    """Cached answers created before cost logging existed (their spend isn't in the log)."""
    if not cache_dir.exists():
        return 0
    n = 0
    for p in cache_dir.glob("*.json"):
        try:
            if "usage" not in json.loads(p.read_text(encoding="utf-8")):
                n += 1
        except json.JSONDecodeError:
            continue
    return n


# ---------------------------------------------------------------------------
# Cached structured call
# ---------------------------------------------------------------------------
def cache_path(prefix: str, key_blob: str):
    return CACHE_DIR / f"{prefix}_{hashlib.sha256(key_blob.encode()).hexdigest()[:16]}.json"


def parse_cached(step: str, prefix: str, key_blob: str, system: str, user: str,
                 schema: type[BaseModel], refresh: bool = False, max_tokens: int = 4000) -> dict | None:
    """
    Structured Claude call with a disk cache. Returns the parsed dict, or None when the
    model declined or produced nothing usable (callers then fall back to their rules).
    API errors propagate so callers can decide; AuthenticationError means "no LLM at all".
    """
    path = cache_path(prefix, key_blob)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    response = get_client().beta.messages.parse(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                 # re-run a declined request on Anthropic's fallback model
        output_config={"effort": "low"},     # short classification / extraction tasks
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    usage = log_usage(step, response)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        log.warning("%s: no usable answer (stop_reason=%s)", step, response.stop_reason)
        return None

    result = response.parsed_output.model_dump()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**result, "usage": usage}, indent=2), encoding="utf-8")
    return result

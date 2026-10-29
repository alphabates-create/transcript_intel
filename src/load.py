"""
load.py — Ingest the raw transcript folders and normalize them into tables.

Each meeting lives in its own folder under data/raw/<meetingId>/ with six JSON files:
    meeting-info.json   title, start/end time, duration, attendee emails
    summary.json        summary, action items, topics, sentiment label + score, key moments
    transcript.json     sentence-level text with speaker, timing, sentiment, ASR confidence
    speakers.json       speaker turns (start/end per turn)
    speaker-meta.json   speaker id -> name
    events.json         join / leave events

This module turns those into four tidy tables:
    calls         one row per meeting      (the main analysis table)
    sentences     one row per sentence     (sentiment, talk time, quotes)
    key_moments   one row per key moment   (churn signals, feature gaps, ...)
    action_items  one row per action item  (owner + task)

Decision: this step only READS and FLATTENS. It does not derive call type, account,
product, etc. — that lives in enrich.py so each step stays small and testable.

Usage:
    python -m src.load                 # build tables and save to data/processed/
    from src.load import load_all      # use in notebooks
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (resolved from this file, so it works from any working directory).
# TI_HOME overrides the folder holding data/, outputs/ and .env — the packaged .exe
# sets it to its own folder so new transcripts can be dropped into data/raw/ there.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOME_DIR = Path(os.getenv("TI_HOME", PROJECT_ROOT))
RAW_DIR = HOME_DIR / "data" / "raw"
PROCESSED_DIR = HOME_DIR / "data" / "processed"
OUTPUT_DIR = HOME_DIR / "outputs"

# LLM-dependent outputs (topics, incidents, insights) are built once per mode so the
# dashboard can switch between them without re-running anything.
MODES = {"hybrid": "Hybrid (LLM + rules)", "rules": "Rules only"}


def mode_dir(mode: str) -> Path:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {list(MODES)}")
    return PROCESSED_DIR / mode

INTERNAL_DOMAIN = "aegiscloud.com"

REQUIRED_FILES = ["meeting-info.json", "summary.json", "transcript.json"]
OPTIONAL_FILES = ["speakers.json", "speaker-meta.json", "events.json"]

# All key-moment types seen in the dataset; each becomes a count column on `calls`
KEY_MOMENT_TYPES = [
    "concern", "positive_pivot", "churn_signal", "technical_issue",
    "feature_gap", "action_item", "praise", "pricing_offer",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _read_json(path: Path):
    """Read a JSON file as UTF-8 (transcripts contain em-dashes and smart quotes)."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_meeting(folder: Path) -> dict | None:
    """
    Load all JSON files for one meeting folder.

    Returns a dict keyed by file stem (e.g. "meeting-info"), or None if a
    required file is missing or unreadable — one bad folder should not stop the run.
    """
    meeting: dict = {"folder": folder.name}
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        path = folder / name
        if not path.exists():
            if name in REQUIRED_FILES:
                log.warning("Skipping %s: missing %s", folder.name, name)
                return None
            meeting[path.stem] = None
            continue
        try:
            meeting[path.stem] = _read_json(path)
        except json.JSONDecodeError as e:
            log.warning("Skipping %s: bad JSON in %s (%s)", folder.name, name, e)
            return None
    return meeting


def iter_meetings(raw_dir: Path = RAW_DIR):
    """Yield every readable meeting under raw_dir, in a stable (sorted) order."""
    folders = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not folders:
        raise FileNotFoundError(f"No meeting folders found in {raw_dir}")
    for folder in folders:
        meeting = read_meeting(folder)
        if meeting is not None:
            yield meeting


def _sentences_of(transcript) -> list[dict]:
    """transcript.json is {"data": [...]} in this dataset; accept a bare list too."""
    if isinstance(transcript, dict):
        return transcript.get("data", [])
    return transcript or []


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def _call_row(m: dict) -> dict:
    """Flatten one meeting into a single `calls` row."""
    info = m["meeting-info"]
    summ = m["summary"]
    sents = _sentences_of(m["transcript"])
    emails = info.get("allEmails", []) or []
    external = [e for e in emails if not e.lower().endswith("@" + INTERNAL_DOMAIN)]

    # Count key moments by type (missing types -> 0)
    km_counts = {f"km_{t}": 0 for t in KEY_MOMENT_TYPES}
    for km in summ.get("keyMoments", []) or []:
        key = f"km_{km.get('type')}"
        km_counts[key] = km_counts.get(key, 0) + 1

    # Sentence-level sentiment mix (provided by the transcription vendor)
    s_types = pd.Series([s.get("sentimentType") for s in sents], dtype="object")
    n = max(len(sents), 1)

    return {
        "call_id": info.get("meetingId", m["folder"]),
        "title": info.get("title", ""),
        "start_time": info.get("startTime"),
        "end_time": info.get("endTime"),
        "duration_min": info.get("duration"),
        "organizer": info.get("organizerEmail"),
        "emails": emails,
        "n_attendees": len(emails),
        "n_internal": len(emails) - len(external),
        "n_external": len(external),
        "external_domains": sorted({e.split("@")[-1].lower() for e in external}),
        "summary": summ.get("summary", ""),
        "topics": summ.get("topics", []) or [],
        "action_items": summ.get("actionItems", []) or [],
        "n_action_items": len(summ.get("actionItems", []) or []),
        "sentiment_label": summ.get("overallSentiment"),
        "sentiment_score": summ.get("sentimentScore"),
        "n_sentences": len(sents),
        "pct_positive_sent": round((s_types == "positive").sum() / n, 3),
        "pct_negative_sent": round((s_types == "negative").sum() / n, 3),
        "pct_neutral_sent": round((s_types == "neutral").sum() / n, 3),
        "n_speakers": len(m.get("speaker-meta") or {}),
        **km_counts,
    }


def build_calls_table(meetings: list[dict]) -> pd.DataFrame:
    """One row per call."""
    df = pd.DataFrame([_call_row(m) for m in meetings])
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    df["date"] = df["start_time"].dt.date
    df["month"] = df["start_time"].dt.strftime("%Y-%m")
    return df.sort_values("start_time").reset_index(drop=True)


def build_sentences_table(meetings: list[dict]) -> pd.DataFrame:
    """One row per sentence, with the speaker's email resolved where possible."""
    rows = []
    for m in meetings:
        call_id = m["meeting-info"].get("meetingId", m["folder"])
        emails = m["meeting-info"].get("allEmails", []) or []
        for s in _sentences_of(m["transcript"]):
            start, end = s.get("time"), s.get("endTime")
            name = s.get("speaker_name", "")
            rows.append({
                "call_id": call_id,
                "idx": s.get("index"),
                "speaker_id": s.get("speaker_id"),
                "speaker_name": name,
                "speaker_email": _match_email(name, emails),
                "start_s": start,
                "end_s": end,
                "duration_s": (end - start) if start is not None and end is not None else None,
                "text": s.get("sentence", ""),
                "sentiment": s.get("sentimentType"),
                "confidence": s.get("averageConfidence"),
            })
    df = pd.DataFrame(rows)
    df["n_words"] = df["text"].str.split().str.len()
    return df.sort_values(["call_id", "idx"]).reset_index(drop=True)


def _match_email(name: str, emails: list[str]) -> str | None:
    """
    Map a speaker name to an attendee email.

    Emails follow two patterns in this dataset:
      first.last@aegiscloud.com   (internal staff)
      f.last@customer.com         (customers)
    Match on last name plus first initial; returns None if ambiguous or not found.
    Handles titles like "Dr." and apostrophes like "O'Brien".
    """
    parts = [p for p in name.replace(".", " ").split() if p.lower() not in {"dr", "mr", "ms", "mrs"}]
    if len(parts) < 2:
        return None
    first = parts[0].lower()
    last = parts[-1].lower().replace("'", "")
    hits = []
    for e in emails:
        local = e.split("@")[0].lower().replace("'", "")
        bits = local.split(".")
        if bits[-1] == last and bits[0] and bits[0][0] == first[0]:
            hits.append(e)
    return hits[0] if len(hits) == 1 else None


def build_key_moments_table(meetings: list[dict]) -> pd.DataFrame:
    """One row per key moment (churn_signal, feature_gap, concern, ...)."""
    rows = []
    for m in meetings:
        call_id = m["meeting-info"].get("meetingId", m["folder"])
        for km in m["summary"].get("keyMoments", []) or []:
            rows.append({
                "call_id": call_id,
                "time_s": km.get("time"),
                "type": km.get("type"),
                "speaker": km.get("speaker"),
                "text": km.get("text", ""),
            })
    return pd.DataFrame(rows, columns=["call_id", "time_s", "type", "speaker", "text"])


def build_action_items_table(meetings: list[dict]) -> pd.DataFrame:
    """One row per action item, split into owner and task ("Owner: task")."""
    rows = []
    for m in meetings:
        call_id = m["meeting-info"].get("meetingId", m["folder"])
        for item in m["summary"].get("actionItems", []) or []:
            owner, sep, task = item.partition(":")
            rows.append({
                "call_id": call_id,
                "owner": owner.strip() if sep else None,
                "task": task.strip() if sep else item.strip(),
            })
    return pd.DataFrame(rows, columns=["call_id", "owner", "task"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read every meeting once and build all four tables."""
    meetings = list(iter_meetings(raw_dir))
    log.info("Loaded %d meetings from %s", len(meetings), raw_dir)
    return {
        "calls": build_calls_table(meetings),
        "sentences": build_sentences_table(meetings),
        "key_moments": build_key_moments_table(meetings),
        "action_items": build_action_items_table(meetings),
    }


def save_tables(tables: dict[str, pd.DataFrame], out_dir: Path = PROCESSED_DIR) -> None:
    """Write each table to <out_dir>/<name>.parquet (list columns are preserved)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("Saved %-13s %6d rows -> %s", name, len(df), path)


def load_tables(in_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Read the saved parquet tables back (used by later steps and the app)."""
    names = ["calls", "sentences", "key_moments", "action_items"]
    return {n: pd.read_parquet(in_dir / f"{n}.parquet") for n in names}


def summarize(tables: dict[str, pd.DataFrame]) -> None:
    """Print a quick sanity report."""
    calls, sents = tables["calls"], tables["sentences"]
    print(f"Calls:        {len(calls)}")
    print(f"Date range:   {calls['start_time'].min():%Y-%m-%d} -> {calls['start_time'].max():%Y-%m-%d}")
    print(f"Sentences:    {len(sents):,}  (speaker email matched: {sents['speaker_email'].notna().mean():.0%})")
    print(f"Key moments:  {len(tables['key_moments'])}")
    print(f"Action items: {len(tables['action_items'])}")
    print(f"Avg sentiment score: {calls['sentiment_score'].mean():.2f}")
    print("\nSentiment labels:")
    print(calls["sentiment_label"].value_counts().to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tables = load_all()
    summarize(tables)
    save_tables(tables)

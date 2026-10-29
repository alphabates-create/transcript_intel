"""
screen.py — keep non-business calls (birthdays, celebrations, farewells, socials…) out of every output.

Runs right after load, so no later stage, table, notebook or dashboard view ever sees them.

    keyword rule   NON_BUSINESS on the title and vendor topics (always; also flags new
                   transcripts in the dashboard before they're processed)
    is_business    Claude's yes/no answer from the title, attendees, summary and topics
                   (hybrid runs; cached). Claude decides when it answers, the rule otherwise.

Every excluded call is recorded in data/exclusions.csv and its raw folder is moved out of
data/raw/ into data/exclude/<meetingId>/. If the same folder is dropped into data/raw/ again,
the load stage and the new-transcript scan still skip it, and the next run moves it out again.
To bring a call back, set its action to "keep": the next run moves the folder back to
data/raw/ (restore_kept), loads it, and never screens it out again.

    action    exclude | keep
    source    keyword | llm | user (Ignore in the dashboard, or edited by hand)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.llm import LLM_MODEL, cache_path, parse_cached
from src.load import HOME_DIR, INTERNAL_DOMAIN, RAW_DIR

EXCLUSIONS = HOME_DIR / "data" / "exclusions.csv"
EXCLUDE_DIR = HOME_DIR / "data" / "exclude"
EXCLUSION_COLS = ["call_id", "title", "date", "action", "source", "reason", "decided_at"]
PROMPT_VERSION = "screen-v1"   # bump to invalidate cached answers when the prompt changes

# Social / non-work meetings. Matched on the title and vendor topics only: summaries often say
# things like "celebrated the renewal", which must not exclude a business call.
NON_BUSINESS = re.compile(
    r"\b(?:birthday|b-?day|celebrations?|happy hour|farewell|going[- ]away|baby shower|bridal shower|"
    r"potluck|holiday party|christmas party|office party|team (?:lunch|dinner|outing|social)|"
    r"game night|trivia night|social hour|retirement party|welcome (?:lunch|party)|cake)\b", re.I)

SYSTEM_PROMPT = (
    "You screen recorded calls at AegisCloud, a B2B cloud-security company (products: Detect, Protect, "
    "Comply, Identity, LogVault), before they go into a business-analytics dashboard.\n\n"
    "is_business is true if any substantive part of the call is work: customers, sales, renewals, "
    "support cases, incidents, products, engineering, planning, operations, hiring or internal strategy. "
    "It is false for social or personal calls with no work content: birthdays, celebrations, farewells, "
    "happy hours, holiday parties, casual chats. A work meeting that starts or ends with a celebration "
    "is still business.\n"
    "reason is one short sentence citing what in the call decided it."
)

log = logging.getLogger(__name__)


class BusinessCheck(BaseModel):
    is_business: bool
    reason: str


def _as_list(v) -> list:
    return list(v) if v is not None and not isinstance(v, float) else []


def keyword_hit(title: str, topics=()) -> str | None:
    """The first non-business keyword in the title or vendor topics, or None."""
    for text in [title or "", *map(str, _as_list(topics))]:
        m = NON_BUSINESS.search(text)
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------------------
# Exclusion list
# ---------------------------------------------------------------------------
def load_exclusions(path: Path = EXCLUSIONS) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EXCLUSION_COLS)
    return pd.read_csv(path, dtype=str).fillna("").reindex(columns=EXCLUSION_COLS, fill_value="")


def ids_with_action(action: str, path: Path = EXCLUSIONS) -> set[str]:
    ex = load_exclusions(path)
    return set(ex.loc[ex["action"].str.strip().str.lower() == action, "call_id"])


def excluded_ids(path: Path = EXCLUSIONS) -> set[str]:
    return ids_with_action("exclude", path)


def _meeting_meta(call_id: str, raw_dir: Path) -> tuple[str, str]:
    """(title, mm/dd/yyyy date) from a raw folder's meeting-info.json; blanks if unreadable."""
    try:
        info = json.loads((raw_dir / call_id / "meeting-info.json").read_text(encoding="utf-8"))
        start = info.get("startTime") or info.get("start_time") or ""
        return info.get("title", ""), (f"{pd.Timestamp(start):%m/%d/%Y}" if start else "")
    except (OSError, ValueError):
        return "", ""


def exclude_call(call_id: str, source: str, reason: str, title: str | None = None, date: str | None = None,
                 raw_dir: Path = RAW_DIR, path: Path = EXCLUSIONS, exclude_dir: Path = EXCLUDE_DIR) -> None:
    """Add (or update) a call's exclusion and move its raw folder to data/exclude/."""
    if title is None or date is None:
        meta = _meeting_meta(call_id, raw_dir)
        title, date = title if title is not None else meta[0], date if date is not None else meta[1]
    ex = load_exclusions(path)
    row = {"call_id": call_id, "title": title, "date": date, "action": "exclude", "source": source,
           "reason": reason, "decided_at": datetime.now(timezone.utc).strftime("%m/%d/%Y %H:%M UTC")}
    ex = pd.concat([ex[ex["call_id"] != call_id], pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ex.to_csv(path, index=False, encoding="utf-8")
    move_folder(raw_dir / call_id, exclude_dir / call_id)
    log.info("Excluded %s (%s: %s)", title or call_id, source, reason)


def move_folder(src: Path, dest: Path) -> bool:
    """Move a meeting folder; if dest already exists (excluded before), merge into it. False if src is gone."""
    if not src.is_dir():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.copytree(src, dest, dirs_exist_ok=True)
        shutil.rmtree(src)
    else:
        shutil.move(str(src), str(dest))
    return True


def sweep_excluded(raw_dir: Path = RAW_DIR, exclude_dir: Path = EXCLUDE_DIR) -> list[str]:
    """Move any excluded folder that is (back) in data/raw/ out to data/exclude/."""
    moved = [i for i in sorted(excluded_ids()) if move_folder(raw_dir / i, exclude_dir / i)]
    if moved:
        log.info("Moved %d excluded folder(s) from %s to %s", len(moved), raw_dir, exclude_dir)
    return moved


def restore_kept(raw_dir: Path = RAW_DIR, exclude_dir: Path = EXCLUDE_DIR) -> list[str]:
    """Move folders marked "keep" back from data/exclude/ to data/raw/ so they're loaded again."""
    restored = [i for i in sorted(ids_with_action("keep"))
                if not (raw_dir / i).exists() and move_folder(exclude_dir / i, raw_dir / i)]
    if restored:
        log.info("Restored %d kept folder(s) to %s: %s", len(restored), raw_dir, ", ".join(restored))
    return restored


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------
def _payload(row) -> dict:
    emails = _as_list(row.get("emails"))
    customers = sorted({e.split("@")[-1] for e in emails if "@" in e and not e.endswith(INTERNAL_DOMAIN)})
    return {"title": row.get("title"), "date": str(row.get("date")),
            "attendees": {"aegiscloud_staff": sum(e.endswith(INTERNAL_DOMAIN) for e in emails),
                          "customer_domains": customers},
            "summary": row.get("summary"), "vendor_topics": [str(t) for t in _as_list(row.get("topics"))]}


def _cache_blob(payload: dict) -> str:
    return json.dumps({"model": LLM_MODEL, "prompt": PROMPT_VERSION, "call": payload}, sort_keys=True, default=str)


def check_with_llm(row, refresh: bool = False) -> dict | None:
    payload = _payload(row)
    return parse_cached("screen", "screen", _cache_blob(payload), SYSTEM_PROMPT,
                        json.dumps(payload, ensure_ascii=False), BusinessCheck, refresh, max_tokens=1000)


def cached_screen(calls: pd.DataFrame) -> pd.DataFrame:
    """
    The screening result per call from Claude's cached answers and the keyword rule only.
    Never calls the API (for notebooks and reports). claude_answer is None where no answer is cached.
    """
    rows = []
    for _, r in calls.iterrows():
        path = cache_path("screen", _cache_blob(_payload(r)))
        ans = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        hit = keyword_hit(r.get("title"), r.get("topics"))
        rows.append({"call_id": r["call_id"], "title": r.get("title"), "keyword": hit,
                     "claude_answer": None if ans is None else bool(ans["is_business"]),
                     "claude_reason": None if ans is None else ans["reason"],
                     "is_business": bool(ans["is_business"]) if ans else hit is None,
                     "decided_by": "llm" if ans else "keyword"})
    return pd.DataFrame(rows)


def screen_calls(calls: pd.DataFrame, use_llm: bool = True, refresh: bool = False,
                 workers: int = 6) -> pd.DataFrame:
    """One row per call: is_business, source (keyword | llm | user) and reason."""
    keep = ids_with_action("keep")
    rows = [r for _, r in calls.iterrows()]
    answers: list[dict | None] = [None] * len(rows)

    if use_llm:
        def one(i: int) -> None:
            if rows[i]["call_id"] in keep:
                return
            try:
                answers[i] = check_with_llm(rows[i], refresh)
            except ValidationError as e:
                log.warning("Invalid screening answer for %s: %s", rows[i]["call_id"], e)
            except anthropic.AuthenticationError:
                raise
            except anthropic.APIError as e:
                log.warning("Screening API error on %s (%s); using the keyword rule", rows[i]["call_id"], e)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one, range(len(rows))))
        except anthropic.AuthenticationError:
            log.warning("No valid Anthropic credentials; screening by keyword rule only")

    out = []
    for row, ans in zip(rows, answers):
        if row["call_id"] in keep:
            res = {"is_business": True, "source": "user", "reason": "Marked keep in the exclusion list"}
        elif ans:
            res = {"is_business": bool(ans["is_business"]), "source": "llm", "reason": ans["reason"]}
        else:
            hit = keyword_hit(row.get("title"), row.get("topics"))
            res = {"is_business": hit is None, "source": "keyword",
                   "reason": f'Non-business keyword "{hit}"' if hit else "No non-business keyword"}
        out.append({"call_id": row["call_id"], "title": row.get("title"), "date": row.get("date"), **res})
    return pd.DataFrame(out, columns=["call_id", "title", "date", "is_business", "source", "reason"])


def drop_calls(tables: dict[str, pd.DataFrame], ids: set[str]) -> dict[str, pd.DataFrame]:
    """The four load tables without the given calls."""
    return {name: df[~df["call_id"].isin(ids)].reset_index(drop=True) for name, df in tables.items()}


def apply_screen(tables: dict[str, pd.DataFrame], use_llm: bool = True,
                 refresh: bool = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Screen the loaded calls, record and copy the non-business ones, and drop them from the tables."""
    result = screen_calls(tables["calls"], use_llm=use_llm, refresh=refresh)
    out = result[~result["is_business"]]
    for _, r in out.iterrows():
        exclude_call(r["call_id"], r["source"], r["reason"], title=r["title"],
                     date=f"{pd.Timestamp(r['date']):%m/%d/%Y}" if r["date"] else "")
    log.info("Screened %d calls: %d business, %d excluded (%s)", len(result), int(result["is_business"].sum()),
             len(out), result["source"].value_counts().to_dict())
    return drop_calls(tables, set(out["call_id"])), result


def flag_folder(folder: Path) -> str | None:
    """Keyword hit for an unprocessed raw folder (title + vendor topics), for the new-transcript notice."""
    title, topics = "", []
    try:
        title = json.loads((folder / "meeting-info.json").read_text(encoding="utf-8")).get("title", "")
        topics = json.loads((folder / "summary.json").read_text(encoding="utf-8")).get("topics", []) or []
    except (OSError, ValueError, AttributeError):
        pass
    return keyword_hit(title, topics)

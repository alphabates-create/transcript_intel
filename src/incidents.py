"""
incidents.py — Incident register: what broke, when it was reported, owned and resolved.

Incidents come from two places:
    major incidents   internal war-room / incident calls ("INCIDENT: Detect Pipeline Failure -
                      War Room"), grouped with the follow-up incident calls and the customer
                      escalations that came in while it was open
    support cases     each "Support Case #NNNN" call; cases opened during a major incident
                      that are about it are linked to it as children

For each incident:
    reported_at   start of the first call where it came up
    owner         Hybrid: Claude picks the owner from the staff on the calls.
                  Rules: the major incident's organizer, or the support agent who led the case.
    owned_at      when the owner first spoke on the reporting call
    resolved_at   end of the call where the fix was made or confirmed. Child cases that weren't
                  fixed on their own close with their major incident. Otherwise still open.
    outage_hours  customer-facing downtime for outages ("six-hour outage")

Hybrid mode asks Claude to read the linked calls (cached, cost logged). Rules mode uses
the keyword patterns below. Both produce the same columns.

SLA: availability per month = 1 - platform outage hours / hours in month, compared with the
SLA_TARGET stated on the calls ("current contract has a 99.5% uptime SLA for Detect").

Usage:
    python -m src.incidents               # hybrid
    python -m src.incidents --rules-only
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import re
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel

from src.enrich import load_enriched
from src.llm import LLM_MODEL, parse_cached
from src.load import mode_dir

SLA_TARGET = 0.995           # standard Detect contract; some accounts have 99.9%
MAJOR_SEED = re.compile(r"^(?:INCIDENT|URGENT)\b|war room", re.I)
MAJOR_FOLLOW_UP = re.compile(r"outage|incident|post-incident|remediation|root cause|escalation bridge", re.I)
MAJOR_FOLLOW_UP_DAYS = 14
CUSTOMER_ESCALATION = re.compile(r"^(?:URGENT|ESCALATION)\b|outage", re.I)
OUTAGE_THEME = "Detect outage & fallout"

RESOLVED = re.compile(r"resolved live|was resolved|resolved on the call|issue is resolved|"
                      r"restoration confirmed|full restoration|recovery confirm|the fix involved|"
                      r"fix (?:was )?(?:applied|deployed|confirmed)|post-incident review|back online", re.I)
PRIORITY_1 = re.compile(r"\bP1\b|sev ?1|business-critical|priority 1", re.I)
CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("Request", re.compile(r"request|question|template", re.I)),
    ("Billing", re.compile(r"invoice|billing|overage|charge|license|credit", re.I)),
    ("Outage", re.compile(r"outage|down\b|not firing|loss of|data gaps|failure$", re.I)),
    ("Degradation", re.compile(r"slow|latency|delay|exceeded|timeout|performance", re.I)),
    ("Defect", re.compile(r"error|fail|bug|issue|false positive|formatting", re.I)),
]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "twenty-four": 24}
OUTAGE_LENGTH = re.compile(r"\b(\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r")[- ](hour|day)s?\b"
                           r"(?=[^.]{0,25}\b(outage|down|downtime|visibility|blind))", re.I)

INCIDENT_COLS = ["incident_id", "kind", "parent_id", "title", "account", "category", "priority",
                 "status", "reported_at", "owner", "owner_team", "owned_at", "resolved_at",
                 "minutes_to_own", "hours_to_resolve", "outage_hours", "resolution_note",
                 "source", "linked_calls", "n_linked_calls"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grouping calls into incidents
# ---------------------------------------------------------------------------
def _themes(row) -> list:
    t = row.get("themes")
    return list(t) if t is not None else []


def group_incidents(calls: pd.DataFrame) -> list[dict]:
    """Candidate incidents with their linked calls (sorted by time)."""
    c = calls.sort_values("start_time")
    incidents, used = [], set()

    seeds = c[(c["call_type"] == "internal") & c["title"].str.contains(MAJOR_SEED)]
    for _, seed in seeds.iterrows():
        if seed["call_id"] in used:
            continue
        horizon = seed["start_time"] + pd.Timedelta(days=MAJOR_FOLLOW_UP_DAYS)
        follow = c[(c["call_type"] == "internal") & c["start_time"].between(seed["start_time"], horizon)
                   & c["title"].str.contains(MAJOR_FOLLOW_UP)]
        end = follow["start_time"].max()
        customer = c[(c["call_type"] == "external") & c["start_time"].between(seed["start_time"], end)
                     & (c["title"].str.contains(CUSTOMER_ESCALATION)
                        | c.apply(lambda r: OUTAGE_THEME in _themes(r), axis=1))]
        linked = pd.concat([follow, customer]).drop_duplicates("call_id").sort_values("start_time")
        name = re.sub(r"^(INCIDENT|URGENT):\s*|\s*-\s*war room$", "", seed["title"], flags=re.I)
        incidents.append({"incident_id": f"INC-{seed['start_time']:%Y%m%d}", "kind": "Major incident",
                          "title": name, "parent_id": None, "calls": linked})
        used |= set(linked["call_id"])

    for _, case in c[c["call_type"] == "support"].iterrows():
        m = re.match(r"Support Case #(\d+)\s*-\s*(.+)", case["title"])
        number, title = (m.group(1), m.group(2)) if m else (case["call_id"][-6:], case["title"])
        parent = next((i["incident_id"] for i in incidents
                       if i["kind"] == "Major incident"
                       and i["calls"]["start_time"].min() <= case["start_time"] <= i["calls"]["start_time"].max()
                       and OUTAGE_THEME in _themes(case)), None)
        incidents.append({"incident_id": f"CASE-{number}", "kind": "Support case", "title": title,
                          "parent_id": parent, "calls": c[c["call_id"] == case["call_id"]]})
    return incidents


# ---------------------------------------------------------------------------
# Rules extraction
# ---------------------------------------------------------------------------
def _category(title: str, kind: str) -> str:
    if kind == "Major incident":
        return "Outage"
    return next((cat for cat, pat in CATEGORY_RULES if pat.search(title)), "Defect")


def outage_hours_from_text(texts: list[str]) -> float | None:
    """Most often stated outage length across the texts, in hours."""
    found = []
    for t in texts:
        for num, unit, _ in OUTAGE_LENGTH.findall(t or ""):
            n = float(NUMBER_WORDS.get(num.lower(), num)) if not num.replace(".", "").isdigit() else float(num)
            found.append(n * (24 if unit.lower() == "day" else 1))
    return pd.Series(found).mode().iat[0] if found else None


def _lead_staff(call_ids: list[str], sentences: pd.DataFrame, team: str | None = None) -> str | None:
    s = sentences[sentences["call_id"].isin(call_ids) & (sentences["speaker_side"] == "aegiscloud")]
    if team:
        s = s[s["speaker_team"] == team] if (s["speaker_team"] == team).any() else s
    talk = s.groupby("speaker_name")["duration_s"].sum()
    return talk.idxmax() if len(talk) else None


def rules_extract(inc: dict, sentences: pd.DataFrame, roster: pd.DataFrame) -> dict:
    calls = inc["calls"]
    texts = calls["summary"].fillna("").tolist()
    if inc["kind"] == "Major incident":
        organizer = calls.iloc[0]["organizer"]
        owner = roster.set_index("email")["name"].get(organizer) or _lead_staff(calls["call_id"].tolist()[:1], sentences)
    else:
        owner = _lead_staff(calls["call_id"].tolist(), sentences, team="Support")
    resolved_idx = next((i for i, (_, r) in enumerate(calls.iterrows())
                         if RESOLVED.search(r["title"]) or RESOLVED.search(r["summary"] or "")), None)
    category = _category(inc["title"], inc["kind"])
    linked_text = texts + sentences.loc[sentences["call_id"].isin(calls["call_id"]), "text"].tolist()
    return {
        "owner": owner,
        "status": "resolved" if resolved_idx is not None else "open",
        "resolved_call_index": resolved_idx,
        "category": category,
        "priority": "P1" if inc["kind"] == "Major incident" or any(PRIORITY_1.search(t) for t in texts)
        else ("P3" if category in ("Billing", "Request") else "P2"),
        "outage_hours": outage_hours_from_text(linked_text) if category == "Outage" else None,
        "resolution_note": "Resolution wording found in the call summary" if resolved_idx is not None
        else "No resolution recorded on the calls",
        "source": "rules",
    }


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------
class IncidentExtraction(BaseModel):
    status: Literal["open", "resolved"]
    resolved_call_index: int | None
    owner: str
    category: Literal["Outage", "Degradation", "Defect", "Billing", "Request"]
    priority: Literal["P1", "P2", "P3"]
    outage_hours: float | None
    resolution_note: str


SYSTEM_PROMPT = (
    "You maintain the incident register at AegisCloud, a B2B cloud-security company. You get one "
    "incident and the recorded calls about it, in time order (index 0 first). Fill in:\n"
    "- status: resolved only if a call says the problem was fixed or service restored and confirmed; "
    "escalations, workarounds, ETAs and follow-ups mean open.\n"
    "- resolved_call_index: index of the call where it was fixed or confirmed fixed, else null.\n"
    "- owner: the AegisCloud person accountable for it, chosen exactly from the staff list given.\n"
    "- category: Outage (service unavailable), Degradation (slow or delayed), Defect (wrong behaviour), "
    "Billing, or Request (feature or how-to).\n"
    "- priority: P1 for business-critical or P1/Sev 1 escalations, P3 for billing and requests, else P2.\n"
    "- outage_hours: customer-facing downtime in hours if the calls state it, else null.\n"
    "- resolution_note: one short sentence on how it was resolved, or what is still pending."
)


def llm_extract(inc: dict, sentences: pd.DataFrame, action_items: pd.DataFrame,
                key_moments: pd.DataFrame, refresh: bool = False) -> dict | None:
    calls = inc["calls"]
    ids = calls["call_id"].tolist()
    staff = sorted(sentences.loc[sentences["call_id"].isin(ids) & (sentences["speaker_side"] == "aegiscloud"),
                                 "speaker_name"].unique())
    payload = {
        "incident": inc["title"], "kind": inc["kind"], "staff": staff,
        "calls": [{"index": i, "title": r["title"], "start": str(r["start_time"]), "summary": r["summary"],
                   "key_moments": key_moments.loc[key_moments["call_id"] == r["call_id"], "text"].tolist(),
                   "action_items": action_items.loc[action_items["call_id"] == r["call_id"], "task"].tolist()}
                  for i, (_, r) in enumerate(calls.iterrows())],
    }
    blob = json.dumps({"model": LLM_MODEL, "prompt": "incidents-v1", "incident": payload},
                      sort_keys=True, default=str)
    result = parse_cached("incidents", "incident", blob, SYSTEM_PROMPT,
                          json.dumps(payload, ensure_ascii=False, default=str), IncidentExtraction, refresh)
    if result is None:
        return None
    result.pop("usage", None)
    if result["owner"] not in staff:
        result["owner"] = None
    if result["resolved_call_index"] is not None and not 0 <= result["resolved_call_index"] < len(calls):
        result["resolved_call_index"] = None
    result["source"] = "llm"
    return result


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
def _owned_at(owner: str | None, calls: pd.DataFrame, sentences: pd.DataFrame):
    """When the owner first spoke on the incident's calls (falls back to the first call start)."""
    if owner:
        s = sentences[sentences["call_id"].isin(calls["call_id"]) & (sentences["speaker_name"] == owner)]
        if len(s):
            first = s.merge(calls[["call_id", "start_time"]], on="call_id").sort_values(["start_time", "start_s"]).iloc[0]
            return first["start_time"] + pd.Timedelta(seconds=float(first["start_s"] or 0))
    return calls["start_time"].iloc[0]


def build_register(tables: dict[str, pd.DataFrame], call_topics: pd.DataFrame,
                   use_llm: bool = True, refresh: bool = False) -> pd.DataFrame:
    calls = tables["calls"].merge(call_topics[["call_id", "themes"]], on="call_id", how="left")
    sentences, roster = tables["sentences"], tables["staff_roster"]
    team_of = roster.set_index("name")["team"].to_dict()
    llm_ok = use_llm
    rows = []
    for inc in group_incidents(calls):
        rules = rules_extract(inc, sentences, roster)
        info = None
        if llm_ok:
            try:
                info = llm_extract(inc, sentences, tables["action_items"], tables["key_moments"], refresh)
            except anthropic.AuthenticationError:
                log.warning("No valid Anthropic credentials; using rules for incidents")
                llm_ok = False
            except anthropic.APIError as e:
                log.warning("LLM incident extraction failed for %s (%s); using rules", inc["incident_id"], e)
        info = {**rules, **{k: v for k, v in (info or {}).items() if v is not None}} if info else rules
        if info.get("status") == "open":
            info["resolved_call_index"] = None

        calls_i = inc["calls"]
        owner = info.get("owner") or rules["owner"]
        reported = calls_i["start_time"].iloc[0]
        owned = _owned_at(owner, calls_i, sentences)
        idx = info.get("resolved_call_index")
        resolved = calls_i["end_time"].iloc[idx] if idx is not None else None
        rows.append({
            "incident_id": inc["incident_id"], "kind": inc["kind"], "parent_id": inc["parent_id"],
            "title": inc["title"],
            "account": ", ".join(sorted(calls_i["account"].dropna().unique())) or None,
            "category": info["category"], "priority": info["priority"],
            "status": "Resolved" if resolved is not None else "Open",
            "reported_at": reported, "owner": owner, "owner_team": team_of.get(owner),
            "owned_at": owned, "resolved_at": resolved,
            "minutes_to_own": round((owned - reported).total_seconds() / 60, 1),
            "hours_to_resolve": round((resolved - reported).total_seconds() / 3600, 1) if resolved is not None else None,
            "outage_hours": info.get("outage_hours") if info["category"] == "Outage" else None,
            "resolution_note": info["resolution_note"], "source": info["source"],
            "linked_calls": calls_i["call_id"].tolist(), "n_linked_calls": len(calls_i),
        })
    reg = pd.DataFrame(rows, columns=INCIDENT_COLS)

    # Child cases that weren't fixed on their own close with their major incident
    parents = reg.set_index("incident_id")
    for i, r in reg.iterrows():
        p = r["parent_id"]
        if (r["status"] == "Open" and p in parents.index and parents.loc[p, "status"] == "Resolved"
                and parents.loc[p, "resolved_at"] > r["reported_at"]):
            reg.at[i, "status"] = "Resolved"
            reg.at[i, "resolved_at"] = parents.loc[p, "resolved_at"]
            reg.at[i, "hours_to_resolve"] = round((reg.at[i, "resolved_at"] - r["reported_at"]).total_seconds() / 3600, 1)
            reg.at[i, "resolution_note"] = f"Closed with {p}"
    reg["resolved_at"] = pd.to_datetime(reg["resolved_at"], utc=True)
    log.info("Incident register: %d incidents, %d open", len(reg), (reg["status"] == "Open").sum())
    return reg.sort_values("reported_at").reset_index(drop=True)


# ---------------------------------------------------------------------------
# SLA / availability
# ---------------------------------------------------------------------------
def monthly_availability(register: pd.DataFrame, start, end, sla_target: float = SLA_TARGET) -> pd.DataFrame:
    """
    One row per calendar month in [start, end]: platform outage hours (major incidents,
    by the month they were reported), share of the month down, availability vs SLA target.
    """
    major = register[(register["kind"] == "Major incident") & register["outage_hours"].notna()]
    start, end = (pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t)
                  for t in (start, end))
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    rows = []
    for m in months:
        hours_in_month = calendar.monthrange(m.year, m.month)[1] * 24
        in_month = major[major["reported_at"].dt.tz_convert(None).dt.to_period("M") == m]
        down = float(in_month["outage_hours"].sum())
        allowed = (1 - sla_target) * hours_in_month
        rows.append({
            "month": m.to_timestamp(), "label": m.strftime("%b %Y"),
            "outage_hours": down, "n_outages": len(in_month),
            "pct_time_down": down / hours_in_month,
            "availability": 1 - down / hours_in_month,
            "allowed_hours": round(allowed, 2), "sla_target": sla_target,
            "sla_breached": down > allowed,
            "hours_over_sla": round(max(0.0, down - allowed), 2),
            "incidents": ", ".join(in_month["incident_id"]),
        })
    return pd.DataFrame(rows)


SLA_CREDIT = re.compile(r"SLA credit|service credit|remediation credit", re.I)


def sla_credit_calls(sentences: pd.DataFrame) -> set[str]:
    """Calls where an SLA / service credit was discussed."""
    return set(sentences.loc[sentences["text"].fillna("").str.contains(SLA_CREDIT), "call_id"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def save_register(register: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    register.to_parquet(out_dir / "incidents.parquet", index=False)
    log.info("Saved incidents.parquet (%d rows) to %s", len(register), out_dir)


def load_register(in_dir) -> pd.DataFrame:
    return pd.read_parquet(in_dir / "incidents.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from src.topics import load_topics

    parser = argparse.ArgumentParser(description="Build the incident register")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    mode = "rules" if args.rules_only else "hybrid"
    reg = build_register(load_enriched(), load_topics(mode_dir(mode))[0],
                         use_llm=not args.rules_only, refresh=args.refresh)
    pd.set_option("display.width", 220, "display.max_colwidth", 50)
    print(reg[["incident_id", "title", "category", "priority", "status", "owner",
               "minutes_to_own", "hours_to_resolve", "outage_hours", "source"]].to_string(index=False))
    save_register(reg, mode_dir(mode))

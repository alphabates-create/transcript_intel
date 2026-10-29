"""
enrich.py — Derive the fields the raw data doesn't have.

Adds to `calls`:
    call_type   support / external / internal
    account     customer company, from the email domain (None for internal calls)
    products    AegisCloud products discussed (keyword match on title + summary + topics)
    period      Pre-outage / Outage / Recovery / Post-Comply-launch

    teams       AegisCloud teams on the call, plus the customer (e.g. "Brightpath (customer)")

Adds to `sentences`:
    speaker_side  aegiscloud / customer / unknown
    speaker_team  the speaker's AegisCloud team, or their company for customers

New table `staff_roster`: one row per AegisCloud person with a derived team and the evidence.
The raw data has no team field, so teams come from role mentions ("Maria Santos (CSM)",
"this is Sarah Chen from Aegis support") and, failing that, the calls each person attends.

Decision: every rule here is a plain, deterministic heuristic (no LLM) so results
are reproducible and each function can be unit-tested on a single value.

Usage:
    python -m src.enrich               # enrich the saved tables from load.py
    from src.enrich import enrich_all  # use in notebooks
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import date as Date

import numpy as np
import pandas as pd

from src.load import INTERNAL_DOMAIN, PROCESSED_DIR, load_tables

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORT_PREFIX = "support case"

# Timeline (UTC dates). Outage window is inclusive on both ends.
OUTAGE_START = Date(2026, 3, 10)
OUTAGE_END = Date(2026, 3, 18)
COMPLY_LAUNCH = Date(2026, 4, 7)   # Comply v2 GA, per the launch-planning calls

PERIODS = ["Pre-outage", "Outage", "Recovery", "Post-Comply-launch"]

# Team derivation, checked in this order against each person's role evidence
TEAM_RULES: list[tuple[str, re.Pattern]] = [
    ("Leadership", re.compile(r"\b(VP|vice president|chief|CEO|CTO|COO|CRO|head of|director)\b", re.I)),
    ("Support", re.compile(r"\bsupport\b", re.I)),
    ("Solutions Engineering", re.compile(r"solutions? (architect|architecture|engineer)", re.I)),
    ("Customer Success & Sales", re.compile(r"\bCSM\b|customer success|account (manager|executive)|\bsales\b", re.I)),
    ("Engineering", re.compile(r"engineer|engineering|\bSRE\b|devops|infrastructure|on-call", re.I)),
    ("Product & Strategy", re.compile(r"\bproduct\b|\bPM\b|marketing|strategy", re.I)),
]
# Internal meeting titles that indicate which team someone works in (fallback only)
ENGINEERING_TITLES = re.compile(r"sprint|standup|retro|root cause|war room|reliability|scalability|"
                                r"engineering|remediation|pipeline", re.I)
PRODUCT_TITLES = re.compile(r"roadmap|design review|launch|adoption|competitive|win/loss|product sync", re.I)
# Corrections where the rules lack evidence, each from what colleagues say on the calls
STAFF_TEAM_OVERRIDES: dict[str, str] = {
    "brian.cho@aegiscloud.com": "Support",        # repeatedly asked for "the support" perspective
    "tom.bradley@aegiscloud.com": "Engineering",  # runs the war room, assigns engineering owners
}

# Product keywords. Product names are matched case-sensitively, because "protect" and
# "detect" are also ordinary verbs. Domain terms are matched case-insensitively.
PRODUCT_PATTERNS: dict[str, list[re.Pattern]] = {
    "Detect": [
        re.compile(r"\bDetect\b"),
        re.compile(r"\bthreat detection\b", re.I),
    ],
    "Protect": [
        re.compile(r"\bProtect\b"),
        re.compile(r"\bbackups?\b|\brestores?\b|\bsnapshots?\b", re.I),
    ],
    "Comply": [
        re.compile(r"\bComply\b"),
        re.compile(r"\bcompliance\b|\bSOC ?2\b|\bPCI(?: DSS)?\b|\bHIPAA\b|\bISO ?27001\b", re.I),
    ],
    "Identity": [
        re.compile(r"\bIdentity\b"),
        re.compile(r"\bSSO\b|\bMFA\b|\bSAML\b|\bSCIM\b|\bLDAP\b"),
    ],
    "LogVault": [
        re.compile(r"\blog ?vault\b", re.I),
    ],
}

# Trailing industry words removed from a domain to get the company's short name,
# e.g. brightpathcommerce -> Brightpath, steelpointmfg -> Steelpoint.
DOMAIN_SUFFIXES = sorted([
    "commerce", "health", "living", "trust", "invest", "platform", "retail", "wealth",
    "clinical", "precision", "goods", "bank", "cloud", "logistics", "capital",
    "software", "mkt", "pharma", "data", "industries", "edge", "brands", "mfg",
    "financial", "med", "ins", "corp", "labs", "systems", "group",
], key=len, reverse=True)   # longest first so "industries" wins over "ins"

log = logging.getLogger(__name__)


def _is_internal(email: str) -> bool:
    return email.lower().endswith("@" + INTERNAL_DOMAIN)


# ---------------------------------------------------------------------------
# Field derivations
# ---------------------------------------------------------------------------
def derive_call_type(row) -> str:
    """
    "support" if the title starts with "Support Case", "external" if any attendee
    is outside aegiscloud.com, otherwise "internal".
    """
    title = (row.get("title") or "").strip().lower()
    if title.startswith(SUPPORT_PREFIX):
        return "support"
    emails = list(row.get("emails") if row.get("emails") is not None else [])
    if any(not _is_internal(e) for e in emails):
        return "external"
    return "internal"


def extract_account(emails) -> str | None:
    """
    The customer's company name, from the most common non-AegisCloud email domain.
    e.g. brightpathcommerce.com -> "Brightpath". Returns None if all attendees are internal.
    """
    domains = [e.split("@")[-1].lower() for e in (emails if emails is not None else [])
               if "@" in e and not _is_internal(e)]
    if not domains:
        return None
    root = Counter(domains).most_common(1)[0][0].split(".")[0]
    for suffix in DOMAIN_SUFFIXES:
        # keep at least 3 characters so a short name is never stripped to nothing
        if root.endswith(suffix) and len(root) - len(suffix) >= 3:
            root = root[: -len(suffix)]
            break
    return root.capitalize()


def tag_product(text: str) -> list[str]:
    """AegisCloud products mentioned in `text`, in a fixed order (empty list if none)."""
    if not text:
        return []
    return [p for p, pats in PRODUCT_PATTERNS.items() if any(pat.search(text) for pat in pats)]


def assign_period(date) -> str:
    """
    Timeline label for a call date (date, datetime, Timestamp or ISO string):
      Pre-outage          before 10 Mar
      Outage              10–18 Mar (inclusive)
      Recovery            19 Mar – 6 Apr
      Post-Comply-launch  from 7 Apr (Comply v2 GA)
    """
    d = pd.Timestamp(date).date()
    if d < OUTAGE_START:
        return "Pre-outage"
    if d <= OUTAGE_END:
        return "Outage"
    if d < COMPLY_LAUNCH:
        return "Recovery"
    return "Post-Comply-launch"


def flag_speaker_side(sentences: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `sentences` with `speaker_side` = "aegiscloud" / "customer" / "unknown".

    Uses the speaker's matched email where available. If there is no email, everyone
    on an internal call must be AegisCloud staff. Otherwise the side is unknown.
    """
    out = sentences.copy()
    email = out["speaker_email"].fillna("").str.lower()
    side = pd.Series("unknown", index=out.index, dtype="object")
    has_email = email != ""
    side[has_email] = email[has_email].map(lambda e: "aegiscloud" if _is_internal(e) else "customer")

    internal_calls = set(calls.loc[calls["n_external"] == 0, "call_id"])
    side[~has_email & out["call_id"].isin(internal_calls)] = "aegiscloud"

    out["speaker_side"] = side
    n_unknown = (side == "unknown").sum()
    if n_unknown:
        log.warning("%d sentences have an unknown speaker side", n_unknown)
    return out


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
def _role_evidence(name: str, calls: pd.DataFrame, own_sentences: pd.Series) -> list[str]:
    """Role phrases for one person: "Name (role)" in summaries and self-introductions."""
    first = re.escape(name.split()[0])
    full = re.escape(name)
    paren = re.compile(rf"\b(?:{full}|{first})\s*\(([^)]{{2,40}})\)")
    evidence = [m.group(1) for text in calls["summary"].fillna("") for m in paren.finditer(text)]
    intro = re.compile(rf"\b(?:this is|I'm|I am)\s+(?:{full}|{first})\b[^.?!]{{0,60}}", re.I)
    evidence += [m.group(0) for text in own_sentences for m in intro.finditer(text)]
    return evidence


def _team_from_evidence(evidence: list[str]) -> str | None:
    votes = Counter(team for text in evidence for team, pat in TEAM_RULES if pat.search(text))
    return votes.most_common(1)[0][0] if votes else None


def build_staff_roster(calls: pd.DataFrame, sentences: pd.DataFrame) -> pd.DataFrame:
    """One row per AegisCloud person: email, name, team, and how the team was decided."""
    staff = sentences[sentences["speaker_side"] == "aegiscloud"].dropna(subset=["speaker_email"])
    call_type = calls.set_index("call_id")["call_type"]
    rows = []
    for email, g in staff.groupby("speaker_email"):
        name = g["speaker_name"].mode().iat[0]
        mix = g["call_id"].map(call_type).value_counts()
        team = STAFF_TEAM_OVERRIDES.get(email)
        basis = "override" if team else ""
        if not team:
            evidence = _role_evidence(name, calls, g["text"].fillna(""))
            team = _team_from_evidence(evidence)
            if team:
                pat = dict(TEAM_RULES)[team]
                basis = "role mention: " + next(e for e in evidence if pat.search(e))[:60]
        if not team and mix.get("support", 0) >= mix.sum() / 2:
            team, basis = "Support", "mostly support calls"
        if not team and mix.get("external", 0) >= mix.sum() / 2:
            team, basis = "Customer Success & Sales", "mostly customer calls"
        if not team:
            titles = calls.loc[calls["emails"].map(lambda e: email in list(e)), "title"]
            eng = int(titles.str.contains(ENGINEERING_TITLES).sum())
            prod = int(titles.str.contains(PRODUCT_TITLES).sum())
            team = "Engineering" if eng >= prod else "Product & Strategy"
            basis = f"internal meetings ({eng} engineering, {prod} product)"
        rows.append({"email": email, "name": name, "team": team, "basis": basis,
                     "n_calls": g["call_id"].nunique()})
    return pd.DataFrame(rows).sort_values(["team", "name"]).reset_index(drop=True)


def add_teams(calls: pd.DataFrame, sentences: pd.DataFrame,
              roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """calls.teams (list) and sentences.speaker_team."""
    team_of = roster.set_index("email")["team"].to_dict()

    def teams(row) -> list[str]:
        ours = sorted({team_of[e] for e in row["emails"] if e in team_of})
        return ours + ([f"{row['account']} (customer)"] if pd.notna(row.get("account")) else [])

    calls = calls.assign(teams=calls.apply(teams, axis=1))
    sentences = sentences.assign(speaker_team=np.where(
        sentences["speaker_side"] == "aegiscloud",
        sentences["speaker_email"].map(team_of), sentences["account"]))
    return calls, sentences


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _product_text(row) -> str:
    topics = row.get("topics")
    return " ".join([row.get("title") or "", row.get("summary") or "",
                     " ".join(list(topics) if topics is not None else [])])


def enrich_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Add call_type, account, products and period to the calls table."""
    out = calls.copy()
    out["call_type"] = out.apply(derive_call_type, axis=1)
    out["account"] = out["emails"].map(extract_account)
    out["products"] = out.apply(lambda r: tag_product(_product_text(r)), axis=1)
    out["period"] = pd.Categorical(out["start_time"].map(assign_period),
                                   categories=PERIODS, ordered=True)
    return out


def enrich_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Enrich calls and sentences, and copy account/period onto the other tables."""
    calls = enrich_calls(tables["calls"])
    lookup = calls[["call_id", "call_type", "account", "period"]]
    sentences = flag_speaker_side(tables["sentences"], calls).merge(lookup, on="call_id", how="left")
    roster = build_staff_roster(calls, sentences)
    calls, sentences = add_teams(calls, sentences, roster)
    out = {"calls": calls, "sentences": sentences, "staff_roster": roster}
    for name in ("key_moments", "action_items"):
        if name in tables:
            out[name] = tables[name].merge(lookup, on="call_id", how="left")
    return out


def save_enriched(tables: dict[str, pd.DataFrame], out_dir=PROCESSED_DIR) -> None:
    """Write each table to <out_dir>/<name>_enriched.parquet (raw tables are left as they are)."""
    for name, df in tables.items():
        path = out_dir / f"{name}_enriched.parquet"
        df.to_parquet(path, index=False)
        log.info("Saved %-13s %6d rows -> %s", name, len(df), path)


def load_enriched(in_dir=PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Read the enriched parquet tables back (used by analysis and the app)."""
    names = ["calls", "sentences", "key_moments", "action_items", "staff_roster"]
    return {n: pd.read_parquet(in_dir / f"{n}_enriched.parquet") for n in names}


def summarize(tables: dict[str, pd.DataFrame]) -> None:
    """Print a quick sanity report."""
    calls, sents = tables["calls"], tables["sentences"]
    print("Call types:\n" + calls["call_type"].value_counts().to_string())
    print("\nPeriods:\n" + calls["period"].value_counts(sort=False).to_string())
    print(f"\nAccounts: {calls['account'].nunique()}")
    print("\nProduct mentions:\n" + calls["products"].explode().value_counts().to_string())
    print(f"\nUntagged calls: {(calls['products'].str.len() == 0).sum()}")
    print("\nSpeaker side:\n" + sents["speaker_side"].value_counts().to_string())
    print("\nStaff teams:\n" + tables["staff_roster"]["team"].value_counts().to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tables = enrich_all(load_tables())
    summarize(tables)
    save_enriched(tables)

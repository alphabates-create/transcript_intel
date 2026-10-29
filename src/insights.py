"""
insights.py — Task 3: stakeholder insights.

    Account health      Sales / CS      per-account risk score from confidence, trend, churn
                                        signals, competitor mentions, support load, outage exposure
    Incident radius     Eng / execs     every call touching the Detect outage, on one timeline,
                                        with lag from the first war room and the accounts hit
    Feature gaps        Product         feature_gap key moments clustered into ranked asks
    Competitive intel   Sales / Product competitor mentions by account and month, with stated reason
    Talk time           Sales managers  customer share of talk time vs call outcome, per rep
    Action items        Ops / Support   owner workload and accounts with commitments across calls

Decision: the risk score is a transparent weighted sum (RISK_WEIGHTS) rather than a model,
so every account's score can be explained by its listed drivers. The only LLM use is naming
feature-gap clusters (cached; top TF-IDF terms are the fallback).

Outputs (data/processed/<mode>/, mode = hybrid or rules):
    account_health.parquet, incident_timeline.parquet, feature_gaps.parquet,
    feature_gap_moments.parquet, competitor_mentions.parquet, talk_time.parquet,
    rep_talk.parquet, owner_workload.parquet, insights_summary.json

Usage:
    python -m src.insights              # LLM names for feature-gap clusters where possible
    python -m src.insights --rules-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re

import anthropic
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.enrich import OUTAGE_END, OUTAGE_START, load_enriched
from src.llm import LLM_MODEL, parse_cached
from src.load import PROCESSED_DIR, mode_dir
from src.sentiment import load_sentiment
from src.topics import COMPETITORS, embed_calls, load_topics

OUTAGE_THEME = "Detect outage & fallout"
RENEWAL_THEME = "Renewal & contracts"

# Points out of 100; each component is scaled 0..1 before weighting
RISK_WEIGHTS = {
    "churn_signals": 25,     # churn_signal key moments, capped at 3
    "competitor": 20,        # a competitor was named on any of the account's calls
    "low_confidence": 20,    # latest call score below 3.5 (full weight at 1.5)
    "declining": 15,         # latest score at least 0.5 below the first
    "support_load": 10,      # 2+ support cases
    "outage_exposure": 10,   # had an outage-related call
}
RISK_TIERS = [(55, "High"), (30, "Medium"), (0, "Low")]

COMPETITOR_REASONS = {
    "Pricing": re.compile(r"pric|cheaper|discount|cost|budget|quote", re.I),
    "Reliability / uptime": re.compile(r"uptime|SLA|outage|reliab|redundan|downtime|99\.9", re.I),
    "Features": re.compile(r"feature|capabilit|integrat|report|dashboard|coverage", re.I),
    "Outreach": re.compile(r"reached out|approached|contacted|reach out|pitch", re.I),
}

log = logging.getLogger(__name__)


def _as_list(v) -> list:
    # NaN when a call has no row in the mode's outputs (e.g. added by a rules-only run)
    return list(v) if v is not None and not isinstance(v, float) else []


# ---------------------------------------------------------------------------
# Account health / churn risk
# ---------------------------------------------------------------------------
def account_health(calls: pd.DataFrame, key_moments: pd.DataFrame,
                   action_items: pd.DataFrame) -> pd.DataFrame:
    """One row per customer account, ranked by risk score (highest first)."""
    c = calls[calls["account"].notna()].sort_values("start_time")
    km = key_moments[key_moments["account"].notna()]
    churn = km[km["type"] == "churn_signal"].groupby("account").size()
    concern = km[km["type"] == "concern"].groupby("account").size()
    items = action_items[action_items["account"].notna()].groupby("account").size()

    rows = []
    for account, g in c.groupby("account"):
        themes = set().union(*g["themes"].map(lambda t: set(_as_list(t))))
        comps = sorted(set().union(*g["competitors"].map(lambda t: set(_as_list(t)))))
        first, last = g["sentiment_score"].iloc[0], g["sentiment_score"].iloc[-1]
        trend = last - first if len(g) > 1 else 0.0
        n_support = int((g["call_type"] == "support").sum())
        n_churn = int(churn.get(account, 0))

        parts = {
            "churn_signals": min(n_churn, 3) / 3,
            "competitor": float(bool(comps)),
            "low_confidence": float(np.clip((3.5 - last) / 2, 0, 1)),
            "declining": float(trend <= -0.5),
            "support_load": float(n_support >= 2),
            "outage_exposure": float(OUTAGE_THEME in themes),
        }
        score = round(sum(RISK_WEIGHTS[k] * v for k, v in parts.items()))
        drivers = []
        if n_churn:
            drivers.append(f"{n_churn} churn signal{'s' * (n_churn > 1)}")
        if comps:
            drivers.append("named " + ", ".join(comps))
        if last < 3.0:
            drivers.append(f"latest call {last:.1f}/5")
        if trend <= -0.5:
            drivers.append(f"down {abs(trend):.1f} since first call")
        if n_support >= 2:
            drivers.append(f"{n_support} support cases")
        if OUTAGE_THEME in themes:
            drivers.append("hit by outage")

        rows.append({
            "account": account,
            "risk_score": score,
            "risk_tier": next(t for floor, t in RISK_TIERS if score >= floor),
            "drivers": "; ".join(drivers) or "no risk signals",
            "n_calls": len(g),
            "n_support": n_support,
            "first_call": g["start_time"].iloc[0],
            "last_call": g["start_time"].iloc[-1],
            "avg_score": round(float(g["sentiment_score"].mean()), 2),
            "first_score": first,
            "last_score": last,
            "score_trend": round(float(trend), 2),
            "customer_vader": round(float(g["customer_vader"].mean()), 3),
            "churn_signals": n_churn,
            "concerns": int(concern.get(account, 0)),
            "competitors": comps,
            "outage_affected": OUTAGE_THEME in themes,
            "renewal_in_play": RENEWAL_THEME in themes,
            "action_items": int(items.get(account, 0)),
            "last_title": g["title"].iloc[-1],
        })
    return pd.DataFrame(rows).sort_values(["risk_score", "avg_score"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Incident blast radius
# ---------------------------------------------------------------------------
def incident_timeline(calls: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every outage-themed call from the outage start on, timed from the first internal war room."""
    is_outage = calls["themes"].map(lambda t: OUTAGE_THEME in _as_list(t))
    t = calls[is_outage & (calls["start_time"].dt.date >= OUTAGE_START)].sort_values("start_time").copy()
    if t.empty:
        return t, {}
    internal = t[t["call_type"] == "internal"]
    t0 = internal["start_time"].iloc[0] if len(internal) else t["start_time"].iloc[0]
    t["hours_from_start"] = ((t["start_time"] - t0).dt.total_seconds() / 3600).round(1)
    t["in_window"] = t["start_time"].dt.date.between(OUTAGE_START, OUTAGE_END)
    t = t[["call_id", "start_time", "hours_from_start", "in_window", "title", "call_type",
           "account", "sentiment_score", "km_churn_signal", "competitors"]]

    customer = t[t["call_type"] != "internal"]
    affected = set(customer["account"].dropna())
    window = calls[calls["start_time"].dt.date.between(OUTAGE_START, OUTAGE_END)]
    before = calls[(calls["start_time"].dt.date < OUTAGE_START) & calls["account"].isin(affected)]
    after = calls[(calls["start_time"].dt.date > OUTAGE_END) & calls["account"].isin(affected)]
    named_after = set(after.loc[after["competitors"].map(len) > 0, "account"])
    first_customer = customer["start_time"].min()
    metrics = {
        "first_war_room": str(t0),
        "n_outage_calls": int(len(t)),
        "n_window_calls": int(len(window)),
        "n_affected_accounts": len(affected),
        "hours_to_first_customer_call": round(float((first_customer - t0).total_seconds() / 3600), 1)
        if pd.notna(first_customer) else None,
        "hours_to_last_window_call": float(t.loc[t["in_window"], "hours_from_start"].max()),
        "window_avg_score": round(float(window["sentiment_score"].mean()), 2),
        "affected_avg_before": round(float(before["sentiment_score"].mean()), 2) if len(before) else None,
        "affected_avg_after": round(float(after["sentiment_score"].mean()), 2) if len(after) else None,
        "affected_naming_competitor_after": sorted(named_after),
    }
    return t.reset_index(drop=True), metrics


# ---------------------------------------------------------------------------
# Voice-of-customer feature gaps
# ---------------------------------------------------------------------------
class ClusterName(BaseModel):
    name: str
    summary: str


GAP_NAME_SYSTEM = ("You name groups of customer feature requests and product gaps for a product "
                   "manager at AegisCloud, a cloud-security company. name: 2-5 words, the ask itself "
                   "(e.g. 'Granular file-level restore'). summary: one sentence on what customers want.")


def _name_cluster(use_llm: bool, texts: list[str], terms: str,
                  refresh: bool = False) -> tuple[str, str, str]:
    """(name, summary, source). Uses Claude when allowed, cached; else the top terms."""
    fallback = (terms.split(", ")[0].capitalize() + " – " + ", ".join(terms.split(", ")[1:3]), "", "terms")
    if not use_llm:
        return fallback
    blob = json.dumps({"model": LLM_MODEL, "prompt": "gap-name-v1", "texts": sorted(texts)})
    try:
        d = parse_cached("feature_gaps", "gap", blob, GAP_NAME_SYSTEM,
                         "\n".join(f"- {t}" for t in texts), ClusterName, refresh, max_tokens=2000)
    except anthropic.AuthenticationError:
        raise
    except anthropic.APIError as e:
        log.warning("Cluster naming failed (%s); using top terms", e)
        return fallback
    return (d["name"], d["summary"], "llm") if d else fallback


def feature_gaps(key_moments: pd.DataFrame, use_llm: bool = True, refresh: bool = False,
                 k: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster feature_gap moments into ranked asks. Returns (asks, moments with ask label)."""
    m = key_moments[key_moments["type"] == "feature_gap"].reset_index(drop=True).copy()
    if len(m) < k:
        k = max(1, len(m))
    if m.empty:
        return pd.DataFrame(), m
    texts = m["text"].tolist()
    m["gap_cluster"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embed_calls(texts))

    tfidf = TfidfVectorizer(stop_words="english", min_df=1)
    X = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())

    rows = []
    for c, g in m.groupby("gap_cluster"):
        weights = np.asarray(X[g.index].mean(axis=0)).ravel()
        top = ", ".join(terms[weights.argsort()[::-1][:5]])
        try:
            name, summary, source = _name_cluster(use_llm, g["text"].tolist(), top, refresh)
        except anthropic.AuthenticationError:
            log.warning("No valid Anthropic credentials; naming feature gaps by top terms")
            use_llm = False
            name, summary, source = _name_cluster(False, g["text"].tolist(), top)
        accounts = sorted(g["account"].dropna().unique())
        rows.append({"gap_cluster": c, "ask": name, "summary": summary, "name_source": source,
                     "n_mentions": len(g), "n_accounts": len(accounts),
                     "accounts": accounts, "top_terms": top, "example": g["text"].iloc[0]})
    asks = pd.DataFrame(rows).sort_values(["n_accounts", "n_mentions"], ascending=False).reset_index(drop=True)
    m = m.merge(asks[["gap_cluster", "ask"]], on="gap_cluster")
    return asks, m[["call_id", "account", "speaker", "text", "ask", "period"]]


# ---------------------------------------------------------------------------
# Competitive intelligence
# ---------------------------------------------------------------------------
def competitor_mentions(sentences: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """
    One row per sentence naming a competitor. The reason is read from that sentence plus
    its neighbours, since "we're looking at SentinelShield" often has the why next to it.
    """
    rows = []
    meta = calls.set_index("call_id")[["start_time", "month", "title", "call_type", "account"]]
    for call_id, g in sentences.sort_values(["call_id", "idx"]).groupby("call_id"):
        texts = g["text"].fillna("").tolist()
        sides = g["speaker_side"].tolist()
        for i, text in enumerate(texts):
            named = [n for n, pat in COMPETITORS.items() if pat.search(text)]
            if not named:
                continue
            context = " ".join(texts[max(0, i - 1): i + 2])
            reasons = [r for r, pat in COMPETITOR_REASONS.items() if pat.search(context)]
            for n in named:
                rows.append({"call_id": call_id, "competitor": n, "speaker_side": sides[i],
                             "reasons": reasons or ["Other"], "quote": text})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.join(meta, on="call_id").sort_values("start_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Talk time / coaching
# ---------------------------------------------------------------------------
def talk_time(sentences: pd.DataFrame, calls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per customer-facing call: customer share of talk time and the lead rep. Plus per-rep rollup."""
    s = sentences[sentences["call_type"].isin(["external", "support"])]
    secs = s.groupby(["call_id", "speaker_side"])["duration_s"].sum().unstack(fill_value=0)
    per_call = pd.DataFrame({
        "customer_share": (secs.get("customer", 0) / secs.sum(axis=1)).round(3),
        "talk_minutes": (secs.sum(axis=1) / 60).round(1),
    })
    staff = s[s["speaker_side"] == "aegiscloud"]
    lead = (staff.groupby(["call_id", "speaker_name"])["duration_s"].sum()
            .reset_index().sort_values("duration_s").groupby("call_id").tail(1)
            .set_index("call_id")["speaker_name"])
    per_call["lead_rep"] = lead
    per_call = per_call.join(calls.set_index("call_id")[["title", "call_type", "account", "sentiment_score"]])
    # Quartiles: customer share sits in a narrow band, so fixed cut-offs leave buckets empty
    edges = per_call["customer_share"].quantile([0, .25, .5, .75, 1])
    if edges.is_unique:
        q = edges.round(2).tolist()
        labels = [f"Q{i + 1} ({q[i]:.0%}–{q[i + 1]:.0%})" for i in range(4)]
        per_call["talk_bucket"] = pd.qcut(per_call["customer_share"], 4, labels=labels)
    else:
        # A few calls with repeated shares (e.g. a short period in the dashboard) can't make four
        # quartiles; use as many groups as the distinct edges allow
        e = sorted(set(edges))
        if len(e) < 2:
            per_call["talk_bucket"] = pd.Categorical([f"Q1 ({e[0]:.0%})"] * len(per_call), ordered=True)
        else:
            labels = [f"Q{i + 1} ({e[i]:.0%}–{e[i + 1]:.0%})" for i in range(len(e) - 1)]
            per_call["talk_bucket"] = pd.cut(per_call["customer_share"], e, labels=labels, include_lowest=True)
    per_call = per_call.reset_index()

    reps = (per_call.groupby("lead_rep")
            .agg(n_calls=("call_id", "size"), avg_customer_share=("customer_share", "mean"),
                 avg_score=("sentiment_score", "mean"))
            .round(3).sort_values("n_calls", ascending=False).reset_index())
    return per_call, reps


# ---------------------------------------------------------------------------
# Action-item follow-through
# ---------------------------------------------------------------------------
def owner_workload(action_items: pd.DataFrame) -> pd.DataFrame:
    a = action_items.dropna(subset=["owner"])
    return (a.groupby("owner")
            .agg(n_items=("task", "size"), n_calls=("call_id", "nunique"),
                 n_accounts=("account", "nunique"),
                 share_customer=("call_type", lambda t: (t != "internal").mean()))
            .round(2).sort_values("n_items", ascending=False).reset_index())


def recurring_commitments(action_items: pd.DataFrame) -> pd.DataFrame:
    """Accounts whose action items span 2+ calls: commitments that keep coming back."""
    a = action_items.dropna(subset=["account"])
    g = (a.groupby("account")
         .agg(n_items=("task", "size"), n_calls=("call_id", "nunique"),
              owners=("owner", lambda o: sorted(o.dropna().unique()))))
    return g[g["n_calls"] >= 2].sort_values(["n_calls", "n_items"], ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def prepare_calls(tables: dict[str, pd.DataFrame], call_topics: pd.DataFrame,
                  call_sent: pd.DataFrame) -> pd.DataFrame:
    """Calls joined with the topic and sentiment outputs the insights depend on."""
    return (tables["calls"]
            .merge(call_topics[["call_id", "primary_theme", "themes", "competitors", "n_competitor_mentions"]],
                   on="call_id", how="left")
            .merge(call_sent[["call_id", "customer_vader", "staff_vader", "arc"]], on="call_id", how="left"))


def build_insights(tables: dict[str, pd.DataFrame], call_topics: pd.DataFrame, call_sent: pd.DataFrame,
                   use_llm: bool = True, refresh: bool = False) -> dict:
    calls = prepare_calls(tables, call_topics, call_sent)
    timeline, incident = incident_timeline(calls)
    asks, gap_moments = feature_gaps(tables["key_moments"], use_llm=use_llm, refresh=refresh)
    comp = competitor_mentions(tables["sentences"], calls)
    per_call_talk, reps = talk_time(tables["sentences"], calls)
    health = account_health(calls, tables["key_moments"], tables["action_items"])

    talk_r = per_call_talk["customer_share"].corr(per_call_talk["sentiment_score"])
    summary = {
        "incident": incident,
        "n_accounts": int(len(health)),
        "risk_tiers": health["risk_tier"].value_counts().to_dict(),
        "top_risk_accounts": health.head(5)["account"].tolist(),
        "n_feature_asks": int(len(asks)),
        "competitor_calls": comp.groupby("competitor")["call_id"].nunique().sort_values(ascending=False).to_dict()
        if len(comp) else {},
        "talk_share_vs_score_r": round(float(talk_r), 3),
        "talk_bucket_scores": per_call_talk.groupby("talk_bucket", observed=True)["sentiment_score"]
        .mean().round(2).to_dict(),
    }
    return {
        "account_health": health, "incident_timeline": timeline, "feature_gaps": asks,
        "feature_gap_moments": gap_moments, "competitor_mentions": comp, "talk_time": per_call_talk,
        "rep_talk": reps, "owner_workload": owner_workload(tables["action_items"]),
        "recurring_commitments": recurring_commitments(tables["action_items"]), "summary": summary,
    }


TABLE_NAMES = ["account_health", "incident_timeline", "feature_gaps", "feature_gap_moments",
               "competitor_mentions", "talk_time", "rep_talk", "owner_workload", "recurring_commitments"]


def save_insights(results: dict, out_dir=PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        results[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "insights_summary.json").write_text(
        json.dumps(results["summary"], indent=2, default=str), encoding="utf-8")
    log.info("Saved %d insight tables and insights_summary.json to %s", len(TABLE_NAMES), out_dir)


def load_insights(in_dir=PROCESSED_DIR) -> dict:
    """Read the saved insight outputs back (used by the app)."""
    out = {n: pd.read_parquet(in_dir / f"{n}.parquet") for n in TABLE_NAMES}
    out["summary"] = json.loads((in_dir / "insights_summary.json").read_text(encoding="utf-8"))
    return out


def summarize(results: dict) -> None:
    print(json.dumps(results["summary"], indent=2, default=str))
    print("\nTop 10 accounts by risk:")
    print(results["account_health"].head(10)[["account", "risk_score", "risk_tier", "drivers"]].to_string(index=False))
    print("\nFeature asks:")
    print(results["feature_gaps"][["ask", "n_mentions", "n_accounts", "name_source"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Build stakeholder insights")
    parser.add_argument("--rules-only", action="store_true", help="name feature-gap clusters by top terms")
    args = parser.parse_args()
    mode = "rules" if args.rules_only else "hybrid"
    results = build_insights(load_enriched(), load_topics(mode_dir(mode))[0], load_sentiment()[1],
                             use_llm=not args.rules_only)
    summarize(results)
    save_insights(results, mode_dir(mode))

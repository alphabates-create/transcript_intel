"""
topics.py — Task 1: categorize calls by theme (hybrid).

    1. Discover  embed each call (title + summary) and cluster with KMeans, so the
                 taxonomy is grounded in the data. Top TF-IDF terms describe each cluster.
    2. Define    a fixed taxonomy of 10 themes (THEMES), consolidated from the clusters
                 and the vendor's 351 free-text topic labels.
    3. Classify  Claude assigns a primary theme, every theme that applies (multi-label),
                 a certainty and a one-line rationale. JSON is validated with pydantic and
                 cached in data/processed/llm_cache/, so re-runs are free and identical.
                 Keyword rules are the fallback when the LLM is unavailable or declines.
    4. Validate  LLM vs keyword-rule agreement, and how well clusters line up with themes.

Outputs (data/processed/<mode>/, mode = hybrid or rules):
    call_topics.parquet      one row per call: cluster, primary_theme, themes, competitors, ...
    topic_clusters.parquet   one row per cluster: size, top terms, dominant theme, purity
    topic_validation.json    agreement metrics

Usage:
    python -m src.topics                # LLM where credentials exist, rules otherwise
    python -m src.topics --rules-only   # no API calls (writes the rules-mode outputs)
    python -m src.topics --refresh      # ignore the LLM cache
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.enrich import load_enriched
from src.llm import LLM_MODEL, parse_cached
from src.load import PROCESSED_DIR, mode_dir

# ---------------------------------------------------------------------------
# Taxonomy: name -> (definition for the LLM, keyword rule on title + summary + topics)
# ---------------------------------------------------------------------------
THEMES: dict[str, tuple[str, re.Pattern]] = {
    "Compliance & audit": (
        "Regulatory or security compliance work: SOC 2, PCI DSS, HIPAA, ISO 27001, audits, "
        "compliance reporting and evidence.",
        re.compile(r"complian|audit|SOC ?2|PCI|HIPAA|ISO ?27001|GDPR", re.I),
    ),
    "Detect outage & fallout": (
        "The March Detect pipeline outage and its consequences: war rooms, escalations, "
        "lost threat visibility, data gaps, root cause, remediation, SLA credits, trust repair.",
        re.compile(r"outage|war room|post-incident|incident|root cause|pipeline failure|"
                   r"loss of threat visibility|dashboard down|alerts not firing|data gaps|SLA credit", re.I),
    ),
    "Comply v2 launch": (
        "The Comply v2 release: design, launch readiness, GA (7 Apr), previews, early access, "
        "deployment of v2 and adoption.",
        re.compile(r"Comply v2|Comply 2|early access|launch readiness|\bGA\b", re.I),
    ),
    "Renewal & contracts": (
        "Commercial conversations: renewals, contract terms, pricing, discounts, multi-year deals, "
        "expansions and account reviews with a commercial outcome.",
        re.compile(r"renewal|contract|multi-year|pricing|discount|expansion|upsell", re.I),
    ),
    "Internal planning": (
        "AegisCloud-only operating meetings: sprint planning, retros, standups, roadmap, "
        "all-hands, quarterly planning, internal strategy reviews.",
        re.compile(r"sprint|standup|stand-up|all hands|retro|roadmap|quarterly planning|win/loss", re.I),
    ),
    "Competitive threat": (
        "Competitors come up as a real risk: the customer is evaluating, comparing or considering "
        "switching to another vendor (SentinelShield, VaultEdge, CyberNova, Fortiguard Pro), or "
        "AegisCloud is planning its competitive response.",
        re.compile(r"competit|vendor comparison|SentinelShield|VaultEdge|CyberNova|Fortiguard|switching", re.I),
    ),
    "Identity / SSO / MFA": (
        "The Identity product and access management: SSO, MFA, SAML, SCIM, LDAP, provisioning, "
        "group policy sync, login failures.",
        re.compile(r"\bIdentity\b|SSO|MFA|SAML|SCIM|LDAP|provisioning|login", re.I),
    ),
    "Backup / Protect": (
        "The Protect product and data protection: backups, restores, snapshots, backup windows "
        "and backup performance.",
        re.compile(r"\bProtect\b|backup|restore|snapshot", re.I),
    ),
    "Onboarding & deployment": (
        "Getting a customer live: onboarding kickoffs, module setup, deployment, configuration, "
        "integration rollout.",
        re.compile(r"onboarding|kickoff|deployment|setup|implementation|rollout", re.I),
    ),
    "Billing disputes": (
        "Invoices, billing errors, overage charges, seat-count discrepancies, refunds or credits "
        "being disputed.",
        re.compile(r"invoice|billing|overage|discrepanc|refund|charges? dispute", re.I),
    ),
}
THEME_NAMES = list(THEMES)
ThemeName = Literal[tuple(THEME_NAMES)]

# Competitors named in the transcripts (see docs/architecture.md, 6.5)
COMPETITORS: dict[str, re.Pattern] = {
    "SentinelShield": re.compile(r"\bSentinel ?Shield\b", re.I),
    "VaultEdge": re.compile(r"\bVault ?Edge\b", re.I),
    "CyberNova": re.compile(r"\bCyber ?Nova\b", re.I),
    "Fortiguard Pro": re.compile(r"\bFortiguard(?: Pro)?\b", re.I),
}

N_CLUSTERS = 10
EMBED_MODEL = "all-MiniLM-L6-v2"
PROMPT_VERSION = "topics-v1"   # bump to invalidate the cache when the prompt changes

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _as_list(value) -> list:
    return list(value) if value is not None else []


def call_text(row) -> str:
    """Title + summary + vendor topic labels: the text both rules and clustering see."""
    return " ".join([row.get("title") or "", row.get("summary") or "",
                     " ".join(_as_list(row.get("topics")))])


# ---------------------------------------------------------------------------
# 1. Discover
# ---------------------------------------------------------------------------
def embed_calls(texts: list[str]) -> np.ndarray:
    """Sentence embeddings (local MiniLM). Falls back to TF-IDF if the model can't load."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:  # offline, missing weights, ...
        log.warning("Embedding model unavailable (%s); clustering on TF-IDF instead", e)
        return TfidfVectorizer(stop_words="english", min_df=2).fit_transform(texts).toarray()


def discover_clusters(calls: pd.DataFrame, k: int = N_CLUSTERS) -> tuple[pd.Series, pd.DataFrame]:
    """
    Cluster calls and describe each cluster by its top TF-IDF terms.
    Returns (cluster id per call, cluster table).
    """
    texts = calls.apply(call_text, axis=1).tolist()
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embed_calls(texts))

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())
    rows = []
    for c in range(k):
        mask = labels == c
        weights = np.asarray(X[mask].mean(axis=0)).ravel()
        rows.append({"cluster": c, "n_calls": int(mask.sum()),
                     "top_terms": ", ".join(terms[weights.argsort()[::-1][:6]])})
    return pd.Series(labels, index=calls.index, name="cluster"), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Classify
# ---------------------------------------------------------------------------
class ThemeAssignment(BaseModel):
    primary_theme: ThemeName
    themes: list[ThemeName]
    certainty: float
    rationale: str


SYSTEM_PROMPT = (
    "You categorize recorded calls at AegisCloud, a B2B cloud-security company whose products "
    "are Detect, Protect, Comply, Identity and LogVault. Assign each call to themes from this "
    "fixed taxonomy:\n\n"
    + "\n".join(f"- {name}: {definition}" for name, (definition, _) in THEMES.items())
    + "\n\nRules:\n"
    "- primary_theme is the single theme the call is mostly about.\n"
    "- themes lists every theme that is a substantive part of the call, including the primary. "
    "Leave out themes that only come up in passing.\n"
    "- certainty is your confidence in primary_theme, from 0 to 1.\n"
    "- rationale is one short sentence that cites what in the call supports the primary theme."
)


def rule_themes(text: str) -> list[str]:
    """Keyword-rule themes, in taxonomy order."""
    return [name for name, (_, pat) in THEMES.items() if pat.search(text)]


def _rule_assignment(row) -> dict:
    """Fallback: all rule themes; primary = most keyword hits, title hits weighted 3x."""
    text, title = call_text(row), row.get("title") or ""
    themes = rule_themes(text)
    if row.get("call_type") == "internal" and "Internal planning" not in themes:
        themes.append("Internal planning")
    themes = themes or ["Internal planning" if row.get("call_type") == "internal" else "Renewal & contracts"]
    score = {t: 3 * len(THEMES[t][1].findall(title)) + len(THEMES[t][1].findall(text)) for t in themes}
    return {"primary_theme": max(themes, key=score.get), "themes": themes, "certainty": None,
            "rationale": "Keyword rules", "source": "rules"}


def _llm_payload(row, key_moments: list[str]) -> dict:
    return {
        "title": row.get("title"),
        "call_type": row.get("call_type"),
        "date": str(row.get("date")),
        "summary": row.get("summary"),
        "vendor_topics": _as_list(row.get("topics")),
        "key_moments": key_moments,
    }


def classify_with_llm(payload: dict, refresh: bool = False) -> dict | None:
    """
    Classify one call with Claude. Returns the assignment dict, or None if the model
    declined or gave nothing usable (the caller then uses the keyword rules).
    """
    blob = json.dumps({"model": LLM_MODEL, "prompt": PROMPT_VERSION, "call": payload},
                      sort_keys=True, default=str)
    result = parse_cached("topics", "topics", blob, SYSTEM_PROMPT,
                          json.dumps(payload, ensure_ascii=False), ThemeAssignment, refresh)
    if result is None:
        return None
    # primary must be in the list, keep taxonomy order, clamp certainty
    result["themes"] = [t for t in THEME_NAMES if t in set(result["themes"]) | {result["primary_theme"]}]
    result["certainty"] = round(min(max(float(result["certainty"]), 0.0), 1.0), 2)
    result["source"] = "llm"
    result.pop("usage", None)
    return result


def classify_calls(calls: pd.DataFrame, key_moments: pd.DataFrame,
                   use_llm: bool = True, refresh: bool = False, workers: int = 6) -> pd.DataFrame:
    """Theme assignment per call: LLM where possible, keyword rules otherwise."""
    km_text = key_moments.groupby("call_id")["text"].apply(list).to_dict()
    rows = [r for _, r in calls.iterrows()]
    results: list[dict | None] = [None] * len(rows)

    if use_llm:
        def one(i: int) -> None:
            row = rows[i]
            try:
                results[i] = classify_with_llm(_llm_payload(row, km_text.get(row["call_id"], [])), refresh)
            except ValidationError as e:
                log.warning("Invalid LLM output for %s: %s", row["call_id"], e)
            except anthropic.AuthenticationError:
                raise
            except anthropic.RateLimitError as e:
                log.warning("Rate limited on %s (%s)", row["call_id"], e.message)
            except anthropic.APIStatusError as e:
                log.warning("API error %s on %s: %s", e.status_code, row["call_id"], e.message)
            except anthropic.APIConnectionError:
                log.warning("Network error on %s", row["call_id"])

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one, range(len(rows))))
        except anthropic.AuthenticationError:
            log.warning("No valid Anthropic credentials; using keyword rules for every call")

    out = []
    for row, res in zip(rows, results):
        res = res or _rule_assignment(row)
        out.append({"call_id": row["call_id"], **res, "rule_themes": rule_themes(call_text(row))})
    df = pd.DataFrame(out)
    log.info("Classified %d calls: %s", len(df), df["source"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# Competitors (feeds the Competitive threat theme and the dashboard)
# ---------------------------------------------------------------------------
def tag_competitors(sentences: pd.DataFrame) -> pd.DataFrame:
    """Per call: which competitors are named in the transcript, and how many times."""
    rows = []
    for call_id, texts in sentences.groupby("call_id")["text"]:
        joined = " ".join(texts)
        counts = {name: len(pat.findall(joined)) for name, pat in COMPETITORS.items()}
        rows.append({"call_id": call_id,
                     "competitors": [n for n, c in counts.items() if c],
                     "n_competitor_mentions": sum(counts.values())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Validate
# ---------------------------------------------------------------------------
def validate(assigned: pd.DataFrame, clusters: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """LLM-vs-rules agreement, plus each cluster's dominant theme and purity."""
    llm = assigned[assigned["source"] == "llm"]

    def jaccard(a, b) -> float:
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if a | b else 1.0

    metrics = {
        "n_calls": int(len(assigned)),
        "n_llm": int(len(llm)),
        "n_rules": int((assigned["source"] == "rules").sum()),
        "primary_in_rule_themes": round(float(llm.apply(
            lambda r: r["primary_theme"] in set(r["rule_themes"]), axis=1).mean()), 3) if len(llm) else None,
        "mean_jaccard_llm_vs_rules": round(float(llm.apply(
            lambda r: jaccard(r["themes"], r["rule_themes"]), axis=1).mean()), 3) if len(llm) else None,
        "model": LLM_MODEL if len(llm) else None,
    }

    dom = (assigned.groupby("cluster")["primary_theme"]
           .agg(lambda s: (s.mode().iat[0], round(float((s == s.mode().iat[0]).mean()), 2))))
    clusters = clusters.assign(dominant_theme=clusters["cluster"].map(lambda c: dom[c][0]),
                               purity=clusters["cluster"].map(lambda c: dom[c][1]))
    metrics["cluster_purity_weighted"] = round(float(
        (clusters["purity"] * clusters["n_calls"]).sum() / clusters["n_calls"].sum()), 3)
    return metrics, clusters


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_topics(tables: dict[str, pd.DataFrame], use_llm: bool = True,
                 refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    calls = tables["calls"]
    cluster_ids, clusters = discover_clusters(calls)
    assigned = classify_calls(calls, tables["key_moments"], use_llm=use_llm, refresh=refresh)
    assigned = (assigned.merge(calls[["call_id"]].assign(cluster=cluster_ids.values), on="call_id")
                .merge(tag_competitors(tables["sentences"]), on="call_id", how="left"))
    assigned["competitors"] = assigned["competitors"].map(lambda v: v if isinstance(v, list) else [])
    assigned["n_competitor_mentions"] = assigned["n_competitor_mentions"].fillna(0).astype(int)
    metrics, clusters = validate(assigned, clusters)
    return assigned, clusters, metrics


def save_topics(call_topics: pd.DataFrame, clusters: pd.DataFrame, metrics: dict,
                out_dir=PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    call_topics.to_parquet(out_dir / "call_topics.parquet", index=False)
    clusters.to_parquet(out_dir / "topic_clusters.parquet", index=False)
    (out_dir / "topic_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved call_topics, topic_clusters and topic_validation to %s", out_dir)


def load_topics(in_dir=PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read the saved topic outputs back (used by the app)."""
    return (pd.read_parquet(in_dir / "call_topics.parquet"),
            pd.read_parquet(in_dir / "topic_clusters.parquet"),
            json.loads((in_dir / "topic_validation.json").read_text(encoding="utf-8")))


def summarize(call_topics: pd.DataFrame, clusters: pd.DataFrame, metrics: dict) -> None:
    print("Calls per theme (multi-label):")
    print(call_topics["themes"].explode().value_counts().reindex(THEME_NAMES, fill_value=0).to_string())
    print("\nPrimary theme:")
    print(call_topics["primary_theme"].value_counts().to_string())
    print("\nClusters:")
    print(clusters.to_string(index=False))
    print("\nValidation:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Categorize calls by theme")
    parser.add_argument("--rules-only", action="store_true", help="skip the LLM, use keyword rules")
    parser.add_argument("--refresh", action="store_true", help="ignore the LLM cache")
    args = parser.parse_args()

    results = build_topics(load_enriched(), use_llm=not args.rules_only, refresh=args.refresh)
    summarize(*results)
    save_topics(*results, out_dir=mode_dir("rules" if args.rules_only else "hybrid"))

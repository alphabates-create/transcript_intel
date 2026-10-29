"""
sentiment.py — Task 2: score sentiment at several levels and slice it.

    Sentence  VADER compound score (-1..1) for every sentence, plus its third of the call.
    Call      the vendor's 1–5 sentimentScore vs our own word-weighted VADER aggregate.
              Both are z-scored; calls where they disagree by > DISAGREE_Z are flagged.
    Speaker   customer vs AegisCloud staff, scored separately (staff stay upbeat on bad
              calls, so the customer score is the truer measure of customer mood).
    Arc       first third vs last third of each call: improved / declined / flat, and
              how well that lines up with the vendor's positive_pivot key moments.
    Slices    call type, month, period, product and theme.

Decision: VADER is a transparent, free baseline. It is used to check the vendor score,
not to replace it; the dashboard leads with the vendor score.

Outputs (data/processed/):
    sentence_sentiment.parquet   call_id, idx, vader, third
    call_sentiment.parquet       one row per call (scores above + disagreement flag)
    sentiment_slices.parquet     dimension, value, n_calls, vendor_mean, vader_mean, customer_mean, staff_mean
    sentiment_validation.json    correlations, disagreement count, arc / pivot alignment

Usage:
    python -m src.sentiment
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.enrich import load_enriched
from src.load import PROCESSED_DIR

DISAGREE_Z = 1.5     # |z(vendor) - z(vader)| above this flags a call
ARC_THRESHOLD = 0.10  # change in mean compound between first and last third

SLICE_DIMENSIONS = ["call_type", "month", "period", "product", "theme"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence level
# ---------------------------------------------------------------------------
def score_sentences(sentences: pd.DataFrame) -> pd.DataFrame:
    """Add `vader` (compound, -1..1) and `third` (1, 2, 3 by position within the call)."""
    analyzer = SentimentIntensityAnalyzer()
    out = sentences.copy()
    out["vader"] = out["text"].fillna("").map(lambda t: analyzer.polarity_scores(t)["compound"])
    position = out.groupby("call_id")["idx"].rank(method="first", pct=True)
    out["third"] = np.ceil(position * 3).clip(1, 3).astype(int)
    return out


def _weighted_mean(df: pd.DataFrame) -> float:
    """Word-weighted mean compound, so "Yeah." doesn't count as much as a full complaint."""
    w = df["n_words"].clip(lower=1)
    return float((df["vader"] * w).sum() / w.sum()) if len(df) else np.nan


# ---------------------------------------------------------------------------
# Call, speaker and arc level
# ---------------------------------------------------------------------------
def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def _arc(delta: float) -> str:
    if pd.isna(delta):
        return "n/a"
    if delta > ARC_THRESHOLD:
        return "improved"
    if delta < -ARC_THRESHOLD:
        return "declined"
    return "flat"


def score_calls(calls: pd.DataFrame, scored: pd.DataFrame, key_moments: pd.DataFrame) -> pd.DataFrame:
    """One row per call: vendor vs VADER, customer vs staff, and the within-call arc."""
    by_call = scored.groupby("call_id")
    side = (scored.groupby(["call_id", "speaker_side"]).apply(_weighted_mean, include_groups=False)
            .unstack())
    thirds = scored.groupby(["call_id", "third"]).apply(_weighted_mean, include_groups=False).unstack()
    cust = scored[scored["speaker_side"] == "customer"]
    cust_thirds = cust.groupby(["call_id", "third"]).apply(_weighted_mean, include_groups=False).unstack()
    pivots = set(key_moments.loc[key_moments["type"] == "positive_pivot", "call_id"])

    out = calls[["call_id", "sentiment_score", "sentiment_label"]].rename(
        columns={"sentiment_score": "vendor_score", "sentiment_label": "vendor_label"}).set_index("call_id")
    out["vader_mean"] = by_call.apply(_weighted_mean, include_groups=False)
    out["customer_vader"] = side.get("customer")
    out["staff_vader"] = side.get("aegiscloud")
    out["side_gap"] = out["customer_vader"] - out["staff_vader"]
    out["arc_first"], out["arc_last"] = thirds.get(1), thirds.get(3)
    out["arc_delta"] = out["arc_last"] - out["arc_first"]
    out["arc"] = out["arc_delta"].map(_arc)
    out["customer_arc_delta"] = cust_thirds.get(3) - cust_thirds.get(1) if len(cust_thirds) else np.nan
    out["has_positive_pivot"] = out.index.isin(pivots)

    gap = _zscore(out["vendor_score"]) - _zscore(out["vader_mean"])
    out["disagreement_z"] = gap.round(2)
    out["disagreement"] = gap.abs() > DISAGREE_Z
    out["disagreement_note"] = np.where(
        ~out["disagreement"], "",
        np.where(gap > 0, "Vendor score more positive than the transcript language",
                 "Transcript language more positive than the vendor score"))
    return out.reset_index()


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------
def with_dimensions(call_sent: pd.DataFrame, calls: pd.DataFrame,
                     call_topics: pd.DataFrame | None) -> pd.DataFrame:
    dims = calls[["call_id", "call_type", "month", "period", "products"]].copy()
    dims["period"] = dims["period"].astype(str)
    if call_topics is not None:
        dims = dims.merge(call_topics[["call_id", "themes"]], on="call_id", how="left")
    return call_sent.merge(dims, on="call_id")


def slice_sentiment(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Mean scores per value of one dimension. product / theme are multi-label (exploded)."""
    source = {"product": "products", "theme": "themes"}.get(dimension, dimension)
    if source not in df:
        return pd.DataFrame()
    d = df.assign(value=df[source])
    if dimension in ("product", "theme"):
        d = d.assign(value=d["value"].map(lambda v: list(v) if v is not None else [])).explode("value")
    d = d.dropna(subset=["value"])
    g = d.groupby("value").agg(
        n_calls=("call_id", "nunique"), vendor_mean=("vendor_score", "mean"),
        vader_mean=("vader_mean", "mean"), customer_mean=("customer_vader", "mean"),
        staff_mean=("staff_vader", "mean"), share_improved=("arc", lambda a: (a == "improved").mean()),
    ).reset_index()
    return g.assign(dimension=dimension).round(3)


def build_slices(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dimension", "value", "n_calls", "vendor_mean", "vader_mean",
            "customer_mean", "staff_mean", "share_improved"]
    parts = [slice_sentiment(df, d) for d in SLICE_DIMENSIONS]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)[cols]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(call_sent: pd.DataFrame) -> dict:
    """How far to trust each signal: vendor vs VADER agreement, and arc vs pivot alignment."""
    cs = call_sent.dropna(subset=["vendor_score", "vader_mean"])
    pivot = cs[cs["has_positive_pivot"]]
    no_pivot = cs[~cs["has_positive_pivot"]]
    both = cs.dropna(subset=["customer_vader", "staff_vader"])
    return {
        "n_calls": int(len(cs)),
        "pearson_vendor_vs_vader": round(float(cs["vendor_score"].corr(cs["vader_mean"])), 3),
        "spearman_vendor_vs_vader": round(float(cs["vendor_score"].corr(cs["vader_mean"], method="spearman")), 3),
        "pearson_vendor_vs_customer": round(float(both["vendor_score"].corr(both["customer_vader"])), 3),
        "n_disagreements": int(cs["disagreement"].sum()),
        "mean_customer_minus_staff": round(float(both["side_gap"].mean()), 3),
        "share_improved_with_pivot": round(float((pivot["arc"] == "improved").mean()), 3) if len(pivot) else None,
        "share_improved_without_pivot": round(float((no_pivot["arc"] == "improved").mean()), 3) if len(no_pivot) else None,
        "n_with_pivot": int(len(pivot)),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_sentiment(tables: dict[str, pd.DataFrame], call_topics: pd.DataFrame | None = None):
    scored = score_sentences(tables["sentences"])
    call_sent = score_calls(tables["calls"], scored, tables["key_moments"])
    slices = build_slices(with_dimensions(call_sent, tables["calls"], call_topics))
    return scored[["call_id", "idx", "vader", "third"]], call_sent, slices, validate(call_sent)


def save_sentiment(sentence_sent, call_sent, slices, metrics, out_dir=PROCESSED_DIR) -> None:
    sentence_sent.to_parquet(out_dir / "sentence_sentiment.parquet", index=False)
    call_sent.to_parquet(out_dir / "call_sentiment.parquet", index=False)
    slices.to_parquet(out_dir / "sentiment_slices.parquet", index=False)
    (out_dir / "sentiment_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved sentence_sentiment, call_sentiment, sentiment_slices and sentiment_validation to %s", out_dir)


def load_sentiment(in_dir=PROCESSED_DIR):
    """Read the saved sentiment outputs back (used by the app)."""
    return (pd.read_parquet(in_dir / "sentence_sentiment.parquet"),
            pd.read_parquet(in_dir / "call_sentiment.parquet"),
            pd.read_parquet(in_dir / "sentiment_slices.parquet"),
            json.loads((in_dir / "sentiment_validation.json").read_text(encoding="utf-8")))


def summarize(sentence_sent, call_sent, slices, metrics) -> None:
    print("Validation:", json.dumps(metrics, indent=2))
    print("\nArc:\n" + call_sent["arc"].value_counts().to_string())
    for dim in ("call_type", "period"):
        print(f"\nBy {dim}:")
        print(slices[slices["dimension"] == dim].drop(columns="dimension").to_string(index=False))
    print("\nDisagreements:")
    print(call_sent.loc[call_sent["disagreement"],
                        ["call_id", "vendor_score", "vader_mean", "disagreement_z"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tables = load_enriched()
    try:
        from src.topics import load_topics
        call_topics = load_topics()[0]
    except FileNotFoundError:
        log.warning("No topic outputs yet; the theme slice is skipped (run python -m src.topics)")
        call_topics = None
    results = build_sentiment(tables, call_topics)
    summarize(*results)
    save_sentiment(*results)

"""
pipeline.py — One-command runner: raw transcripts to every output the dashboard and deck use.

Stages, in order (each saves its outputs before the next one starts):
    load       data/raw/ -> calls, sentences, key_moments, action_items       (load.py)
               (excluded folders are moved out to data/exclude/ first; folders marked
                keep in data/exclusions.csv are moved back to data/raw/)
    screen     keyword rule + Claude's is_business answer; non-business calls are
               added to the exclusion list, moved to data/exclude/ and dropped   (screen.py)
    enrich     call type, account, products, period, teams                    (enrich.py)
    topics     clusters + theme classification                                (topics.py)
    sentiment  vendor vs transcript scores, customer vs staff, arcs, slices   (sentiment.py)
    incidents  incident register: reported / owned / resolved, outage hours   (incidents.py)
    insights   account health, incident radius, feature gaps, competitors, …  (insights.py)
    export     CSVs for the deck in outputs/tables/, plus outputs/run_summary.json

topics, incidents and insights run once per mode, into data/processed/<mode>/:
    rules    keyword rules only (always built, no API calls)
    hybrid   Claude + rules (built unless --rules-only)
so the dashboard can switch modes instantly. LLM answers are cached in
data/processed/llm_cache/ and every real API call's cost is logged to llm_usage.jsonl.

Usage:
    python -m src.pipeline                    # full run, both modes (uses the LLM cache)
    python -m src.pipeline --refresh          # ignore the LLM cache, call the API again
    python -m src.pipeline --rules-only       # rules mode only, no API calls
    python -m src.pipeline --from sentiment   # re-run from a stage, reading earlier outputs from disk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import pandas as pd

from src import enrich, incidents, insights, load, screen, sentiment, topics
from src.llm import LLM_MODEL
from src.load import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, mode_dir

STAGES = ["load", "screen", "enrich", "topics", "sentiment", "incidents", "insights", "export"]
TABLES_DIR = OUTPUT_DIR / "tables"
RUN_SUMMARY = OUTPUT_DIR / "run_summary.json"
PIPELINE_LOG = OUTPUT_DIR / "logs" / "pipeline.log"   # every run, from the CLI or the dashboard
NOISY_LOGGERS = ("httpx", "httpx2", "sentence_transformers", "huggingface_hub")

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------
@contextmanager
def capture_logs(*extra: logging.Handler):
    """
    INFO-level logging for the duration of a run: appended to PIPELINE_LOG (rotated at 1 MB,
    3 old files kept) and sent to any extra handlers, e.g. the dashboard's live view.
    """
    PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(PIPELINE_LOG, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handlers = [file_handler, *extra]
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    root = logging.getLogger()
    old_levels = {name: logging.getLogger(name).level for name in ("", *NOISY_LOGGERS)}
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        yield
    finally:
        for h in handlers:
            root.removeHandler(h)
        file_handler.close()
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)


def read_log_tail(n: int = 200) -> str:
    """The last n lines of PIPELINE_LOG ("" if there's no log yet)."""
    if not PIPELINE_LOG.exists():
        return ""
    return "".join(PIPELINE_LOG.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-n:])


# ---------------------------------------------------------------------------
# New-transcript detection (used by the dashboard's periodic scan and refresh button)
# ---------------------------------------------------------------------------
SETTLE_SECONDS = 30   # a folder must be unchanged this long before it's processed (copy finished)


def folder_signature(folder) -> float:
    """Latest modification time of anything in a meeting folder."""
    return max([folder.stat().st_mtime] + [p.stat().st_mtime for p in folder.iterdir()])


def last_run() -> dict:
    return json.loads(RUN_SUMMARY.read_text(encoding="utf-8")) if RUN_SUMMARY.exists() else {}


def scan_raw(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR) -> dict[str, list[str]]:
    """
    Meeting folders in data/raw/ that the processed tables don't include yet, split into:
        new         ready to process
        arriving    changed in the last SETTLE_SECONDS (probably still being copied)
        unreadable  the last run couldn't read them and they haven't changed since
    plus flagged: {folder: keyword} for new folders whose title or topics look non-business.
    Folders on the exclusion list are left out entirely.
    """
    out = {"new": [], "arriving": [], "unreadable": [], "flagged": {}}
    if not raw_dir.exists():
        return out
    calls_path = processed_dir / "calls.parquet"
    done = set(pd.read_parquet(calls_path, columns=["call_id"])["call_id"]) if calls_path.exists() else set()
    known_bad = last_run().get("unreadable", {})
    excluded = screen.excluded_ids()
    now = time.time()
    for p in sorted(raw_dir.iterdir()):
        if not p.is_dir() or p.name in done | excluded or not (p / "meeting-info.json").exists():
            continue
        sig = folder_signature(p)
        if now - sig < SETTLE_SECONDS:
            out["arriving"].append(p.name)
        elif known_bad.get(p.name) == sig:
            out["unreadable"].append(p.name)
        else:
            out["new"].append(p.name)
            hit = screen.flag_folder(p)
            if hit:
                out["flagged"][p.name] = hit
    return out


def pending_transcripts(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR) -> list[str]:
    """Meeting folders ready to process (see scan_raw)."""
    return scan_raw(raw_dir, processed_dir)["new"]


def unreadable_folders(calls: pd.DataFrame, raw_dir=RAW_DIR) -> dict[str, float]:
    """Folders with a meeting-info.json that the load stage couldn't read, with their signature."""
    loaded = set(calls["call_id"]) | screen.excluded_ids()
    return {p.name: folder_signature(p) for p in raw_dir.iterdir()
            if p.is_dir() and (p / "meeting-info.json").exists() and p.name not in loaded}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """List columns -> "a; b" strings so the CSVs open cleanly in Excel / Sheets."""
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: hasattr(v, "__len__") and not isinstance(v, str)).any():
            out[col] = out[col].map(lambda v: "; ".join(map(str, v)) if v is not None else "")
    return out


def export_tables(enriched: dict, call_topics: pd.DataFrame, call_sent: pd.DataFrame,
                  slices: pd.DataFrame, register: pd.DataFrame, ins: dict, out_dir=TABLES_DIR) -> dict[str, int]:
    """Write the deck-facing CSVs. Returns {file name: rows}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = (enriched["calls"][["call_id", "date", "title", "call_type", "account", "teams", "period",
                                "products", "sentiment_label", "sentiment_score", "duration_min"]]
             .merge(call_topics[["call_id", "primary_theme", "themes", "competitors"]], on="call_id", how="left")
             .merge(call_sent[["call_id", "vader_mean", "customer_vader", "staff_vader", "arc",
                               "disagreement"]], on="call_id", how="left")
             .rename(columns={"sentiment_score": "confidence_score", "vader_mean": "transcript_score",
                              "customer_vader": "customer_score", "staff_vader": "staff_score"}))
    theme_counts = (call_topics["themes"].map(list).explode().value_counts()
                    .rename_axis("theme").rename("n_calls").reset_index())
    c = enriched["calls"]
    exports = {
        "calls_summary": calls.sort_values("date"),
        "theme_counts": theme_counts,
        "sentiment_slices": slices,
        "incidents": register,
        "monthly_availability": incidents.monthly_availability(register, c["start_time"].min(), c["start_time"].max()),
        "staff_roster": enriched["staff_roster"],
        "account_health": ins["account_health"],
        "incident_timeline": ins["incident_timeline"],
        "feature_gaps": ins["feature_gaps"],
        "competitor_mentions": ins["competitor_mentions"],
        "rep_talk": ins["rep_talk"],
        "owner_workload": ins["owner_workload"],
    }
    rows = {}
    for name, df in exports.items():
        _flatten(df).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        rows[f"{name}.csv"] = len(df)
    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(start: str = "load", use_llm: bool = True, refresh: bool = False) -> dict:
    """Run the stages from `start` onward. Returns the run summary (also saved to disk)."""
    todo = STAGES[STAGES.index(start):]
    modes = ["rules"] + (["hybrid"] if use_llm else [])
    primary = modes[-1]
    timings: dict[str, float] = {}
    state: dict = {"topics": {}, "incidents": {}, "insights": {}}

    def stage(name: str, fn) -> None:
        if name not in todo:
            return
        log.info("── %s", name)
        t = time.perf_counter()
        try:
            fn()
        except Exception:
            log.exception("Stage '%s' failed; earlier stages' outputs are saved. "
                          "Fix and re-run with --from %s", name, name)
            raise
        timings[name] = round(time.perf_counter() - t, 1)
        log.info("   %s done in %.1fs", name, timings[name])

    def enriched() -> dict:
        if "enriched" not in state:
            state["enriched"] = enrich.load_enriched()
        return state["enriched"]

    def call_topics(mode: str) -> pd.DataFrame:
        if mode not in state["topics"]:
            state["topics"][mode] = topics.load_topics(mode_dir(mode))
        return state["topics"][mode][0]

    def call_sent() -> pd.DataFrame:
        if "sentiment" not in state:
            state["sentiment"] = sentiment.load_sentiment()
        return state["sentiment"][1]

    def do_load():
        screen.restore_kept()
        screen.sweep_excluded()
        tables = load.load_all()
        state["unreadable"] = unreadable_folders(tables["calls"])
        excluded = screen.excluded_ids() & set(tables["calls"]["call_id"])
        state["raw"] = screen.drop_calls(tables, excluded)
        load.save_tables(state["raw"])
        if excluded:
            log.info("Skipped %d call(s) on the exclusion list", len(excluded))
        if state["unreadable"]:
            log.warning("Skipped %d unreadable folder(s): %s", len(state["unreadable"]),
                        ", ".join(state["unreadable"]))

    def do_screen():
        state["raw"], result = screen.apply_screen(state.get("raw") or load.load_tables(),
                                                   use_llm=use_llm, refresh=refresh)
        load.save_tables(state["raw"])
        state["screened"] = {"n": int(len(result)), "excluded": int((~result["is_business"]).sum()),
                             "sources": result["source"].value_counts().to_dict()}

    def do_enrich():
        state["enriched"] = enrich.enrich_all(state.get("raw") or load.load_tables())
        enrich.save_enriched(state["enriched"])

    def do_topics():
        for mode in modes:
            state["topics"][mode] = topics.build_topics(enriched(), use_llm=mode == "hybrid", refresh=refresh)
            topics.save_topics(*state["topics"][mode], out_dir=mode_dir(mode))

    def do_sentiment():
        state["sentiment"] = sentiment.build_sentiment(enriched(), call_topics(primary))
        sentiment.save_sentiment(*state["sentiment"])

    def do_incidents():
        for mode in modes:
            state["incidents"][mode] = incidents.build_register(
                enriched(), call_topics(mode), use_llm=mode == "hybrid", refresh=refresh)
            incidents.save_register(state["incidents"][mode], mode_dir(mode))

    def do_insights():
        for mode in modes:
            state["insights"][mode] = insights.build_insights(
                enriched(), call_topics(mode), call_sent(), use_llm=mode == "hybrid", refresh=refresh)
            insights.save_insights(state["insights"][mode], mode_dir(mode))

    def do_export():
        d = mode_dir(primary)
        t = state["topics"].get(primary) or topics.load_topics(d)
        s = state.get("sentiment") or sentiment.load_sentiment()
        reg = state["incidents"].get(primary)
        reg = reg if reg is not None else incidents.load_register(d)
        ins = state["insights"].get(primary) or insights.load_insights(d)
        state["exported"] = export_tables(enriched(), t[0], s[1], s[2], reg, ins)
        state["metrics"] = {"topics": t[2], "sentiment": s[3], "insights": ins["summary"],
                            "incidents": {"n": int(len(reg)), "open": int((reg["status"] == "Open").sum())}}

    for name, fn in [("load", do_load), ("screen", do_screen), ("enrich", do_enrich), ("topics", do_topics),
                     ("sentiment", do_sentiment), ("incidents", do_incidents),
                     ("insights", do_insights), ("export", do_export)]:
        stage(name, fn)

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages_run": todo,
        "seconds": timings,
        "modes_built": modes,
        "options": {"use_llm": use_llm, "refresh": refresh, "llm_model": LLM_MODEL if use_llm else None},
        "rows": {n: len(df) for n, df in enriched().items()},
        "exports": state.get("exported", {}),
        "metrics": state.get("metrics", {}),
        "screen": state.get("screened", {}),
        "excluded_total": len(screen.excluded_ids()),
        # folders the load stage couldn't read, so the dashboard's scan doesn't retry them forever
        "unreadable": state.get("unreadable", last_run().get("unreadable", {})),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def report(summary: dict) -> None:
    m = summary.get("metrics", {})
    t, s, i, inc = m.get("topics", {}), m.get("sentiment", {}), m.get("insights", {}), m.get("incidents", {})
    print("\nPipeline complete")
    print(f"  Stages:    {', '.join(f'{k} {v}s' for k, v in summary['seconds'].items())}")
    print(f"  Modes:     {', '.join(summary['modes_built'])}")
    print(f"  Rows:      {summary['rows']}")
    if summary.get("screen"):
        print(f"  Screen:    {summary['screen']['excluded']} non-business call(s) excluded this run; "
              f"{summary['excluded_total']} on the exclusion list in total")
    if t:
        print(f"  Topics:    {t['n_llm']}/{t['n_calls']} by LLM, {t['primary_in_rule_themes']:.0%} agree with rules"
              if t.get("primary_in_rule_themes") is not None else f"  Topics:    {t['n_rules']} by keyword rules")
    if s:
        print(f"  Sentiment: vendor vs transcript r = {s['pearson_vendor_vs_vader']}, {s['n_disagreements']} disagreements")
    if inc:
        print(f"  Incidents: {inc['n']} in register, {inc['open']} open")
    if i:
        print(f"  Accounts:  {i['risk_tiers']}; top risk {', '.join(i['top_risk_accounts'][:3])}")
    print(f"  Outputs:   {PROCESSED_DIR}  ·  {TABLES_DIR}  ·  {RUN_SUMMARY}")
    print("  Next:      streamlit run app/app.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the transcript intelligence pipeline end to end")
    parser.add_argument("--from", dest="start", choices=STAGES, default="load",
                        help="start at this stage, reading earlier outputs from disk")
    parser.add_argument("--rules-only", action="store_true", help="rules mode only, no LLM calls")
    parser.add_argument("--refresh", action="store_true", help="ignore the LLM cache and call the API again")
    args = parser.parse_args()
    if args.refresh and args.rules_only:
        parser.error("--refresh and --rules-only can't be combined")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        with capture_logs():
            summary = run(args.start, use_llm=not args.rules_only, refresh=args.refresh)
        report(summary)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

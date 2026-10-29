"""
app.py — Streamlit dashboard over the precomputed tables.

Reads only the files the pipeline writes (see docs/architecture.md, "Precompute, then
present"). Mode-dependent outputs (themes, incidents, feature-gap names) are read from
data/processed/<mode>/ for the mode picked in Config; everything that depends on the
filters (account health, incident timeline, SLA, competitors, talk time) is recomputed
here from the filtered calls, so every number follows the selected time period.

The Data section of the sidebar re-runs the pipeline when new transcripts are dropped
into data/raw/.

Usage:
    streamlit run app/app.py
"""

from __future__ import annotations

import html
import json
import logging
import os
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# `streamlit run app/app.py` puts app/ on the path, not the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.enrich import COMPLY_LAUNCH, OUTAGE_END, OUTAGE_START, PERIODS, PRODUCT_PATTERNS, load_enriched  # noqa: E402
from src.incidents import SLA_TARGET, load_register, monthly_availability, sla_credit_calls  # noqa: E402
from src.insights import (  # noqa: E402
    RISK_WEIGHTS, account_health, competitor_mentions, incident_timeline, load_insights,
    owner_workload, recurring_commitments, talk_time,
)
from src.llm import (  # noqa: E402
    BUDGET_SOURCE, BUDGET_WARN_SHARE, LLM_MODEL, MONTHLY_BUDGET_USD, load_usage, uncosted_cache_entries)
from src.load import MODES, PROCESSED_DIR, RAW_DIR, mode_dir  # noqa: E402
from src.pipeline import PIPELINE_LOG, capture_logs, last_run, read_log_tail, scan_raw  # noqa: E402
from src.pipeline import run as run_pipeline  # noqa: E402
from src.screen import EXCLUSIONS, exclude_call, excluded_ids  # noqa: E402
from src.sentiment import load_sentiment, slice_sentiment  # noqa: E402
from src.topics import THEME_NAMES, load_topics  # noqa: E402

# ---------------------------------------------------------------------------
# Colour: fixed per entity, so a call type keeps its colour under any filter
# ---------------------------------------------------------------------------
CALL_TYPES = ["external", "support", "internal"]
CALL_TYPE_COLORS = {"external": "#2a78d6", "support": "#eb6834", "internal": "#8a8f98"}
SINGLE_SERIES = "#2a78d6"
RISK = "#eb6834"          # risk-type key moments
NEUTRAL_BAR = "#64748b"
OUTAGE_FILL = "rgba(235,104,52,0.10)"
SIDE_COLORS = {"Customer": "#1baf7a", "AegisCloud staff": "#6250d6"}
# Status colours are reserved for state and always shown with a label
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
TIER_COLORS = {"High": CRITICAL, "Medium": WARNING, "Low": GOOD}
TIERS = list(TIER_COLORS)
DIVERGING = [[0, "#eb6834"], [0.5, "#e8e6e1"], [1, "#2a78d6"]]   # confidence 1..5, neutral at 3

SENTIMENT_LABELS = ["very-positive", "positive", "mixed-positive",
                    "mixed-negative", "negative", "very-negative"]
KEY_MOMENT_TYPES = ["concern", "positive_pivot", "churn_signal", "technical_issue",
                    "feature_gap", "action_item", "praise", "pricing_offer"]
RISK_MOMENTS = {"concern", "churn_signal", "technical_issue", "feature_gap"}
OUTAGE_THEME = "Detect outage & fallout"
RENEWAL_THEME = "Renewal & contracts"
LAUNCH_THEME = "Comply v2 launch"
SCAN_INTERVALS = {"Every 30 seconds": 30, "Every minute": 60, "Every 5 minutes": 300, "Off": None}
# Dates are shown as mm/dd/yyyy everywhere (text, charts, tables, date picker)
DATE_FMT, DATETIME_FMT = "%m/%d/%Y", "%m/%d/%Y %H:%M"
INTERNAL_ACCOUNT = "— (internal meeting)"
PLOTLY_DATE = "%m/%d/%Y"
INCIDENT_CATEGORIES = ["Outage", "Degradation", "Defect"]
ALL_CATEGORIES = INCIDENT_CATEGORIES + ["Billing", "Request"]
SLICE_LABELS = {"call_type": "Call type", "month": "Month", "period": "Period",
                "product": "Product", "theme": "Theme"}
CALL_SENTIMENT_COLS = ["call_id", "vader_mean", "customer_vader", "staff_vader", "side_gap",
                       "arc_first", "arc_last", "arc_delta", "arc", "has_positive_pivot",
                       "disagreement_z", "disagreement", "disagreement_note"]

st.set_page_config(page_title="Transcript Intelligence", layout="wide")

CSS = """
<style>
.ti-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:16px 18px; height:100%;}
.ti-kpi-label {font-size:.85rem; opacity:.75;}
.ti-kpi-value {font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:1.75rem; font-weight:600; margin:2px 0;}
.ti-kpi-sub {font-size:.8rem; opacity:.65;}
.ti-banner {background:#1f2229; color:#f0efec; border-radius:12px; padding:18px 22px; display:flex; gap:28px; margin:8px 0 4px;}
.ti-banner-tag {color:#fab219; font-weight:700; font-size:.8rem; letter-spacing:.06em; white-space:nowrap; padding-top:3px;}
.ti-banner-text {font-size:1rem; line-height:1.55;}
.ti-banner-text b {color:#fab219;}
.ti-row {display:grid; grid-template-columns:minmax(120px,38%) 1fr 36px; align-items:center; gap:12px; margin:7px 0; font-size:.9rem;}
.ti-track {background:rgba(128,128,128,.18); border-radius:6px; height:8px; overflow:hidden;}
.ti-fill {height:8px; border-radius:6px;}
.ti-num {font-family:ui-monospace,SFMono-Regular,Menlo,monospace; text-align:right;}
.ti-mix {display:flex; gap:2px; height:12px; border-radius:6px; overflow:hidden; margin:10px 0 14px;}
.ti-mix-stats {display:grid; grid-template-columns:repeat(3,1fr); gap:8px;}
.ti-mix-n {font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:1.5rem; font-weight:600;}
.ti-note {font-size:.8rem; opacity:.65; margin-top:8px;}
.ti-list {margin:6px 0 0; padding-left:0; list-style:none;}
.ti-list li {margin:8px 0; padding-left:26px; position:relative; line-height:1.45; font-size:.92rem;}
.ti-list li .ti-ico {position:absolute; left:0; top:0; font-weight:700;}
.ti-badge {display:inline-block; border-radius:999px; padding:1px 9px; font-size:.75rem; font-weight:600; color:#fff;}
.ti-budget {height:10px; border-radius:6px; background:rgba(128,128,128,.18); overflow:hidden; margin:8px 0 4px;}
.ti-budget > div {height:10px; border-radius:6px;}
</style>
"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def available_modes() -> list[str]:
    return [m for m in MODES if (mode_dir(m) / "call_topics.parquet").exists()
            and (mode_dir(m) / "incidents.parquet").exists()]


@st.cache_data
def load_base():
    tables = load_enriched()
    _, call_sent, _, sent_metrics = load_sentiment()
    return tables, call_sent, sent_metrics, last_run().get("finished_at")


@st.cache_data
def load_mode(mode: str):
    d = mode_dir(mode)
    call_topics, clusters, topic_metrics = load_topics(d)
    return call_topics, clusters, topic_metrics, load_register(d), load_insights(d)


def merged_calls(tables, call_sent, call_topics) -> pd.DataFrame:
    return (tables["calls"]
            .merge(call_topics.drop(columns=["rule_themes"]), on="call_id", how="left")
            .merge(call_sent[CALL_SENTIMENT_COLS], on="call_id", how="left"))


def _as_list(v) -> list:
    return list(v) if v is not None and not isinstance(v, float) else []


# ---------------------------------------------------------------------------
# Header: config + data refresh
# ---------------------------------------------------------------------------
def config_popover(modes: list[str]) -> tuple[str, float]:
    st.session_state.setdefault("mode", "hybrid" if "hybrid" in modes else (modes[0] if modes else "rules"))
    st.session_state.setdefault("sla_target", SLA_TARGET)
    st.session_state.setdefault("scan_every", "Every minute")
    with st.popover("⚙ Config", width="stretch"):
        st.markdown("**Analysis mode**")
        st.radio("Analysis mode", modes, format_func=MODES.get, key="mode", label_visibility="collapsed",
                 help="Hybrid uses Claude for themes, incident details and feature-ask names, with "
                      "keyword rules as the fallback. Rules only uses the keyword rules for everything.")
        missing = [MODES[m] for m in MODES if m not in modes]
        if missing:
            st.caption(f"Not built yet: {', '.join(missing)}. Use Data → Refresh to build it.")
        st.markdown("**SLA target**")
        st.radio("SLA target", [0.995, 0.999], key="sla_target", horizontal=True,
                 format_func=lambda v: f"{v:.1%} uptime", label_visibility="collapsed",
                 help="99.5% is the standard Detect contract; some accounts have 99.9%.")
        st.markdown("**New transcripts**")
        st.selectbox("Check data/raw for new transcripts", list(SCAN_INTERVALS), key="scan_every",
                     help="The check only notifies you; nothing is loaded until you press Refresh dashboard data.")
    return st.session_state["mode"], st.session_state["sla_target"]


def has_llm_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


@st.cache_resource
def pipeline_lock() -> threading.Lock:
    """Shared by every browser session, so only one pipeline run happens at a time."""
    return threading.Lock()


class LiveLogHandler(logging.Handler):
    """
    Shows the latest pipeline log lines in the refresh status box. Lines logged from worker
    threads (no Streamlit context) are kept and appear with the next line from the script thread.
    """
    def __init__(self, box, n_lines: int = 25):
        super().__init__()
        self.box, self.n_lines, self.lines = box, n_lines, []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))
        if get_script_run_ctx(suppress_warning=True) is not None:
            self.box.code("\n".join(self.lines[-self.n_lines:]), language=None)


def run_refresh(use_llm: bool, reason: str) -> bool | None:
    """Run the pipeline. True = done, False = failed, None = another refresh is already running."""
    lock = pipeline_lock()
    if not lock.acquire(blocking=False):
        st.info("A refresh is already running in another tab; use Load latest data when it finishes.")
        return None
    try:
        with st.status(f"{reason} — re-running the pipeline…", expanded=True) as status:
            st.caption(f"Full log: `{PIPELINE_LOG}`")
            try:
                with capture_logs(LiveLogHandler(st.empty())):
                    summary = run_pipeline("load", use_llm=use_llm)
            except Exception as e:  # surface any stage failure in the UI
                status.update(label=f"Refresh failed: {e} (details in the pipeline log)", state="error")
                return False
            status.update(label=f"Done: {summary['rows']['calls']} calls processed", state="complete")
    finally:
        lock.release()
    st.cache_data.clear()
    return True


def _title(folder: str) -> str:
    try:
        return json.loads((RAW_DIR / folder / "meeting-info.json").read_text(encoding="utf-8")).get("title", folder)
    except (OSError, json.JSONDecodeError):
        return folder


def _ignore(folder: str, title: str, hit: str | None) -> None:
    """Ignore button callback: runs before the rerun, so the list redraws without the folder."""
    exclude_call(folder, "user", "Ignored in the dashboard" + (f' (keyword "{hit}")' if hit else ""))
    st.session_state["ignored_toast"] = f"Ignored: {title}"   # shown by transcript_watch on the redraw


def new_transcript_list(new: list[str], flagged: dict[str, str], limit: int = 10) -> None:
    """One row per new transcript: title, a non-business flag, and an Ignore button."""
    for f in new[:limit]:
        title, hit = _title(f), flagged.get(f)
        text, button = st.columns([4, 2], vertical_alignment="center")
        text.markdown(f"{html.escape(title)}" + (f"  \n:orange[⚠ possibly not business (\"{hit}\")]" if hit else ""))
        button.button("Ignore", key=f"ignore_{f}", type="primary" if hit else "secondary",
                      on_click=_ignore, args=(f, title, hit),
                      help="Exclude this transcript: it's added to the exclusion list, moved from "
                           "data/raw/ to data/exclude/ and never processed.")
    if len(new) > limit:
        st.caption(f"… and {len(new) - limit} more.")


def transcript_watch() -> None:
    """
    Sidebar fragment re-run on a timer. It only scans data/raw and notifies; the pipeline
    runs, and the page reloads, only when the viewer presses a button.
    """
    scan, run = scan_raw(), last_run()

    if run:
        finished = pd.Timestamp(run["finished_at"]).tz_convert("UTC")
        st.caption(f"Last pipeline run: {finished:{DATETIME_FMT}} UTC · modes: {', '.join(run.get('modes_built', []))}")
    every = SCAN_INTERVALS[st.session_state["scan_every"]]
    st.caption(f"Last check {datetime.now():%H:%M:%S}" + (f" · next in {every}s" if every else " · automatic check off"))

    # Newer data was produced elsewhere (another tab or the command line): offer it, don't force it.
    if "data_version" in st.session_state and run.get("finished_at") != st.session_state["data_version"]:
        st.info("Newer processed data is available.")
        if st.button("Load latest data", key="load_latest", width="stretch"):
            st.cache_data.clear()
            st.rerun(scope="app")

    if st.session_state.get("ignored_toast"):
        st.toast(st.session_state.pop("ignored_toast"), icon="🚫")
    new, flagged = scan["new"], scan["flagged"]
    if new:
        st.warning(f"{len(new)} new transcript{'s' * (len(new) > 1)} available"
                   + (f", {len(flagged)} possibly not business" if flagged else "")
                   + ". Press **Refresh dashboard data** to load them, or **Ignore** any that shouldn't be.")
        new_transcript_list(new, flagged)
        if set(new) != st.session_state.get("notified"):          # one pop-up per new batch
            st.session_state["notified"] = set(new)
            st.toast(f"{len(new)} new transcript{'s' * (len(new) > 1)} available"
                     + (f" ({len(flagged)} possibly not business)" if flagged else "")
                     + " — use Refresh dashboard data in the sidebar to load them.", icon="🆕")
    elif not scan["arriving"]:
        st.caption("No new transcripts waiting.")
    n_excluded = len(excluded_ids())
    if n_excluded:
        st.caption(f"{n_excluded} non-business transcript{'s' * (n_excluded > 1)} excluded "
                   f"(list: `{EXCLUSIONS}`).")
    if scan["arriving"]:
        st.info(f"{len(scan['arriving'])} transcript folder(s) still arriving; they'll be picked up once the copy "
                "has finished.")
    if scan["unreadable"]:
        st.error(f"{len(scan['unreadable'])} folder(s) couldn't be read (missing or invalid JSON): "
                 + ", ".join(scan["unreadable"]) + ". Fix the files and they'll be retried.")

    if st.button("⟳ Refresh dashboard data", key="refresh_now", width="stretch",
                 type="primary" if new else "secondary"):
        if run_refresh(bool(st.session_state.get("use_llm")) and has_llm_key(),
                       f"Loading {len(new)} new transcript{'s' * (len(new) != 1)}" if new else "Refreshing"):
            st.session_state["notified"] = None
            st.rerun(scope="app")

    tail = read_log_tail()
    with st.expander("Pipeline log"):
        if tail:
            st.caption(f"Last 200 lines of `{PIPELINE_LOG}`")
            st.code(tail, language=None)
            st.download_button("Download log", PIPELINE_LOG.read_bytes(), file_name="pipeline.log",
                               mime="text/plain", key="dl_log")
        else:
            st.caption("No pipeline runs logged yet.")


def data_section() -> None:
    """Sidebar: transcript folder, Claude switch, and the periodic scan / refresh."""
    st.markdown("**DATA**")
    st.caption(f"Transcript folder: `{RAW_DIR}`")
    has_key = has_llm_key()
    st.session_state.setdefault("use_llm", has_key)
    st.checkbox("Use Claude for Hybrid mode", key="use_llm", disabled=not has_key,
                help="Cached answers are reused, so only new transcripts cost anything."
                if has_key else "No Anthropic credentials found in .env; only Rules mode will be rebuilt.")
    st.fragment(run_every=SCAN_INTERVALS[st.session_state["scan_every"]])(transcript_watch)()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def quick_ranges(first: date, last: date) -> dict[str, tuple[date, date]]:
    def clip(a, b):
        return max(a, first), min(b, last)
    return {
        "All time": (first, last),
        "Pre-outage": clip(first, OUTAGE_START - timedelta(days=1)),
        f"Outage ({OUTAGE_START:%m/%d/%Y}–{OUTAGE_END:%m/%d/%Y})": clip(OUTAGE_START, OUTAGE_END),
        "Recovery": clip(OUTAGE_END + timedelta(days=1), COMPLY_LAUNCH - timedelta(days=1)),
        "Post-Comply-launch": clip(COMPLY_LAUNCH, last),
        "Last 30 days": clip(last - timedelta(days=29), last),
    }


def sidebar_filters(calls: pd.DataFrame) -> tuple[pd.DataFrame, date, date]:
    first, last = calls["start_time"].min().date(), calls["start_time"].max().date()
    ranges = quick_ranges(first, last)
    defaults = {"f_external": True, "f_support": True, "f_internal": True, "f_product": "All products",
                "f_account": "", "f_quick": "All time", "f_dates": (first, last)}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    # When a refresh brings in calls outside the old range, move the period with the data:
    # quick ranges are recomputed, a custom range is kept but clipped to the new bounds.
    if st.session_state.get("f_bounds") != (first, last):
        st.session_state["f_bounds"] = (first, last)
        quick = st.session_state["f_quick"]
        if quick in ranges:
            st.session_state["f_dates"] = ranges[quick]
        else:
            dates = tuple(st.session_state["f_dates"]) or (first, last)
            st.session_state["f_dates"] = (min(max(dates[0], first), last), max(min(dates[-1], last), first))

    def on_quick():
        if st.session_state["f_quick"] in ranges:
            st.session_state["f_dates"] = ranges[st.session_state["f_quick"]]

    def on_dates():
        st.session_state["f_quick"] = next((k for k, v in ranges.items()
                                            if v == tuple(st.session_state["f_dates"])), "Custom")

    def reset():
        for k, v in defaults.items():
            st.session_state[k] = v

    counts = calls["call_type"].value_counts()
    with st.sidebar:
        st.markdown("**TIME PERIOD**")
        st.selectbox("Quick range", [*ranges, "Custom"], key="f_quick", on_change=on_quick)
        st.date_input("From – to", min_value=first, max_value=last, key="f_dates", on_change=on_dates,
                      format="MM/DD/YYYY")
        st.markdown("**FILTERS**")
        st.markdown("Call type")
        types = [t for t in CALL_TYPES if st.checkbox(f"{t.capitalize()} · {counts.get(t, 0)}", key=f"f_{t}")]
        product = st.selectbox("Product", ["All products", *PRODUCT_PATTERNS], key="f_product")
        account = st.text_input("Account", placeholder="Search accounts...", key="f_account")
        st.button("Reset filters", on_click=reset, width="stretch")
        st.divider()
        data_section()

    dates = tuple(st.session_state["f_dates"])
    start, end = (dates[0], dates[-1]) if dates else (first, last)
    return filter_calls(calls, start, end, types, product, account), start, end


def filter_calls(calls: pd.DataFrame, start: date, end: date, types=CALL_TYPES,
                 product: str = "All products", account: str = "") -> pd.DataFrame:
    """The sidebar filters: period (inclusive), call types, product, account name contains."""
    mask = calls["start_time"].dt.date.between(start, end) & calls["call_type"].isin(types)
    if product != "All products":
        mask &= calls["products"].map(lambda p: product in _as_list(p))
    if account.strip():
        mask &= calls["account"].fillna("").str.contains(account.strip(), case=False, regex=False)
    return calls[mask]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def kpi(label: str, value, sub: str, color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="ti-card"><div class="ti-kpi-label">{html.escape(label)}</div>'
            f'<div class="ti-kpi-value"{style}>{value}</div><div class="ti-kpi-sub">{html.escape(sub)}</div></div>')


def show_table(df: pd.DataFrame, **kwargs):
    """
    st.dataframe with every date column shown as MM/DD/YYYY (sorting still works). Internal
    meetings have no customer account; label them rather than let Streamlit print "None".
    """
    if "account" in df.columns and df["account"].isna().any():
        df = df.assign(account=df["account"].astype("object").fillna(INTERNAL_ACCOUNT))
    config = {}
    for col in df.columns:
        values = df[col].dropna()
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            config[col] = st.column_config.DatetimeColumn(col, format="MM/DD/YYYY HH:mm")
        elif len(values) and isinstance(values.iloc[0], date):
            config[col] = st.column_config.DateColumn(col, format="MM/DD/YYYY")
    config.update(kwargs.pop("column_config", None) or {})
    return st.dataframe(df, column_config=config, **kwargs)


def kpi_row(cards: list[str]) -> None:
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def bar_list(items: list[tuple[str, int, str]]) -> str:
    """Rows of label · bar · value. Values stay in text ink; the bar carries the colour."""
    top = max((v for _, v, _ in items), default=0) or 1
    return "".join(
        f'<div class="ti-row"><span>{html.escape(label)}</span>'
        f'<div class="ti-track"><div class="ti-fill" style="width:{100 * v / top:.1f}%;background:{color}"></div></div>'
        f'<span class="ti-num">{v}</span></div>'
        for label, v, color in items)


def icon_list(items: list[str], icon: str, color: str) -> str:
    if not items:
        return '<div class="ti-note">Nothing stands out for this period.</div>'
    return '<ul class="ti-list">' + "".join(
        f'<li><span class="ti-ico" style="color:{color}">{icon}</span>{t}</li>' for t in items) + "</ul>"


def badge(text: str, color: str) -> str:
    return f'<span class="ti-badge" style="background:{color}">{html.escape(text)}</span>'


def money(v: float) -> str:
    return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"


def budget_source_note() -> str:
    """Where the budget settings came from: .env, or defaults because they aren't set there."""
    defaults = [k for k, src in BUDGET_SOURCE.items() if src != ".env"]
    if not defaults:
        return "Set in .env (LLM_MONTHLY_BUDGET_USD, LLM_BUDGET_WARN_SHARE)."
    return f"{' and '.join(defaults)} not set in .env (or not a valid number), so the default is used."


def fmt_ts(ts) -> str:
    return "—" if ts is None or pd.isna(ts) else f"{pd.Timestamp(ts):%m/%d/%Y %H:%M}"


def n_outage_calls(calls: pd.DataFrame) -> int:
    """Outage-themed calls from the outage start on (same rule as the incident timeline)."""
    tagged = calls["themes"].map(lambda t: OUTAGE_THEME in _as_list(t))
    return int((tagged & (calls["start_time"].dt.date >= OUTAGE_START)).sum())


def has_theme(calls: pd.DataFrame, theme: str) -> pd.Series:
    return calls["themes"].map(lambda t: theme in _as_list(t))


def theme_counts(calls: pd.DataFrame) -> pd.Series:
    return (calls["themes"].map(_as_list).explode().dropna().value_counts()
            .reindex(THEME_NAMES, fill_value=0).sort_values(ascending=False))


def bucket(calls: pd.DataFrame, start: date, end: date) -> pd.Series:
    """Weekly buckets for short periods, monthly otherwise."""
    t = calls["start_time"].dt.tz_convert(None)
    return (t.dt.to_period("W") if (end - start).days <= 45 else t.dt.to_period("M")).dt.start_time


def by_type_over_time(calls: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return (calls.assign(bucket=bucket(calls, start, end))
            .groupby(["bucket", "call_type"])["sentiment_score"].agg(["mean", "size"]).reset_index())


def staff_people(tables) -> list[str]:
    return sorted(tables["staff_roster"]["name"].tolist())


def handle(name: str | None) -> str:
    return "@" + (name or "owner").lower().replace("'", "").replace(" ", ".")


# ---------------------------------------------------------------------------
# Narrative: headline, accomplishments, improvements
# ---------------------------------------------------------------------------
def headline_insight(calls: pd.DataFrame, start: date, end: date, sla: pd.DataFrame) -> str:
    if calls.empty:
        return ""
    in_outage = calls["start_time"].dt.date.between(OUTAGE_START, OUTAGE_END)
    before = calls[calls["start_time"].dt.date < OUTAGE_START]
    if in_outage.any():
        during = calls[in_outage]["sentiment_score"].mean()
        text = (f"The Detect outage ({OUTAGE_START:%m/%d/%Y}–{OUTAGE_END:%m/%d/%Y}) pulled confidence to "
                f"<b>{during:.2f}</b>" + (f" from {before['sentiment_score'].mean():.2f} before it" if len(before) else "")
                + f", and came up in {n_outage_calls(calls)} of {len(calls)} calls in this period.")
        breach = sla[sla["sla_breached"]]
        if len(breach):
            r = breach.iloc[0]
            text += (f" It breached the {r['sla_target']:.1%} SLA in {r['label']} "
                     f"({r['outage_hours']:.1f} h down vs {r['allowed_hours']:.1f} h allowed).")
        after = calls[calls["start_time"].dt.date > OUTAGE_END]
        if len(after):
            by_type = after.groupby("call_type")["sentiment_score"].mean()
            worst = by_type.idxmin()
            text += (f" Since then confidence is back to {after['sentiment_score'].mean():.2f}; "
                     f"<b>{worst} calls are still the lowest at {by_type[worst]:.2f}.</b>")
        return text
    by_type = calls.groupby("call_type")["sentiment_score"].mean()
    return (f"Across {len(calls)} calls from {start:%m/%d/%Y} to {end:%m/%d/%Y}, average confidence was "
            f"<b>{calls['sentiment_score'].mean():.2f}</b>. {by_type.idxmax().capitalize()} calls led at "
            f"{by_type.max():.2f}; <b>{by_type.idxmin()} calls trailed at {by_type.min():.2f}.</b>")


def accomplishments(calls, km, register, health) -> list[str]:
    out = []
    renew = calls[has_theme(calls, RENEWAL_THEME) & (calls["sentiment_score"] >= 4)]
    if len(renew):
        names = ", ".join(sorted(renew["account"].dropna().unique())[:4])
        out.append(f"<b>{len(renew)} renewal / contract conversation{'s' * (len(renew) > 1)} ended positively</b>"
                   + (f" ({names})" if names else "") + ".")
    launch = calls[has_theme(calls, LAUNCH_THEME) & (calls["start_time"].dt.date >= COMPLY_LAUNCH)]
    if len(launch) >= 2 and launch["sentiment_score"].mean() >= 3.5:
        out.append(f"<b>Comply v2 launch landed well</b>: average confidence {launch['sentiment_score'].mean():.2f} "
                   f"across {len(launch)} calls since GA on {COMPLY_LAUNCH:%m/%d/%Y}.")
    resolved = register[register["status"] == "Resolved"]
    if len(resolved):
        major = resolved[resolved["kind"] == "Major incident"]
        out.append(f"<b>{len(resolved)} incident{'s' * (len(resolved) > 1)} resolved</b>"
                   + (f", including the {major.iloc[0]['title']} (restored {fmt_ts(major.iloc[0]['resolved_at'])} UTC)"
                      if len(major) else "") + ".")
    improved = (calls["arc"] == "improved").mean() if len(calls) else 0
    if improved >= 0.4:
        out.append(f"<b>{improved:.0%} of calls ended better than they started</b>, so teams are turning "
                   "difficult conversations around.")
    up = health[health["score_trend"] >= 0.5]
    if len(up):
        out.append(f"<b>{len(up)} account{'s' * (len(up) > 1)} trending up</b>: {', '.join(up['account'].head(4))}.")
    praise = int((km["type"] == "praise").sum())
    if praise:
        out.append(f"<b>{praise} moments of customer praise</b> captured on the calls.")
    return out[:5]


def improvements(calls, register, health, sla, comp, asks) -> list[str]:
    out = []
    by_type = calls.groupby("call_type")["sentiment_score"].mean()
    if len(by_type) > 1 and by_type.min() < 3.3:
        out.append(f"<b>{by_type.idxmin().capitalize()} confidence trails at {by_type.min():.2f}</b> "
                   f"(vs {by_type.max():.2f} for {by_type.idxmax()} calls).")
    high = health[health["risk_tier"] == "High"]
    if len(high):
        out.append(f"<b>{len(high)} high-risk account{'s' * (len(high) > 1)}</b>: "
                   f"{', '.join(high['account'].head(4))} — churn signals, competitors or falling confidence.")
    open_p1 = register[(register["status"] == "Open") & (register["priority"] == "P1")
                       & register["category"].isin(INCIDENT_CATEGORIES)]
    if len(open_p1):
        out.append(f"<b>{len(open_p1)} P1 incident{'s' * (len(open_p1) > 1)} still open</b>, "
                   f"e.g. {open_p1.iloc[0]['title']}.")
    breach = sla[sla["sla_breached"]]
    if len(breach):
        out.append(f"<b>SLA breached in {', '.join(breach['label'])}</b> — "
                   f"{breach['outage_hours'].sum():.1f} h of platform downtime; SLA credits are being negotiated.")
    if len(comp):
        top = comp.groupby("competitor")["call_id"].nunique().sort_values(ascending=False)
        out.append(f"<b>Competitive pressure</b>: {top.index[0]} named on {top.iloc[0]} calls.")
    if len(asks):
        a = asks.iloc[0]
        out.append(f"<b>Top product ask: {a['ask']}</b> ({a['n_accounts']} accounts).")
    return out[:5]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _style(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def _outage_band(fig: go.Figure, start: date, end: date, label: bool = True, tz: bool = False) -> None:
    if end < OUTAGE_START or start > OUTAGE_END:
        return
    x0, x1 = pd.Timestamp(OUTAGE_START), pd.Timestamp(OUTAGE_END) + pd.Timedelta(days=1)
    if tz:
        x0, x1 = x0.tz_localize("UTC"), x1.tz_localize("UTC")
    fig.add_vrect(x0=x0, x1=x1, fillcolor=OUTAGE_FILL, line_width=0,
                  annotation_text="Detect outage" if label else None, annotation_position="top",
                  annotation_font_color=RISK)


def confidence_by_type_chart(calls: pd.DataFrame, start: date, end: date) -> go.Figure:
    m = by_type_over_time(calls, start, end)
    weekly = (end - start).days <= 45
    m["x"] = m["bucket"] + pd.Timedelta(days=3 if weekly else 14)
    fig = go.Figure()
    _outage_band(fig, start, end)
    for t in CALL_TYPES:
        d = m[m["call_type"] == t]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["mean"], name=t.capitalize(), mode="lines+markers",
            line=dict(color=CALL_TYPE_COLORS[t], width=2),
            marker=dict(size=9, color="white", line=dict(color=CALL_TYPE_COLORS[t], width=2)),
            customdata=d["size"],
            hovertemplate=("Week of %{x|%m/%d/%Y}" if weekly else "%{x|%b}") +
                          ": %{y:.2f} (%{customdata} calls)<extra>" + t.capitalize() + "</extra>",
        ))
        last = d.iloc[-1]
        fig.add_annotation(x=last["x"], y=last["mean"], text=f"{last['mean']:.2f}", showarrow=False,
                           xanchor="left", xshift=10, font=dict(color=CALL_TYPE_COLORS[t], family="monospace"))
    fig.update_xaxes(tickformat="%m/%d/%Y" if weekly else "%b", dtick=None if weekly else "M1")
    fig.update_yaxes(title=None)
    return _style(fig)


def sla_chart(sla: pd.DataFrame) -> go.Figure:
    """Platform downtime per month vs the downtime the SLA allows (same unit, one axis)."""
    fig = go.Figure(go.Bar(
        x=sla["label"], y=sla["outage_hours"], name="Downtime",
        marker_color=[CRITICAL if b else SINGLE_SERIES for b in sla["sla_breached"]],
        text=[f"{h:.1f} h" for h in sla["outage_hours"]], textposition="outside",
        customdata=sla[["pct_time_down", "availability", "allowed_hours"]],
        hovertemplate="%{x}: %{y:.1f} h down (%{customdata[0]:.2%} of the month)<br>"
                      "availability %{customdata[1]:.3%} · SLA allows %{customdata[2]:.1f} h<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=sla["label"], y=sla["allowed_hours"], mode="lines+markers", name="SLA allowance",
                             line=dict(color=NEUTRAL_BAR, dash="dash", width=2), marker=dict(size=8),
                             hovertemplate="%{x}: SLA allows %{y:.1f} h<extra></extra>"))
    fig.update_yaxes(title="Hours", rangemode="tozero",
                     range=[0, max(1.0, float(sla[["outage_hours", "allowed_hours"]].max().max()) * 1.3)])
    fig.update_layout(bargap=0.45)
    return _style(fig, height=280)


def account_heatmap(calls: pd.DataFrame, start: date, end: date) -> go.Figure:
    c = calls[calls["account"].notna()]
    grid = (c.assign(bucket=bucket(c, start, end))
            .pivot_table(index="account", columns="bucket", values="sentiment_score", aggfunc="mean"))
    grid = grid.loc[grid.mean(axis=1).sort_values().index]
    weekly = (end - start).days <= 45
    cols = [f"{b:%m/%d/%Y}" if weekly else f"{b:%b %Y}" for b in grid.columns]
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=cols, y=grid.index, colorscale=DIVERGING, zmin=1, zmax=5, zmid=3,
        xgap=2, ygap=2, text=grid.round(1).values, texttemplate="%{text}",
        colorbar=dict(title="Confidence", thickness=10, len=0.6),
        hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>",
    ))
    return _style(fig, height=max(320, 22 * len(grid) + 80))


def theme_period_heatmap(calls: pd.DataFrame) -> go.Figure:
    long = calls[["period", "themes"]].assign(themes=calls["themes"].map(_as_list)).explode("themes")
    grid = (pd.crosstab(long["themes"], long["period"].astype(str))
            .reindex(index=THEME_NAMES[::-1], columns=PERIODS, fill_value=0))
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=grid.columns, y=grid.index, colorscale=[[0, "#eef4fd"], [1, "#184f95"]],
        xgap=2, ygap=2, text=grid.values, texttemplate="%{text}", showscale=False,
        hovertemplate="%{y}<br>%{x}: %{z} calls<extra></extra>",
    ))
    return _style(fig, height=380)


def confidence_by_theme_chart(calls: pd.DataFrame) -> go.Figure:
    long = calls[["sentiment_score", "themes"]].assign(themes=calls["themes"].map(_as_list)).explode("themes")
    avg = long.groupby("themes")["sentiment_score"].agg(["mean", "size"]).sort_values("mean")
    fig = go.Figure(go.Bar(
        x=avg["mean"], y=avg.index, orientation="h", marker_color=SINGLE_SERIES,
        text=avg["mean"].round(2), textposition="outside", customdata=avg["size"],
        hovertemplate="%{y}: %{x:.2f} avg (%{customdata} calls)<extra></extra>",
    ))
    fig.update_xaxes(range=[1, 5], title="Avg confidence (1–5)", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return _style(fig, height=380)


def primary_theme_by_type_chart(calls: pd.DataFrame) -> go.Figure:
    counts = calls.groupby(["primary_theme", "call_type"]).size().rename("calls").reset_index()
    order = calls["primary_theme"].value_counts().index.tolist()
    fig = px.bar(counts, x="calls", y="primary_theme", color="call_type", orientation="h",
                 category_orders={"primary_theme": order, "call_type": CALL_TYPES},
                 color_discrete_map=CALL_TYPE_COLORS, labels={"primary_theme": "", "calls": "Calls"})
    fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.8)",
                      hovertemplate="%{y}<br>%{fullData.name}: %{x} calls<extra></extra>")
    return _style(fig, height=380)


def vendor_vs_transcript_chart(calls: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for t in CALL_TYPES:
        d = calls[calls["call_type"] == t]
        if d.empty:
            continue
        flagged = d["disagreement"].fillna(False).astype(bool)
        fig.add_trace(go.Scatter(
            x=d["vader_mean"], y=d["sentiment_score"], mode="markers", name=t.capitalize(),
            marker=dict(size=10, color=CALL_TYPE_COLORS[t], opacity=0.85,
                        line=dict(width=flagged.map({True: 2.5, False: 1}).tolist(),
                                  color=flagged.map({True: "#0b0b0b", False: "white"}).tolist())),
            customdata=d[["title"]],
            hovertemplate="%{customdata[0]}<br>Vendor %{y:.1f} · transcript %{x:.2f}<extra></extra>",
        ))
    fig.update_xaxes(title="Transcript language score (−1 negative … +1 positive)", showgrid=True,
                     gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(title="Vendor confidence (1–5)", range=[1, 5])
    return _style(fig, height=360)


def side_by_period_chart(calls: pd.DataFrame) -> go.Figure:
    d = calls[calls["call_type"] != "internal"]
    g = (d.groupby(d["period"].astype(str))[["customer_vader", "staff_vader"]].mean()
         .reindex(PERIODS).dropna(how="all"))
    fig = go.Figure()
    for col, name in (("customer_vader", "Customer"), ("staff_vader", "AegisCloud staff")):
        fig.add_trace(go.Bar(x=g.index, y=g[col], name=name, marker_color=SIDE_COLORS[name],
                             hovertemplate="%{x}<br>" + name + ": %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    top = float(g.max().max()) if len(g) else 0.0
    fig.update_yaxes(title="Transcript language score", range=[0, max(0.6, top + 0.05)])
    return _style(fig, height=360)


def slice_chart(calls: pd.DataFrame, dimension: str) -> tuple[go.Figure, pd.DataFrame]:
    s = slice_sentiment(calls.assign(vendor_score=calls["sentiment_score"]), dimension)
    if dimension == "period":
        s = s.set_index("value").reindex(PERIODS).dropna(subset=["n_calls"]).reset_index()
    elif dimension == "month":
        s = s.sort_values("value")
    else:
        s = s.sort_values("vendor_mean")
    fig = go.Figure(go.Bar(
        x=s["vendor_mean"], y=s["value"], orientation="h", marker_color=SINGLE_SERIES,
        text=s["vendor_mean"].round(2), textposition="outside", customdata=s["n_calls"],
        hovertemplate="%{y}: %{x:.2f} (%{customdata} calls)<extra></extra>",
    ))
    fig.update_xaxes(range=[1, 5], title="Avg confidence (1–5)", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(autorange="reversed" if dimension in ("period", "month") else True)
    return _style(fig, height=max(240, 46 * len(s) + 80)), s


def arc_chart(calls: pd.DataFrame) -> go.Figure:
    g = calls.groupby("call_type")[["arc_first", "arc_last"]].mean().reindex(CALL_TYPES).dropna()
    fig = go.Figure()
    for t, r in g.iterrows():
        fig.add_trace(go.Scatter(x=[r["arc_first"], r["arc_last"]], y=[t.capitalize()] * 2, mode="lines",
                                 line=dict(color="rgba(128,128,128,0.5)", width=2), showlegend=False, hoverinfo="skip"))
    for col, name, symbol in (("arc_first", "First third", "circle-open"), ("arc_last", "Last third", "circle")):
        fig.add_trace(go.Scatter(
            x=g[col], y=[t.capitalize() for t in g.index], mode="markers", name=name,
            marker=dict(size=12, symbol=symbol, color=[CALL_TYPE_COLORS[t] for t in g.index],
                        line=dict(width=2, color=[CALL_TYPE_COLORS[t] for t in g.index])),
            hovertemplate="%{y} · " + name + ": %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title="Transcript language score", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return _style(fig, height=260)


def risk_ranking_chart(health: pd.DataFrame, top: int = 15) -> go.Figure:
    h = health.head(top).iloc[::-1]
    fig = go.Figure()
    for tier in TIERS:
        d = h[h["risk_tier"] == tier]
        if d.empty:
            continue
        fig.add_trace(go.Bar(
            x=d["risk_score"], y=d["account"], orientation="h", name=f"{tier} risk",
            marker_color=TIER_COLORS[tier], customdata=d[["drivers", "avg_score"]],
            hovertemplate="%{y}: %{x}/100<br>avg confidence %{customdata[1]:.2f}<br>%{customdata[0]}<extra></extra>",
        ))
    fig.update_layout(barmode="overlay", bargap=0.3)
    fig.update_yaxes(categoryorder="array", categoryarray=h["account"].tolist())
    fig.update_xaxes(range=[0, 100], title="Risk score (0–100)", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return _style(fig, height=max(300, 26 * len(h) + 90))


def account_history_chart(calls: pd.DataFrame, start: date, end: date) -> go.Figure:
    d = calls.sort_values("start_time")
    fig = go.Figure(go.Scatter(
        x=d["start_time"], y=d["sentiment_score"], mode="lines+markers",
        line=dict(color="rgba(128,128,128,0.6)", width=2),
        marker=dict(size=11, color=[CALL_TYPE_COLORS[t] for t in d["call_type"]], line=dict(width=2, color="white")),
        customdata=d[["title", "call_type"]],
        hovertemplate="%{x|%m/%d/%Y}: %{y:.1f}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    ))
    _outage_band(fig, start, end, label=False, tz=True)
    fig.update_yaxes(range=[1, 5], title="Confidence (1–5)")
    return _style(fig, height=260)


def incident_chart(timeline: pd.DataFrame, start: date, end: date) -> go.Figure:
    fig = go.Figure()
    _outage_band(fig, start, end, tz=True)
    for t in CALL_TYPES:
        d = timeline[timeline["call_type"] == t]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["start_time"], y=[t.capitalize()] * len(d), mode="markers", name=t.capitalize(),
            marker=dict(size=8 + 4 * d["km_churn_signal"].clip(upper=3), color=CALL_TYPE_COLORS[t],
                        opacity=0.85, line=dict(width=1.5, color="white")),
            customdata=d[["title", "account", "sentiment_score", "hours_from_start"]].fillna(""),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]} · confidence %{customdata[2]}"
                          "<br>+%{customdata[3]} h from the war room<extra></extra>",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=[t.capitalize() for t in CALL_TYPES][::-1])
    return _style(fig, height=260)


def talk_bucket_chart(talk: pd.DataFrame) -> go.Figure:
    g = talk.groupby("talk_bucket", observed=True)["sentiment_score"].agg(["mean", "size"]).reset_index()
    fig = go.Figure(go.Bar(
        x=g["talk_bucket"].astype(str), y=g["mean"], marker_color=SINGLE_SERIES,
        text=g["mean"].round(2), textposition="outside", customdata=g["size"],
        hovertemplate="%{x}: %{y:.2f} (%{customdata} calls)<extra></extra>",
    ))
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(title="Customer share of talk time (quarter of calls)")
    fig.update_yaxes(range=[1, 5], title="Avg confidence (1–5)")
    return _style(fig, height=300)


def monthly_cost_chart(usage: pd.DataFrame, budget: float, warn: float) -> go.Figure:
    m = (usage.assign(month=usage["ts"].dt.tz_convert(None).dt.to_period("M").dt.start_time)
         .groupby("month")["cost_usd"].sum().reset_index())
    colors = [CRITICAL if c >= warn * budget else SINGLE_SERIES for c in m["cost_usd"]]
    fig = go.Figure(go.Bar(x=m["month"], y=m["cost_usd"], marker_color=colors,
                           text=[f"${c:,.2f}" for c in m["cost_usd"]], textposition="outside",
                           hovertemplate="%{x|%b %Y}: $%{y:,.4f}<extra></extra>"))
    fig.update_xaxes(tickformat="%b %Y", dtick="M1")
    fig.update_yaxes(title="USD", rangemode="tozero")
    return _style(fig, height=240)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def render_overview(ctx: dict) -> None:
    calls, start, end = ctx["calls"], ctx["start"], ctx["end"]
    if calls.empty:
        st.info("No calls match the current filters.")
        return
    banner = headline_insight(calls, start, end, ctx["sla"])
    if banner:
        st.markdown(f'<div class="ti-banner"><div class="ti-banner-tag">HEADLINE INSIGHT</div>'
                    f'<div class="ti-banner-text">{banner}</div></div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown(f"**Major accomplishments** · {start:%m/%d/%Y} – {end:%m/%d/%Y}"
                    + icon_list(accomplishments(calls, ctx["km"], ctx["register"], ctx["health"]), "✓", GOOD),
                    unsafe_allow_html=True)
    with right, st.container(border=True):
        st.markdown("**Scope for improvement**"
                    + icon_list(improvements(calls, ctx["register"], ctx["health"], ctx["sla"], ctx["comp"],
                                             ctx["asks"]), "!", WARNING), unsafe_allow_html=True)

    participants = calls["emails"].map(_as_list).explode().dropna().nunique()
    comp = calls["competitors"].map(_as_list)
    top_comp = comp.explode().dropna().value_counts()
    kpi_row([
        kpi("Calls analysed", len(calls), f"{participants} participants · {calls['account'].nunique()} accounts"),
        kpi("Avg confidence", f"{calls['sentiment_score'].mean():.2f}", "Scale 1–5"),
        kpi("Outage-related calls", n_outage_calls(calls), f"Detect outage + fallout, from {OUTAGE_START:%m/%d/%Y}"),
        kpi("Churn signals", int(calls["km_churn_signal"].sum()), "Key moments flagged"),
        kpi("Competitor mentions", int((comp.map(len) > 0).sum()),
            f"Calls · {top_comp.index[0]} leads" if len(top_comp) else "Calls naming a competitor"),
    ])

    st.write("")
    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        weekly = (end - start).days <= 45
        st.markdown(f"**Average confidence by call type** · score 1–5, {'weekly' if weekly else 'monthly'} mean")
        st.plotly_chart(confidence_by_type_chart(calls, start, end), width="stretch")
    with right, st.container(border=True):
        mix = calls.groupby("call_type")["sentiment_score"].agg(["size", "mean"]).reindex(CALL_TYPES).dropna()
        segments = "".join(f'<div style="flex:{r["size"]};background:{CALL_TYPE_COLORS[t]}"></div>' for t, r in mix.iterrows())
        stats = "".join(f'<div><div class="ti-mix-n">{int(r["size"])}</div>'
                        f'<div class="ti-kpi-sub">{t.capitalize()} · avg {r["mean"]:.2f}</div></div>' for t, r in mix.iterrows())
        labels = calls["sentiment_label"].value_counts()
        rows = bar_list([(lbl.replace("-", " ").capitalize(), int(labels.get(lbl, 0)), NEUTRAL_BAR) for lbl in SENTIMENT_LABELS])
        st.markdown(f'<b>Call mix</b><div class="ti-kpi-sub">Derived from title prefix + email domains</div>'
                    f'<div class="ti-mix">{segments}</div><div class="ti-mix-stats">{stats}</div>'
                    f'<hr style="margin:14px 0 8px;opacity:.3"><b>Overall confidence labels</b>{rows}',
                    unsafe_allow_html=True)

    sla = ctx["sla"]
    with st.container(border=True):
        st.markdown(f"**Outage duration & SLA impact** · platform downtime by month vs the "
                    f"{ctx['sla_target']:.1%} uptime SLA")
        left, right = st.columns([1.1, 1])
        with left:
            st.plotly_chart(sla_chart(sla), width="stretch")
        with right:
            table = sla.assign(
                **{"Downtime (h)": sla["outage_hours"].round(1),
                   "% of month down": (sla["pct_time_down"] * 100).round(3),
                   "Availability %": (sla["availability"] * 100).round(3),
                   "SLA allows (h)": sla["allowed_hours"],
                   "SLA": sla["sla_breached"].map({True: "Breached", False: "Met"}),
                   "Over SLA (h)": sla["hours_over_sla"]})
            show_table(table[["label", "Downtime (h)", "% of month down", "Availability %", "SLA allows (h)",
                                "SLA", "Over SLA (h)"]].rename(columns={"label": "Month"}),
                         hide_index=True, width="stretch")
            credits = ctx["credit_calls"] & set(calls["call_id"])
            tl = ctx["timeline"]
            affected = tl.loc[tl["call_type"] != "internal", "account"].nunique() if len(tl) else 0
            st.caption(f"{affected} customer accounts were on outage-related calls; SLA or service credits came up "
                       f"on {len(credits)} calls in this period.")

    left, right = st.columns([1.7, 1])
    with left, st.container(border=True):
        st.markdown("**Account details** · confidence over the period, oldest → newest call")
        acc = calls[calls["account"].notna()].sort_values("start_time")
        if len(acc):
            g = acc.groupby("account")
            details = pd.DataFrame({
                "calls": g.size(),
                "confidence_trend": g["sentiment_score"].apply(list),
                "avg_confidence": g["sentiment_score"].mean().round(2),
                "latest": g["sentiment_score"].last(),
                "change": (g["sentiment_score"].last() - g["sentiment_score"].first()).round(1),
                "last_call": g["start_time"].max().dt.date,
            }).join(ctx["health"].set_index("account")[["risk_tier", "churn_signals"]]).reset_index()
            show_table(details.sort_values("avg_confidence"), hide_index=True, width="stretch", height=400,
                         column_config={"confidence_trend": st.column_config.LineChartColumn(
                             "Confidence trend", y_min=1, y_max=5, width="medium")})
        else:
            st.caption("No customer accounts in this selection.")
    with right, st.container(border=True):
        teams = calls["teams"].map(_as_list).explode().dropna()
        ours = teams[~teams.str.endswith("(customer)")].value_counts()
        st.markdown("**Teams on calls** · AegisCloud teams by calls joined"
                    + bar_list([(t, int(n), SINGLE_SERIES) for t, n in ours.items()])
                    + f'<div class="ti-note">Plus {teams[teams.str.endswith("(customer)")].nunique()} customer teams. '
                      "Teams are derived from role mentions and meeting attendance.</div>",
                    unsafe_allow_html=True)

    if calls["account"].notna().any():
        with st.container(border=True):
            st.markdown("**Account confidence over time** · average per account and "
                        f"{'week' if (end - start).days <= 45 else 'month'} (orange low, blue high)")
            st.plotly_chart(account_heatmap(calls, start, end), width="stretch")

    left, right = st.columns(2)
    with left, st.container(border=True):
        themes = theme_counts(calls)
        st.markdown("**Top themes** · calls per theme, multi-label"
                    + bar_list([(t, int(v), SINGLE_SERIES) for t, v in themes.items()]), unsafe_allow_html=True)
    with right, st.container(border=True):
        km = ctx["km"]["type"].value_counts()
        items = sorted(((t.replace("_", " ").capitalize(), int(km.get(t, 0)),
                         RISK if t in RISK_MOMENTS else SINGLE_SERIES) for t in KEY_MOMENT_TYPES), key=lambda x: -x[1])
        st.markdown(f"**Key moments detected** · across {len(calls)} calls{bar_list(items)}"
                    f'<div class="ti-note">Orange = risk signals · Blue = positive / neutral</div>',
                    unsafe_allow_html=True)


def render_topics(ctx: dict) -> None:
    calls = ctx["calls"]
    if calls.empty:
        st.info("No calls match the current filters.")
        return
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Themes by period** · calls per theme")
        st.plotly_chart(theme_period_heatmap(calls), width="stretch")
    with right, st.container(border=True):
        st.markdown("**Average confidence by theme** · calls tagged with each theme")
        st.plotly_chart(confidence_by_theme_chart(calls), width="stretch")
    with st.container(border=True):
        st.markdown("**Primary theme by call type**")
        st.plotly_chart(primary_theme_by_type_chart(calls), width="stretch")
    with st.container(border=True):
        st.markdown("**Call explorer**")
        theme = st.selectbox("Theme", ["All themes", *THEME_NAMES], key="topics_theme")
        shown = calls if theme == "All themes" else calls[has_theme(calls, theme)]
        table = (shown[["date", "title", "call_type", "teams", "primary_theme", "themes", "sentiment_score"]]
                 .rename(columns={"sentiment_score": "confidence_score"}))
        show_table(table.sort_values("date"), hide_index=True, width="stretch")
        st.caption("Classification certainty and reasons are in Tool Telemetry.")


def render_confidence(ctx: dict) -> None:
    calls = ctx["calls"]
    if calls.empty:
        st.info("No calls match the current filters.")
        return
    ext = calls[calls["call_type"] != "internal"]
    r = calls["sentiment_score"].corr(calls["vader_mean"]) if len(calls) > 2 else float("nan")
    support = calls[calls["call_type"] == "support"]
    kpi_row([
        kpi("Avg confidence", f"{calls['sentiment_score'].mean():.2f}", "Vendor score, 1–5"),
        kpi("Transcript agreement", f"{r:.2f}" if pd.notna(r) else "–",
            "How closely the vendor score tracks the words used (1 = perfectly)"),
        kpi("Customer − staff", f"{ext['side_gap'].mean():+.2f}" if len(ext) else "–",
            f"Support calls: {support['side_gap'].mean():+.2f}" if len(support) else "External + support calls"),
        kpi("Improved during call", f"{(calls['arc'] == 'improved').mean():.0%}", "Last third more positive than the first"),
        kpi("Scores to double-check", int(calls["disagreement"].fillna(False).astype(bool).sum()),
            "Vendor score and the words used far apart"),
    ])
    st.write("")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Vendor confidence vs transcript language** · one dot per call")
        st.plotly_chart(vendor_vs_transcript_chart(calls), width="stretch")
        st.caption("The two largely agree, so the vendor score is safe to use for trends. "
                   "Ringed dots are calls where they disagree.")
    with right, st.container(border=True):
        st.markdown("**Customer vs AegisCloud staff** · by period, external + support calls")
        st.plotly_chart(side_by_period_chart(calls), width="stretch")
        st.caption("Staff sound slightly more positive than customers on support calls and during "
                   "the outage, so the customer score is the truer read of customer mood.")
    left, right = st.columns([1.2, 1])
    with left, st.container(border=True):
        st.markdown("**Confidence by slice**")
        dim = st.radio("Slice by", list(SLICE_LABELS), format_func=SLICE_LABELS.get, horizontal=True,
                       key="conf_slice", label_visibility="collapsed")
        fig, table = slice_chart(calls, dim)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Table view"):
            show_table(table.drop(columns="dimension").rename(columns={
                "value": SLICE_LABELS[dim], "vendor_mean": "confidence_mean", "vader_mean": "transcript_mean",
                "customer_mean": "customer_transcript", "staff_mean": "staff_transcript"}),
                hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.markdown("**Within-call arc** · first third vs last third of the transcript")
        st.plotly_chart(arc_chart(calls), width="stretch")
        arcs = calls["arc"].value_counts()
        st.markdown(f"{int(arcs.get('improved', 0))} improved · {int(arcs.get('flat', 0))} flat · "
                    f"{int(arcs.get('declined', 0))} declined.")
        pivot = calls["has_positive_pivot"].fillna(False).astype(bool)
        if pivot.any() and (~pivot).any():
            st.caption(f"{(calls.loc[pivot, 'arc'] == 'improved').mean():.0%} of calls with a vendor *positive pivot* "
                       f"moment improved, vs {(calls.loc[~pivot, 'arc'] == 'improved').mean():.0%} without one, so the "
                       "pivot tag alone doesn't predict a better ending.")
    with st.container(border=True):
        st.markdown("**Largest gaps between vendor score and transcript language** · check before quoting")
        gaps = calls.reindex(calls["disagreement_z"].abs().sort_values(ascending=False).index).head(8)
        show_table(gaps[["date", "title", "call_type", "sentiment_score", "vader_mean", "disagreement_note"]]
                     .rename(columns={"sentiment_score": "confidence_score", "vader_mean": "transcript_score",
                                      "disagreement_note": "note"}), hide_index=True, width="stretch")


def account_brief(account: str, ctx: dict) -> tuple[str, pd.DataFrame]:
    """Markdown brief of one account's state for the selected period, plus its action items."""
    row = ctx["health"].set_index("account").loc[account]
    calls = ctx["calls"][ctx["calls"]["account"] == account].sort_values("start_time")
    items = (ctx["ai"][ctx["ai"]["account"] == account]
             .merge(calls[["call_id", "date", "title"]], on="call_id", how="left")
             .sort_values("date")[["date", "owner", "task", "title"]]
             .rename(columns={"title": "from_call"}))
    km = ctx["km"]
    risk_km = km[(km["account"] == account) & km["type"].isin(RISK_MOMENTS)]
    reg = ctx["register_all"]
    open_inc = reg[(reg["status"] == "Open") & reg["account"].fillna("").str.contains(account, regex=False)]

    lines = [
        f"# Account brief: {account}",
        f"_Period {ctx['start']:%m/%d/%Y} – {ctx['end']:%m/%d/%Y} · generated "
        f"{datetime.now(timezone.utc):%m/%d/%Y %H:%M} UTC_",
        "",
        "## Current state",
        f"- **Risk:** {row['risk_tier']} ({row['risk_score']}/100) — {row['drivers']}",
        f"- **Calls:** {row['n_calls']} ({row['n_support']} support), "
        f"{row['first_call']:%m/%d/%Y} → {row['last_call']:%m/%d/%Y}",
        f"- **Confidence:** average {row['avg_score']:.2f}, latest {row['last_score']:.1f} "
        f"({row['score_trend']:+.1f} since the first call in the period)",
        f"- **Churn signals:** {row['churn_signals']} · **concerns:** {row['concerns']}",
        f"- **Competitors named:** {', '.join(row['competitors']) or 'none'}",
        f"- **Affected by the Detect outage:** {'yes' if row['outage_affected'] else 'no'} · "
        f"**renewal in play:** {'yes' if row['renewal_in_play'] else 'no'}",
        f"- **Latest call:** {row['last_title']}",
        "",
        f"## Open incidents ({len(open_inc)})",
    ]
    lines += [f"- {r['incident_id']} · {r['priority']} {r['category'].lower()} · {r['title']} "
              f"(reported {fmt_ts(r['reported_at'])} UTC, owner {r['owner'] or 'unassigned'})"
              for _, r in open_inc.iterrows()] or ["- None"]
    lines += ["", f"## Action items ({len(items)})"]
    lines += [f"- [ ] **{r['owner'] or 'Unassigned'}:** {r['task']} _({r['date']:%m/%d/%Y}, {r['from_call']})_"
              for _, r in items.iterrows()] or ["- None recorded"]
    lines += ["", "## Risk moments"]
    lines += [f"- {r['type'].replace('_', ' ').capitalize()} — {r['text']}" for _, r in risk_km.head(6).iterrows()] \
        or ["- None"]
    lines += ["", "## Recent calls"]
    lines += [f"- {r['date']:%m/%d/%Y} · {r['title']} · {r['call_type']} · confidence {r['sentiment_score']:.1f}"
              for _, r in calls.tail(5).iloc[::-1].iterrows()]
    return "\n".join(lines), items


@st.dialog("Account brief", width="large")
def account_brief_dialog(account: str, ctx: dict) -> None:
    brief, items = account_brief(account, ctx)
    slug = account.lower().replace(" ", "_")
    st.markdown(brief)
    st.divider()
    a, b = st.columns(2)
    a.download_button("⬇ Download brief (.md)", brief, file_name=f"{slug}_account_brief.md",
                      mime="text/markdown", width="stretch")
    csv = items.assign(date=items["date"].map(lambda d: f"{d:{DATE_FMT}}" if pd.notna(d) else ""))
    b.download_button("⬇ Download action items (.csv)", csv.to_csv(index=False).encode("utf-8-sig"),
                      file_name=f"{slug}_action_items.csv", mime="text/csv", width="stretch",
                      disabled=items.empty)
    with st.expander("Copy as text"):
        st.code(brief, language="markdown", wrap_lines=True)
        st.caption("Use the copy icon at the top right of the box.")


def render_accounts(ctx: dict) -> None:
    calls, health = ctx["calls"], ctx["health"]
    if health.empty:
        st.info("No customer accounts match the current filters.")
        return
    tiers = health["risk_tier"].value_counts()
    kpi_row([
        kpi("Accounts", len(health), "Customer email domains"),
        kpi("High risk", int(tiers.get("High", 0)), "Risk score 55+"),
        kpi("Medium risk", int(tiers.get("Medium", 0)), "Risk score 30–54"),
        kpi("Named a competitor", int(health["competitors"].map(len).gt(0).sum()), "In this period"),
        kpi("Hit by outage", int(health["outage_affected"].sum()), "Had an outage-related call"),
    ])
    st.write("")
    left, right = st.columns([1, 1.25])
    with left, st.container(border=True):
        st.markdown("**Churn risk ranking** · top 15 accounts")
        st.plotly_chart(risk_ranking_chart(health), width="stretch")
        st.caption("Risk score = " + " · ".join(f"{k.replace('_', ' ')} {v}" for k, v in RISK_WEIGHTS.items()) + ".")
    with right, st.container(border=True):
        st.markdown("**Account health table** · ranked by risk")
        st.caption("Click an account name to open its brief: current state and action items, ready to copy or download.")
        table = (health[["account", "risk_tier", "risk_score", "drivers", "n_calls", "avg_score", "last_score",
                         "score_trend", "churn_signals", "competitors", "action_items"]]
                 .rename(columns={"avg_score": "avg_confidence", "last_score": "latest_confidence"})
                 .reset_index(drop=True))
        event = show_table(
            table, hide_index=True, width="stretch", height=520, key="acct_table",
            on_select="rerun", selection_mode="single-cell",
            column_config={
                "account": st.column_config.TextColumn("account 🔎", help="Click to open the account brief"),
                "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=100, format="%d"),
                "action_items": st.column_config.NumberColumn(
                    "action_items", help="Commitments made on this account's calls; click the account to see them"),
            })
        cells = event.selection.cells if event else []
        if cells and cells[0][1] == "account":
            picked = table.iloc[cells[0][0]]["account"]
            if st.session_state.get("acct_brief_open") != picked:   # open once per click, not on every rerun
                st.session_state["acct_brief_open"] = picked
                account_brief_dialog(picked, ctx)
        else:
            st.session_state["acct_brief_open"] = None
    with st.container(border=True):
        st.markdown("**Account drill-down**")
        account = st.selectbox("Account", health["account"].tolist(), key="acct_pick")
        row = health.set_index("account").loc[account]
        st.markdown(f"**{account}** · {badge(row['risk_tier'] + ' risk', TIER_COLORS[row['risk_tier']])} "
                    f"({row['risk_score']}/100) · {html.escape(row['drivers'])}", unsafe_allow_html=True)
        a_calls = calls[calls["account"] == account]
        left, right = st.columns([1.3, 1])
        with left:
            st.plotly_chart(account_history_chart(a_calls, ctx["start"], ctx["end"]), width="stretch")
            show_table(a_calls.sort_values("start_time")[["date", "title", "call_type", "teams", "primary_theme",
                                                            "sentiment_score"]]
                         .rename(columns={"sentiment_score": "confidence_score"}), hide_index=True, width="stretch")
        with right:
            km = ctx["km"]
            st.markdown("Risk moments")
            show_table(km[(km["account"] == account) & km["type"].isin(RISK_MOMENTS)][["type", "speaker", "text"]],
                         hide_index=True, width="stretch", height=200)
            comp = ctx["comp"]
            comp = comp[comp["account"] == account] if len(comp) else comp
            if len(comp):
                st.markdown("Competitor mentions")
                show_table(comp[["competitor", "reasons", "quote"]], hide_index=True, width="stretch", height=160)
            ai = ctx["ai"]
            st.markdown("Open commitments")
            show_table(ai[ai["account"] == account][["owner", "task"]], hide_index=True, width="stretch", height=200)


@st.dialog("Call out", width="large")
def call_out_dialog(inc: dict, people: list[str]) -> None:
    st.markdown(f"**{inc['incident_id']} · {inc['title']}**")
    st.caption(f"{inc['priority']} {inc['category'].lower()} · reported {fmt_ts(inc['reported_at'])} UTC · "
               f"owner {inc['owner'] or 'unassigned'}")
    options = [handle(p) for p in people]
    default = [handle(inc["owner"])] if inc["owner"] and handle(inc["owner"]) in options else []
    to = st.multiselect("Address", options, default=default, placeholder="@user_name")
    msg = st.text_area("Message", height=140, value=(
        f"{inc['incident_id']} ({inc['title']}) is still open since {fmt_ts(inc['reported_at'])} UTC. "
        "Can you share the current status and an ETA for resolution?"))
    st.markdown("**Preview**")
    st.info(f"{' '.join(to) or '@…'} {msg}")
    if st.button("Send call-out", type="primary", disabled=not to):
        st.success("Demo only: this call-out was not sent or stored.")


def render_incidents(ctx: dict, tables) -> None:
    reg, start, end = ctx["register"], ctx["start"], ctx["end"]
    st.subheader("Incident register")
    st.caption("Every incident raised on a call: when it was reported, who owned it and when, and when it was "
               "resolved. Times are UTC.")
    left, right = st.columns([2, 1])
    cats = left.multiselect("Category", ALL_CATEGORIES, default=INCIDENT_CATEGORIES, key="inc_cats")
    status = right.segmented_control("Status", ["All", "Open", "Resolved"], default="All", key="inc_status")
    r = reg[reg["category"].isin(cats)]
    if status in ("Open", "Resolved"):
        r = r[r["status"] == status]
    open_r = r[r["status"] == "Open"]
    resolved = r[r["status"] == "Resolved"]
    kpi_row([
        kpi("Incidents", len(r), f"Reported {start:%m/%d/%Y} – {end:%m/%d/%Y}"),
        kpi("Open", len(open_r), f"{int((open_r['priority'] == 'P1').sum())} of them P1",
            CRITICAL if (open_r["priority"] == "P1").any() else None),
        kpi("Median time to own", f"{r['minutes_to_own'].median():.0f} min" if len(r) else "–", "Reported → owner engaged"),
        kpi("Median time to resolve", f"{resolved['hours_to_resolve'].median():.1f} h" if len(resolved) else "–",
            f"{len(resolved)} resolved"),
        kpi("Platform downtime", f"{ctx['sla']['outage_hours'].sum():.1f} h", "Major incidents in this period"),
    ])
    st.write("")
    with st.container(border=True):
        view = r.assign(
            reported=r["reported_at"].map(fmt_ts), owned=r["owned_at"].map(fmt_ts), resolved=r["resolved_at"].map(fmt_ts),
            owner=r["owner"].fillna("—") + " (" + r["owner_team"].fillna("?") + ")")
        show_table(view[["incident_id", "status", "priority", "category", "title", "account", "reported",
                           "owner", "owned", "resolved", "minutes_to_own", "hours_to_resolve", "outage_hours",
                           "parent_id", "resolution_note"]],
                     hide_index=True, width="stretch", height=380,
                     column_config={"minutes_to_own": "min to own", "hours_to_resolve": "h to resolve",
                                    "outage_hours": "outage h", "parent_id": "part of"})

    with st.container(border=True):
        st.markdown(f"**Open incidents** · {len(open_r)} — use Call Out to chase the owner")
        people = staff_people(tables)
        order = {"P1": 0, "P2": 1, "P3": 2}
        for _, inc in open_r.sort_values(by="priority", key=lambda s: s.map(order)).iterrows():
            a, b = st.columns([6, 1])
            color = CRITICAL if inc["priority"] == "P1" else (WARNING if inc["priority"] == "P2" else NEUTRAL_BAR)
            age = (datetime.now(timezone.utc) - inc["reported_at"]).days
            a.markdown(f"{badge(inc['priority'], color)} **{inc['incident_id']}** · {html.escape(inc['title'])} · "
                       f"reported {fmt_ts(inc['reported_at'])} · owner {inc['owner'] or '—'} · open {age} days",
                       unsafe_allow_html=True)
            if b.button("Call Out", key=f"co_{inc['incident_id']}", width="stretch"):
                call_out_dialog(inc.to_dict(), people)
        if open_r.empty:
            st.caption("No open incidents for these filters.")

    st.subheader("Blast radius of the Detect outage")
    tl, inc = ctx["timeline"], ctx["incident_metrics"]
    if tl.empty or not inc:
        st.info("The selected period doesn't include the outage.")
        return
    before, after = inc.get("affected_avg_before"), inc.get("affected_avg_after")
    named = inc.get("affected_naming_competitor_after", [])
    kpi_row([
        kpi("Outage-related calls", len(tl), f"{inc.get('n_window_calls', 0)} inside {OUTAGE_START:%m/%d/%Y}–{OUTAGE_END:%m/%d/%Y}"),
        kpi("Accounts affected", tl.loc[tl["call_type"] != "internal", "account"].nunique(), "Customer calls about the outage"),
        kpi("First customer escalation", f"+{inc.get('hours_to_first_customer_call') or 0:.0f} h", "After the first war room"),
        kpi("Affected accounts' confidence", f"{before:.2f} → {after:.2f}" if before and after else "–", "Before vs after"),
        kpi("Then named a competitor", len(named), "Affected accounts, after the outage"),
    ])
    with st.container(border=True):
        st.markdown("**Outage call timeline** · marker size = churn signals on the call")
        st.plotly_chart(incident_chart(tl, start, end), width="stretch")
        if named:
            st.caption("Affected accounts that named a competitor afterwards: " + ", ".join(named))


def render_insights(ctx: dict) -> None:
    st.subheader("Voice of the customer")
    left, right = st.columns(2)
    with left, st.container(border=True):
        asks, moments = ctx["asks"], ctx["gap_moments"]
        st.markdown("**Top feature asks** · accounts asking, from feature-gap moments"
                    + bar_list([(a, int(n), SINGLE_SERIES) for a, n in zip(asks["ask"], asks["n_accounts"])]),
                    unsafe_allow_html=True)
        if len(asks):
            with st.expander("What customers said"):
                ask = st.selectbox("Ask", asks["ask"].tolist(), key="ins_ask")
                st.caption(asks.set_index("ask").loc[ask, "summary"] or "")
                show_table(moments[moments["ask"] == ask][["account", "speaker", "text"]], hide_index=True, width="stretch")
    with right, st.container(border=True):
        comp = ctx["comp"]
        by_comp = comp.groupby("competitor")["call_id"].nunique().sort_values(ascending=False) if len(comp) else pd.Series(dtype=int)
        reasons = comp["reasons"].map(_as_list).explode().value_counts() if len(comp) else pd.Series(dtype=int)
        st.markdown("**Competitive intelligence** · calls naming each competitor"
                    + bar_list([(c, int(n), RISK) for c, n in by_comp.items()])
                    + "<br><b>Why they come up</b> · mentions by stated reason"
                    + bar_list([(r, int(n), NEUTRAL_BAR) for r, n in reasons.items()]), unsafe_allow_html=True)
        if len(comp):
            with st.expander("Quotes"):
                show_table(comp[["start_time", "account", "competitor", "reasons", "quote"]], hide_index=True, width="stretch")

    st.subheader("Coaching and follow-through")
    left, right = st.columns(2)
    with left, st.container(border=True):
        talk, reps = ctx["talk"], ctx["reps"]
        st.markdown("**Customer talk time vs outcome** · external + support calls")
        if len(talk) >= 8:
            st.plotly_chart(talk_bucket_chart(talk), width="stretch")
            st.caption("Calls where the customer does the most talking score lowest: long customer turns are "
                       "usually complaints, so a rising share is an early warning.")
        else:
            st.caption("Not enough customer calls in this period.")
        if len(reps):
            with st.expander("By lead rep"):
                show_table(reps.rename(columns={"avg_score": "avg_confidence"}), hide_index=True, width="stretch")
    with right, st.container(border=True):
        owners = owner_workload(ctx["ai"]).head(12)
        st.markdown("**Action-item workload** · items owned, top 12"
                    + bar_list([(o, int(n), SINGLE_SERIES) for o, n in zip(owners["owner"], owners["n_items"])]),
                    unsafe_allow_html=True)
        with st.expander("Accounts with commitments across several calls"):
            show_table(recurring_commitments(ctx["ai"]), hide_index=True, width="stretch")


def render_telemetry(ctx: dict, tables, mode: str, topic_metrics: dict, clusters: pd.DataFrame,
                     sent_metrics: dict) -> None:
    usage = load_usage()
    now = pd.Timestamp.now(tz="UTC")
    in_month = usage["ts"].dt.tz_convert(None).dt.to_period("M") == now.tz_convert(None).to_period("M") \
        if len(usage) else pd.Series(dtype=bool)
    month_cost = float(usage.loc[in_month, "cost_usd"].sum()) if len(usage) else 0.0
    total = float(usage["cost_usd"].sum()) if len(usage) else 0.0
    share = month_cost / MONTHLY_BUDGET_USD if MONTHLY_BUDGET_USD else 0.0
    alert = share >= BUDGET_WARN_SHARE

    st.subheader("LLM cost")
    kpi_row([
        kpi(f"This month ({now:%b %Y})", f"${month_cost:,.2f}", f"of {money(MONTHLY_BUDGET_USD)} monthly budget",
            CRITICAL if alert else None),
        kpi("Total spend so far", f"${total:,.2f}", f"{len(usage)} API requests logged"),
        kpi("Tokens in / out", f"{int(usage['input_tokens'].sum()):,} / {int(usage['output_tokens'].sum()):,}"
            if len(usage) else "0 / 0", "Across all logged requests"),
        kpi("Model", LLM_MODEL, "Set by LLM_MODEL in .env"),
    ])
    bar_color = CRITICAL if alert else (WARNING if share >= BUDGET_WARN_SHARE * 0.75 else GOOD)
    st.markdown(f'<div class="ti-budget"><div style="width:{min(share, 1) * 100:.2f}%;background:{bar_color}"></div></div>'
                f'<div class="ti-note">{share:.2%} of this month\'s budget used. Turns red at '
                f'{BUDGET_WARN_SHARE:.0%} ({money(BUDGET_WARN_SHARE * MONTHLY_BUDGET_USD)}). '
                f'{budget_source_note()}</div>', unsafe_allow_html=True)
    if alert:
        st.error(f"LLM spend this month is ${month_cost:,.2f} — {share:.0%} of the {money(MONTHLY_BUDGET_USD)} budget.")
    uncosted = uncosted_cache_entries()
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Spend by month**")
        if len(usage):
            st.plotly_chart(monthly_cost_chart(usage, MONTHLY_BUDGET_USD, BUDGET_WARN_SHARE), width="stretch")
        else:
            st.caption("No API requests logged yet.")
        if uncosted:
            st.caption(f"{uncosted} cached answers were created before cost logging was added; their cost isn't "
                       "included. They're reused for free on every run.")
    with right, st.container(border=True):
        st.markdown("**Spend by step**")
        if len(usage):
            by_step = usage.groupby("step").agg(requests=("cost_usd", "size"), input_tokens=("input_tokens", "sum"),
                                                output_tokens=("output_tokens", "sum"), cost_usd=("cost_usd", "sum"))
            show_table(by_step.round(4).reset_index(), hide_index=True, width="stretch")
            with st.expander("Request log"):
                show_table(usage.sort_values("ts", ascending=False), hide_index=True, width="stretch")

    st.subheader(f"Classification quality · {MODES[mode]}")
    agree = topic_metrics.get("primary_in_rule_themes")
    reg = ctx["register_all"]
    kpi_row([
        kpi("Classified by LLM", f"{topic_metrics['n_llm']} / {topic_metrics['n_calls']}",
            topic_metrics.get("model") or "Keyword rules only"),
        kpi("LLM vs rules agreement", f"{agree:.0%}" if agree is not None else "–", "Primary theme also found by rules"),
        kpi("Cluster purity", f"{topic_metrics['cluster_purity_weighted']:.0%}", "Calls matching their group's main theme"),
        kpi("Vendor vs transcript", f"{sent_metrics['pearson_vendor_vs_vader']:.2f}",
            f"{sent_metrics['n_disagreements']} calls flagged to double-check"),
        kpi("Incidents by LLM", f"{int((reg['source'] == 'llm').sum())} / {len(reg)}", "Rest by keyword rules"),
    ])
    st.write("")
    with st.container(border=True):
        st.markdown("**Discovered clusters** · calls grouped automatically by similarity, before the themes were applied")
        show_table(clusters[["cluster", "n_calls", "top_terms", "dominant_theme", "purity"]], hide_index=True, width="stretch")
    with st.container(border=True):
        st.markdown("**Call classification details**")
        c = ctx["calls"]
        show_table(c[["date", "title", "call_type", "teams", "primary_theme", "themes", "certainty",
                        "rationale", "source"]].sort_values("date"), hide_index=True, width="stretch",
                     column_config={"certainty": st.column_config.ProgressColumn("certainty", min_value=0, max_value=1, format="%.2f")})
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Incident extraction details**")
        show_table(reg[["incident_id", "source", "category", "priority", "status", "owner", "resolution_note"]],
                     hide_index=True, width="stretch")
    with right, st.container(border=True):
        st.markdown("**Staff roster** · how each person's team was decided")
        show_table(tables["staff_roster"][["name", "team", "basis", "n_calls"]], hide_index=True, width="stretch")

    st.subheader("Pipeline")
    run = last_run()
    with st.container(border=True):
        if run:
            st.markdown(f"Last run **{pd.Timestamp(run['finished_at']):%m/%d/%Y %H:%M} UTC** · stages "
                        + " · ".join(f"{k} {v}s" for k, v in run.get("seconds", {}).items())
                        + f" · modes built: {', '.join(run.get('modes_built', []))}")
        st.markdown(
            "- **Themes:** Claude classifies each call into 10 fixed themes; keyword rules are the fallback and the check.\n"
            "- **Transcript language score:** an open-source word-list scorer reads every sentence (−1 to +1), "
            "used to check the vendor's confidence score.\n"
            "- **Clusters:** calls grouped by meaning with a small local language model, to ground the themes in the data.\n"
            "- **Incidents:** Claude reads the calls about each incident for status, owner and outage length; "
            "keyword rules otherwise.\n"
            f"- **Cache:** Claude answers are stored in `{PROCESSED_DIR.name}/llm_cache`, so re-runs cost nothing.")


def build_context(tables, register, ins, calls: pd.DataFrame, start: date, end: date,
                  sla_target: float) -> dict:
    """Everything the tabs show, recomputed for the filtered calls and period."""
    ids = set(calls["call_id"])
    sents = tables["sentences"][tables["sentences"]["call_id"].isin(ids)]
    km = tables["key_moments"][tables["key_moments"]["call_id"].isin(ids)]
    ai = tables["action_items"][tables["action_items"]["call_id"].isin(ids)]
    timeline, incident_metrics = incident_timeline(calls) if len(calls) else (pd.DataFrame(), {})
    customer_calls = calls[calls["call_type"] != "internal"]
    gap_moments = ins["feature_gap_moments"][ins["feature_gap_moments"]["call_id"].isin(ids)]
    asks = (gap_moments.groupby("ask").agg(n_mentions=("text", "size"), n_accounts=("account", "nunique"))
            .reset_index().merge(ins["feature_gaps"][["ask", "summary"]], on="ask", how="left")
            .sort_values(["n_accounts", "n_mentions"], ascending=False))
    talk, reps = (talk_time(sents, calls) if len(customer_calls) >= 4
                  else (pd.DataFrame(columns=["talk_bucket", "sentiment_score"]), pd.DataFrame()))
    return {
        "calls": calls, "start": start, "end": end, "km": km, "ai": ai, "sla_target": sla_target,
        "health": account_health(calls, km, ai) if calls["account"].notna().any() else pd.DataFrame(
            columns=["account", "risk_tier", "risk_score", "score_trend", "competitors", "outage_affected",
                     "churn_signals"]),
        "timeline": timeline if len(timeline) else pd.DataFrame(columns=["call_type", "account"]),
        "incident_metrics": incident_metrics,
        "comp": competitor_mentions(sents, calls) if len(calls) else pd.DataFrame(),
        "talk": talk, "reps": reps,
        "register": register[register["reported_at"].dt.date.between(start, end)],
        "register_all": register,
        "sla": monthly_availability(register, start, end, sla_target),
        "credit_calls": sla_credit_calls(tables["sentences"]),
        "asks": asks, "gap_moments": gap_moments,
    }


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    modes = available_modes()
    head, cfg = st.columns([5, 1])
    with cfg:
        mode, sla_target = config_popover(modes or ["rules"])
    try:
        tables, call_sent, sent_metrics, version = load_base()
        st.session_state["data_version"] = version   # the pipeline run this page's (cached) data came from
    except FileNotFoundError:
        head.title("Transcript Intelligence")
        st.warning("No processed data yet. Use **Data → Refresh dashboard data** in the sidebar to run the pipeline.")
        with st.sidebar:
            data_section()
        return
    if mode not in modes:
        head.title("Transcript Intelligence")
        st.warning("This analysis mode hasn't been built yet. Use Data → Refresh in the sidebar.")
        with st.sidebar:
            data_section()
        return
    call_topics, clusters, topic_metrics, register, ins = load_mode(mode)
    all_calls = merged_calls(tables, call_sent, call_topics)

    first, last = all_calls["start_time"].min(), all_calls["start_time"].max()
    with head:
        st.title("Transcript Intelligence")
        st.caption(f"AegisCloud · {len(all_calls)} calls · {first:%m/%d/%Y} – {last:%m/%d/%Y} · "
                   f"mode: {MODES[mode]}")

    calls, start, end = sidebar_filters(all_calls)
    ctx = build_context(tables, register, ins, calls, start, end, sla_target)

    tabs = st.tabs(["Overview", "Topics", "Confidence", "Accounts", "Incidents", "Insights", "Tool Telemetry"])
    with tabs[0]:
        render_overview(ctx)
    with tabs[1]:
        render_topics(ctx)
    with tabs[2]:
        render_confidence(ctx)
    with tabs[3]:
        render_accounts(ctx)
    with tabs[4]:
        render_incidents(ctx, tables)
    with tabs[5]:
        render_insights(ctx)
    with tabs[6]:
        render_telemetry(ctx, tables, mode, topic_metrics, clusters, sent_metrics)


# `streamlit run` executes this file as __main__; importing it (e.g. notebooks/05_dashboard) doesn't start the app
if __name__ == "__main__":
    main()

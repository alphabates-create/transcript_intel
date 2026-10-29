# Transcript Intelligence — Architecture

> A pipeline and dashboard that turns raw call transcripts into decisions for support,
> sales, product and engineering leaders.

---

## 1. Purpose and scope

**Goal.** Process call transcripts from a B2B SaaS company (AegisCloud): 100 original calls plus
13 synthetic ones, and any new ones dropped into `data/raw/`. Then:

1. **Categorize** calls by topic/theme (Task 1)
2. **Analyze sentiment** across call types and over time, and explain the trends (Task 2)
3. **Surface additional insights** for specific stakeholders (Task 3)

**Out of scope for this submission.** Production deployment, real-time ingestion,
authentication, multi-tenant data. Section 11 describes how the system would be productionized.

---

## 2. Design principles

| Principle | What it means here |
|---|---|
| **Insights over infrastructure** | Runs locally. No cloud services required to review or demo. |
| **Precompute, then present** | The pipeline runs once and writes files. The dashboard only reads those files, so the demo cannot fail on an API outage, rate limit or slow LLM response. |
| **Small, testable steps** | Each module does one job and writes a table the next module reads. |
| **Reproducible** | Dependencies in `req_item.txt`, structured LLM output, LLM responses cached to disk (`data/processed/llm_cache/`). Re-runs give the same output at no cost. |
| **Hybrid AI** | Rules where the answer is deterministic, clustering to discover structure, an LLM for nuance — each validated against the others. |
| **Trust but verify the source data** | Provided summaries and sentiment are used as one signal and checked against our own scoring. |
| **Fail soft** | A missing or malformed transcript folder is logged and skipped, never crashes the run. |

---

## 3. Source data

Each meeting is a folder `data/raw/<meetingId>/` with six JSON files:

| File | Content | Used for |
|---|---|---|
| `meeting-info.json` | Title, start/end time, duration, attendee emails | Call type, account, timeline |
| `summary.json` | Summary, action items, free-text topics, sentiment label + 1–5 score, key moments | Topics, sentiment, insights |
| `transcript.json` | Sentence text, speaker, timing, sentence sentiment, ASR confidence | Sentiment validation, talk time, quotes |
| `speakers.json` | Speaker turns | Talk time |
| `speaker-meta.json` | Speaker id → name | Speaker resolution |
| `events.json` | Join/leave times | Attendance (optional) |

**Dataset profile** (as of 10/29/2026)

- **113 calls, 3 Feb – 16 Jun 2026**: the original 100 (3 Feb – 28 Apr, 85 days) plus 13
  synthetic calls (6 May – 16 Jun)
- Call types: 47 external, 32 support, 34 internal (original 100: 43 / 27 / 30)
- 78 participants: 27 internal (`aegiscloud.com`), 51 customer contacts across 32 accounts (customer email domains)
- 4,725 sentences, 479 key moments, 456 action items
- Call type is **not labelled** in the data and is derived (Section 6.2)
- Topics are free text: **351 unique labels** across the original 100 calls, so they need normalizing

**Data handling decision.** The original 100 calls already contain all three call types and a
clear narrative (a major outage and a product launch). 13 synthetic transcripts in the same
format were added later (source notes in `C:\dev\New_Input\README.md`) to prove the pipeline
handles new input and to continue the storylines into May–June. The notebooks flag them
(`is_synthetic`, any call after 28 Apr) and keep headline numbers to the original 100; the
dashboard shows all calls.

---

## 4. System overview

```
            ┌───────────────────────────────────── src/pipeline.py ──────────────────────────────────────┐
            │                                                                                             │
 data/raw/ ─► load.py ─► screen.py ─► enrich.py ─► topics.py ─► sentiment.py ─► incidents.py ─► insights.py ─► export
 (113       flatten     drop non-     call type,   cluster +    vendor vs       register,       account        CSVs +
 folders)   to tables   business      account,     classify     transcript,     owner, SLA      health, VoC,   run
    ▲       (skips the  calls ──┐     product,        │         customer vs        │            competitors    summary
    │       exclusion   │       │     period, teams   │         staff, arcs        │                 │
    │       list)       │       └──────────────────── src/llm.py ◄─┴───────────────┴─────────────────┘
    │                   ▼                       Claude API · llm_cache/ · llm_usage.jsonl
    │      data/exclusions.csv · data/exclude/<id>/  (also written by the dashboard's Ignore button)
            │          topics / incidents / insights run twice: rules and hybrid                         │
            └───────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                        ▼
                         data/processed/*.parquet  ·  data/processed/{hybrid,rules}/
                outputs/tables/*.csv  ·  outputs/run_summary.json  ·  outputs/logs/pipeline.log
                                                        │
                         ┌──────────────────────────────┼──────────────────────────────┐
                         ▼                              ▼                              ▼
                  app/app.py                  notebooks/01–05                    Slide deck
          (Streamlit; also packaged as   (exploration, Tasks 1–3, and a         (presentation)
           TranscriptIntel.exe via        tour of app.py that imports
           run_app.py)                    its functions)
```

---

## 5. Repository layout

```
transcript_intel/
├── README.md                 # what it is, how to run it, key findings
├── req_item.txt              # dependencies (pip install -r req_item.txt)
├── .env.example              # ANTHROPIC_API_KEY, LLM_MODEL, LLM_MONTHLY_BUDGET_USD, LLM_BUDGET_WARN_SHARE (copy to .env)
├── .gitignore                # .env, .venv, caches, outputs/logs/, build/, dist/
├── check_setup.py            # one-off environment smoke test (data, libraries, API key)
├── data/
│   ├── raw/                  # 113 transcript folders (source; excluded calls are moved out)
│   ├── processed/            # parquet tables, llm_cache/, llm_usage.jsonl,
│   │   ├── hybrid/           #   per-mode outputs: Claude + rules
│   │   └── rules/            #   per-mode outputs: rules only
│   ├── labels/               # topic_gold_template.csv: hand-labelling sample (02_topics, 8c)
│   ├── exclusions.csv        # non-business calls kept out of every output (Section 6.1a)
│   └── exclude/              # excluded transcript folders, moved here from data/raw/
├── src/
│   ├── __init__.py
│   ├── load.py               # ingest and normalize; paths (TI_HOME override), modes
│   ├── enrich.py             # derive call type, account, product, period, teams
│   ├── llm.py                # the one Claude client: cache, fallbacks, cost log
│   ├── topics.py             # Task 1: topic categorization
│   ├── sentiment.py          # Task 2: sentiment analysis and trends
│   ├── screen.py             # drop non-business calls; exclusion list
│   ├── incidents.py          # incident register: reported / owned / resolved, SLA
│   ├── insights.py           # Task 3: stakeholder insights
│   └── pipeline.py           # end-to-end runner (both modes), new-transcript scan, run log
├── notebooks/
│   ├── 01_explore.ipynb      # data profile, quality, non-business screening, derived fields
│   ├── 02_topics.ipynb       # Task 1: approach, results, validation
│   ├── 03_sentiment.ipynb    # Task 2: trends and what they mean
│   ├── 04_insights.ipynb     # Task 3: stakeholder insights
│   └── 05_dashboard.ipynb    # tour of app/app.py using its own functions
├── app/app.py                # Streamlit dashboard
├── run_app.py                # launcher; entry point of TranscriptIntel.exe
├── TranscriptIntel.spec      # PyInstaller build spec
├── build_exe.ps1             # builds dist/TranscriptIntel/ and copies data next to the .exe
├── outputs/
│   ├── tables/               # CSV exports for the deck (written by the pipeline's export stage)
│   ├── charts/               # reserved for chart images (not generated yet)
│   ├── logs/pipeline.log     # every pipeline run, CLI or dashboard (git-ignored)
│   └── run_summary.json      # last pipeline run: timings, row counts, metrics
├── build/, dist/             # PyInstaller output (git-ignored)
└── docs/architecture.md      # this document
```

---

## 6. Pipeline stages

### 6.1 `src/load.py` — ingest and normalize

Reads each folder once and flattens it into four tables. It does **not** derive
business fields; that is `enrich.py`'s job.

| Function | Purpose |
|---|---|
| `read_meeting(folder)` | Load one folder's JSON files (UTF-8); return `None` if a required file is missing or invalid |
| `iter_meetings(raw_dir)` | Yield readable meetings in stable sorted order |
| `build_calls_table()` | One row per call |
| `build_sentences_table()` | One row per sentence, speaker resolved to email |
| `build_key_moments_table()` | One row per key moment |
| `build_action_items_table()` | One row per action item, split into owner and task |
| `load_all()` / `save_tables()` / `load_tables()` | Orchestrate, write and read parquet |

Speaker → email matching uses first initial + last name, which covers both email styles
(`first.last@aegiscloud.com` and `f.last@customer.com`) and names such as
"Dr. Linda Cheng" and "O'Brien". Result: 100% of sentences matched.

Before reading, the load stage moves folders marked `keep` in `data/exclusions.csv` back from
`data/exclude/` and moves any excluded folder that has reappeared in `data/raw/` out again. Calls
on the exclusion list are also dropped from the loaded tables, so they never reach any table.

### 6.1a `src/screen.py` — keep non-business calls out

Birthdays, celebrations, farewells, happy hours and other social calls would distort every
metric (a very positive internal call lifts confidence, adds action items, stretches the time
range), so they're removed before enrichment.

| Check | How |
|---|---|
| Keyword rule | `NON_BUSINESS` regex on the title and vendor topics only (summaries often say "celebrated the renewal"). Always available; also used to flag new transcripts in the dashboard before they're processed (`flag_folder()`). |
| `is_business` (Hybrid) | Claude answers `{is_business, reason}` from the title, attendee mix (staff count, customer domains), summary and topics. Cached under `screen_*`, so each call is checked once (~$0.002 per call). A work meeting that ends with a celebration counts as business. Claude's answer wins when it answers; the keyword rule decides otherwise. |
| Exclusion list | `data/exclusions.csv`: `call_id`, `title`, `date`, `action` (exclude / keep), `source` (keyword / llm / user), `reason`, `decided_at`. Each excluded folder is **moved** from `data/raw/` to `data/exclude/<id>/` (`move_folder()`; merged if a copy is already there). If an excluded folder is dropped into `data/raw/` again, the new-transcript scan ignores it and the next run moves it out (`sweep_excluded()`). `action = keep` forces a call in: the next run moves its folder back to `data/raw/` (`restore_kept()`), and it's never screened out again. |

The first Hybrid run with screening checked all 113 existing calls: none were non-business.

### 6.2 `src/enrich.py` — derive missing fields

| Field | Rule |
|---|---|
| `call_type` | Title starts with "Support Case" → **support**; any non-`aegiscloud.com` attendee → **external**; otherwise → **internal**. Result: 32 / 47 / 34 (original 100: 27 / 43 / 30). Edge cases (e.g. "URGENT: Cobalt Software…") are spot-checked. |
| `account` | Customer email domain with the trailing industry word removed (e.g. `brightpathcommerce.com` → Brightpath). Joins support and account-manager calls for the same customer. 32 accounts. |
| `products` | Keyword map (`PRODUCT_PATTERNS`) for Detect, Protect, Comply (incl. v2), Identity, LogVault on title + summary + topics |
| `period` | `Pre-outage` · `Outage` (10–18 Mar) · `Recovery` · `Post-Comply-launch` (from 7 Apr) |
| `speaker_side` | Staff vs customer, from the speaker's email domain |
| `teams` / `staff_roster` | No team field exists, so each AegisCloud person's team (Support, Customer Success & Sales, Solutions Engineering, Engineering, Product & Strategy, Leadership) comes from role mentions ("Maria Santos (CSM)", "this is Sarah Chen from Aegis support"), then the calls they attend; two documented overrides (`STAFF_TEAM_OVERRIDES`). Calls list the teams present plus the customer. |

Outputs: `data/processed/<table>_enriched.parquet` for calls, sentences, key_moments,
action_items and `staff_roster`.

### 6.3 `src/topics.py` — Task 1: categorization (hybrid)

1. **Discover** — embed each call (title + summary + vendor topics) with `all-MiniLM-L6-v2`
   (local; TF-IDF vectors if the model can't load), cluster with KMeans (k = 10), describe each
   cluster by its top TF-IDF terms.
2. **Define** — a fixed taxonomy of 10 themes (`THEMES`): Compliance & audit · Detect outage &
   fallout · Comply v2 launch · Renewal & contracts · Internal planning · Competitive threat ·
   Identity / SSO / MFA · Backup / Protect · Onboarding & deployment · Billing disputes. Each theme
   has a definition (sent to Claude) and a keyword pattern (the rules).
3. **Classify** — Claude assigns `primary_theme`, all applicable `themes`, `certainty` and a
   one-line `rationale`. Structured output validated with `pydantic`, low effort, cached so re-runs
   are identical. Keyword rules fill in when Claude is unavailable or declines.
4. **Validate** — agreement between Claude and the keyword rules (same primary theme for 75% of
   calls; Claude's primary is among the rule themes for 96%), and cluster purity against the
   assigned themes (58%). `02_topics` also writes a stratified 20-call template to
   `data/labels/` for a hand-labelled check; no labels have been filled in yet.

Outputs per mode: `call_topics.parquet`, `topic_clusters.parquet`, `topic_validation.json`.

**Why hybrid:** clustering grounds the taxonomy in the data rather than assumptions; the LLM
handles nuance and multi-topic calls; rules cover deterministic fields cheaply.

### 6.4 `src/sentiment.py` — Task 2: sentiment and trends

| Level | Method |
|---|---|
| Sentence | VADER compound score (−1 to 1) for every sentence, plus which third of the call it is in |
| Call | Provided `sentimentScore` vs our word-weighted VADER mean (`vader_mean`); both z-scored, gaps over 1.5 flagged (`disagreement`) |
| Speaker | Customer (`customer_vader`) separated from AegisCloud staff (`staff_vader`) — the truer measure of customer mood |
| Within call | First third vs last third (`arc`: improved / flat / declined), compared with `positive_pivot` key moments |
| Trend | Sliced by call type, month, period, product and theme (`slice_sentiment`) |

Outputs: `sentence_sentiment.parquet`, `call_sentiment.parquet`, `sentiment_slices.parquet`,
`sentiment_validation.json`. In the dashboard the vendor score is called **confidence** and the
VADER score **transcript language score**.

### 6.5 `src/insights.py` — Task 3: stakeholder insights

| Insight | Stakeholder | Approach |
|---|---|---|
| Account health / churn risk | Sales, Customer Success | Per account: sentiment trend, churn signals, support volume, competitor mentions → ranked list |
| Incident blast radius | Engineering leads, execs | Timeline of every call touching the outage; lag from first war room to customer escalations; affected accounts |
| Voice-of-customer feature gaps | Product managers | Cluster the `feature_gap` key moments into ranked asks, linked to accounts |
| Competitive intelligence | Sales, Product | Competitor mentions (SentinelShield, VaultEdge, CyberNova, Fortiguard Pro) by account, month and stated reason |
| Talk-time / coaching | Sales managers | Customer-vs-rep talk ratio correlated with outcome sentiment |
| Action-item follow-through | Ops, Support | Owner workload; commitments that recur across calls for the same account |

The risk score is a transparent weighted sum (`RISK_WEIGHTS`: churn signals 25, competitor 20,
low latest confidence 20, declining 15, support load 10, outage exposure 10; High ≥ 55,
Medium ≥ 30). Feature-gap clusters are named by Claude (cached) or by their top terms.
Outputs per mode: `account_health`, `incident_timeline`, `feature_gaps`, `feature_gap_moments`,
`competitor_mentions`, `talk_time`, `rep_talk`, `owner_workload`, `recurring_commitments`
(`.parquet`) and `insights_summary.json`. The dashboard recomputes the filter-dependent ones
for the selected period.

### 6.6 `src/incidents.py` — incident register

Major incidents come from internal war-room calls, grouped with follow-up incident calls and
the customer escalations while they were open; every support case is its own incident, linked
to a major incident as a child when it is about it. For each: **reported** (first call start),
**owner** and **owned** (when the owner first spoke), **resolved** (end of the call where the fix
was made or confirmed; children close with their parent), priority, category and outage hours.
Hybrid mode asks Claude to read the linked calls; rules mode uses keyword patterns.
`monthly_availability()` turns platform outage hours into % of month down and availability vs
the SLA stated on the calls (99.5% standard, 99.9% for some accounts).
Output per mode: `incidents.parquet`.

### 6.7 `src/pipeline.py` — runner

`python -m src.pipeline` runs every stage in order and writes all outputs. topics, incidents
and insights are built twice, into `data/processed/rules/` and `data/processed/hybrid/`, so the
dashboard can switch modes instantly. `--rules-only` skips Claude; `--refresh` bypasses the
LLM cache; `--from <stage>` resumes. The dashboard's **Refresh dashboard data** button calls
the same `run()`. `scan_raw()` reports new, still-arriving and unreadable folders in
`data/raw/` for the dashboard's periodic check, leaves out excluded ones, and lists the new ones
whose title or topics hit the non-business keyword rule (`flagged`). Stage order:
load → screen → enrich → topics → sentiment → incidents → insights → export. The export stage writes `outputs/tables/*.csv`
and `outputs/run_summary.json`.

**Logging.** `capture_logs()` wraps every run, from the CLI or the dashboard: INFO-level messages
go to `outputs/logs/pipeline.log` (rotated at 1 MB, three old files kept, timestamps
mm/dd/yyyy) and to any extra handler passed in, which is how the dashboard shows the run live.
A failing stage is logged with its traceback before the error is raised. `read_log_tail()`
returns the last lines for display.

### 6.8 `src/llm.py` — shared Claude access

`parse_cached()` is the only function that calls Claude: structured output against a pydantic
schema, server-side fallbacks on, disk cache, and a usage log entry with tokens and cost for
every real request.

---

## 7. Data model (`data/processed/`)

| File | Grain | Key columns |
|---|---|---|
| `calls.parquet` → `calls_enriched.parquet` | 1 row / call | `call_id`, `title`, `start_time`, `end_time`, `duration_min`, `organizer`, `emails`, `external_domains`, `summary`, `topics`, `action_items`, `sentiment_label`, `sentiment_score`, `pct_*_sent`, `km_<type>` counts, `date`, `month` — plus enrichment: `call_type`, `account`, `products`, `period`, `teams` |
| `sentences.parquet` → `sentences_enriched.parquet` | 1 row / sentence | `call_id`, `idx`, `speaker_name`, `speaker_email`, `start_s`, `end_s`, `duration_s`, `text`, `sentiment`, `confidence` (ASR), `n_words` — plus `speaker_side`, `speaker_team`, `call_type`, `account`, `period` |
| `key_moments.parquet` → `key_moments_enriched.parquet` | 1 row / moment | `call_id`, `time_s`, `type`, `speaker`, `text` |
| `action_items.parquet` → `action_items_enriched.parquet` | 1 row / item | `call_id`, `owner`, `task` |
| `staff_roster_enriched.parquet` | 1 row / AegisCloud person | `email`, `name`, `team`, `basis`, `n_calls` |
| `call_sentiment.parquet` | 1 row / call | `vendor_score`, `vader_mean`, `customer_vader`, `staff_vader`, `side_gap`, `arc_*`, `disagreement*` |
| `<mode>/call_topics.parquet` | 1 row / call | `primary_theme`, `themes`, `certainty`, `rationale`, `source`, `rule_themes`, `cluster`, `competitors` |
| `<mode>/incidents.parquet` | 1 row / incident | `incident_id`, `kind`, `parent_id`, `status`, `priority`, `category`, `reported_at`, `owner`, `owned_at`, `resolved_at`, `outage_hours`, `linked_calls` |
| `data/exclusions.csv` | 1 row / excluded (or force-kept) call | `call_id`, `title`, `date`, `action`, `source`, `reason`, `decided_at` |
| `llm_cache/*.json`, `llm_usage.jsonl` | 1 file / cached answer, 1 line / API call | cached Claude output; tokens and `cost_usd` |

`<mode>` is `hybrid` or `rules`. `call_id` (the meeting id, which is also the raw folder name)
joins every table. Parquet preserves list columns (`topics`, `themes`, `teams`, …), which come
back as NumPy arrays.

---

## 8. LLM usage

| Aspect | Decision |
|---|---|
| Provider / model | Anthropic Claude via the `anthropic` SDK; model name set by `LLM_MODEL` in `.env` |
| Where used | Business / non-business screening; topic classification; incident extraction (status, owner, outage hours); naming feature-gap clusters. All through `src/llm.py`. |
| Output contract | Structured output (`client.beta.messages.parse`) validated with `pydantic`; the SDK retries transient errors; a refusal or invalid answer falls back to the keyword rules for that item |
| Determinism | Low effort, structured output; the cache makes re-runs identical |
| Caching | Each response stored in `data/processed/llm_cache/` (`screen_*`, `topics_*`, `incident_*`, `gap_*`) keyed by a hash of (model, prompt version, input). Re-runs are free and identical. |
| Cost | Every real API call's tokens and cost are logged to `data/processed/llm_usage.jsonl` and shown in the Tool Telemetry tab against a monthly budget. Both budget settings live in `.env`: `LLM_MONTHLY_BUDGET_USD` (USD; default 1000) and `LLM_BUDGET_WARN_SHARE` (0–1, the share at which spend turns red; default 0.8). `src/llm.py` validates them (`_env_number()`), falls back to the default with a warning if one is missing or invalid, and records where each came from in `BUDGET_SOURCE`, which Tool Telemetry shows under the budget bar. Cached re-runs cost nothing. |
| Secrets | `ANTHROPIC_API_KEY` in `.env` (git-ignored); `.env.example` committed. The .exe build copies `.env.example` next to the .exe but never copies `.env`. `.env` is read from next to the .exe (`TI_HOME`) first, then the project root; settings are read at start-up, so a running dashboard needs a restart to pick up changes. |

---

## 9. Presentation layer — `app/app.py` (Streamlit)

Reads only the precomputed files (`@st.cache_data`); never calls the LLM except when the user
presses **Refresh dashboard data**. Global controls: **Config** (Hybrid vs Rules only, SLA target,
check interval), **time period** (quick ranges or custom dates; every number follows it),
filters, and **Data** (new-transcript detection, refresh and the pipeline log). The UI says
"confidence" for the vendor's sentiment score and avoids method names.

**Structure.** `main()` loads the files (`load_base()`, `load_mode()`, `merged_calls()`), applies
the sidebar filters (`sidebar_filters()` → `filter_calls()`), then `build_context()` recomputes
everything filter-dependent (account health, incident timeline, competitor mentions, talk time,
the register for the period, monthly availability, feature asks) into one dict that every
`render_<tab>()` function reads. Chart (`*_chart`) and narrative functions (`headline_insight`,
`accomplishments`, `improvements`, `account_brief`) take DataFrames and return a figure or text,
with no Streamlit calls inside. `main()` only runs when the file is executed by
`streamlit run`, so `notebooks/05_dashboard.ipynb` imports the module and calls the same
functions on the data.

| Tab | Content |
|---|---|
| Overview | Headline insight, accomplishments vs scope for improvement, KPIs, confidence by call type, call mix, outage duration & SLA impact by month, account details with confidence sparklines, account × month heatmap, teams on calls, themes, key moments |
| Topics | Themes by period, confidence by theme, primary theme by call type, call explorer with teams |
| Confidence | Vendor score vs transcript language, customer vs staff, slices, within-call arc, gaps to double-check |
| Accounts | Risk ranking and table (recomputed for the period); click an account for a brief with its action items (copy, .md or .csv download); account drill-down |
| Incidents | Register with reported / owned / resolved times, open incidents with a demo **Call Out** dialog (@user + message, not sent), blast radius of the outage |
| Insights | Feature asks, competitive intelligence, talk time vs outcome, action-item workload |
| Tool Telemetry | LLM cost vs budget, classification quality, discovered clusters, per-call certainty and rationale, incident extraction details, staff roster, pipeline timings |

`TranscriptIntel.exe` (PyInstaller, one-folder) runs the same app with its own copy of the code,
so code changes need a rebuild (`build_exe.ps1`). `run_app.py` sets `TI_HOME`: the project root
while the .exe is in the project's `dist/` folder, otherwise the folder next to the .exe.
`TranscriptIntel.exe --pipeline` runs the pipeline without the UI.

Sidebar: time period (quick ranges and custom dates), call type, product, account search,
reset, and the Data section.

**New-transcript scan.** The Data section is a Streamlit fragment re-run on a timer (Config:
30 s / 1 min / 5 min / off). Each run calls `scan_raw()` in `src/pipeline.py`, which splits
unprocessed folders in `data/raw/` into *new*, *arriving* (changed in the last 30 s, still being
copied) and *unreadable* (skipped by the last run and unchanged since; recorded in
`outputs/run_summary.json`). The scan only notifies (sidebar message plus one pop-up per new
batch). Each new transcript is listed with an **Ignore** button, and keyword hits are marked
**⚠ possibly not business**; Ignore adds the folder to the exclusion list (`source = user`) and
moves its folder from `data/raw/` to `data/exclude/` without running anything. Excluded calls never appear in the
dashboard; the sidebar only shows how many there are; the pipeline runs only when the viewer presses **Refresh dashboard data**, under a lock
shared across browser sessions so one run happens at a time. While it runs, a `LiveLogHandler`
shows the latest log lines in the status box (lines from worker threads appear with the next
line from the script thread); afterwards the **Pipeline log** section shows the last 200 lines
of `outputs/logs/pipeline.log` with a download button. When `run_summary.json` shows newer data
than a page is displaying, that page offers **Load latest data** rather than reloading itself.

**Tables.** Every table goes through `show_table()`, which formats date columns as mm/dd/yyyy
(the date picker uses MM/DD/YYYY too) and labels a missing account, which only internal
meetings have, as "— (internal meeting)" rather than "None".

---

## 10. Technology stack

| Layer | Choice |
|---|---|
| Language / runtime | Python 3.12, `venv` |
| Data | `pandas`, `pyarrow` (parquet) |
| Validation | `pydantic` |
| LLM | `anthropic` SDK |
| Embeddings / ML | `sentence-transformers` (`all-MiniLM-L6-v2`, CPU `torch`), `scikit-learn` (KMeans, TF-IDF). `umap-learn` is listed in `req_item.txt` but not used. |
| Baseline sentiment | `vaderSentiment` |
| Visualization | `plotly` |
| Dashboard | `streamlit` |
| Packaging | `pyinstaller` (`TranscriptIntel.spec`, `build_exe.ps1`) |
| Notebooks | `jupyter`, `ipykernel` |
| Config / secrets | `python-dotenv` |
| Dev tooling | VS Code (Python, Jupyter, Claude Code extensions), Git, private GitHub repo |
| Demo recording | OBS Studio or Windows Game Bar |

**Run it**

```powershell
pip install -r req_item.txt
python -m src.pipeline        # optional: processed outputs are already in data/processed/
streamlit run app/app.py      # or: python run_app.py
```

**Deliberately not used:** AWS/GCP/Azure, databases, Docker (optional extra). They add setup
risk without improving the insights; Section 11 covers the production design instead.
Optional hosting: Streamlit Community Cloud from the GitHub repo.

---

## 11. How this would be productionized

| Concern | Production design |
|---|---|
| **Ingestion** | Transcription vendor webhook → object storage (S3/GCS) → queue (SQS/PubSub). Idempotent processing keyed by `meetingId`. |
| **Processing** | Stateless workers per stage (containers on ECS/Cloud Run or a workflow engine such as Airflow/Step Functions). Each stage writes versioned outputs, as in this repo. |
| **LLM layer** | Batch API for backfills, streaming for near-real-time; prompt and model versioning; response cache; per-tenant cost budgets; fallbacks on provider errors. |
| **Storage** | Warehouse (Snowflake/BigQuery) for the four tables; vector store for embeddings and semantic search ("show me calls like this one"). |
| **Serving** | BI or a web app with role-based views per stakeholder (support, sales, product, engineering). |
| **Reliability** | SLOs on freshness (e.g. call analysed within 15 min) and pipeline success rate; dead-letter queue for failed transcripts; alerting on classifier drift and LLM error rates. |
| **Quality** | Golden labelled set; scheduled evaluation of topic accuracy and sentiment agreement; human review for low-confidence classifications. |
| **Security & privacy** | PII redaction before LLM calls, encryption at rest and in transit, access by role, retention policies, audit logging — calls contain customer security posture and compliance details. |
| **Observability** | Per-stage metrics, traces and cost dashboards. |

Lesson from the data itself: the outage in this dataset was caused by a known single point of
failure and health checks that shared fate with the system they monitored. The production
design avoids both — redundant workers, and pipeline monitoring that is independent of the pipeline.

---

## 12. Key decisions and trade-offs

| Decision | Alternative considered | Why |
|---|---|---|
| Derive call type with rules | LLM classification | Deterministic, explainable, 100% coverage from title prefix + email domain |
| Hybrid topic method | Pure clustering / pure LLM / pure rules | Combines data-grounded discovery with nuanced labelling, and gives a cross-check |
| Keep provided sentiment, add our own | Trust it / ignore it | Lets us measure and report agreement instead of assuming it |
| Separate customer and staff sentiment | Call-level only | Staff speak ~2/3 of the words and their tone barely moves, which dampens the call-level signal; customer language tracks call outcome more closely |
| Precompute, read-only dashboard | Live LLM in the app | Demo reliability, cost, reproducibility |
| Local + GitHub | Cloud deployment | The brief values insight quality, not infrastructure |
| A few labelled synthetic calls | Original data only / many synthetic calls | 13 calls prove new input works and extend the storylines; flagged and kept out of the notebooks' headline numbers so they don't dilute the real narrative |
| Transparent risk score | Trained churn model | No churn outcomes to train on, and an account manager needs to see why an account is flagged |
| Notify-only new-transcript check | Auto-refresh | A viewer's page never changes under them; the pipeline runs only on request |
| Screen out non-business calls before enrichment | Tag them and filter in the dashboard | Nothing downstream (tables, notebooks, CSV exports) can count them by accident; the exclusion list keeps an audit trail and an override |

---

## 13. Limitations

- **Synthetic, single-vendor dataset** over 12 weeks (plus 7 weeks of synthetic follow-on calls):
  trends are illustrative, not statistically robust.
- **Inconsistent facts across transcripts** — e.g. outage duration reported from under 1 hour to
  several days and conflicting start times. Treated as a finding (engineering's 6 hours vs the
  customer's experienced outage), with the post-incident reviews used as the source of truth.
- **Provided sentiment and summaries are vendor-generated** and not independently verified
  beyond our own sentence-level check.
- **No hand-labelled validation set yet.** Themes are validated against the keyword rules and the
  clusters; the template for a human-labelled sample exists in `data/labels/` but hasn't been filled in.
- **Feature-gap groups are re-clustered on every run**, so their names and counts shift as
  transcripts are added.
- **Teams, incident ownership and resolution times are derived from the calls**; there is no CRM or ticketing data.
- **Account names derived from email domains** may not match how the business names accounts.

---

## 14. Build status

| Component | Status |
|---|---|
| Environment setup, `check_setup.py` | Done |
| `src/load.py` | Done — 113 calls, 4,725 sentences, 479 key moments, 456 action items |
| `src/enrich.py` | Done — call type, account (32), products, period, teams (27 staff) |
| `src/topics.py` | Done — 113/113 classified by Claude; Claude's primary among the rule themes for 96% |
| `src/sentiment.py` | Done — vendor vs transcript correlation 0.81 (original 100), 0.74 (all 113) |
| `src/incidents.py` | Done — 33 incidents (hybrid: 1 major, 32 support cases; 26 open), 6 h outage, March SLA breach |
| `src/insights.py`, `src/llm.py` | Done — 10 of 32 accounts high risk |
| `src/screen.py` | Done — keyword rule + Claude is_business, exclusion list, folders moved to data/exclude (moved back on keep); 113/113 existing calls are business |
| `src/pipeline.py` | Done — both modes, new-transcript scan, run log |
| `app/app.py` | Done — 7 tabs, notify-only refresh, live and saved pipeline log |
| `run_app.py`, `TranscriptIntel.spec`, `build_exe.ps1` | Done — `dist/TranscriptIntel/TranscriptIntel.exe` (last built before the pipeline log and the `build_context()` refactor; rebuild to include them) |
| `notebooks/01`–`05` | Done — run without errors; commentary checked 10/29/2026 |
| Chart exports (`outputs/charts/`), slide deck, demo video | Not started |

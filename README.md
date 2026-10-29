# Transcript Intelligence

Turns recorded calls from AegisCloud, a B2B cloud-security company, into a dashboard that
support, sales, product and engineering leaders can act on. It categorizes every call, measures
customer confidence over time, tracks incidents against the SLA, and ranks accounts by churn risk.

The data is 115 business calls from 3 Feb to 1 Jul 2026: 48 customer calls, 32 support cases and
35 internal meetings. 100 are the original dataset (3 Feb – 28 Apr), which includes a major Detect
outage (10–18 Mar) and the Comply v2 launch (7 Apr). 15 are later transcripts (6 May – 1 Jul),
added to test new input and continue the original storylines. One more transcript, a birthday
celebration, was excluded as non-business and is in `data\exclude\`. The notebooks keep their
headline numbers to the original 100; the dashboard and the findings below use all 115.

> **"Confidence"** in the dashboard is the transcription vendor's 1–5 call sentiment score,
> checked against our own reading of the transcript text.

---

## Quick start

Requires Python 3.12 on Windows, macOS or Linux.

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r req_item.txt

python -m src.pipeline              # build every output (about a minute with the LLM cache)
streamlit run app/app.py            # open the dashboard (or: python run_app.py)
```

Processed outputs are already in `data/processed/`, so you can skip the pipeline step and open
the dashboard straight away. Opening the dashboard never runs the pipeline by itself.

### Claude API key (optional)

The **Hybrid** mode uses Claude for screening out non-business calls, theme classification,
incident extraction and naming feature requests. Copy `.env.example` to `.env` in the project
root and fill it in:

```
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-5-5
# LLM spend: monthly budget in USD, and the share of it at which the dashboard turns red
LLM_MONTHLY_BUDGET_USD=1000
LLM_BUDGET_WARN_SHARE=0.8
```

| Setting | What it does | If missing |
|---|---|---|
| `ANTHROPIC_API_KEY` | Enables Hybrid mode | Rules only |
| `LLM_MODEL` | Claude model for every LLM step | `claude-opus-5-5` |
| `LLM_MONTHLY_BUDGET_USD` | Monthly LLM budget shown in Tool Telemetry | 1000 |
| `LLM_BUDGET_WARN_SHARE` | Share of the budget (0–1) at which this month's spend turns red | 0.8 |

Tool Telemetry says whether the budget values came from `.env` or are defaults; a value that
isn't a valid number also falls back to the default, with a warning in the console. Settings
are read when the dashboard or pipeline starts, so restart the dashboard after changing `.env`.

Without a key, everything runs in **Rules only** mode using keyword rules. Claude answers are cached
in `data/processed/llm_cache/`, so re-runs are free; only new transcripts cost anything.
`python check_setup.py` checks the data, libraries and API key in one go.

### Windows app (no Python needed)

`dist\TranscriptIntel\TranscriptIntel.exe` runs the same dashboard in your browser. While it sits
in the project's `dist\` folder it uses the project's own `data\`, `outputs\` and `.env`, so there
is a single `data\raw` for both. If you copy the `TranscriptIntel` folder somewhere else, it uses
the `data\` folder next to the .exe (the build includes a copy); add a `.env` there for Hybrid mode.
`TranscriptIntel.exe --pipeline` runs the pipeline without the dashboard.

The .exe contains its own copy of the code. After changing anything in `src/` or `app/`, rebuild
it with `.\build_exe.ps1` (needs PyInstaller, included in `req_item.txt`), with the .exe closed.

### New transcripts

Drop new meeting folders into `data\raw\` (the sidebar shows the exact path). The dashboard checks
that folder every minute and, when it finds new transcripts, shows a short pop-up and lists them in
the sidebar. It never reloads on its own: press **Refresh dashboard data** to run the pipeline and
load them. In **⚙ Config** you can change how often it checks (30 s, 1 min, 5 min or off).
Folders still being copied wait until they've been unchanged for 30 seconds, and folders that
can't be read are reported in the sidebar. If another tab or the command line has already
processed newer data, the sidebar offers **Load latest data** instead of reloading the page.

### Non-business calls (birthdays, celebrations, socials)

These are kept out of every number, chart and table. Three things catch them:

1. **Flag on arrival.** A new transcript whose title or topics contain a social keyword
   (birthday, celebration, happy hour, farewell, holiday party, …) is marked
   **⚠ possibly not business** in the sidebar. Every new transcript has an **Ignore** button.
2. **Screening in the pipeline.** Right after loading, each call is screened: in Hybrid mode
   Claude answers *is this business?* (cached, so once per call); otherwise the keyword rule
   decides. Non-business calls are dropped before any analysis.
3. **Exclusion list.** Ignored and screened-out calls are recorded in `data\exclusions.csv`
   (with who decided and why) and their folder is **moved** out of `data\raw\` into
   `data\exclude\`. If the same folder is dropped into `data\raw\` again, it's skipped and
   moved out again on the next run.

To bring a call back, set its `action` to `keep` in `data\exclusions.csv` and refresh: the
folder is moved back to `data\raw\`, loaded again, and never screened out. The keyword list is `NON_BUSINESS` in `src/screen.py`.

### Pipeline logs

Every pipeline run, from the dashboard or the command line, is logged to
`outputs\logs\pipeline.log` (a new file starts at 1 MB; three old ones are kept). During a
dashboard refresh the latest lines are shown live; afterwards the sidebar's **Pipeline log**
section shows the last 200 lines and offers a download. If a refresh fails, the error and its
details are in the log.

### Changing the rules

Thresholds and keyword rules are constants at the top of each `src/` module, for example the
themes in `src/topics.py` (`THEMES`), the risk weights in `src/insights.py` (`RISK_WEIGHTS`) or
the disagreement threshold in `src/sentiment.py` (`DISAGREE_Z`). Edit the value, run
`python -m src.pipeline` (or `--from <stage>` to start at the stage that uses it), then press
**Load latest data** in the dashboard. In the dashboard itself, ⚙ Config sets only the mode, the
SLA target and the check interval.

Dates in the dashboard are shown as mm/dd/yyyy.

---

## Key findings

All 115 business calls, as the dashboard shows them with *All time* selected (checked 10/29/2026).

| Finding | Evidence |
|---|---|
| **The Detect outage was the defining event of the period.** | Confidence fell from 3.78 before the outage to 2.18 during it. 51 of 115 calls dealt with it or its fallout, and 24 customer accounts were affected. |
| **It breached the SLA.** | 6 hours of platform downtime in March against 3.7 hours allowed at 99.5% uptime. SLA or service credits came up on 8 calls. |
| **It opened the door to competitors.** | 10 affected accounts named a competitor afterwards. SentinelShield came up on 28 calls, nearly twice as many as VaultEdge (16), mostly on reliability and pricing. |
| **Customer calls recovered; support calls didn't keep up.** | Support averages 2.93 against 3.75 for external calls, and is still the lowest call type after the outage (3.02 vs 3.85). Many of the later support cases were side effects of the outage fixes. |
| **10 of 33 accounts are high churn risk.** | Summit, Silverline, Ridgeline, Meridian, Northstar and Brightpath lead, each combining repeated churn signals, outage exposure and usually a named competitor or falling confidence. |
| **The Comply v2 launch landed well; customers want more of it.** | Average confidence 3.76 across 24 calls since GA. The most requested capability is compliance reporting and mapping across frameworks: the top ask (9 accounts) and two other groups, together 18 of 33 accounts. |
| **When customers talk more, calls go worse.** | Calls where the customer has more than about 42% of the talk time average 3.21, against 3.66 for the rest: long customer turns usually mean a complaint. The link is weak call by call, so it's a triage signal, not a rule. |
| **The vendor's confidence score is usable for trends.** | It agrees with our own transcript scoring (0.72 correlation across all 115 calls, 0.81 on the original 100); 6 calls disagree enough to double-check. |

---

## The dashboard

Global controls sit on every tab: **⚙ Config** switches between Hybrid and Rules only and sets the
SLA target and check interval; the sidebar sets the **time period** (every number follows it),
call type, product and account filters; and **Data** shows new transcripts, **Refresh dashboard
data** and the pipeline log. In tables, calls from internal meetings show "— (internal meeting)"
as their account.

| Tab | What it answers |
|---|---|
| **Overview** | What happened this period? Headline insight, accomplishments vs areas to improve, confidence by call type, outage duration and SLA impact by month, account details and teams on calls |
| **Topics** | What are calls about? 10 themes by period, confidence by theme, call explorer |
| **Confidence** | How do customers feel, and can we trust the score? Customer vs staff, slices, how calls end vs start |
| **Accounts** | Which accounts are at risk and why? Risk ranking, click an account for a copyable brief with its action items |
| **Incidents** | What broke, who owns it, is it fixed? Register with reported / owned / resolved times, a demo **Call Out** for open incidents, outage blast radius |
| **Insights** | What should product and sales do? Feature asks, competitor intelligence, talk time, action-item workload |
| **Tool Telemetry** | How well is the tooling working and what does it cost? LLM spend vs monthly budget, classification quality, clusters, per-call certainty |

---

## How it works

```
data/raw/ ─► load ─► screen ─► enrich ─► topics ─► sentiment ─► incidents ─► insights ─► export
            flatten  drop non-  call type,  themes     vendor vs     register,    account     CSVs and
            to       business   account,    (Claude +  transcript,   SLA          risk, VoC,  run
            tables   calls      teams,      rules)     customer vs                competitors summary
                                period                 staff, arcs
```

| Step | Module | Approach |
|---|---|---|
| Load | `src/load.py` | Reads each meeting's six JSON files into four tables: calls, sentences, key moments, action items |
| Screen | `src/screen.py` | Drops non-business calls (Claude's is_business answer, keyword rule as fallback) and records them in the exclusion list |
| Enrich | `src/enrich.py` | Rules for call type, account, product, period and each person's team (from role mentions and meetings attended) |
| Topics | `src/topics.py` | Calls grouped by similarity to ground a 10-theme taxonomy, then Claude assigns themes with keyword rules as fallback and check |
| Sentiment | `src/sentiment.py` | Every sentence scored, customer and staff scored separately, first vs last third of each call compared |
| Incidents | `src/incidents.py` | War-room calls and support cases become incidents with owner, timings, outage hours and monthly availability |
| Insights | `src/insights.py` | Transparent risk score per account, feature-gap clusters, competitor mentions with reasons, talk time |
| Run all | `src/pipeline.py` | Runs every stage for both modes and logs the run; `--rules-only`, `--refresh` (ignore cache) and `--from <stage>` |
| LLM access | `src/llm.py` | One place for Claude calls: structured output, caching, fallbacks and cost logging |

The dashboard only reads precomputed files, so a demo never depends on a live API call. What
depends on the filters (account risk, incident timeline, SLA, competitors, talk time) is
recomputed from the filtered calls by `build_context()` in `app/app.py`. Design decisions, data
model and limitations are in [docs/architecture.md](docs/architecture.md).

---

## Notebooks

Each notebook reads the pipeline's saved outputs, makes no LLM calls, and runs top to bottom
(**Restart → Run All**) after `python -m src.pipeline`.

| Notebook | Question it answers |
|---|---|
| `01_explore` | What's in the data, is it trustworthy, which calls are screened out as non-business, and how are the derived fields built? |
| `02_topics` | What are the calls about, and should we trust the labels? (Task 1) |
| `03_sentiment` | How do people feel, and what's changing? (Task 2) |
| `04_insights` | What should each stakeholder do? (Task 3) |
| `05_dashboard` | How does `app/app.py` work? Runs the dashboard's own functions on the data |

The written commentary was checked against the data on 10/29/2026 (115 calls, 1 excluded); tables and charts
always reflect the current data.

**PDF copies:** `python notebooks/export_pdfs.py` runs every notebook and saves it as a PDF in
`outputs/notebooks_pdf/` (or name notebooks, e.g. `python notebooks/export_pdfs.py 02_topics`).
Charts load Plotly from its CDN, so it needs an internet connection; the first time, you may need
`python -m playwright install chromium`.

---

## Project layout

```
app/app.py              Streamlit dashboard
src/                    pipeline modules (above)
data/raw/               115 transcript folders (source; excluded calls are moved out)
data/processed/         tables, per-mode outputs (hybrid/, rules/), LLM cache and usage log
data/labels/            hand-labelling template for checking topics (02_topics, 8c)
data/exclusions.csv     non-business calls kept out of everything (created on first exclusion)
data/exclude/           excluded transcript folders (moved here from data/raw/)
outputs/tables/         CSV exports for slides; outputs/run_summary.json
outputs/logs/           pipeline.log
notebooks/              01–05 (above)
docs/architecture.md    architecture and decisions
check_setup.py          environment check
run_app.py              launcher, and the entry point of TranscriptIntel.exe
TranscriptIntel.spec    PyInstaller build; build_exe.ps1 wraps it
```

## Limitations

- Teams, incident ownership and resolution times are derived from the calls; there is no CRM or
  ticketing data, so "time to own" means when the owner first spoke on the call, and a case only
  counts as resolved if a call says so.
- Account names come from email domains, which gives 33 accounts.
- 15 of the 115 calls are later, synthetic transcripts. They test that new input works, but findings that include
  them partly reflect data written for this project.
- Feature-request groups are re-clustered on every run, so their names and counts can shift as
  transcripts are added.
- The .exe leaves out the local embedding model to stay small; when it re-runs the pipeline, call
  grouping uses keyword similarity instead.
- LLM spend from before cost logging was added isn't counted in Tool Telemetry.

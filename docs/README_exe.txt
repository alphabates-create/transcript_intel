TRANSCRIPT INTELLIGENCE - USER GUIDE
====================================

Transcript Intelligence turns recorded call transcripts into a dashboard for support,
sales, product and engineering leaders. It groups calls by theme, measures sentiment,
ranks accounts by churn risk, tracks incidents against the SLA, and surfaces feature
requests and competitor mentions.

Everything runs on this computer. No installation and no Python are needed.


REQUIREMENTS
------------
- Windows 10 or 11, 64-bit
- A web browser (Edge, Chrome or Firefox)
- About 400 MB of free disk space
- Optional: an Anthropic API key, only if you want Claude to analyse new transcripts
  (see "Using Claude" below)


QUICK START
-----------
1. If you received a zip file: right-click it > Properties > tick "Unblock" > OK.
   Then extract it.

2. Put the TranscriptIntel folder somewhere you can write to, for example
   C:\TranscriptIntel or your Documents folder.
   - Do NOT put it in "Program Files": the app saves its results in its own folder.
   - Keep everything together. TranscriptIntel.exe needs the "_internal", "data" and
     "outputs" folders next to it.

3. Double-click TranscriptIntel.exe.
   - If Windows shows "Windows protected your PC", click "More info" > "Run anyway".
     (The app isn't code-signed, so Windows asks the first time.)
   - If a firewall prompt appears, you can allow it or cancel. The dashboard only runs
     on this computer either way.

4. Wait 20-60 seconds the first time. A black console window opens, then the dashboard
   opens in your browser. If the browser doesn't open by itself, copy the address shown
   in the console window (usually http://localhost:8501) into your browser.

5. Keep the console window open while you use the dashboard. Closing it stops the app.


USING THE DASHBOARD
-------------------
Left sidebar
  TIME PERIOD   Quick ranges (All time, Pre-outage, Outage, Recovery, Post-Comply-launch,
                Last 30 days) or your own dates.
  FILTERS       Call type (External, Support, Internal), Product, and Account search.
                Every tab follows these filters.
  DATA          New-transcript alerts and the "Refresh dashboard data" button
                (see "Adding new transcripts").

Config (gear button at the top)
  Analysis mode   Hybrid (Claude + keyword rules) or Rules only (keyword rules).
                  Both come pre-built.
  SLA target      99.5% (standard) or 99.9% uptime.
  New transcripts How often to check for new transcripts: every 30 seconds, every
                  minute (default), every 5 minutes, or off.

Tabs
  Overview        Headline numbers, call mix and the sentiment trend.
  Topics          Themes by call type and over time, plus a call explorer.
  Confidence      The vendor's sentiment score vs the transcript language, customers
                  vs staff, and whether calls improved from start to finish.
  Accounts        Churn-risk ranking. Click an account name for a brief you can copy
                  or download, with its risk drivers and open commitments.
  Incidents       Incident register (reported / owned / resolved), monthly availability
                  against the SLA, and the outage's blast radius. "Call Out" drafts a
                  status request to an incident's owner. (Demo only: nothing is sent.)
  Insights        Feature requests, competitor mentions, and coaching views.
  Tool Telemetry  Claude spend against the monthly budget, classification quality and
                  pipeline runs.


ADDING NEW TRANSCRIPTS
----------------------
1. Copy one or more meeting folders into the "data\raw" folder next to
   TranscriptIntel.exe. Each meeting folder must contain these six files:
       meeting-info.json   summary.json   transcript.json
       speakers.json       speaker-meta.json   events.json

2. Within the check interval (one minute by default), the sidebar shows
   "N new transcripts available" with each call's title, and a pop-up appears.
   - "Still arriving": the folder changed in the last 30 seconds, so the app waits for
     the copy to finish.
   - "Couldn't be read": a file is missing or isn't valid JSON. Fix it and it's retried
     automatically.
   - "Possibly not business": the title or topics look like a social call (birthday,
     farewell, celebration...). Click "Ignore" to keep it out of every analysis.

3. Click "Refresh dashboard data". The analysis re-runs (usually under a minute) and
   the dashboard reloads with the new calls.

Non-business calls
  Calls screened out as non-business are listed in data\exclusions.csv, and their
  folders are moved to data\exclude\. To bring one back, open exclusions.csv, change
  its "action" from "exclude" to "keep", save, and click Refresh. The call is moved back
  and won't be screened out again.


USING CLAUDE (OPTIONAL)
-----------------------
Without an API key, everything still works: you can view both pre-built modes, and
Refresh re-analyses new transcripts with keyword rules only.

To let Refresh use Claude (Hybrid mode):
1. In the TranscriptIntel folder, make a copy of ".env.example" and name it ".env".
2. Open .env in Notepad and fill in your key:
       ANTHROPIC_API_KEY=your-key-here
   Optional settings in the same file:
       LLM_MODEL                 the Claude model to use
       LLM_MONTHLY_BUDGET_USD    monthly spending budget (default 1000)
       LLM_BUDGET_WARN_SHARE     share of the budget at which Tool Telemetry turns red
                                 (default 0.8 = 80%)
3. Restart TranscriptIntel.exe. In the sidebar, tick "Use Claude for Hybrid mode".

Claude's answers are cached, so only new transcripts cost anything.

Tip: if you refresh WITHOUT a key, only Rules mode is updated. To see the new calls,
set Config > Analysis mode to "Rules only".


RUNNING THE ANALYSIS WITHOUT THE DASHBOARD
------------------------------------------
Open a terminal in the TranscriptIntel folder (in File Explorer: type "cmd" in the
address bar and press Enter), then run:

    TranscriptIntel.exe --pipeline               (both modes; uses Claude if a key is set)
    TranscriptIntel.exe --pipeline --rules-only  (keyword rules only, no Claude)

Results are written to data\processed\ and the CSV exports to outputs\tables\.


WHAT'S IN THIS FOLDER
---------------------
    TranscriptIntel.exe     the app
    _internal\              files the app needs (don't change or move)
    data\raw\               transcript folders: drop new ones here
    data\processed\         analysis results (rebuilt by Refresh)
    data\exclude\           non-business transcripts that were screened out
    data\exclusions.csv     the list of screened-out calls (created when needed)
    outputs\tables\         CSV exports (account health, incidents, themes, ...)
    outputs\run_summary.json   details of the last analysis run
    .env.example            settings template for using Claude
    README.txt              this guide


STOPPING THE APP
----------------
Close the console window, or click it and press Ctrl+C. Closing the browser tab alone
doesn't stop the app.


TROUBLESHOOTING
---------------
Nothing happens, or antivirus removes the .exe
    Allow TranscriptIntel.exe in your antivirus, or unblock the zip (Quick start, step 1)
    and extract it again.

Browser says "can't connect"
    Wait until the console window shows the address, then refresh the page.

Dashboard says there's no processed data
    The "data" folder isn't next to TranscriptIntel.exe. Extract the whole zip again,
    or click "Refresh dashboard data" to rebuild it.

"Port already in use" or a different address than 8501
    Another copy is already running. Close the other console window, or use the address
    the console shows.

A refresh fails
    The sidebar shows the error. Earlier results are kept, so the dashboard still works.
    Check the folders listed as "couldn't be read".

The "Use Claude" box is greyed out
    No API key was found. See "Using Claude" and restart the app.


ABOUT THE DATA AND LIMITATIONS
------------------------------
- The included calls come from the assignment dataset (3 Feb - 28 Apr 2026), plus
  synthetic calls written to test new input and continue the storylines. Synthetic
  calls are clearly labelled in the project and excluded from the headline findings.
- To keep the app small, it doesn't include the local embedding model. When you
  Refresh, calls are grouped by keyword similarity instead, so topic clusters can differ
  slightly from the pre-built results.
- Sentiment scores and summaries come partly from the transcription vendor. The app
  checks them against its own scoring and flags large disagreements.
- Transcripts cover only part of each meeting, so time-based figures (such as talk
  time) are samples rather than full-meeting totals.

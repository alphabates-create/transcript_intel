"""
run_app.py — Launcher for the dashboard, and the entry point of TranscriptIntel.exe.

    python run_app.py                          # same as: streamlit run app/app.py
    TranscriptIntel.exe                        # packaged build (see build_exe.ps1)
    TranscriptIntel.exe --pipeline [options]   # run the pipeline without the UI (same options
                                               # as python -m src.pipeline, e.g. --rules-only)

Where the data lives: a build still inside the project's dist/ folder reads the project's own
data/, outputs/ and .env, so there is one data/raw/ for both `streamlit run` and the .exe.
A copy moved anywhere else keeps its data next to the .exe. Either way the dashboard scans
data/raw/ on a timer and runs the pipeline when new transcripts arrive:

    TranscriptIntel/
        TranscriptIntel.exe
        .env                  ANTHROPIC_API_KEY=..., LLM_MODEL=...  (optional; Rules mode works without it)
        data/raw/             transcript folders
        data/processed/       pipeline outputs (shipped pre-built)
        outputs/              CSV exports and run summary
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)


def _home() -> Path:
    """
    Folder holding data/, outputs/ and .env. A build still sitting in the project's dist/
    folder uses the project's data, so there is only one data/raw to drop transcripts into;
    a copy moved anywhere else uses its own folder.
    """
    if not FROZEN:
        return Path(__file__).resolve().parent
    exe_dir = Path(sys.executable).parent
    project = exe_dir.parent.parent
    if exe_dir.parent.name == "dist" and (project / "data" / "raw").is_dir() and (project / "src").is_dir():
        return project
    return exe_dir


HOME = _home()
BUNDLE = Path(getattr(sys, "_MEIPASS", HOME))
os.environ.setdefault("TI_HOME", str(HOME))   # read by src/load.py: data/, outputs/, .env


def main() -> int:
    # Imported here so PyInstaller bundles everything app.py needs (Streamlit runs app.py as a
    # script, so the bundler can't see its imports on its own).
    import plotly.express  # noqa: F401
    import plotly.graph_objects  # noqa: F401
    import src.enrich  # noqa: F401
    import src.incidents  # noqa: F401
    import src.insights  # noqa: F401
    import src.llm  # noqa: F401
    import src.load  # noqa: F401
    import src.pipeline  # noqa: F401
    import src.sentiment  # noqa: F401
    import src.topics  # noqa: F401
    if "--pipeline" in sys.argv:
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--pipeline"]
        return src.pipeline.main()

    from streamlit.web import cli as stcli

    # Opens the browser by default; set STREAMLIT_SERVER_HEADLESS=true to run without one.
    sys.argv = [
        "streamlit", "run", str(BUNDLE / "app" / "app.py"),
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())

"""
export_pdfs.py — run every notebook and save it as a PDF in outputs/notebooks_pdf/.

nbconvert's own `--to webpdf` fails on Windows (it switches asyncio to the selector event loop,
which can't start the headless browser), so this does it in two steps: nbconvert executes each
notebook into HTML, then Playwright's Chromium prints the HTML to PDF. Plotly charts load
plotly.js from its CDN, so the export needs an internet connection.

Usage (from the project root, .venv active):
    python notebooks/export_pdfs.py                  # all notebooks
    python notebooks/export_pdfs.py 02_topics        # just the ones named
First time only:  python -m playwright install chromium
"""

import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

NOTEBOOKS = Path(__file__).resolve().parent
OUT = NOTEBOOKS.parent / "outputs" / "notebooks_pdf"


def to_html(nb: Path) -> Path:
    """Execute the notebook and write HTML next to the PDFs (the notebook file is not changed)."""
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "--execute",
                    "--ExecutePreprocessor.timeout=600", "--output-dir", str(OUT), str(nb)],
                   check=True, cwd=NOTEBOOKS)
    return OUT / f"{nb.stem}.html"


def main() -> int:
    wanted = set(sys.argv[1:])
    notebooks = [p for p in sorted(NOTEBOOKS.glob("*.ipynb")) if not wanted or p.stem in wanted]
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        for nb in notebooks:
            print(f"{nb.name}: running…", flush=True)
            html = to_html(nb)
            page.goto(html.as_uri(), wait_until="networkidle")
            page.wait_for_timeout(1500)   # let Plotly finish drawing
            pdf = OUT / f"{nb.stem}.pdf"
            page.pdf(path=str(pdf), format="A4", landscape=True, print_background=True,
                     margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"})
            html.unlink()
            print(f"  → {pdf}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Verifies data, libraries, and the LLM API are all working."""
import os, json, pathlib
from dotenv import load_dotenv

load_dotenv()
raw = pathlib.Path("data/raw")

# 1. Data
folders = [p for p in raw.iterdir() if p.is_dir()]
sample = json.loads((folders[0] / "meeting-info.json").read_text(encoding="utf-8"))
print(f"[OK] {len(folders)} transcripts found. Sample: {sample['title']}")

# 2. Libraries
import pandas, sklearn, plotly, streamlit
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")   # downloads ~90 MB on first run
print(f"[OK] Embedding dim: {model.encode(['hello']).shape[1]}")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
print(f"[OK] VADER: {SentimentIntensityAnalyzer().polarity_scores('This outage is terrible')['compound']}")

# 3. LLM
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
resp = client.messages.create(
    model=os.getenv("LLM_MODEL"),
    max_tokens=20,
    messages=[{"role": "user", "content": "Reply with just: setup ok"}],
)
print(f"[OK] LLM: {resp.content[0].text}")
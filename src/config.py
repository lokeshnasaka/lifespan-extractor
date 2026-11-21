"""Paths and settings shared by the pipeline."""

import tomllib
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DRUGAGE_CSV = DATA_DIR / "drugage.csv"
DB_PATH = DATA_DIR / "extractions.db"

load_dotenv(ROOT / ".env")

with open(ROOT / "config.toml", "rb") as f:
    _cfg = tomllib.load(f)

PROVIDER: str = _cfg["llm"].get("provider", "anthropic")
MODEL: str = _cfg["llm"]["model"]
MAX_TOKENS: int = _cfg["llm"]["max_tokens"]
EFFORT: str = _cfg["llm"].get("effort", "high")
OLLAMA_URL: str = _cfg["llm"].get("ollama_url", "http://localhost:11434")
NUM_CTX: int = _cfg["llm"].get("num_ctx", 32768)

EVAL_N_PAPERS: int = _cfg["eval"]["n_papers"]
EVAL_SEED: int = _cfg["eval"]["seed"]
LIFESPAN_TOLERANCE_PCT: float = _cfg["eval"]["lifespan_tolerance_pct"]

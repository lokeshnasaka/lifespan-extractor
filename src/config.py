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

MODEL: str = _cfg["llm"]["model"]
EFFORT: str = _cfg["llm"]["effort"]
MAX_TOKENS: int = _cfg["llm"]["max_tokens"]

EVAL_N_PAPERS: int = _cfg["eval"]["n_papers"]
EVAL_SEED: int = _cfg["eval"]["seed"]
LIFESPAN_TOLERANCE_PCT: float = _cfg["eval"]["lifespan_tolerance_pct"]

"""Download DrugAge and save a clean copy to data/drugage.csv.

Run:  python -m src.drugage
"""

import io
import zipfile

import pandas as pd
import requests

from src.config import DATA_DIR, DRUGAGE_CSV

DRUGAGE_ZIP_URL = "https://genomics.senescence.info/drugs/dataset.zip"
RAW_DIR = DATA_DIR / "raw" / "drugage"
PAPER_ID_COLUMN = "pubmed_id"  # links each DrugAge entry to its source paper


def download_raw() -> pd.DataFrame:
    """Return the raw DrugAge table, downloading the zip only if it is not on disk."""
    raw_csv = RAW_DIR / "drugage.csv"
    if not raw_csv.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(DRUGAGE_ZIP_URL, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw_csv.write_bytes(zf.read("drugage.csv"))
    # keep_default_na=False: the significance columns use the literal string "NA"
    return pd.read_csv(raw_csv, dtype=str, keep_default_na=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for col in df.columns:
        df[col] = df[col].str.strip()
    df = df.replace({"": pd.NA, "NA": pd.NA})

    for col in ("avg_lifespan_change_percent", "max_lifespan_change_percent", "weight_change_percent"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[PAPER_ID_COLUMN] = pd.to_numeric(df[PAPER_ID_COLUMN], errors="coerce").astype("Int64")

    df = df.dropna(subset=[PAPER_ID_COLUMN, "compound_name", "species"])
    return df.drop_duplicates().reset_index(drop=True)


def load_drugage() -> pd.DataFrame:
    """Load the clean CSV written by main()."""
    return pd.read_csv(DRUGAGE_CSV, dtype={PAPER_ID_COLUMN: "Int64"})


def main() -> None:
    raw = download_raw()
    print("Columns:", list(raw.columns))
    print(f"Paper link column: {PAPER_ID_COLUMN}")
    df = clean(raw)
    df.to_csv(DRUGAGE_CSV, index=False)
    print(f"Raw rows: {len(raw)}  clean rows: {len(df)}  papers: {df[PAPER_ID_COLUMN].nunique()}")
    print(f"Saved {DRUGAGE_CSV}")


if __name__ == "__main__":
    main()

"""Extract lifespan experiments from a paper with the LLM and store them in SQLite.

Run:  python -m src.extract            (extracts the eval sample)
      python -m src.extract 12345 678  (extracts specific PubMed IDs)
"""

import os
import sqlite3
import sys
from datetime import datetime, timezone

import anthropic

from src.config import DB_PATH, EFFORT, EVAL_N_PAPERS, EVAL_SEED, MAX_TOKENS, MODEL
from src.fetch import Paper, fetch_paper
from src.schema import ExtractionResult, LifespanExperiment

SYSTEM_PROMPT = """\
You extract lifespan experiments from aging research papers into structured records.

A lifespan experiment is one comparison of a treated group against its control: one \
compound, in one species, strain and sex, at one dose. Return one record per experiment. \
If the paper tests several compounds, doses, strains or sexes, return one record for each.

Rules:
- Extract only. Every value must be stated in the paper text you are given.
- Do not guess, infer from background knowledge, or calculate values the paper does not \
state. If a value is not in the text, use null.
- avg_lifespan_change_pct and max_lifespan_change_pct are percent changes versus control. \
Use the paper's own percentage. Use a negative number when lifespan was shortened.
- significant is true or false only when the paper states the statistical result for \
that comparison (a p-value, or wording such as "significantly"). Otherwise null.
- evidence_quote must be copied word for word from the paper: the sentence (or table \
row) that supports the numbers in the record.
- Only include experiments where lifespan or survival of the whole organism was measured \
after giving a compound. Skip genetic-only interventions and healthspan-only results.
- If the text contains no such experiment, return an empty list.
"""

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    pmid INTEGER NOT NULL,
    model TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    source TEXT NOT NULL            -- 'full_text', 'abstract' or 'title'
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    pmid INTEGER NOT NULL,
    model TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    compound TEXT NOT NULL,
    species TEXT NOT NULL,
    strain TEXT,
    sex TEXT,
    dose TEXT,
    avg_lifespan_change_pct REAL,
    max_lifespan_change_pct REAL,
    n_animals INTEGER,
    significant INTEGER,
    evidence_quote TEXT NOT NULL
);
"""
_FIELDS = list(LifespanExperiment.model_fields)


class ExtractionError(Exception):
    pass


def make_client() -> anthropic.Anthropic:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return anthropic.Anthropic()


def extract_experiments(paper: Paper, client: anthropic.Anthropic) -> list[LifespanExperiment]:
    """Send the paper text to the LLM and return the experiments it found."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<paper>\n{paper.text}\n</paper>"}],
        output_format=ExtractionResult,
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise ExtractionError(f"PMID {paper.pmid}: stop_reason={response.stop_reason}")
    return response.parsed_output.experiments


def connect(path=DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_SQL)
    return conn


def save_run(
    conn: sqlite3.Connection, pmid: int, model: str, source: str, experiments: list[LifespanExperiment]
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        run_id = conn.execute(
            "INSERT INTO runs (pmid, model, extracted_at, source) VALUES (?, ?, ?, ?)",
            (pmid, model, now, source),
        ).lastrowid
        conn.executemany(
            f"INSERT INTO experiments (run_id, pmid, model, extracted_at, {', '.join(_FIELDS)}) "
            f"VALUES (?, ?, ?, ?, {', '.join('?' * len(_FIELDS))})",
            [(run_id, pmid, model, now, *e.model_dump().values()) for e in experiments],
        )


def load_latest(conn: sqlite3.Connection, pmid: int, model: str) -> list[LifespanExperiment] | None:
    """Experiments from the newest run for this paper and model, or None if never extracted."""
    run = conn.execute(
        "SELECT id FROM runs WHERE pmid = ? AND model = ? ORDER BY id DESC LIMIT 1", (pmid, model)
    ).fetchone()
    if run is None:
        return None
    rows = conn.execute(
        f"SELECT {', '.join(_FIELDS)} FROM experiments WHERE run_id = ? ORDER BY id", (run[0],)
    ).fetchall()
    out = []
    for row in rows:
        rec = dict(zip(_FIELDS, row))
        if rec["significant"] is not None:
            rec["significant"] = bool(rec["significant"])
        out.append(LifespanExperiment(**rec))
    return out


def extract_paper(
    pmid: int, conn: sqlite3.Connection, client: anthropic.Anthropic
) -> list[LifespanExperiment] | None:
    """Extract one paper, reusing a stored run for the configured model if there is one.

    Returns None when Europe PMC does not have the paper.
    """
    stored = load_latest(conn, pmid, MODEL)
    if stored is not None:
        return stored
    paper = fetch_paper(pmid)
    if paper is None:
        return None
    experiments = extract_experiments(paper, client)
    source = "full_text" if paper.full_text else "abstract" if paper.abstract else "title"
    save_run(conn, pmid, MODEL, source, experiments)
    return experiments


def main() -> None:
    from src.drugage import sample_pmids

    pmids = [int(a) for a in sys.argv[1:]] or sample_pmids(EVAL_N_PAPERS, EVAL_SEED)
    conn = connect()
    client = make_client()
    for pmid in pmids:
        try:
            experiments = extract_paper(pmid, conn, client)
        except ExtractionError as e:
            print(f"FAILED {e}")
            continue
        n = "not in Europe PMC" if experiments is None else f"{len(experiments)} experiments"
        print(f"{pmid}: {n}")
    print(f"\nModel: {MODEL}  database: {DB_PATH}")


if __name__ == "__main__":
    main()

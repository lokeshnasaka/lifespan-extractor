from src.extract import connect, load_latest, save_run
from src.schema import ExtractionResult, LifespanExperiment


def make(**overrides) -> LifespanExperiment:
    base = dict(
        compound="Rapamycin",
        species="Mus musculus",
        strain=None,
        sex="Female",
        dose="14 ppm",
        avg_lifespan_change_pct=14.0,
        max_lifespan_change_pct=None,
        n_animals=None,
        significant=True,
        evidence_quote="Rapamycin extended median lifespan by 14% in females.",
    )
    return LifespanExperiment(**(base | overrides))


def test_parses_llm_json_with_nulls():
    raw = (
        '{"experiments": [{"compound": "Metformin", "species": "Caenorhabditis elegans", '
        '"strain": "N2", "sex": null, "dose": "50 mM", "avg_lifespan_change_pct": 36, '
        '"max_lifespan_change_pct": null, "n_animals": null, "significant": null, '
        '"evidence_quote": "Metformin at 50 mM increased mean lifespan by 36%."}]}'
    )
    [exp] = ExtractionResult.model_validate_json(raw).experiments
    assert exp.compound == "Metformin"
    assert exp.avg_lifespan_change_pct == 36.0
    assert exp.sex is None and exp.significant is None


def test_sqlite_round_trip(tmp_path):
    conn = connect(tmp_path / "t.db")
    exps = [make(), make(sex="Male", avg_lifespan_change_pct=9.0, significant=False)]
    save_run(conn, 19587680, "model-a", "abstract", exps)
    assert load_latest(conn, 19587680, "model-a") == exps

    row = conn.execute("SELECT pmid, model, extracted_at FROM experiments").fetchone()
    assert row[:2] == (19587680, "model-a") and row[2]


def test_load_latest_distinguishes_empty_run_from_no_run(tmp_path):
    conn = connect(tmp_path / "t.db")
    assert load_latest(conn, 1, "model-a") is None
    save_run(conn, 1, "model-a", "abstract", [])
    assert load_latest(conn, 1, "model-a") == []
    assert load_latest(conn, 1, "model-b") is None


def test_load_latest_returns_newest_run(tmp_path):
    conn = connect(tmp_path / "t.db")
    save_run(conn, 1, "model-a", "abstract", [make(compound="Old")])
    save_run(conn, 1, "model-a", "abstract", [make(compound="New")])
    assert [e.compound for e in load_latest(conn, 1, "model-a")] == ["New"]

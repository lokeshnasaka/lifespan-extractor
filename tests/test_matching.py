from evals.matching import (
    compound_level_counts,
    compounds_match,
    field_accuracy,
    lifespan_correct,
    match_paper,
    norm_compound,
    quote_in_text,
    species_key,
)
from evals.run_eval import worst_errors
from src.schema import LifespanExperiment


def truth(**overrides) -> dict:
    base = dict(
        compound_name="Rapamycin",
        species="Mus musculus",
        strain="UM-HET3",
        dosage="14 ppm",
        avg_lifespan_change_percent=14.0,
        avg_lifespan_significance="S",
        max_lifespan_change_percent=None,
        gender="Female",
    )
    return base | overrides


def pred(**overrides) -> LifespanExperiment:
    base = dict(
        compound="rapamycin",
        species="Mus musculus",
        strain="UM-HET3",
        sex="female",
        dose="14 ppm",
        avg_lifespan_change_pct=14.0,
        max_lifespan_change_pct=None,
        n_animals=None,
        significant=True,
        evidence_quote="q",
    )
    return LifespanExperiment(**(base | overrides))


def test_norm_compound_handles_case_punctuation_and_greek():
    assert norm_compound("Alpha‐linolenic acid") == norm_compound("α-Linolenic Acid")


def test_compounds_match_allows_containment_but_not_short_or_different_names():
    assert compounds_match("Lithium Chloride", "lithium")
    assert compounds_match("Rhodiola rosea extract", "Rhodiola rosea")
    assert not compounds_match("Sodium Chloride", "Potassium Chloride")
    assert not compounds_match("Glucose", "D-glucosamine")
    assert not compounds_match("NAD", "NADH")


def test_species_key_abbreviations_and_common_names():
    assert species_key("Caenorhabditis elegans") == species_key("C. elegans")
    assert species_key("mice") == species_key("Mus musculus")
    assert species_key("Drosophila melanogaster") != species_key("Drosophila mojavensis")


def test_match_requires_compound_and_species():
    m = match_paper([truth()], [pred(species="Rattus norvegicus"), pred(compound="Metformin")])
    assert m.pairs == [] and len(m.missed) == 1 and len(m.extra) == 2


def test_match_is_one_to_one_and_prefers_same_sex():
    rows = [truth(gender="Male", avg_lifespan_change_percent=9.0), truth(gender="Female")]
    preds = [pred(sex="Female"), pred(sex="Male", avg_lifespan_change_pct=9.0), pred(sex="Male")]
    m = match_paper(rows, preds)
    assert len(m.pairs) == 2 and len(m.extra) == 1 and m.missed == []
    for t, p in m.pairs:
        assert t["gender"].lower() == p.sex.lower()


def test_compound_level_counts_collapse_doses():
    rows = [truth(dosage="14 ppm"), truth(dosage="42 ppm"), truth(compound_name="Aspirin")]
    assert compound_level_counts(rows, [pred(), pred(compound="Metformin")]) == (1, 2, 1, 2)


def test_lifespan_tolerance_boundary():
    assert lifespan_correct(10.0, 12.0, 2.0)
    assert lifespan_correct(10.0, 8.0, 2.0)
    assert not lifespan_correct(10.0, 12.1, 2.0)
    assert not lifespan_correct(10.0, None, 2.0)


def test_field_accuracy_skips_fields_drugage_does_not_have():
    pairs = [
        (truth(), pred()),
        (truth(gender="Unknown", strain=None), pred(avg_lifespan_change_pct=20.0, significant=None)),
    ]
    acc = field_accuracy(pairs, 2.0)
    assert acc["avg_lifespan_change_pct"] == (1, 2)
    assert acc["max_lifespan_change_pct"] == (0, 0)
    assert acc["sex"] == (1, 1)
    assert acc["strain"] == (1, 1)
    assert acc["dose"] == (2, 2)
    assert acc["significant"] == (1, 2)


def test_quote_in_text_ignores_case_and_whitespace():
    assert quote_in_text("lifespan  by 14%", "Median Lifespan\nby 14% in females")
    assert not quote_in_text("lifespan by 15%", "Median lifespan by 14%")
    assert not quote_in_text("", "anything")


def test_worst_errors_ranks_largest_gap_first_then_nulls():
    pairs = [
        (1, truth(), pred(avg_lifespan_change_pct=15.0)),  # within tolerance
        (2, truth(), pred(avg_lifespan_change_pct=30.0)),
        (3, truth(), pred(avg_lifespan_change_pct=None)),
        (4, truth(), pred(avg_lifespan_change_pct=-40.0)),
    ]
    assert [e["pmid"] for e in worst_errors(pairs, 2.0)] == [4, 2, 3]

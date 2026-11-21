from evals.matching import (
    compound_level_counts,
    compounds_match,
    doses_agree,
    field_accuracy,
    lifespan_correct,
    match_paper,
    norm_compound,
    parse_dose,
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


def test_parse_dose_converts_units_and_notation():
    assert parse_dose("10 µM") == parse_dose("0.01 mM") == parse_dose("10 μm") == ("molar", 10.0)
    assert parse_dose("0.5 mg/ml") == parse_dose("500 µg/mL") == parse_dose("0.5 g/L")
    assert parse_dose("1*10^-6 %") == parse_dose("1 x 10(-6) wt.% of culture medium")
    assert parse_dose("1 gr") == parse_dose("1 gr of formulation in 10 mL culture") == ("mass", 1.0)
    assert parse_dose("25 mg/mL (9% dietary yeast)") == ("mass/vol", 25.0)
    assert parse_dose("about 26 mg/mL") == ("mass/vol", 26.0)


def test_parse_dose_rejects_ranges_combinations_and_missing_units():
    for dose in ["2 to 10 mm", "5 mM NAM + 100 µM RSV", "0.1 μm arsenite with 1 mm NAC", "10",
                 "clinically relevant concentrations", "daf-2 RNAi", None]:
        assert parse_dose(dose) is None, dose


def test_doses_agree_is_three_valued():
    assert doses_agree("10 µM", "0.01 mM") is True
    assert doses_agree("25 mM", "15 mM") is False
    assert doses_agree("1 mg/ml", "1 mM") is None  # different unit families
    assert doses_agree("25 mM", None) is None


def test_match_never_pairs_clearly_different_doses():
    m = match_paper([truth(dosage="25 mM")], [pred(dose="15 mM")])
    assert m.pairs == [] and len(m.missed) == 1 and len(m.extra) == 1


def test_match_pairs_same_dose_written_differently_and_unknown_doses():
    m = match_paper([truth(dosage="0.01 mM"), truth(dosage=None, gender="Male")],
                    [pred(dose="10 µM"), pred(dose="clinically relevant", sex="Male")])
    assert len(m.pairs) == 2
    assert field_accuracy(m.pairs, 2.0)["dose"] == (1, 1)


def test_compound_level_counts_ignore_dose():
    assert compound_level_counts([truth(dosage="25 mM")], [pred(dose="15 mM")]) == (1, 1, 1, 1)

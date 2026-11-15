"""Match extracted experiments to DrugAge records and score them. Pure functions, no I/O."""

import re
import unicodedata
from dataclasses import dataclass, field

from src.schema import LifespanExperiment

_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ω": "omega"}
_COMMON_SPECIES = {
    "mouse": "mus musculus",
    "mice": "mus musculus",
    "rat": "rattus norvegicus",
    "rats": "rattus norvegicus",
    "yeast": "saccharomyces cerevisiae",
    "budding yeast": "saccharomyces cerevisiae",
    "fruit fly": "drosophila melanogaster",
    "fruit flies": "drosophila melanogaster",
    "drosophila": "drosophila melanogaster",
}


def norm_compound(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).lower()
    for letter, word in _GREEK.items():
        s = s.replace(letter, word)
    return re.sub(r"[^a-z0-9]", "", s)


def compounds_match(a: str, b: str) -> bool:
    """Same normalised name, or one name contains the other ('lithium' ~ 'lithium chloride')."""
    na, nb = norm_compound(a), norm_compound(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return min(len(na), len(nb)) >= 5 and (na in nb or nb in na)


def species_key(name: str) -> tuple[str, ...]:
    """('c', 'elegans') for both 'Caenorhabditis elegans' and 'C. elegans'."""
    s = name.strip().lower()
    s = _COMMON_SPECIES.get(s, s)
    parts = re.findall(r"[a-z]+", s)
    return (parts[0][0], parts[1]) if len(parts) >= 2 else (s,)


def _norm_text(value: str | None) -> str | None:
    """Lowercase alphanumerics only, for comparing strain and dose strings."""
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", value).lower().replace("μ", "u").replace("µ", "u")
    return re.sub(r"[^a-z0-9.]", "", s) or None


def _norm_sex(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip().lower().rstrip("s")
    return None if s in ("", "unknown") else s


@dataclass
class PaperMatch:
    pairs: list[tuple[dict, LifespanExperiment]] = field(default_factory=list)
    missed: list[dict] = field(default_factory=list)  # DrugAge entries with no extracted record
    extra: list[LifespanExperiment] = field(default_factory=list)  # extracted, not in DrugAge


def _is_candidate(truth: dict, pred: LifespanExperiment) -> bool:
    return compounds_match(truth["compound_name"], pred.compound) and species_key(
        truth["species"]
    ) == species_key(pred.species)


def _pair_cost(truth: dict, pred: LifespanExperiment) -> tuple:
    """Lower is a better pairing: agreeing sex, then agreeing dose, then closest lifespan change."""
    t_sex, p_sex = _norm_sex(truth["gender"]), _norm_sex(pred.sex)
    sex_conflict = t_sex is not None and p_sex is not None and t_sex != p_sex
    dose_mismatch = _norm_text(truth["dosage"]) != _norm_text(pred.dose)
    t_avg, p_avg = truth["avg_lifespan_change_percent"], pred.avg_lifespan_change_pct
    avg_diff = abs(t_avg - p_avg) if t_avg is not None and p_avg is not None else float("inf")
    return (sex_conflict, dose_mismatch, avg_diff)


def match_paper(drugage_rows: list[dict], extracted: list[LifespanExperiment]) -> PaperMatch:
    """Pair DrugAge rows and extracted records of one paper, one-to-one, on compound + species."""
    candidates = sorted(
        (_pair_cost(t, p), i, j)
        for i, t in enumerate(drugage_rows)
        for j, p in enumerate(extracted)
        if _is_candidate(t, p)
    )
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    result = PaperMatch()
    for _, i, j in candidates:
        if i not in used_truth and j not in used_pred:
            used_truth.add(i)
            used_pred.add(j)
            result.pairs.append((drugage_rows[i], extracted[j]))
    result.missed = [t for i, t in enumerate(drugage_rows) if i not in used_truth]
    result.extra = [p for j, p in enumerate(extracted) if j not in used_pred]
    return result


def compound_level_counts(
    drugage_rows: list[dict], extracted: list[LifespanExperiment]
) -> tuple[int, int, int, int]:
    """Counts over unique compound + species per paper, ignoring how many doses/sexes each has.

    Returns (drugage_found, drugage_total, extracted_matched, extracted_total).
    """
    truth = {(norm_compound(t["compound_name"]), species_key(t["species"])): t for t in drugage_rows}
    preds = {(norm_compound(p.compound), species_key(p.species)): p for p in extracted}
    found = sum(any(_is_candidate(t, p) for p in preds.values()) for t in truth.values())
    matched = sum(any(_is_candidate(t, p) for t in truth.values()) for p in preds.values())
    return found, len(truth), matched, len(preds)


def lifespan_correct(truth: float, pred: float | None, tolerance: float) -> bool:
    return pred is not None and abs(pred - truth) <= tolerance


def field_accuracy(
    pairs: list[tuple[dict, LifespanExperiment]], tolerance: float
) -> dict[str, tuple[int, int]]:
    """(correct, total) per field over matched pairs. A pair counts only if DrugAge has the value."""
    acc = {f: [0, 0] for f in ("avg_lifespan_change_pct", "max_lifespan_change_pct", "strain", "sex", "dose", "significant")}

    def tally(name: str, correct: bool) -> None:
        acc[name][1] += 1
        acc[name][0] += bool(correct)

    for truth, pred in pairs:
        if truth["avg_lifespan_change_percent"] is not None:
            tally(
                "avg_lifespan_change_pct",
                lifespan_correct(truth["avg_lifespan_change_percent"], pred.avg_lifespan_change_pct, tolerance),
            )
        if truth["max_lifespan_change_percent"] is not None:
            tally(
                "max_lifespan_change_pct",
                lifespan_correct(truth["max_lifespan_change_percent"], pred.max_lifespan_change_pct, tolerance),
            )
        if _norm_text(truth["strain"]) is not None:
            tally("strain", _norm_text(truth["strain"]) == _norm_text(pred.strain))
        if _norm_sex(truth["gender"]) is not None:
            tally("sex", _norm_sex(truth["gender"]) == _norm_sex(pred.sex))
        if _norm_text(truth["dosage"]) is not None:
            tally("dose", _norm_text(truth["dosage"]) == _norm_text(pred.dose))
        if truth["avg_lifespan_significance"] in ("S", "NS"):
            tally("significant", pred.significant == (truth["avg_lifespan_significance"] == "S"))
    return {name: (c, n) for name, (c, n) in acc.items()}


def quote_in_text(quote: str, text: str) -> bool:
    """True if the evidence quote appears in the paper text, ignoring case and whitespace."""
    squash = lambda s: " ".join(s.lower().split())
    return bool(quote.strip()) and squash(quote) in squash(text)

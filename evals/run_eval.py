"""Score LLM extractions against DrugAge and write evals/results.md.

Run:  python -m evals.run_eval
"""

import sys
from datetime import date
from pathlib import Path

import anthropic

from evals.matching import compound_level_counts, field_accuracy, match_paper, quote_in_text
from src.config import EVAL_N_PAPERS, EVAL_SEED, LIFESPAN_TOLERANCE_PCT, MODEL
from src.drugage import PAPER_ID_COLUMN, load_drugage, sample_pmids
from src.extract import ExtractionError, connect, extract_paper, make_client
from src.fetch import fetch_paper

RESULTS_PATH = Path(__file__).parent / "results.md"


def pct(correct: int, total: int) -> str:
    return f"{100 * correct / total:.1f}% ({correct}/{total})" if total else "n/a (0/0)"


def worst_errors(pairs: list[tuple[int, dict, object]], tolerance: float, n: int = 5) -> list[dict]:
    """The n matched pairs whose average lifespan change is furthest from DrugAge."""
    errors = []
    for pmid, truth, pred in pairs:
        t = truth["avg_lifespan_change_percent"]
        if t is None:
            continue
        p = pred.avg_lifespan_change_pct
        diff = abs(p - t) if p is not None else None
        if diff is None or diff > tolerance:
            errors.append({"pmid": pmid, "truth": truth, "pred": pred, "diff": diff})
    # Largest numeric differences first, then records where the extractor returned null.
    errors.sort(key=lambda e: (e["diff"] is None, -(e["diff"] or 0)))
    return errors[:n]


def main() -> None:
    drugage = load_drugage()
    drugage = drugage.astype(object).where(drugage.notna(), None)
    pmids = sample_pmids(EVAL_N_PAPERS, EVAL_SEED)
    conn = connect()
    client = make_client()

    all_pairs: list[tuple[int, dict, object]] = []
    n_truth = n_pred = 0
    cl_found = cl_truth = cl_matched = cl_pred = 0
    quotes_ok = quotes_total = 0
    sources = {"full text": 0, "abstract only": 0, "title only": 0, "not in Europe PMC": 0}
    failed: list[int] = []
    per_paper: list[tuple[int, int, int, int]] = []

    for pmid in pmids:
        truth_rows = drugage[drugage[PAPER_ID_COLUMN] == pmid].to_dict("records")
        try:
            extracted = extract_paper(pmid, conn, client)
        except ExtractionError as e:
            print(f"FAILED {e}")
            failed.append(pmid)
            extracted = []
        except anthropic.AnthropicError as e:
            sys.exit(
                f"LLM call failed on PMID {pmid}: {e}\n"
                "Check that ANTHROPIC_API_KEY is set in .env and the model in config.toml is valid."
            )

        paper = fetch_paper(pmid)
        if paper is None:
            sources["not in Europe PMC"] += 1
        elif paper.full_text:
            sources["full text"] += 1
        elif paper.abstract:
            sources["abstract only"] += 1
        else:
            sources["title only"] += 1
        extracted = extracted or []

        m = match_paper(truth_rows, extracted)
        all_pairs += [(pmid, t, p) for t, p in m.pairs]
        n_truth += len(truth_rows)
        n_pred += len(extracted)
        found, total_t, matched, total_p = compound_level_counts(truth_rows, extracted)
        cl_found, cl_truth, cl_matched, cl_pred = (
            cl_found + found, cl_truth + total_t, cl_matched + matched, cl_pred + total_p
        )
        if paper is not None:
            quotes_total += len(extracted)
            quotes_ok += sum(quote_in_text(p.evidence_quote, paper.text) for p in extracted)
        per_paper.append((pmid, len(truth_rows), len(extracted), len(m.pairs)))
        print(f"{pmid}: DrugAge {len(truth_rows):>3}  extracted {len(extracted):>3}  matched {len(m.pairs):>3}")

    n_matched = len(all_pairs)
    accuracy = field_accuracy([(t, p) for _, t, p in all_pairs], LIFESPAN_TOLERANCE_PCT)
    errors = worst_errors(all_pairs, LIFESPAN_TOLERANCE_PCT)

    lines = [
        "# Eval results",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Model: `{MODEL}`",
        f"- Papers: {len(pmids)} (random sample of DrugAge PubMed IDs, seed {EVAL_SEED})",
        "- Paper text available: " + ", ".join(f"{v} {k}" for k, v in sources.items() if v),
    ]
    if failed:
        lines.append(f"- Extraction failed (counted as zero records): {', '.join(map(str, failed))}")
    lines += [
        "",
        "## Recall and precision",
        "",
        "Records are matched within a paper on compound + species.",
        "",
        "| Level | Recall (DrugAge entries found) | Precision (extracted entries that match) |",
        "|---|---|---|",
        f"| Entry (one-to-one pairs) | {pct(n_matched, n_truth)} | {pct(n_matched, n_pred)} |",
        f"| Compound + species (unique per paper) | {pct(cl_found, cl_truth)} | {pct(cl_matched, cl_pred)} |",
        "",
        "## Per-field accuracy",
        "",
        f"Over the {n_matched} matched pairs, counting only pairs where DrugAge has a value for the field. "
        f"Lifespan change is correct within ±{LIFESPAN_TOLERANCE_PCT:g} points.",
        "",
        "| Field | Accuracy |",
        "|---|---|",
    ]
    lines += [f"| {name} | {pct(c, n)} |" for name, (c, n) in accuracy.items()]
    lines += [
        "",
        "DrugAge has no animal counts, so `n_animals` is not scored. `species` and `compound` are "
        "the match keys, so they are correct by construction.",
        "",
        f"`evidence_quote` found word for word in the paper text: {pct(quotes_ok, quotes_total)}",
        "",
        "## 5 worst errors",
        "",
        "Matched pairs with the largest gap in average lifespan change.",
        "",
    ]
    if not errors:
        lines.append("None: no matched pair is outside the tolerance.")
    for i, e in enumerate(errors, 1):
        t, p = e["truth"], e["pred"]
        gap = "extractor returned null" if e["diff"] is None else f"off by {e['diff']:.1f} points"
        lines += [
            f"### {i}. PMID {e['pmid']}: {t['compound_name']} in {t['species']} ({gap})",
            "",
            f"- DrugAge: {t['avg_lifespan_change_percent']}% (dose {t['dosage']}, sex {t['gender']}, strain {t['strain']})",
            f"- Extracted: {p.avg_lifespan_change_pct}% (dose {p.dose}, sex {p.sex}, strain {p.strain})",
            f"- Evidence quote: \"{p.evidence_quote}\"",
            "",
        ]
    lines += [
        "## Per paper",
        "",
        "| PMID | DrugAge entries | Extracted | Matched |",
        "|---|---|---|---|",
    ]
    lines += [f"| {pmid} | {a} | {b} | {c} |" for pmid, a, b, c in per_paper]

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nModel: {MODEL}   papers: {len(pmids)}")
    print(f"Recall:    {pct(n_matched, n_truth)}")
    print(f"Precision: {pct(n_matched, n_pred)}")
    print("Per-field accuracy:")
    for name, (c, n) in accuracy.items():
        print(f"  {name:26} {pct(c, n)}")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()

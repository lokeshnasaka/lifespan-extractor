# Lifespan Study Extractor

A small Python pipeline that reads aging research papers and extracts structured
lifespan-experiment data (compound, species, dose, lifespan change, ...) with an LLM.
Accuracy is measured against [DrugAge](https://genomics.senescence.info/drugs/), a
hand-curated database of compounds that extend lifespan in model organisms.

## Layout

```
config.toml        model name and eval settings
src/drugage.py     download DrugAge, save clean data/drugage.csv
src/schema.py      LifespanExperiment Pydantic model
src/fetch.py       Europe PMC: title, abstract, open-access full text (cached on disk)
src/extract.py     LLM extraction -> SQLite (data/extractions.db)
evals/run_eval.py  precision / recall / per-field accuracy -> evals/results.md
tests/             unit tests for parsing and matching
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The LLM provider and model are set in `config.toml`.

**Local (default, free):** runs the model on your machine with [Ollama](https://ollama.com).

```bash
brew install ollama && brew services start ollama
ollama pull gpt-oss:20b   # ~13 GB; needs roughly 24 GB of RAM
```

Papers that do not fit the `num_ctx` context window fail loudly rather than being
truncated; raise `num_ctx` (more memory) or use a smaller model if that happens.

**Anthropic API:** set `provider = "anthropic"` and a Claude model name in `config.toml`, then

```bash
cp .env.example .env   # then add your ANTHROPIC_API_KEY (.env is git-ignored)
```

## Run

```bash
uv run python -m src.drugage      # 1. download + clean DrugAge -> data/drugage.csv
uv run python -m src.fetch        # 2. fetch the 30 eval papers from Europe PMC (cached)
uv run python -m src.extract      # 3. extract experiments with the LLM -> SQLite
uv run python -m evals.run_eval   # 4. score against DrugAge -> evals/results.md
uv run pytest                     # unit tests
```

Steps 2 and 3 are also run on demand by `evals.run_eval`, so step 4 alone is enough.
Europe PMC responses are cached in `data/cache/`, and a paper that already has an
extraction for the configured model is not sent to the LLM again.

## How the eval works

- The 30 eval papers are a fixed random sample (seed in `config.toml`) of the PubMed IDs
  in DrugAge.
- Extracted records are matched to DrugAge records of the same paper on
  compound + species (normalised names).
- **Recall** = DrugAge entries that were found. **Precision** = extracted entries that
  match a DrugAge entry.
- **Per-field accuracy** is computed over matched pairs. A lifespan change within
  ±2 percentage points of the DrugAge value counts as correct.

"""Schema for one lifespan experiment extracted from a paper."""

from pydantic import BaseModel, Field


class LifespanExperiment(BaseModel):
    compound: str = Field(description="Compound or intervention tested, as named in the paper")
    species: str = Field(description="Scientific name of the organism, e.g. 'Caenorhabditis elegans'")
    strain: str | None = Field(description="Strain or genetic background, e.g. 'N2', 'C57BL/6'")
    sex: str | None = Field(description="Male, Female, Hermaphrodite, or Pooled")
    dose: str | None = Field(description="Dose or concentration with units, as written")
    avg_lifespan_change_pct: float | None = Field(
        description="Percent change in mean or median lifespan vs control; negative if shortened"
    )
    max_lifespan_change_pct: float | None = Field(
        description="Percent change in maximum lifespan vs control; negative if shortened"
    )
    n_animals: int | None = Field(description="Number of animals in the treated group")
    significant: bool | None = Field(
        description="Whether the lifespan change was reported as statistically significant"
    )
    evidence_quote: str = Field(description="The exact sentence from the paper that supports the numbers")


class ExtractionResult(BaseModel):
    """Top-level object returned by the LLM: one paper can contain many experiments."""

    experiments: list[LifespanExperiment]

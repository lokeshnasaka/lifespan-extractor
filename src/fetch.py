"""Fetch a paper from Europe PMC by PubMed ID: title, abstract, open-access full text.

Every API response is cached under data/cache/europepmc/, so a paper is only
requested once.

Run:  python -m src.fetch            (fetches the eval sample)
      python -m src.fetch 12345 678  (fetches specific PubMed IDs)
"""

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

from src.config import CACHE_DIR, EVAL_N_PAPERS, EVAL_SEED

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EPMC_CACHE = CACHE_DIR / "europepmc"

# Tags after which we insert a line break / cell separator when flattening JATS XML.
_BLOCK_TAGS = {"p", "title", "tr", "caption", "label", "sec", "table-wrap"}
_CELL_TAGS = {"td", "th"}


@dataclass
class Paper:
    pmid: int
    title: str
    abstract: str | None
    full_text: str | None  # None when the paper is not open access

    @property
    def text(self) -> str:
        """The text sent to the LLM: full text when available, otherwise the abstract."""
        parts = [f"Title: {self.title}"]
        if self.abstract:
            parts.append(f"Abstract:\n{self.abstract}")
        if self.full_text:
            parts.append(f"Full text:\n{self.full_text}")
        return "\n\n".join(parts)


def _get_cached(cache_name: str, url: str, params: dict | None = None) -> str | None:
    """GET a URL with an on-disk cache. A 404 is cached as an empty file and returns None."""
    path = EPMC_CACHE / cache_name
    if not path.exists():
        EPMC_CACHE.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 404:
            body = ""
        else:
            resp.raise_for_status()
            body = resp.text
        path.write_text(body, encoding="utf-8")
    return path.read_text(encoding="utf-8") or None


def parse_search_result(payload: dict, pmid: int) -> dict | None:
    """Pick the record for this PubMed ID out of a Europe PMC search response."""
    for rec in payload.get("resultList", {}).get("result", []):
        if rec.get("pmid") == str(pmid):
            return rec
    return None


def fulltext_xml_to_text(xml: str) -> str | None:
    """Flatten the <body> of a JATS full-text XML to plain text, keeping table cells."""
    body = ET.fromstring(xml).find(".//body")
    if body is None:
        return None

    out: list[str] = []

    def walk(el: ET.Element) -> None:
        if el.text:
            out.append(el.text)
        for child in el:
            walk(child)
            if child.tag in _BLOCK_TAGS:
                out.append("\n")
            elif child.tag in _CELL_TAGS:
                out.append(" | ")
            if child.tail:
                out.append(child.tail)

    walk(body)
    lines = (" ".join(line.split()) for line in "".join(out).splitlines())
    return "\n".join(line for line in lines if line) or None


def fetch_paper(pmid: int) -> Paper | None:
    """Return the paper for a PubMed ID, or None if Europe PMC does not have it."""
    raw = _get_cached(
        f"{pmid}.json",
        f"{API}/search",
        {"query": f"EXT_ID:{pmid} AND SRC:MED", "resultType": "core", "format": "json"},
    )
    rec = parse_search_result(json.loads(raw), pmid) if raw else None
    if rec is None:
        return None

    full_text = None
    pmcid = rec.get("pmcid")
    if pmcid and rec.get("isOpenAccess") == "Y":
        xml = _get_cached(f"{pmcid}.xml", f"{API}/{pmcid}/fullTextXML")
        if xml:
            full_text = fulltext_xml_to_text(xml)

    return Paper(
        pmid=pmid,
        title=rec.get("title", ""),
        abstract=rec.get("abstractText"),
        full_text=full_text,
    )


def main() -> None:
    from src.drugage import sample_pmids

    pmids = [int(a) for a in sys.argv[1:]] or sample_pmids(EVAL_N_PAPERS, EVAL_SEED)
    counts = {"found": 0, "abstract": 0, "full_text": 0}
    for pmid in pmids:
        paper = fetch_paper(pmid)
        if paper is None:
            print(f"{pmid}: not in Europe PMC")
            continue
        counts["found"] += 1
        counts["abstract"] += bool(paper.abstract)
        counts["full_text"] += bool(paper.full_text)
        kind = "full text" if paper.full_text else "abstract" if paper.abstract else "title only"
        print(f"{pmid}: {kind:10} {len(paper.text):>7} chars  {paper.title[:70]}")
    print(f"\n{len(pmids)} PubMed IDs: {counts}")


if __name__ == "__main__":
    main()

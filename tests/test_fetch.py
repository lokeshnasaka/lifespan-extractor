from src.fetch import Paper, fulltext_xml_to_text, parse_search_result

JATS = """<article>
  <front><article-meta><title-group><article-title>Ignored</article-title></title-group></article-meta></front>
  <body>
    <sec><title>Results</title>
      <p>Rapamycin extended <italic>median</italic> lifespan by 14%.</p>
      <table-wrap><table>
        <tr><th>Dose</th><th>Change</th></tr>
        <tr><td>14 ppm</td><td>+9%</td></tr>
      </table></table-wrap>
    </sec>
  </body>
</article>"""


def test_fulltext_keeps_inline_text_together():
    text = fulltext_xml_to_text(JATS)
    assert "Rapamycin extended median lifespan by 14%." in text


def test_fulltext_separates_table_cells_and_skips_front_matter():
    text = fulltext_xml_to_text(JATS)
    assert "14 ppm | +9% |" in text
    assert "Ignored" not in text


def test_fulltext_without_body_is_none():
    assert fulltext_xml_to_text("<article><front/></article>") is None


def test_parse_search_result_picks_matching_pmid():
    payload = {"resultList": {"result": [{"pmid": "111", "title": "a"}, {"pmid": "222", "title": "b"}]}}
    assert parse_search_result(payload, 222)["title"] == "b"
    assert parse_search_result(payload, 333) is None
    assert parse_search_result({}, 333) is None


def test_paper_text_uses_what_is_available():
    assert Paper(1, "T", None, None).text == "Title: T"
    full = Paper(1, "T", "A", "F").text
    assert "Abstract:\nA" in full and "Full text:\nF" in full

import pytest

from app.generation.citations import parse_citations
from app.models import Chunk, SearchResult


@pytest.fixture
def sources() -> list[SearchResult]:
    return [
        SearchResult(Chunk(f"d:{i}", "d", f"text {i + 1}", page=10 + i, index=i), 0.9 - i / 10)
        for i in range(3)
    ]  # cited as [1], [2], [3] -> pages 10, 11, 12


def test_valid_citation_maps_to_the_right_chunk(sources):
    parsed = parse_citations("Hagrid is a giant [2].", sources)
    assert parsed.text == "Hagrid is a giant [2]."
    assert [(c.ref, c.chunk_id, c.page) for c in parsed.citations] == [(2, "d:1", 11)]
    assert parsed.invalid_refs == []


def test_adjacent_and_comma_markers(sources):
    parsed = parse_citations("A [1][3]. B [2, 3].", sources)
    assert [c.ref for c in parsed.citations] == [1, 3, 2]  # first-appearance order
    assert parsed.text == "A [1][3]. B [2, 3]."


def test_duplicates_collapse_to_one_citation(sources):
    parsed = parse_citations("A [1]. B [1]. C [1].", sources)
    assert [c.ref for c in parsed.citations] == [1]


def test_invalid_ref_is_removed_and_reported(sources):
    parsed = parse_citations("Hagrid is a giant [9].", sources)
    assert parsed.text == "Hagrid is a giant."
    assert parsed.citations == []
    assert parsed.invalid_refs == [9]


def test_zero_is_invalid(sources):
    parsed = parse_citations("Claim [0].", sources)
    assert parsed.invalid_refs == [0]
    assert parsed.citations == []


def test_partially_invalid_marker_is_repaired(sources):
    parsed = parse_citations("Fact [1, 9].", sources)
    assert parsed.text == "Fact [1]."
    assert [c.ref for c in parsed.citations] == [1]
    assert parsed.invalid_refs == [9]


def test_invalid_marker_mid_sentence_leaves_clean_spacing(sources):
    parsed = parse_citations("Claim [8] continues here [1].", sources)
    assert parsed.text == "Claim continues here [1]."


def test_answer_without_markers_has_no_citations(sources):
    parsed = parse_citations("Hagrid is a giant.", sources)
    assert parsed.text == "Hagrid is a giant."
    assert parsed.citations == []
    assert parsed.invalid_refs == []


def test_bracketed_years_are_not_treated_as_citations(sources):
    parsed = parse_citations("Published in [2024] by the school [1].", sources)
    assert parsed.text == "Published in [2024] by the school [1]."
    assert [c.ref for c in parsed.citations] == [1]
    assert parsed.invalid_refs == []
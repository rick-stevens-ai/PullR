"""
Unit tests for openalex.py.

Run with:
    pytest tests/test_openalex.py -v

Live tests are gated by OPENALEX_LIVE_TESTS=1 so CI does not hammer the API.
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

import openalex as oa


# ---- _reconstruct_abstract -------------------------------------------------

def test_reconstruct_abstract_basic():
    idx = {"the": [0, 2], "quick": [1], "brown": [3]}
    assert oa._reconstruct_abstract(idx) == "the quick the brown"


def test_reconstruct_abstract_empty():
    assert oa._reconstruct_abstract(None) is None
    assert oa._reconstruct_abstract({}) is None


# ---- _pick_pdf_url ---------------------------------------------------------

def test_pick_pdf_best_oa_pdf():
    w = {"best_oa_location": {"pdf_url": "https://x.org/p.pdf", "host_type": "publisher", "license": "cc-by"}}
    assert oa._pick_pdf_url(w) == ("https://x.org/p.pdf", "publisher", "cc-by")


def test_pick_pdf_falls_back_to_primary():
    w = {
        "best_oa_location": None,
        "primary_location": {"is_oa": True, "pdf_url": "https://prim.org/p.pdf",
                             "host_type": "repository", "license": None},
    }
    url, host, lic = oa._pick_pdf_url(w)
    assert url == "https://prim.org/p.pdf"
    assert host == "repository"


def test_pick_pdf_walks_locations():
    w = {
        "best_oa_location": None,
        "primary_location": None,
        "locations": [
            {"landing_page_url": "https://x"},
            {"pdf_url": "https://repo.org/p.pdf", "host_type": "repository"},
        ],
    }
    assert oa._pick_pdf_url(w)[0] == "https://repo.org/p.pdf"


def test_pick_pdf_landing_fallback():
    w = {"best_oa_location": {"landing_page_url": "https://landing.org", "host_type": "publisher"}}
    assert oa._pick_pdf_url(w)[0] == "https://landing.org"


def test_pick_pdf_returns_none_when_nothing():
    assert oa._pick_pdf_url({}) == (None, None, None)


# ---- _short_doi_from_url ---------------------------------------------------

def test_short_doi_strips_prefix():
    assert oa._short_doi_from_url("https://doi.org/10.1234/abc") == "10.1234/abc"
    assert oa._short_doi_from_url("http://doi.org/10.1234/abc") == "10.1234/abc"
    assert oa._short_doi_from_url("doi.org/10.1234/abc") == "10.1234/abc"
    assert oa._short_doi_from_url("10.1234/abc") == "10.1234/abc"
    assert oa._short_doi_from_url(None) is None


# ---- _to_s2_shape ----------------------------------------------------------

def test_to_s2_shape_basic_fields():
    w = {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/abc",
        "title": "A paper",
        "publication_year": 2020,
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}},
        ],
        "primary_location": {"source": {"display_name": "Some Journal"}},
        "best_oa_location": {"pdf_url": "https://x/p.pdf"},
        "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345"},
        "type": "article",
        "cited_by_count": 42,
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "abstract_inverted_index": {"hello": [0], "world": [1]},
    }
    s = oa._to_s2_shape(w)
    assert s["paperId"] == "oa_W12345"
    assert s["title"] == "A paper"
    assert s["year"] == 2020
    assert [a["name"] for a in s["authors"]] == ["Alice Smith", "Bob Jones"]
    assert s["externalIds"]["DOI"] == "10.1234/abc"
    assert s["externalIds"]["PMID"] == "12345"
    assert s["externalIds"]["OpenAlex"] == "W12345"
    assert s["venue"] == "Some Journal"
    assert s["openAccessPdf"] == {"url": "https://x/p.pdf", "status": "open"}
    assert s["abstract"] == "hello world"
    assert s["_openalex"]["cited_by_count"] == 42
    assert s["_openalex"]["oa_status"] == "gold"


def test_to_s2_shape_no_oa():
    w = {"id": "https://openalex.org/W9", "title": "t", "publication_year": 2010}
    s = oa._to_s2_shape(w)
    assert s["openAccessPdf"] is None
    assert s["paperId"] == "oa_W9"


# ---- query_openalex_by_doi (mocked) ----------------------------------------

def _mock_response(status, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    return r


def test_query_doi_strips_url_prefix():
    with patch.object(oa, "_request") as m:
        m.return_value = {"id": "https://openalex.org/W1", "title": "t"}
        oa.query_openalex_by_doi("https://doi.org/10.1/abc", "x@example.org")
        # Confirm the URL passed didn't double-prefix
        called_url = m.call_args[0][0]
        assert "https://doi.org/10.1/abc" in called_url
        assert "https://doi.org/https://doi.org/" not in called_url


def test_query_doi_returns_none_when_no_id():
    with patch.object(oa, "_request", return_value=None):
        assert oa.query_openalex_by_doi("10.1/x", "e@e") is None


def test_query_doi_returns_paper_dict():
    with patch.object(oa, "_request") as m:
        m.return_value = {
            "id": "https://openalex.org/W42",
            "title": "test",
            "publication_year": 2024,
            "doi": "https://doi.org/10.1/x",
        }
        s = oa.query_openalex_by_doi("10.1/x", "e@e")
        assert s["title"] == "test"
        assert s["paperId"] == "oa_W42"


# ---- _title_similarity -----------------------------------------------------

def test_title_similarity_basic():
    assert oa._title_similarity("hello world", "hello world") == 1.0
    assert oa._title_similarity("hello", "goodbye") == 0.0
    sim = oa._title_similarity("attention is all you need", "attention is all you need in speech")
    assert 0.5 < sim < 1.0


# ---- search_openalex_by_title (mocked) -------------------------------------

def test_search_by_title_ranks_by_similarity_and_filters_low():
    with patch.object(oa, "_request") as m:
        m.return_value = {
            "results": [
                {"id": "https://openalex.org/W1", "title": "random junk",
                 "publication_year": 2024, "cited_by_count": 5},
                {"id": "https://openalex.org/W2", "title": "Attention Is All You Need",
                 "publication_year": 2017, "cited_by_count": 80000,
                 "authorships": [{"author": {"display_name": "Ashish Vaswani"}}]},
            ]
        }
        hits = oa.search_openalex_by_title("Attention Is All You Need", "e@e",
                                            year=2017, author="Vaswani", limit=3)
        assert len(hits) >= 1
        # Original paper should be ranked first thanks to title sim + author + year + citation bonuses
        assert hits[0]["paperId"] == "oa_W2"


def test_search_by_title_too_short_returns_empty():
    assert oa.search_openalex_by_title("ab", "e@e") == []
    assert oa.search_openalex_by_title("", "e@e") == []


# ---- extract_doi_from_ref ---------------------------------------------------

def test_extract_doi():
    assert oa.extract_doi_from_ref("Smith et al. doi: 10.1234/abc.5678") == "10.1234/abc.5678"
    assert oa.extract_doi_from_ref("no doi here") is None


# ---- Live test (gated) ------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("OPENALEX_LIVE_TESTS"),
                    reason="set OPENALEX_LIVE_TESTS=1 to run live network tests")
def test_live_doi_lookup():
    r = oa.query_openalex_by_doi("10.1038/nature14539", "stevens@anl.gov")
    assert r is not None
    assert "deep learning" in (r["title"] or "").lower()
    assert r["year"] == 2015

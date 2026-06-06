"""
Unit tests for unpaywall.py.

Run with:
    pytest tests/test_unpaywall.py -v

The mocked tests run offline. The live smoke test (test_live_known_oa) is
gated by the UNPAYWALL_LIVE_TESTS env var so CI does not hammer the public
API; set it to "1" to enable.
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

import unpaywall as uw


# ---- _pick_pdf_url ---------------------------------------------------------

def test_pick_pdf_url_from_best_url_for_pdf():
    rec = {"best_oa_location": {"url_for_pdf": "https://x.org/p.pdf", "host_type": "publisher", "evidence": "ev"}}
    assert uw._pick_pdf_url(rec) == ("https://x.org/p.pdf", "publisher", "ev")


def test_pick_pdf_url_falls_back_to_url():
    rec = {"best_oa_location": {"url": "https://x.org/landing", "host_type": "repository", "evidence": "ev2"}}
    assert uw._pick_pdf_url(rec) == ("https://x.org/landing", "repository", "ev2")


def test_pick_pdf_url_walks_oa_locations():
    rec = {
        "best_oa_location": None,
        "oa_locations": [
            {"url": "https://landing.org", "host_type": "repository"},
            {"url_for_pdf": "https://second.org/p.pdf", "host_type": "publisher"},
        ],
    }
    # First loc has no url_for_pdf; should fall through to second
    url, host, ev = uw._pick_pdf_url(rec)
    assert url == "https://second.org/p.pdf"
    assert host == "publisher"


def test_pick_pdf_url_returns_none_when_no_oa():
    assert uw._pick_pdf_url({}) == (None, None, None)
    assert uw._pick_pdf_url({"best_oa_location": None, "oa_locations": []}) == (None, None, None)


# ---- _to_s2_shape ----------------------------------------------------------

def test_to_s2_shape_basic_fields():
    rec = {
        "doi": "10.1234/abc",
        "title": "A paper",
        "year": 2020,
        "doi_url": "https://doi.org/10.1234/abc",
        "journal_name": "Some Journal",
        "z_authors": [{"given": "Jane", "family": "Doe"}, {"name": "Bob Loblaw"}],
        "is_oa": True,
        "oa_status": "green",
    }
    out = uw._to_s2_shape(rec, "https://x.org/p.pdf", "repository")
    assert out["title"] == "A paper"
    assert out["year"] == 2020
    assert out["venue"] == "Some Journal"
    assert out["paperId"].startswith("uw_")
    assert out["externalIds"]["DOI"] == "10.1234/abc"
    assert out["openAccessPdf"]["url"] == "https://x.org/p.pdf"
    assert {a["name"] for a in out["authors"]} == {"Jane Doe", "Bob Loblaw"}
    assert out["_unpaywall"]["host_type"] == "repository"


def test_to_s2_shape_handles_missing_optional_fields():
    rec = {"doi": "10.1/x", "title": "T", "is_oa": True}
    out = uw._to_s2_shape(rec, "https://p.pdf", None)
    assert out["title"] == "T"
    assert out["authors"] == []
    assert out["year"] is None
    assert out["openAccessPdf"] == {"url": "https://p.pdf", "status": "open"}


# ---- extract_doi_from_ref --------------------------------------------------

@pytest.mark.parametrize("ref,expected", [
    ("Smith 2020. Nature. doi:10.1038/nature12373.", "10.1038/nature12373"),
    ("Title. Journal 2020. https://doi.org/10.1234/abc-def.", "10.1234/abc-def"),
    ("Bare 10.5555/xyz123 in middle of text.", "10.5555/xyz123"),
    ("No DOI here at all.", None),
    ("", None),
    (None, None),
])
def test_extract_doi_from_ref(ref, expected):
    assert uw.extract_doi_from_ref(ref) == expected


def test_extract_doi_strips_trailing_punctuation():
    assert uw.extract_doi_from_ref("Foo (10.1234/bar.baz).") == "10.1234/bar.baz"


# ---- query_unpaywall_by_doi (mocked) ---------------------------------------

def _mock_response(status_code, body=None):
    r = MagicMock()
    r.status_code = status_code
    if body is not None:
        r.json.return_value = body
    return r


def test_query_returns_none_without_doi():
    assert uw.query_unpaywall_by_doi("", "test@example.com") is None


def test_query_returns_none_without_email():
    assert uw.query_unpaywall_by_doi("10.1234/x", "") is None


def test_query_strips_doi_prefixes():
    """The function should normalize https://doi.org/ and doi: prefixes."""
    body = {"is_oa": True, "title": "T", "doi": "10.1/x",
            "best_oa_location": {"url_for_pdf": "https://p.pdf", "host_type": "publisher"}}
    with patch.object(uw.requests, "get", return_value=_mock_response(200, body)) as mock_get:
        result = uw.query_unpaywall_by_doi("https://doi.org/10.1/x", "test@example.com")
        assert result is not None
        # confirm we stripped the prefix before encoding
        called_url = mock_get.call_args[0][0]
        assert "10.1/x" in called_url
        assert "https%3A" not in called_url  # prefix not encoded into the path


def test_query_returns_oa_paper():
    body = {
        "is_oa": True, "title": "Open access paper", "doi": "10.1/x", "year": 2020,
        "best_oa_location": {"url_for_pdf": "https://arxiv.org/pdf/x.pdf",
                             "host_type": "repository", "evidence": "oa-repo"},
        "z_authors": [{"given": "A", "family": "B"}],
        "journal_name": "Test Journal",
    }
    with patch.object(uw.requests, "get", return_value=_mock_response(200, body)):
        result = uw.query_unpaywall_by_doi("10.1/x", "test@example.com")
    assert result is not None
    assert result["title"] == "Open access paper"
    assert result["openAccessPdf"]["url"] == "https://arxiv.org/pdf/x.pdf"
    assert result["venue"] == "Test Journal"


def test_query_returns_none_when_not_oa():
    body = {"is_oa": False, "title": "Paywalled", "doi": "10.1/x"}
    with patch.object(uw.requests, "get", return_value=_mock_response(200, body)):
        assert uw.query_unpaywall_by_doi("10.1/x", "test@example.com") is None


def test_query_handles_404():
    with patch.object(uw.requests, "get", return_value=_mock_response(404)):
        assert uw.query_unpaywall_by_doi("10.1/missing", "test@example.com") is None


def test_query_handles_422_malformed_doi():
    with patch.object(uw.requests, "get", return_value=_mock_response(422)):
        assert uw.query_unpaywall_by_doi("not-a-doi", "test@example.com") is None


def test_query_retries_on_429_then_succeeds():
    body = {"is_oa": True, "title": "T", "doi": "10.1/x",
            "best_oa_location": {"url_for_pdf": "https://p.pdf", "host_type": "publisher"}}
    responses = [_mock_response(429), _mock_response(200, body)]
    with patch.object(uw.requests, "get", side_effect=responses), \
         patch("unpaywall.time.sleep"):  # don't actually sleep in tests
        result = uw.query_unpaywall_by_doi("10.1/x", "test@example.com")
    assert result is not None
    assert result["title"] == "T"


def test_query_gives_up_after_max_retries():
    with patch.object(uw.requests, "get", return_value=_mock_response(503)), \
         patch("unpaywall.time.sleep"):
        result = uw.query_unpaywall_by_doi("10.1/x", "test@example.com", max_retries=2)
    assert result is None


def test_query_handles_network_exception():
    with patch.object(uw.requests, "get",
                      side_effect=uw._RequestException("connection refused")), \
         patch("unpaywall.time.sleep"):
        assert uw.query_unpaywall_by_doi("10.1/x", "test@example.com", max_retries=2) is None


# ---- live smoke test (opt-in) ----------------------------------------------

@pytest.mark.skipif(
    os.environ.get("UNPAYWALL_LIVE_TESTS") != "1",
    reason="set UNPAYWALL_LIVE_TESTS=1 to hit the live Unpaywall API",
)
def test_live_known_oa_paper():
    """Hit the live API for a paper that has been OA on Nature since 2013."""
    result = uw.query_unpaywall_by_doi("10.1038/nature12373", "stevens@anl.gov")
    assert result is not None
    assert "thermometry" in result["title"].lower()
    assert result["openAccessPdf"]["url"].endswith(".pdf")

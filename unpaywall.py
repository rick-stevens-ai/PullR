"""
Unpaywall API integration for PullR.

Unpaywall (https://unpaywall.org) is a free, open database of ~50 million
open-access scholarly articles. The REST API is keyless and only requires
an email address in the query string (per their terms of use).

This module exposes:
    query_unpaywall_by_doi(doi, email, ...) -> S2-shaped paper dict | None
    available()                              -> True (Unpaywall is keyless)

The returned dict mirrors the relevant subset of Semantic Scholar's paper
schema so it slots directly into PullR's existing pipeline (download_pdf,
save_abstract, etc.) without further translation.

Rate limits: ~100,000 requests/day per IP. We implement modest backoff for
429 and 5xx responses. No per-second cap is enforced — Unpaywall's docs say
"polite" usage is expected; PullR's thread count (default 1, max 10) keeps
us well inside that.
"""

import json
import time
import urllib.parse

try:
    import requests
    _HAS_REQUESTS = True
    _RequestException = requests.RequestException
except ImportError:
    requests = None  # type: ignore[assignment]
    _HAS_REQUESTS = False
    _RequestException = Exception  # type: ignore[misc,assignment]

API_BASE = "https://api.unpaywall.org/v2"
DEFAULT_TIMEOUT = 20
DEFAULT_USER_AGENT = "PullR/1.0 (research paper recovery)"


def available():
    """Unpaywall is always available (keyless API, only needs an email)."""
    return _HAS_REQUESTS


def _pick_pdf_url(uw_record):
    """
    Pick the best PDF URL from an Unpaywall record.

    Preference order:
        1. best_oa_location.url_for_pdf  (direct PDF link)
        2. best_oa_location.url          (landing page that often serves PDF)
        3. first oa_location with url_for_pdf
        4. first oa_location with url

    Returns (url, host_type, evidence) or (None, None, None).
    """
    best = uw_record.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        return best["url_for_pdf"], best.get("host_type"), best.get("evidence")
    if best.get("url"):
        return best["url"], best.get("host_type"), best.get("evidence")
    for loc in uw_record.get("oa_locations") or []:
        if loc.get("url_for_pdf"):
            return loc["url_for_pdf"], loc.get("host_type"), loc.get("evidence")
    for loc in uw_record.get("oa_locations") or []:
        if loc.get("url"):
            return loc["url"], loc.get("host_type"), loc.get("evidence")
    return None, None, None


def _to_s2_shape(uw_record, pdf_url, host_type):
    """
    Convert an Unpaywall record into the subset of the Semantic Scholar
    paper schema that PullR's downstream code reads.

    Fields produced (matching pullr.FIELDS):
        paperId       (synthesized from DOI: "uw_<doi-slug>")
        title         from uw_record.title
        authors       [{"name": "Given Family"}, ...]
        year          uw_record.year
        externalIds   {"DOI": ..., "PMID": ..., "PMCID": ...}
        url           uw_record.doi_url (landing page)
        venue         uw_record.journal_name
        openAccessPdf {"url": pdf_url, "status": "open"} or None
        abstract      None (Unpaywall does not provide abstracts)
        _unpaywall    raw Unpaywall record (kept for debugging)
    """
    doi = uw_record.get("doi") or ""
    paper_id = "uw_" + doi.replace("/", "_").replace(":", "_")[:80] if doi else None

    authors = []
    for a in uw_record.get("z_authors") or []:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        name = (given + " " + family).strip() or a.get("name") or ""
        if name:
            authors.append({"name": name})

    ext_ids = {}
    if doi:
        ext_ids["DOI"] = doi
    # Unpaywall does not return PMID/PMCID directly, but oa_locations may include them.
    for loc in uw_record.get("oa_locations") or []:
        if loc.get("pmh_id"):
            ext_ids.setdefault("PMH", loc["pmh_id"])

    return {
        "paperId": paper_id,
        "title": uw_record.get("title") or "",
        "authors": authors,
        "year": uw_record.get("year"),
        "externalIds": ext_ids,
        "url": uw_record.get("doi_url") or "",
        "venue": uw_record.get("journal_name") or "",
        "openAccessPdf": ({"url": pdf_url, "status": "open"} if pdf_url else None),
        "abstract": None,
        "_unpaywall": {
            "host_type": host_type,
            "is_oa": uw_record.get("is_oa"),
            "oa_status": uw_record.get("oa_status"),
        },
    }


def query_unpaywall_by_doi(doi, email, timeout=DEFAULT_TIMEOUT, max_retries=3, verbose=False):
    """
    Look up a DOI in Unpaywall and return an S2-shaped paper dict if an OA
    copy is available, else None.

    Args:
        doi:         e.g. "10.1038/nature12373" (case-insensitive, Unpaywall normalizes)
        email:       contact email required by Unpaywall terms of use
        timeout:     per-request timeout in seconds
        max_retries: retry budget for 429/5xx (exponential backoff)
        verbose:     print progress lines

    Returns:
        dict in S2 paper shape (see _to_s2_shape), or None if no DOI / not
        found / not open access / network failure.
    """
    if not _HAS_REQUESTS:
        if verbose:
            print("    [unpaywall] requests not installed, skipping")
        return None
    if not doi:
        return None
    if not email:
        if verbose:
            print("    [unpaywall] no email provided, skipping (required by TOS)")
        return None

    doi_clean = doi.strip()
    # strip common "doi:" / "DOI:" / "https://doi.org/" prefixes
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:", "DOI:"):
        if doi_clean.lower().startswith(prefix.lower()):
            doi_clean = doi_clean[len(prefix):].strip()
            break

    enc = urllib.parse.quote(doi_clean, safe="/")
    url = f"{API_BASE}/{enc}?email={urllib.parse.quote(email)}"
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)  # type: ignore[union-attr]
        except _RequestException as e:
            if verbose:
                print(f"    [unpaywall] network error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

        if r.status_code == 200:
            try:
                data = r.json()
            except (json.JSONDecodeError, ValueError) as e:
                if verbose:
                    print(f"    [unpaywall] bad JSON for DOI {doi_clean}: {e}")
                return None
            if not data.get("is_oa"):
                if verbose:
                    print(f"    [unpaywall] DOI {doi_clean}: not open access")
                return None
            pdf_url, host_type, evidence = _pick_pdf_url(data)
            if not pdf_url:
                if verbose:
                    print(f"    [unpaywall] DOI {doi_clean}: is_oa but no usable URL")
                return None
            if verbose:
                print(f"    [unpaywall] DOI {doi_clean}: OA via {host_type or '?'} ({evidence or '?'})")
            return _to_s2_shape(data, pdf_url, host_type)

        if r.status_code == 404:
            if verbose:
                print(f"    [unpaywall] DOI {doi_clean}: not in Unpaywall (404)")
            return None

        if r.status_code == 422:
            # Malformed DOI per Unpaywall
            if verbose:
                print(f"    [unpaywall] DOI {doi_clean}: rejected as malformed (422)")
            return None

        if r.status_code == 429 or 500 <= r.status_code < 600:
            wait = (attempt + 1) * 3
            if verbose:
                print(f"    [unpaywall] HTTP {r.status_code} on attempt {attempt+1}, waiting {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
                continue
            return None

        if verbose:
            print(f"    [unpaywall] unexpected HTTP {r.status_code} for DOI {doi_clean}")
        return None

    return None


def extract_doi_from_ref(ref_text):
    """
    Heuristic DOI extractor from a freeform citation. Returns the first
    plausible DOI string, or None.

    Matches:
        10.NNNN/anything-non-whitespace  (the CrossRef DOI shape)
    """
    if not ref_text:
        return None
    import re
    # Trim trailing punctuation that often clings to citation DOIs
    m = re.search(r"\b(10\.\d{4,9}/[^\s)>\]]+)", ref_text)
    if not m:
        return None
    doi = m.group(1).rstrip(".,;)>]")
    return doi

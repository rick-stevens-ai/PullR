"""
OpenAlex API integration for PullR.

OpenAlex (https://openalex.org) is a free, open catalog of >250 million
scholarly works. The REST API is keyless; an email in the User-Agent or
query string (the "polite pool") gets you 100k req/day with stricter SLAs.

This module exposes:
    query_openalex_by_doi(doi, email, ...) -> S2-shaped paper dict | None
    search_openalex_by_title(title, email, ..., year=None, author=None)
        -> list of S2-shaped paper dicts (best-match first), possibly empty
    search_openalex_full_text(query, email, ..., limit=10)
        -> list of S2-shaped paper dicts (full-text/abstract search)
    available()                            -> True (OpenAlex is keyless)

Returned dicts mirror the relevant subset of Semantic Scholar's paper
schema so the rest of PullR's pipeline runs unchanged.

Coverage notes vs S2 + Unpaywall:
  - OpenAlex has stronger coverage of European/non-English journals,
    preprints, gray literature, conference papers, books.
  - OpenAlex provides reconstructed abstracts (from an inverted index) for
    most works — unlike Unpaywall (DOI-only, no abstracts).
  - OpenAlex returns OA links even when Unpaywall doesn't because it
    indexes more repositories.
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

API_BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT = 25
DEFAULT_USER_AGENT_TMPL = "PullR/1.0 (mailto:{email})"


def available():
    """OpenAlex is always available (keyless API; email is recommended but optional)."""
    return _HAS_REQUESTS


def _ua(email):
    return DEFAULT_USER_AGENT_TMPL.format(email=email or "anonymous@example.org")


def _reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index:
      {"word": [pos1, pos2, ...], ...}
    Reconstruct the original text. Returns None if input is empty/missing.
    """
    if not inverted_index:
        return None
    try:
        positions = []
        for word, pos_list in inverted_index.items():
            for p in pos_list:
                positions.append((p, word))
        positions.sort()
        return " ".join(w for _, w in positions)
    except Exception:
        return None


def _pick_pdf_url(work):
    """
    Pick the best PDF URL from an OpenAlex work record.

    Preference:
        1. best_oa_location.pdf_url
        2. primary_location.pdf_url    (if is_oa)
        3. any locations[*].pdf_url    (first with pdf_url)
        4. best_oa_location.landing_page_url
        5. primary_location.landing_page_url (if is_oa)

    Returns (url, host_type, evidence) — host_type/evidence may be None.
    """
    best = work.get("best_oa_location") or {}
    if best.get("pdf_url"):
        return best["pdf_url"], best.get("host_type") or (best.get("source") or {}).get("type"), best.get("license")
    primary = work.get("primary_location") or {}
    if primary.get("is_oa") and primary.get("pdf_url"):
        return primary["pdf_url"], primary.get("host_type") or (primary.get("source") or {}).get("type"), primary.get("license")
    for loc in work.get("locations") or []:
        if loc.get("pdf_url"):
            return loc["pdf_url"], loc.get("host_type") or (loc.get("source") or {}).get("type"), loc.get("license")
    if best.get("landing_page_url"):
        return best["landing_page_url"], best.get("host_type"), best.get("license")
    if primary.get("is_oa") and primary.get("landing_page_url"):
        return primary["landing_page_url"], primary.get("host_type"), primary.get("license")
    return None, None, None


def _short_doi_from_url(doi_field):
    """OpenAlex returns DOIs as 'https://doi.org/10.x/y' — strip to bare DOI."""
    if not doi_field:
        return None
    s = doi_field
    for p in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if s.lower().startswith(p):
            return s[len(p):]
    return s


def _to_s2_shape(work):
    """
    Convert an OpenAlex work record into the subset of the Semantic Scholar
    paper schema that PullR's downstream code reads.
    """
    doi = _short_doi_from_url(work.get("doi"))
    oa_id = work.get("id", "")  # like 'https://openalex.org/W123...'
    short_id = oa_id.rsplit("/", 1)[-1] if oa_id else None
    paper_id = ("oa_" + short_id) if short_id else None

    authors = []
    for a in work.get("authorships") or []:
        au = a.get("author") or {}
        name = au.get("display_name") or ""
        if name:
            authors.append({"name": name})

    ext_ids = {}
    ids_block = work.get("ids") or {}
    if doi:
        ext_ids["DOI"] = doi
    if ids_block.get("pmid"):
        pm = ids_block["pmid"]
        ext_ids["PMID"] = pm.rsplit("/", 1)[-1] if "/" in pm else pm
    if ids_block.get("pmcid"):
        ext_ids["PMCID"] = ids_block["pmcid"]
    if ids_block.get("mag"):
        ext_ids["MAG"] = str(ids_block["mag"])
    if short_id:
        ext_ids["OpenAlex"] = short_id

    pdf_url, host_type, license_ = _pick_pdf_url(work)
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    venue = source.get("display_name") or ""

    return {
        "paperId": paper_id,
        "title": work.get("title") or work.get("display_name") or "",
        "authors": authors,
        "year": work.get("publication_year"),
        "externalIds": ext_ids,
        "url": work.get("doi") or oa_id or "",
        "venue": venue,
        "openAccessPdf": ({"url": pdf_url, "status": "open"} if pdf_url else None),
        "abstract": abstract,
        "_openalex": {
            "id": short_id,
            "host_type": host_type,
            "license": license_,
            "type": work.get("type"),
            "cited_by_count": work.get("cited_by_count"),
            "is_oa": (work.get("open_access") or {}).get("is_oa"),
            "oa_status": (work.get("open_access") or {}).get("oa_status"),
        },
    }


def _request(url, email, timeout=DEFAULT_TIMEOUT, max_retries=3, verbose=False):
    """Wrapper around requests.get with retries on 429/5xx."""
    if not _HAS_REQUESTS:
        return None
    headers = {"User-Agent": _ua(email), "Accept": "application/json"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)  # type: ignore[union-attr]
        except _RequestException as e:
            if verbose:
                print(f"    [openalex] network error attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        if r.status_code == 200:
            try:
                return r.json()
            except (ValueError, json.JSONDecodeError) as e:
                if verbose:
                    print(f"    [openalex] bad JSON: {e}")
                return None
        if r.status_code == 404:
            return None
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            wait = (attempt + 1) * 3
            if verbose:
                print(f"    [openalex] HTTP {r.status_code} attempt {attempt+1}, waiting {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
                continue
            return None
        if verbose:
            print(f"    [openalex] HTTP {r.status_code} {r.text[:120]}")
        return None
    return None


def query_openalex_by_doi(doi, email, timeout=DEFAULT_TIMEOUT, max_retries=3, verbose=False):
    """
    Look up a DOI in OpenAlex. Returns an S2-shaped paper dict if found,
    else None. Unlike Unpaywall, OpenAlex returns abstracts too.
    """
    if not _HAS_REQUESTS or not doi:
        return None
    doi_clean = doi.strip()
    for p in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:", "DOI:"):
        if doi_clean.lower().startswith(p.lower()):
            doi_clean = doi_clean[len(p):].strip()
            break
    url = f"{API_BASE}/works/https://doi.org/{urllib.parse.quote(doi_clean, safe='/')}"
    if email:
        url += f"?mailto={urllib.parse.quote(email)}"
    data = _request(url, email, timeout, max_retries, verbose)
    if data is None or not data.get("id"):
        if verbose:
            print(f"    [openalex] DOI {doi_clean}: not found")
        return None
    if verbose:
        print(f"    [openalex] DOI {doi_clean}: found '{(data.get('title') or '')[:60]}...'")
    return _to_s2_shape(data)


def _title_similarity(a, b):
    """Simple normalized token overlap, 0..1. Used to rank title matches."""
    if not a or not b:
        return 0.0
    import re as _re
    def toks(s):
        return set(_re.findall(r"[a-z0-9]+", s.lower()))
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    return len(inter) / max(len(ta | tb), 1)


def search_openalex_by_title(title, email, year=None, author=None, limit=5,
                              timeout=DEFAULT_TIMEOUT, max_retries=3, verbose=False):
    """
    Title-based search. Returns a list of S2-shaped paper dicts (best
    title-similarity first), possibly empty. If `year` is provided, filters
    to ±1 year of the requested year. If `author` provided, biases ranking.
    """
    if not _HAS_REQUESTS or not title:
        return []
    title_clean = title.strip().strip('"').strip()
    if len(title_clean) < 5:
        return []
    # OpenAlex title search is a filter, not a top-level query param.
    # `search=` is full-text; `filter=title.search:...` matches against title only.
    filters = [f"title.search:{title_clean}"]
    # Year filter is NOT applied at API level here because OpenAlex sometimes
    # re-indexes classic papers under newer DOIs with the original year lost.
    # We rely on title-similarity ranking + post-filter to handle year.
    params = {
        "filter": ",".join(filters),
        "per-page": min(max(limit * 4, 10), 25),  # over-fetch for better re-ranking
        "sort": "cited_by_count:desc",
        "select": "id,doi,title,display_name,publication_year,publication_date,authorships,"
                  "primary_location,best_oa_location,locations,open_access,ids,"
                  "abstract_inverted_index,type,cited_by_count",
    }
    if email:
        params["mailto"] = email
    url = API_BASE + "/works?" + urllib.parse.urlencode(params, safe=":|")
    data = _request(url, email, timeout, max_retries, verbose)
    if not data or not data.get("results"):
        if verbose:
            print(f"    [openalex] title search '{title_clean[:60]}': 0 results")
        return []
    # Rank by title similarity (OpenAlex's own search is decent but noisy)
    scored = []
    for w in data["results"]:
        cand_title = w.get("title") or w.get("display_name") or ""
        score = _title_similarity(title_clean, cand_title)
        # Drop obvious non-matches (sim < 0.4 unless it's a near-perfect substring match)
        if score < 0.4 and title_clean.lower() not in cand_title.lower():
            continue
        # Author bonus
        if author:
            author_str = author.split(",")[0].split()[-1].lower() if author else ""
            for a in w.get("authorships") or []:
                au_name = ((a.get("author") or {}).get("display_name") or "").lower()
                if author_str and author_str in au_name:
                    score += 0.15
                    break
        # Year bonus / penalty (post-filter rather than API filter)
        if year:
            try:
                target_y = int(year)
                pub_y = w.get("publication_year")
                if pub_y is not None:
                    delta = abs(int(pub_y) - target_y)
                    if delta <= 1:
                        score += 0.10
                    elif delta <= 3:
                        score += 0.03
            except (TypeError, ValueError):
                pass
        # Slight bonus for high citation count (helps surface the canonical version)
        cites = w.get("cited_by_count") or 0
        if cites > 1000: score += 0.05
        if cites > 10000: score += 0.05
        scored.append((score, w))
    scored.sort(reverse=True, key=lambda x: x[0])
    if verbose:
        top_score = scored[0][0] if scored else 0
        print(f"    [openalex] title search '{title_clean[:60]}': "
              f"{len(scored)} results, top sim={top_score:.2f}")
    return [_to_s2_shape(w) for _, w in scored[:limit]]


def search_openalex_full_text(query, email, limit=10, year_min=None, year_max=None,
                               timeout=DEFAULT_TIMEOUT, max_retries=3, verbose=False):
    """
    Full-text/abstract search via OpenAlex's `search` parameter. Useful for
    topic-discovery (find papers about X) rather than reference matching.
    """
    if not _HAS_REQUESTS or not query:
        return []
    params = {
        "search": query.strip(),
        "per-page": min(max(limit, 5), 50),
        "sort": "relevance_score:desc",
        "select": "id,doi,title,display_name,publication_year,publication_date,authorships,"
                  "primary_location,best_oa_location,locations,open_access,ids,"
                  "abstract_inverted_index,type,cited_by_count",
    }
    if email:
        params["mailto"] = email
    filters = []
    if year_min:
        try:
            filters.append(f"from_publication_date:{int(year_min)}-01-01")
        except (TypeError, ValueError): pass
    if year_max:
        try:
            filters.append(f"to_publication_date:{int(year_max)}-12-31")
        except (TypeError, ValueError): pass
    if filters:
        params["filter"] = ",".join(filters)
    url = API_BASE + "/works?" + urllib.parse.urlencode(params)
    data = _request(url, email, timeout, max_retries, verbose)
    if not data or not data.get("results"):
        return []
    if verbose:
        print(f"    [openalex] topic search '{query[:60]}': {len(data['results'])} results")
    return [_to_s2_shape(w) for w in data["results"][:limit]]


def extract_doi_from_ref(ref_text):
    """Convenience re-export for parity with unpaywall.extract_doi_from_ref."""
    if not ref_text:
        return None
    import re as _re
    m = _re.search(r"\b(10\.\d{4,9}/[^\s)>\]]+)", ref_text)
    if not m:
        return None
    return m.group(1).rstrip(".,;)>]")

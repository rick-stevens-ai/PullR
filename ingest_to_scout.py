#!/usr/bin/env python3
"""
Ingest PullR-tree PDFs into the SCOUT paper-index DB.

Adds (or updates) papers from any directory tree under PullR/ into
~/paper-index/data/papers.sqlite. Designed to be run periodically
or after a fresh PullR session.

Handles three kinds of PDFs:

  1. SHA1-named (40 hex chars) -> these are Semantic Scholar paperIDs.
     If already in SCOUT, we just add the new path as another known copy.
     If not, we look up via S2 batch API.

  2. OpenAlex-named (oa_W<digits>.pdf) -> from PullR's new openalex
     strategy. We look up via OpenAlex REST and ingest with paper_id=oa_W...

  3. Unpaywall-named (uw_<...>.pdf) -> from PullR's unpaywall strategy.
     We re-fetch by stored DOI via OpenAlex (better metadata) or Unpaywall.

  4. Other (anything else) -> compute sha1 of bytes for de-dup; ingest with
     paper_id = "p:" + that hash; try to extract title/DOI from first 2
     pages via pdftotext.

Records a 'corpus' label of 'pullr-<subdir>' so PullR sources stay grouped.

Usage:
    python ingest_pullr_to_scout.py              # default: ~/Dropbox/PullR
    python ingest_pullr_to_scout.py /path/dir    # custom root
    python ingest_pullr_to_scout.py --dry-run    # report only
"""
import os, sys, re, sqlite3, hashlib, time, subprocess, json, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DROPBOX = Path.home() / "Dropbox"
SCOUT_DB = Path.home() / "paper-index" / "data" / "papers.sqlite"

SHA1_RE = re.compile(r"^([0-9a-f]{40})$", re.I)
OA_ID_RE = re.compile(r"^oa_(W\d+)$", re.I)
UW_ID_RE = re.compile(r"^uw_(.+)$", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)


def classify_filename(path):
    """Decide what kind of paper_id this PDF maps to.

    Returns dict with keys: kind, paper_id, ext_lookup (DOI/OA-id/None).
    """
    stem = Path(path).stem
    # Strip trailing " 2", "(1)", etc. artifacts
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"\s+\d+$", "", stem)
    m = SHA1_RE.match(stem)
    if m:
        return {"kind": "s2_sha1", "paper_id": m.group(1).lower(), "ext_lookup": m.group(1).lower()}
    m = OA_ID_RE.match(stem)
    if m:
        return {"kind": "openalex", "paper_id": "oa_" + m.group(1), "ext_lookup": m.group(1)}
    m = UW_ID_RE.match(stem)
    if m:
        return {"kind": "unpaywall", "paper_id": "uw_" + m.group(1), "ext_lookup": None}
    # arxiv_NNNN.NNNNN or arxiv_cat_NNNNNNN
    m = re.match(r"^arxiv[_\-]([a-z\-]+[_\-])?([0-9]{4}\.[0-9]{4,5}|[0-9]{7})", stem, re.I)
    if m:
        prefix, num = m.group(1), m.group(2)
        arxiv_id = (prefix.rstrip("_-") + "/" + num) if prefix else num
        return {"kind": "arxiv", "paper_id": "ax_" + arxiv_id.replace("/", "_"),
                "ext_lookup": arxiv_id}
    return {"kind": "unknown", "paper_id": None, "ext_lookup": None}


def file_sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def extract_pdf_meta(path, timeout=15):
    """Quick pdftotext on first 2 pages; return (title_guess, doi)."""
    try:
        r = subprocess.run(
            ["pdftotext", "-l", "2", "-nopgbrk", str(path), "-"],
            capture_output=True, text=True, timeout=timeout
        )
        txt = r.stdout[:5000]
    except Exception:
        return None, None
    if not txt or len(txt) < 100:
        return None, None
    # DOI
    m = DOI_RE.search(txt)
    doi = m.group(1).rstrip(".,;)") if m else None
    # Title heuristic
    title = None
    for line in (l.strip() for l in txt.split("\n")[:12]):
        if not line or len(line) < 20: continue
        if re.match(r"^[\d\.\s]+$", line): continue
        low = line.lower()
        if any(s in low for s in ("doi:", "http", "vol.", "©", "received", "abstract")): continue
        title = line[:250]
        break
    return title, doi


def s2_batch_lookup(sha1s, fields="title,abstract,year,venue,authors,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,referenceCount,citationCount,influentialCitationCount,openAccessPdf,journal", batch=400, verbose=False):
    """Bulk lookup of S2 paper IDs (SHA1 hex). Returns dict sha1 -> S2 record or None."""
    import requests
    out = {}
    for i in range(0, len(sha1s), batch):
        chunk = sha1s[i:i+batch]
        for attempt in range(6):
            try:
                r = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": fields},
                    json={"ids": chunk}, timeout=60,
                )
                if r.status_code == 200:
                    for sha, item in zip(chunk, r.json()):
                        out[sha] = item
                    break
                if r.status_code in (429, 502, 503, 504):
                    if verbose: print(f"  [s2] HTTP {r.status_code}, backing off")
                    time.sleep(3 + attempt * 3)
                    continue
                if verbose: print(f"  [s2] HTTP {r.status_code}: {r.text[:120]}")
                break
            except Exception as e:
                if verbose: print(f"  [s2] err {e}")
                time.sleep(3)
        time.sleep(1.2)  # polite spacing between batches
        if verbose: print(f"  [s2] processed {min(i+batch, len(sha1s)):,}/{len(sha1s):,}")
    return out


def openalex_lookup_by_id(oa_short_id, email="stevens@anl.gov", verbose=False):
    """oa_short_id like 'W3177828909'. Returns OpenAlex work record or None."""
    import requests
    url = f"https://api.openalex.org/works/{oa_short_id}?mailto={email}"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": f"PullR/1.0 (mailto:{email})"})
        if r.status_code == 200: return r.json()
    except Exception as e:
        if verbose: print(f"  [openalex] err {e}")
    return None


def openalex_lookup_by_doi(doi, email="stevens@anl.gov", verbose=False):
    import requests, urllib.parse
    if doi.lower().startswith(("https://doi.org/", "http://doi.org/", "doi:", "doi.org/")):
        for p in ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/"):
            if doi.lower().startswith(p.lower()):
                doi = doi[len(p):].strip(); break
    url = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(doi, safe='/')}?mailto={email}"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": f"PullR/1.0 (mailto:{email})"})
        if r.status_code == 200: return r.json()
    except Exception as e:
        if verbose: print(f"  [openalex-doi] err {e}")
    return None


def _reconstruct_abstract(inv):
    if not inv: return None
    try:
        pairs = []
        for w, ps in inv.items():
            for p in ps: pairs.append((p, w))
        pairs.sort()
        return " ".join(w for _, w in pairs)
    except Exception:
        return None


def s2_record_to_paper_row(sha, item):
    """Map an S2 API record into our papers-table row dict."""
    if item is None:
        return None
    ext = item.get("externalIds") or {}
    authors = item.get("authors") or []
    s2fos = item.get("s2FieldsOfStudy") or []
    return {
        "paper_id": sha,
        "sha1": sha,
        "source_kind": "sha1_named",
        "title": item.get("title"),
        "abstract": item.get("abstract"),
        "year": item.get("year"),
        "venue": item.get("venue"),
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
        "pmid": ext.get("PubMed"),
        "corpus_id": ext.get("CorpusId"),
        "citationCount": item.get("citationCount"),
        "influentialCitationCount": item.get("influentialCitationCount"),
        "referenceCount": item.get("referenceCount"),
        "publicationDate": item.get("publicationDate"),
        "publicationTypes": ", ".join(item.get("publicationTypes") or []) if item.get("publicationTypes") else None,
        "author_names": ", ".join(a.get("name", "") for a in authors[:30]) if authors else None,
        "s2_fos": ", ".join(f.get("category") for f in s2fos if f.get("category")) if s2fos else None,
    }


def oa_record_to_paper_row(work, paper_id):
    if work is None:
        return None
    ids = work.get("ids") or {}
    doi = work.get("doi")
    if doi and doi.startswith("https://doi.org/"): doi = doi[len("https://doi.org/"):]
    authors = work.get("authorships") or []
    source = (work.get("primary_location") or {}).get("source") or {}
    return {
        "paper_id": paper_id,
        "sha1": None,
        "source_kind": "openalex",
        "title": work.get("title") or work.get("display_name"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "venue": source.get("display_name"),
        "doi": doi,
        "arxiv_id": None,
        "pmid": (ids.get("pmid") or "").rsplit("/", 1)[-1] if ids.get("pmid") else None,
        "corpus_id": None,
        "citationCount": work.get("cited_by_count"),
        "influentialCitationCount": None,
        "referenceCount": None,
        "publicationDate": work.get("publication_date"),
        "publicationTypes": work.get("type"),
        "author_names": ", ".join((a.get("author") or {}).get("display_name", "") for a in authors[:30]) if authors else None,
        "s2_fos": None,
    }


def heur_record_for_unknown(paper_id, title, doi, source_kind="pullr_unknown"):
    return {
        "paper_id": paper_id,
        "sha1": None,
        "source_kind": source_kind,
        "title": title,
        "abstract": None,
        "year": None,
        "venue": None,
        "doi": doi,
        "arxiv_id": None,
        "pmid": None,
        "corpus_id": None,
        "citationCount": None,
        "influentialCitationCount": None,
        "referenceCount": None,
        "publicationDate": None,
        "publicationTypes": None,
        "author_names": None,
        "s2_fos": None,
    }


PAPER_COLS = ["paper_id","sha1","source_kind","title","abstract","year","year_bucket",
              "venue","doi","arxiv_id","pmid","corpus_id","citationCount",
              "influentialCitationCount","referenceCount","publicationDate",
              "publicationTypes","author_names","s2_fos","home_dir","paths_str",
              "n_copies","abstract_source"]


def year_bucket(y):
    if y is None: return "unknown"
    try: y = int(y)
    except: return "unknown"
    if y < 2000: return "pre-2000"
    if y < 2010: return "2000-2009"
    if y < 2015: return "2010-2014"
    if y < 2020: return "2015-2019"
    if y < 2023: return "2020-2022"
    if y < 2025: return "2023-2024"
    return "2025-plus"


def upsert_paper(con, row, path_rel):
    """INSERT-OR-UPDATE the paper, append path to paths_str list."""
    pid = row["paper_id"]
    if not pid: return False
    cur = con.execute("SELECT paths_str, n_copies FROM papers WHERE paper_id=?", (pid,)).fetchone()
    if cur is not None:
        # Already there; just add this path if not present
        existing_paths = (cur[0] or "").split(" | ") if cur[0] else []
        if path_rel in existing_paths:
            return False
        new_paths = existing_paths + [path_rel]
        new_n = (cur[1] or 1) + 1
        con.execute(
            "UPDATE papers SET paths_str=?, n_copies=?, "
            "home_dir = COALESCE(home_dir, ?) "
            "WHERE paper_id=?",
            (" | ".join(new_paths), new_n, "/".join(path_rel.split("/")[:-1]), pid)
        )
        return False  # not "new"
    # New row
    row["year_bucket"] = year_bucket(row.get("year"))
    row["home_dir"] = "/".join(path_rel.split("/")[:-1])
    row["paths_str"] = path_rel
    row["n_copies"] = 1
    row.setdefault("abstract_source", None)
    cols = [c for c in PAPER_COLS if c in row]
    placeholders = ",".join("?" * len(cols))
    con.execute(
        f"INSERT INTO papers ({','.join(cols)}) VALUES ({placeholders})",
        tuple(row[c] for c in cols)
    )
    return True


def add_corpus_label(con, paper_id, corpus_value, source="pullr_ingest"):
    """Add (or refresh) a corpus label for a PullR-sourced paper."""
    existing = con.execute(
        "SELECT 1 FROM labels WHERE paper_id=? AND facet='corpus' AND value=?",
        (paper_id, corpus_value)
    ).fetchone()
    if existing: return False
    con.execute(
        "INSERT INTO labels (paper_id, facet, value, confidence, source) VALUES (?,?,?,?,?)",
        (paper_id, "corpus", corpus_value, 1.0, source)
    )
    return True


def main():
    ap = argparse.ArgumentParser(description="Ingest PullR PDFs into SCOUT")
    ap.add_argument("root", nargs="?", default=str(Path.home() / "Dropbox" / "PullR"),
                    help="root directory to scan (default: ~/Dropbox/PullR)")
    ap.add_argument("--dry-run", action="store_true",
                    help="just report what would be ingested, do not write to DB")
    ap.add_argument("--db", default=str(SCOUT_DB),
                    help=f"SCOUT sqlite path (default: {SCOUT_DB})")
    ap.add_argument("--corpus-prefix", default="pullr",
                    help="corpus label prefix (default: 'pullr'); subdir name appended")
    ap.add_argument("--email", default="stevens@anl.gov",
                    help="OpenAlex polite-pool email")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--limit", type=int, help="limit number of new papers (for testing)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr); sys.exit(2)
    if not Path(args.db).exists():
        print(f"ERROR: SCOUT db {args.db} does not exist", file=sys.stderr); sys.exit(2)

    t0 = time.time()
    # Walk
    pdfs = []
    for p in root.rglob("*.pdf"):
        if p.is_file() and p.stat().st_size > 1000:
            pdfs.append(p)
    print(f"Found {len(pdfs):,} PDFs under {root}", flush=True)

    # Classify
    classes = {"s2_sha1": [], "openalex": [], "unpaywall": [], "arxiv": [], "unknown": []}
    for p in pdfs:
        info = classify_filename(p)
        info["path"] = p
        classes[info["kind"]].append(info)
    print(f"  s2_sha1:   {len(classes['s2_sha1']):,}", flush=True)
    print(f"  openalex:  {len(classes['openalex']):,}", flush=True)
    print(f"  unpaywall: {len(classes['unpaywall']):,}", flush=True)
    print(f"  arxiv:     {len(classes['arxiv']):,}", flush=True)
    print(f"  unknown:   {len(classes['unknown']):,}", flush=True)

    if args.dry_run:
        print("\n(dry-run; no DB writes)")
        return

    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=OFF")
    # Ensure indexes exist
    con.execute("CREATE INDEX IF NOT EXISTS idx_papers_pid ON papers(paper_id)")
    con.commit()

    new_count = 0; updated_count = 0; lookup_count = 0
    DROPBOX_R = Path.home() / "Dropbox"

    def rel_path(p):
        try: return str(Path(p).relative_to(DROPBOX_R))
        except ValueError: return str(p)

    # ----- s2_sha1: figure out which need lookup -----
    already_known = set(
        r[0] for r in con.execute("SELECT paper_id FROM papers WHERE source_kind='sha1_named'")
    )
    s2_to_lookup = sorted(set(i["paper_id"] for i in classes["s2_sha1"]
                              if i["paper_id"] not in already_known))
    if args.limit:
        s2_to_lookup = s2_to_lookup[:args.limit]
    print(f"\nS2 SHA1s to look up via API: {len(s2_to_lookup)}", flush=True)
    s2_records = {}
    if s2_to_lookup:
        s2_records = s2_batch_lookup(s2_to_lookup, verbose=args.verbose)
        lookup_count += len(s2_records)

    # Process s2_sha1 group
    for info in classes["s2_sha1"]:
        pid = info["paper_id"]
        if pid in already_known:
            # Just add this path
            if upsert_paper(con, {"paper_id": pid}, rel_path(info["path"])):
                pass  # never returns True for known papers
            else:
                updated_count += 1
        else:
            rec = s2_records.get(pid)
            row = s2_record_to_paper_row(pid, rec)
            if row is None:
                # S2 doesn't know it; create a minimal stub so SCOUT still tracks the file
                row = heur_record_for_unknown(pid, None, None, source_kind="sha1_named_no_s2")
                row["sha1"] = pid
            if upsert_paper(con, row, rel_path(info["path"])):
                new_count += 1
        # Corpus label
        subdir = info["path"].relative_to(root).parts[0] if info["path"].is_relative_to(root) else "root"
        add_corpus_label(con, pid, f"{args.corpus_prefix}-{subdir.lower()}")
    con.commit()
    print(f"After s2_sha1: new={new_count}, updated={updated_count}", flush=True)

    # ----- openalex group -----
    already_oa = set(r[0] for r in con.execute("SELECT paper_id FROM papers WHERE paper_id LIKE 'oa_%'"))
    oa_to_lookup = [i for i in classes["openalex"] if i["paper_id"] not in already_oa]
    if args.limit: oa_to_lookup = oa_to_lookup[:args.limit]
    print(f"\nOpenAlex IDs to look up: {len(oa_to_lookup)}", flush=True)
    for i, info in enumerate(oa_to_lookup):
        if args.verbose and i % 20 == 0: print(f"  [oa] {i}/{len(oa_to_lookup)}", flush=True)
        work = openalex_lookup_by_id(info["ext_lookup"], email=args.email, verbose=args.verbose)
        row = oa_record_to_paper_row(work, info["paper_id"])
        if row is None:
            row = heur_record_for_unknown(info["paper_id"], None, None, source_kind="openalex_unresolved")
        if upsert_paper(con, row, rel_path(info["path"])):
            new_count += 1
        subdir = info["path"].relative_to(root).parts[0] if info["path"].is_relative_to(root) else "root"
        add_corpus_label(con, info["paper_id"], f"{args.corpus_prefix}-{subdir.lower()}")
        time.sleep(0.15)  # polite spacing
    # Also: existing oa_ papers that show up again -> add new path
    for info in classes["openalex"]:
        if info["paper_id"] in already_oa:
            if upsert_paper(con, {"paper_id": info["paper_id"]}, rel_path(info["path"])):
                pass
            else:
                updated_count += 1
    con.commit()

    # ----- unknown group: hash for de-dup, pdftotext for title/DOI -----
    print(f"\nUnknown PDFs to process: {len(classes['unknown'])}", flush=True)
    for info in classes["unknown"]:
        path = info["path"]
        rel = rel_path(path)
        # First check: is this path already in the DB? (e.g. from the original
        # Dropbox walk that used p:<sha1(path)> as the paper_id.)
        existing_by_path = con.execute(
            "SELECT paper_id FROM papers WHERE paths_str LIKE ? LIMIT 1",
            ('%' + rel + '%',)
        ).fetchone()
        if existing_by_path:
            pid = existing_by_path[0]
            # Make sure it has a corpus label
            subdir = path.relative_to(root).parts[0] if path.is_relative_to(root) else "root"
            corpus_val = f"{args.corpus_prefix}-{subdir.lower()}" if subdir != path.name else f"{args.corpus_prefix}-root"
            add_corpus_label(con, pid, corpus_val)
            updated_count += 1
            continue
        # Hash to detect content-level dup
        try: sha = file_sha256(path)
        except Exception: continue
        pid = "p:" + sha[:40]
        existing = con.execute("SELECT 1 FROM papers WHERE paper_id=?", (pid,)).fetchone()
        if existing:
            if upsert_paper(con, {"paper_id": pid}, rel_path(path)):
                pass
            else:
                updated_count += 1
            continue
        # Extract title + DOI
        title, doi = extract_pdf_meta(path)
        # If we got a DOI, hand off to OpenAlex for proper metadata
        if doi:
            work = openalex_lookup_by_doi(doi, email=args.email, verbose=args.verbose)
            if work:
                row = oa_record_to_paper_row(work, pid)
                if row: row["source_kind"] = "pullr_pdf_doi"
            else:
                row = heur_record_for_unknown(pid, title, doi, source_kind="pullr_pdf_doi_unresolved")
        else:
            row = heur_record_for_unknown(pid, title, None, source_kind="pullr_pdf_heuristic")
        if upsert_paper(con, row, rel_path(path)):
            new_count += 1
        subdir = path.relative_to(root).parts[0] if path.is_relative_to(root) else "root"
        corpus_val = f"{args.corpus_prefix}-{subdir.lower()}" if subdir != path.name else f"{args.corpus_prefix}-root"
        add_corpus_label(con, pid, corpus_val)
    con.commit()

    # arxiv group: lookup via OpenAlex by arxiv DOI
    print(f"\narXiv PDFs to look up: {len(classes['arxiv'])}", flush=True)
    for info in classes["arxiv"]:
        pid = info["paper_id"]
        existing = con.execute("SELECT 1 FROM papers WHERE paper_id=?", (pid,)).fetchone()
        if existing:
            if upsert_paper(con, {"paper_id": pid}, rel_path(info["path"])):
                pass
            else:
                updated_count += 1
            continue
        # ArXiv IDs in OpenAlex use the DOI 10.48550/arXiv.NNNN.NNNNN
        arxiv = info["ext_lookup"]
        doi_guess = f"10.48550/arXiv.{arxiv}"
        work = openalex_lookup_by_doi(doi_guess, email=args.email, verbose=args.verbose)
        if work:
            row = oa_record_to_paper_row(work, pid)
        else:
            row = heur_record_for_unknown(pid, None, None, source_kind="arxiv_unresolved")
            row["arxiv_id"] = arxiv
        if upsert_paper(con, row, rel_path(info["path"])):
            new_count += 1
        subdir = info["path"].relative_to(root).parts[0] if info["path"].is_relative_to(root) else "root"
        add_corpus_label(con, pid, f"{args.corpus_prefix}-{subdir.lower()}")
        time.sleep(0.15)
    con.commit()

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"INGEST COMPLETE in {elapsed:.0f}s")
    print(f"  new papers added:   {new_count}")
    print(f"  existing updated:   {updated_count}")
    print(f"  API lookups:        {lookup_count}")
    n_total = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_pullr = con.execute("SELECT COUNT(DISTINCT paper_id) FROM labels WHERE facet='corpus' AND value LIKE 'pullr-%'").fetchone()[0]
    print(f"  total papers in DB: {n_total:,}")
    print(f"  pullr-labeled:      {n_pullr:,}")
    print(f"\nPullR corpus distribution:")
    for v, n in con.execute("SELECT value, COUNT(*) FROM labels WHERE facet='corpus' AND value LIKE 'pullr-%' GROUP BY value ORDER BY 2 DESC LIMIT 10"):
        print(f"  {v:35s} {n:>5}")
    con.close()


if __name__ == "__main__":
    main()

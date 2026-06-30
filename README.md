# PullR - Research Paper Processor

PullR is an intelligent academic paper processing tool that extracts references from PDFs and downloads related papers using AI-powered content analysis. It combines web scraping, LLM processing, and academic database search to build comprehensive research collections.

## 🚀 Features

### Core Functionality
- **PDF Reference Extraction**: Extract references from academic PDFs using intelligent LLM-based parsing
- **Multi-Strategy Search**: Find papers using exact matching, fuzzy search, and fallback strategies
- **Web Content Processing**: Enhanced web scraping with AI-powered content extraction and formatting
- **Parallel Processing**: Fast multi-threaded downloading and processing
- **Smart Rate Limiting**: Respectful API usage with automatic backoff

### AI-Enhanced Processing
- **Intelligent Content Extraction**: LLM-powered extraction from web pages when abstracts are missing
- **Document Cleanup**: Second-pass LLM formatting for professional, readable output
- **Title Optimization**: Automatically chooses the most informative titles
- **Reference Preprocessing**: Cleans and normalizes references for better matching

### Data Sources
- **Semantic Scholar API**: Primary source for academic papers
- **OpenAlex API**: >250M works including titles, authors, abstracts, OA links (keyless; enabled by default)
- **Unpaywall API**: Free, keyless lookup of ~50 million open-access papers by DOI (enabled by default)
- **Web Scraping**: Fallback for non-DOI URLs (arXiv landing pages, ResearchGate, etc.)
- **Open Access PDFs**: Automatic download when available
- **Multiple Formats**: Handles various citation styles and formats

Search strategy order (per reference, exact mode):
  1. `openalex_doi`  — DOI → OpenAlex (instant, ~99% precise; provides abstract)
  2. `unpaywall_doi` — DOI → Unpaywall (best OA-link coverage at some publishers)
  3. `exact`         — S2 exact title + author + year search
  4. `title`         — S2 title-only search
  5. `openalex_title`— OpenAlex title.search filter with year/author re-rank
  6. (... S2 fuzzy fallbacks: author_year, year, keyword_*, text_*, etc.)

## 📋 Requirements

```bash
pip install requests pyyaml openai PyPDF2 tqdm beautifulsoup4
```

## ⚙️ Configuration

Create a `model_servers.yaml` file to configure your LLM providers:

```yaml
servers:
  - shortname: "gpt4"
    openai_api_key: "${OPENAI_API_KEY}"
    openai_api_base: "https://api.openai.com/v1"
    openai_model: "gpt-4"
    
  - shortname: "claude"
    openai_api_key: "${ANTHROPIC_API_KEY}"
    openai_api_base: "https://api.anthropic.com/v1"
    openai_model: "claude-3-sonnet-20240229"
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-key"
export SEMANTIC_SCHOLAR_API_KEY="your-ss-key"  # Optional but recommended
export UNPAYWALL_EMAIL="you@example.org"       # Optional; overrides default
```

### Unpaywall integration

PullR queries [Unpaywall](https://unpaywall.org) by DOI as one of its search
strategies. Unpaywall is a free, keyless database of ~50 million open-access
papers; it requires only a contact email per their terms of use.

The Unpaywall lookup runs **after** Semantic Scholar's exact-bibliographic
search and **before** the title-fuzzy fallbacks, so it catches papers that S2
knows about but lists as non-OA, plus papers S2 does not have at all. When a
DOI can be parsed from the reference text (or returned from `extract_reference_info`),
PullR will:

1. Hit `https://api.unpaywall.org/v2/{doi}?email=...`
2. If `is_oa=true`, pick the best PDF URL (`best_oa_location.url_for_pdf`,
   falling back to landing URLs and other `oa_locations`)
3. Wrap the response in a Semantic-Scholar-shaped paper dict so the rest of
   the pipeline (abstract save, PDF download) runs unchanged

Flags:

```bash
--no-unpaywall                     # disable Unpaywall (default: enabled)
--unpaywall-email you@example.org  # contact email (default: stevens@anl.gov,
                                   # env: UNPAYWALL_EMAIL)
--no-openalex                      # disable OpenAlex (default: enabled)
--openalex-email you@example.org   # "polite pool" email (env: OPENALEX_EMAIL,
                                   # falls back to UNPAYWALL_EMAIL)
```

### OpenAlex integration

[OpenAlex](https://openalex.org) is a free, keyless catalog of >250M works.
Where Unpaywall is DOI-only and returns no abstracts, OpenAlex returns:

- Full bibliographic record (title, authors, year, venue, journal)
- Reconstructed abstract from an inverted index
- Multiple OA-location options (best_oa_location, oa_locations[], primary)
- External IDs: DOI, PMID, PMCID, MAG, ArXiv (when known)
- Citation count and reference count

PullR uses OpenAlex two ways:

1. **DOI lookup** (`openalex_doi` strategy) — instant and high-precision; runs
   first in the strategy ladder when any DOI is detected (LLM-extracted or
   regex-matched) in the reference text.

2. **Title search** (`openalex_title` strategy) — uses `filter=title.search:…`
   with `cited_by_count` sort, then re-ranks locally by title-token overlap
   plus year and author bonuses. Useful when no DOI is available.

### Adding PullR output to SCOUT

After pulling papers, you can fold them into the SCOUT paper-index DB at
`~/paper-index/data/papers.sqlite`:

```bash
python ingest_to_scout.py              # scan ~/Dropbox/PullR (default)
python ingest_to_scout.py /custom/dir  # scan a different root
python ingest_to_scout.py --dry-run    # report only, no DB writes
```

Ingest handles four filename conventions:
  - `<sha1>.pdf`        — Semantic Scholar paper IDs (bulk batch lookup)
  - `oa_W<digits>.pdf`  — OpenAlex (resolved via OpenAlex REST)
  - `arxiv_NNNN.NNNNN.pdf` — arXiv (resolved via OpenAlex DOI)
  - other names         — sha256 content de-dup + pdftotext title/DOI extract

Every ingested paper gets a `corpus = pullr-<subdir>` label in SCOUT so you
can filter by source collection in the Datasette UI.

## 🎯 Usage

### PDF Mode - Extract References from PDF
Extract references from an academic paper and download related papers:

```bash
python pullr.py paper.pdf --model gpt4 --output-dir ./papers --mode pdf
```

### Directory Mode - Batch PDF Processing
Process all PDFs in a directory:

```bash
python pullr.py ./pdf_directory --model gpt4 --output-dir ./papers --mode pdf
```

### Sampling Mode - Test on Subsets
Randomly sample N PDFs from a large directory:

```bash
python pullr.py ./large_directory --model gpt4 --output-dir ./sample_results --mode pdf --sample 10
```

### Exact Mode - Precise Paper Matching
Find papers that exactly match each reference in a file:

```bash
python pullr.py references.txt --model gpt4 --output-dir ./papers --mode exact
```

### Fuzzy Mode - Similar Paper Discovery
Find multiple similar papers for each reference:

```bash
python pullr.py references.txt --model gpt4 --output-dir ./papers --mode fuzzy --max-papers 5
```

### Extract-Only Mode - Reference Cleaning
Extract and clean references without downloading papers:

```bash
python pullr.py paper.pdf --model gpt4 --output-dir ./refs --extract-only
python pullr.py ./pdf_directory --model gpt4 --output-dir ./batch_refs --extract-only
python pullr.py ./large_directory --model gpt4 --output-dir ./sample_refs --extract-only --sample 5
python pullr.py references.txt --model gpt4 --output-dir ./cleaned --extract-only
```

### Parallel Processing
Speed up processing with multiple threads:

```bash
python pullr.py paper.pdf --model gpt4 --output-dir ./papers --mode pdf --parallel 4 --verbose
```

## 📖 Detailed Examples

### Processing a Research Paper
```bash
# Extract references from a PDF and download related papers
python pullr.py "machine_learning_survey.pdf" \
  --model gpt4 \
  --output-dir ./ml_papers \
  --mode pdf \
  --parallel 3 \
  --verbose
```

### Batch Processing Multiple PDFs
```bash
# Process all PDFs in a directory
python pullr.py "./research_papers/" \
  --model gpt4 \
  --output-dir ./batch_results \
  --mode pdf \
  --parallel 4 \
  --verbose
```

### Sample Testing on Large Collections
```bash
# Test processing with a random sample of 10 PDFs
python pullr.py "./large_collection/" \
  --model gpt4 \
  --output-dir ./test_sample \
  --mode pdf \
  --sample 10 \
  --verbose
```

### Building a Research Collection
```bash
# Process a list of references to build a comprehensive collection
python pullr.py "ai_references.txt" \
  --model gpt4 \
  --output-dir ./ai_collection \
  --mode exact \
  --ss-api-key "your-semantic-scholar-key"
```

### Cleaning Reference Lists
```bash
# Extract and clean references from a PDF without downloading
python pullr.py "research_paper.pdf" \
  --model gpt4 \
  --output-dir ./cleaned_refs \
  --extract-only \
  --verbose

# Batch clean references from multiple PDFs
python pullr.py "./papers_to_clean/" \
  --model gpt4 \
  --output-dir ./batch_cleaned \
  --extract-only

# Sample and clean references from a large collection
python pullr.py "./large_paper_collection/" \
  --model gpt4 \
  --output-dir ./sample_cleaned \
  --extract-only \
  --sample 20

# Clean an existing reference list
python pullr.py "messy_references.txt" \
  --model gpt4 \
  --output-dir ./cleaned_refs \
  --extract-only
```

## 📁 Output Structure

PullR creates organized output with detailed metadata:

```
output_dir/
├── extracted_references.txt          # References found in PDF (single file)
├── cleaned_references.txt            # LLM-processed references (single file)
├── original_references.txt           # Original references (extract-only mode)
├── processing_summary.txt            # Processing report (extract-only mode)
├── batch_processing_summary.txt      # Summary for directory processing
├── pdf_001_filename/                 # Individual PDF results (directory mode)
│   ├── extracted_references.txt      
│   ├── [paperID]_[title].txt         
│   └── [paperID].pdf                 
├── pdf_002_filename/                 # Second PDF results
└── [paperID]_[title].txt             # Paper abstracts with metadata (single mode)
```

### Sample Output File
```
Title: Deep Learning for Natural Language Processing: A Survey
Authors: John Smith, Jane Doe et al. (and 3 more)
Year: 2023
Paper ID: 1234567
URL: https://semanticscholar.org/paper/1234567
Venue: Journal of AI Research
Content Status: LLM_PROCESSED, DOCUMENT_CLEANED

Abstract:
This comprehensive survey reviews recent advances in deep learning
approaches for natural language processing tasks...
```

## 🔧 Advanced Features

### Web Content Enhancement
When processing URLs that don't have clear abstracts:
1. **Comprehensive Scraping**: Captures full page content
2. **LLM Processing**: Extracts structured academic information
3. **Document Cleanup**: Formats content professionally
4. **Quality Optimization**: Improves titles and removes redundancy

### Multiple Search Strategies
PullR uses intelligent fallback strategies:
1. **Exact bibliographic search** (title + author + year)
2. **Title-only search** with variants
3. **Author + year combinations**
4. **Keyword extraction and search**
5. **Web scraping for URLs**

### Rate Limiting & Error Handling
- Automatic exponential backoff for rate limits
- Retry logic for temporary failures
- Thread-safe API call coordination
- Graceful handling of missing content

## 🛠️ Development

### Project Structure
```
pullr.py                 # Main application
model_servers.yaml       # LLM configuration
requirements.txt         # Python dependencies
README.md               # Documentation
examples/               # Example files and configs
```

### Key Functions
- `extract_references_from_text()`: PDF reference extraction
- `search_with_fallbacks()`: Multi-strategy paper search
- `try_web_scraping()`: Enhanced web content extraction
- `cleanup_document_with_llm()`: Document formatting
- `process_single_reference()`: Thread-safe reference processing

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional academic database integrations
- Enhanced reference parsing algorithms
- Better web scraping for specific domains
- Performance optimizations
- Documentation improvements

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Semantic Scholar**: Primary academic database
- **OpenAI/Anthropic**: LLM providers for content processing
- **PyPDF2**: PDF text extraction
- **BeautifulSoup**: Web scraping capabilities

## 🔮 Roadmap

- [ ] Support for more academic databases (PubMed, arXiv API)
- [ ] Better citation format detection
- [ ] Integration with reference managers
- [ ] GUI interface
- [ ] Batch processing workflows
- [ ] Advanced deduplication
- [ ] Citation network analysis

---

**PullR** - Making academic research more accessible, one paper at a time.
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
- **Web Scraping**: Fallback for URLs (arXiv, DOI, ResearchGate, etc.)
- **Open Access PDFs**: Automatic download when available
- **Multiple Formats**: Handles various citation styles and formats

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
```

## 🎯 Usage

### PDF Mode - Extract References from PDF
Extract references from an academic paper and download related papers:

```bash
python pullr.py paper.pdf --model gpt4 --output-dir ./papers --mode pdf
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

### Building a Research Collection
```bash
# Process a list of references to build a comprehensive collection
python pullr.py "ai_references.txt" \
  --model gpt4 \
  --output-dir ./ai_collection \
  --mode exact \
  --ss-api-key "your-semantic-scholar-key"
```

## 📁 Output Structure

PullR creates organized output with detailed metadata:

```
output_dir/
├── extracted_references.txt          # References found in PDF
├── cleaned_references.txt            # LLM-processed references
├── [paperID]_[title].txt             # Paper abstracts with metadata
└── [paperID].pdf                     # Downloaded PDFs when available
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
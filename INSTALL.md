# Installation Guide for PullR

## Quick Install

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rick-stevens-ai/PullR.git
   cd PullR
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API access:**
   ```bash
   cp model_servers.yaml.example model_servers.yaml
   # Edit model_servers.yaml with your API keys
   ```

4. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-key"  # Optional
   ```

5. **Test the installation:**
   ```bash
   python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./test --mode exact
   ```

## Detailed Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- OpenAI API key (or compatible LLM API)

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv pullr-env

# Activate virtual environment
# On macOS/Linux:
source pullr-env/bin/activate
# On Windows:
pullr-env\Scripts\activate

# Install PullR
pip install -r requirements.txt
```

### API Key Configuration

#### OpenAI API
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add to environment: `export OPENAI_API_KEY="sk-..."`

#### Semantic Scholar API (Optional but Recommended)
1. Get your API key from [Semantic Scholar](https://www.semanticscholar.org/product/api)
2. Add to environment: `export SEMANTIC_SCHOLAR_API_KEY="your-key"`

#### Alternative LLM Providers
PullR supports any OpenAI-compatible API. Configure in `model_servers.yaml`:

```yaml
servers:
  - shortname: "custom"
    openai_api_key: "your-api-key"
    openai_api_base: "https://your-llm-provider.com/v1"
    openai_model: "your-model-name"
```

### Verification

Test your installation:

```bash
# Test basic functionality
python pullr.py --help

# Test with sample references
python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./test --mode exact --verbose

# Check output
ls -la test/
```

Expected output:
- Abstract files (.txt)
- Downloaded PDFs (if available)
- Processing logs (with --verbose)

## Troubleshooting

### Common Issues

1. **"No module named 'openai'"**
   - Run: `pip install -r requirements.txt`

2. **"Model configuration not found"**
   - Ensure `model_servers.yaml` exists and is properly configured
   - Check API key environment variables

3. **"Rate limit exceeded"**
   - Add Semantic Scholar API key
   - Reduce parallel threads: `--parallel 1`
   - Add delays between requests

4. **"PyPDF2 not found" (PDF mode)**
   - Run: `pip install PyPDF2`

5. **"BeautifulSoup not available" (web scraping)**
   - Run: `pip install beautifulsoup4`

### Performance Tips

- **Use Semantic Scholar API key** for better rate limits
- **Parallel processing**: Use `--parallel 3-5` for optimal speed
- **Model selection**: `gpt-3.5-turbo` is faster and cheaper than `gpt-4`
- **Batch processing**: Process multiple references in one run

### System Requirements

- **Memory**: 2GB+ RAM recommended
- **Storage**: Varies by collection size (PDFs can be large)
- **Network**: Stable internet connection for API calls
- **Rate Limits**: 
  - OpenAI: 3000 requests/minute (varies by plan)
  - Semantic Scholar: 100 requests/minute (without API key)

## Docker Installation (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "pullr.py", "--help"]
```

Build and run:
```bash
docker build -t pullr .
docker run -e OPENAI_API_KEY="your-key" -v $(pwd)/output:/app/output pullr
```

## Development Setup

For development and testing:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests (when available)
pytest

# Format code
black pullr.py

# Lint code
flake8 pullr.py
```

## Support

If you encounter issues:
1. Check this installation guide
2. Review the [README.md](README.md)
3. Open an issue on [GitHub](https://github.com/rick-stevens-ai/PullR/issues)
#!/bin/bash
# PullR Usage Examples
# Make sure to configure model_servers.yaml first

echo "PullR Usage Examples"
echo "==================="

# Basic PDF processing
echo "1. Extract references from PDF and download papers:"
echo "python pullr.py paper.pdf --model gpt4 --output-dir ./papers --mode pdf"
echo ""

# Directory PDF processing
echo "2. Process all PDFs in a directory:"
echo "python pullr.py ./pdf_directory --model gpt4 --output-dir ./batch_papers --mode pdf"
echo ""

# Sample processing - test with a subset
echo "3. Sample processing - randomly select N PDFs from directory:"
echo "python pullr.py ./large_pdf_directory --model gpt4 --output-dir ./sample_papers --mode pdf --sample 5"
echo ""

# Exact reference matching
echo "4. Find exact matches for references:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./exact_papers --mode exact"
echo ""

# Fuzzy searching for similar papers
echo "5. Find similar papers (fuzzy mode):"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./fuzzy_papers --mode fuzzy --max-papers 3"
echo ""

# Parallel processing for speed
echo "6. Fast processing with parallel threads:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./fast_papers --mode exact --parallel 4"
echo ""

# Verbose mode for debugging
echo "7. Detailed output with verbose mode:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./debug_papers --mode exact --verbose"
echo ""

# Using different models
echo "8. Using different LLM models:"
echo "python pullr.py examples/sample_references.txt --model gpt35 --output-dir ./gpt35_papers --mode fuzzy"
echo "python pullr.py examples/sample_references.txt --model claude --output-dir ./claude_papers --mode exact"
echo ""

# With Semantic Scholar API key
echo "9. Using Semantic Scholar API key for better rate limits:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./papers --mode exact --ss-api-key 'your-key'"
echo ""

# Extract-only mode
echo "10. Extract and clean references without downloading (extract-only mode):"
echo "python pullr.py paper.pdf --model gpt4 --output-dir ./refs --extract-only"
echo "python pullr.py ./pdf_directory --model gpt4 --output-dir ./batch_refs --extract-only"
echo "python pullr.py ./pdf_directory --model gpt4 --output-dir ./sample_refs --extract-only --sample 10"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./cleaned --extract-only"
echo ""

echo "Tips:"
echo "- Set OPENAI_API_KEY environment variable"
echo "- Set SEMANTIC_SCHOLAR_API_KEY for better performance"
echo "- Use --parallel 3-5 for optimal speed"
echo "- PDF mode works best for reference extraction"
echo "- Directory processing handles all PDFs in a folder"
echo "- Use --sample N to test processing on large directories"
echo "- Sampling is reproducible (same directory = same sample)"
echo "- Exact mode for precise matching, fuzzy for discovery"
echo "- Extract-only mode is perfect for cleaning reference lists"
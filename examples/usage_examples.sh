#!/bin/bash
# PullR Usage Examples
# Make sure to configure model_servers.yaml first

echo "PullR Usage Examples"
echo "==================="

# Basic PDF processing
echo "1. Extract references from PDF and download papers:"
echo "python pullr.py paper.pdf --model gpt4 --output-dir ./papers --mode pdf"
echo ""

# Exact reference matching
echo "2. Find exact matches for references:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./exact_papers --mode exact"
echo ""

# Fuzzy searching for similar papers
echo "3. Find similar papers (fuzzy mode):"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./fuzzy_papers --mode fuzzy --max-papers 3"
echo ""

# Parallel processing for speed
echo "4. Fast processing with parallel threads:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./fast_papers --mode exact --parallel 4"
echo ""

# Verbose mode for debugging
echo "5. Detailed output with verbose mode:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./debug_papers --mode exact --verbose"
echo ""

# Using different models
echo "6. Using different LLM models:"
echo "python pullr.py examples/sample_references.txt --model gpt35 --output-dir ./gpt35_papers --mode fuzzy"
echo "python pullr.py examples/sample_references.txt --model claude --output-dir ./claude_papers --mode exact"
echo ""

# With Semantic Scholar API key
echo "7. Using Semantic Scholar API key for better rate limits:"
echo "python pullr.py examples/sample_references.txt --model gpt4 --output-dir ./papers --mode exact --ss-api-key 'your-key'"
echo ""

echo "Tips:"
echo "- Set OPENAI_API_KEY environment variable"
echo "- Set SEMANTIC_SCHOLAR_API_KEY for better performance"
echo "- Use --parallel 3-5 for optimal speed"
echo "- PDF mode works best for reference extraction"
echo "- Exact mode for precise matching, fuzzy for discovery"
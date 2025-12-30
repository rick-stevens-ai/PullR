#!/bin/bash
# run-pullr.sh - Convenience script for topic-based paper discovery
# Uses LLM to generate relevant search queries and downloads papers using PullR
# Will retry until target number of PDFs are successfully downloaded

# Don't exit on error - we want to retry
set +e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_MODEL="oss120"
DEFAULT_CONFIG="model_servers.yaml"
DEFAULT_OUTPUT_DIR="./pullr_output"
DEFAULT_PARALLEL=4
MAX_ITERATIONS=10  # Prevent infinite loops

# Display usage
usage() {
    echo -e "${BLUE}Usage:${NC} $0 <topic> <num_pdfs> [options]"
    echo ""
    echo "Arguments:"
    echo "  topic         High-level research topic (quoted string)"
    echo "  num_pdfs      TARGET number of PDFs to download (will retry until reached)"
    echo ""
    echo "Options:"
    echo "  --model MODEL          Model to use (default: $DEFAULT_MODEL)"
    echo "  --config FILE          Path to model config (default: $DEFAULT_CONFIG)"
    echo "  --output-dir DIR       Output directory (default: $DEFAULT_OUTPUT_DIR)"
    echo "  --parallel N           Parallel threads (default: $DEFAULT_PARALLEL)"
    echo "  --queries N            Number of search queries per iteration (default: 5)"
    echo "  --max-iter N           Maximum iterations before giving up (default: $MAX_ITERATIONS)"
    echo "  --papers-per-query N   Papers to request per query (default: 5)"
    echo "  --verbose              Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 \"quantum computing algorithms\" 50"
    echo "  $0 \"machine learning optimization\" 100 --queries 8 --parallel 6"
    echo "  $0 \"deep learning transformers\" 75 --model llama70 --verbose"
    echo ""
    echo "Note: Script will continue generating new queries and searching until"
    echo "      the target number of PDFs is reached or max iterations exceeded."
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    usage
fi

TOPIC="$1"
TARGET_PDFS="$2"
shift 2

# Parse optional arguments
MODEL="$DEFAULT_MODEL"
CONFIG="$DEFAULT_CONFIG"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
PARALLEL="$DEFAULT_PARALLEL"
NUM_QUERIES=5
PAPERS_PER_QUERY=5
MAX_ITER="$MAX_ITERATIONS"
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --papers-per-query)
            PAPERS_PER_QUERY="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate inputs
if ! [[ "$TARGET_PDFS" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: num_pdfs must be a positive integer${NC}"
    exit 1
fi

if ! [[ "$NUM_QUERIES" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: --queries must be a positive integer${NC}"
    exit 1
fi

if ! [[ "$PAPERS_PER_QUERY" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: --papers-per-query must be a positive integer${NC}"
    exit 1
fi

if ! [[ "$MAX_ITER" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: --max-iter must be a positive integer${NC}"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Load and export API keys from api_config.yaml
if [ -f "api_config.yaml" ]; then
    echo -e "${CYAN}Loading API keys from api_config.yaml...${NC}"

    # Export Semantic Scholar API key
    SS_KEY=$(python3 -c "import yaml; c=yaml.safe_load(open('api_config.yaml')); print(c['apis']['semantic_scholar'].get('api_key', ''))" 2>/dev/null || echo "")
    if [ ! -z "$SS_KEY" ] && [ "$SS_KEY" != "None" ]; then
        export SEMANTIC_SCHOLAR_API_KEY="$SS_KEY"
        echo -e "${GREEN}✓ Semantic Scholar API key loaded${NC}"
    fi

    # Set user email for OpenAlex and Unpaywall if not already set
    if [ -z "$USER_EMAIL" ]; then
        export USER_EMAIL="research@example.com"
    fi

    echo ""
fi

# Create temporary files
TEMP_DIR=$(mktemp -d)
QUERY_FILE="$TEMP_DIR/search_queries.txt"
trap "rm -rf $TEMP_DIR" EXIT

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PullR Topic-Based Paper Discovery${NC}"
echo -e "${BLUE}TARGET: $TARGET_PDFS PDFs${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Topic:${NC} $TOPIC"
echo -e "${GREEN}Target PDFs:${NC} $TARGET_PDFS"
echo -e "${GREEN}Papers per query:${NC} $PAPERS_PER_QUERY"
echo -e "${GREEN}Queries per iteration:${NC} $NUM_QUERIES"
echo -e "${GREEN}Max iterations:${NC} $MAX_ITER"
echo -e "${GREEN}Model:${NC} $MODEL"
echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Parallel threads:${NC} $PARALLEL"
echo ""

# Function to count PDFs
count_pdfs() {
    find "$OUTPUT_DIR" -name "*.pdf" -type f 2>/dev/null | wc -l
}

# Function to generate queries
generate_queries() {
    local iteration=$1

echo -e "${YELLOW}Generating $NUM_QUERIES search queries (iteration $iteration)...${NC}"

    python3 - <<PYTHON_SCRIPT
import yaml
import sys
from openai import OpenAI

# Load model configuration
with open('$CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Find the specified model
model_config = None
for server in config['servers']:
    if server['shortname'] == '$MODEL':
        model_config = server
        break

if not model_config:
    print(f"Error: Model '$MODEL' not found in $CONFIG", file=sys.stderr)
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(
    api_key=model_config['openai_api_key'],
    base_url=model_config['openai_api_base']
)

# Generate search queries with variation based on iteration
iteration_context = ""
if $iteration > 1:
    iteration_context = f" (Iteration {$iteration}: Try exploring different angles, related topics, or alternative terminology.)"

prompt = f"""Given the research topic: "$TOPIC"{iteration_context}

Generate $NUM_QUERIES diverse and specific search queries that would help find relevant academic papers. Each query should:
1. Focus on different aspects or subtopics of the main topic
2. Use academic language and terminology
3. Be specific enough to find relevant papers
4. Cover different approaches, methodologies, or perspectives

Return ONLY the search queries, one per line, without numbering or explanations."""

try:
    response = client.chat.completions.create(
        model=model_config['openai_model'],
        messages=[
            {"role": "system", "content": "You are an expert research assistant helping to find relevant academic papers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7 + ($iteration * 0.05),  # Increase diversity each iteration
        max_tokens=500
    )

    # Validate response structure
    if not response or not response.choices:
        print("Error: Empty response from API", file=sys.stderr)
        sys.exit(1)

    content = response.choices[0].message.content
    if content is None:
        print("Error: API returned None for message content (possible timeout or model error)", file=sys.stderr)
        sys.exit(1)

    queries = content.strip()

    if not queries:
        print("Error: API returned empty content", file=sys.stderr)
        sys.exit(1)

    # Clean up the output - remove numbering if present
    lines = []
    for line in queries.split('\n'):
        line = line.strip()
        # Remove common list markers
        line = line.lstrip('0123456789.-) ')
        if line:
            lines.append(line)

    if not lines:
        print("Error: No valid queries extracted from response", file=sys.stderr)
        sys.exit(1)

    # Write to file
    with open('$QUERY_FILE', 'w') as f:
        f.write('\\n'.join(lines))

    print(f"✓ Generated {len(lines)} search queries")
    sys.exit(0)

except Exception as e:
    print(f"Error generating queries: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

    return $?
}

# Main loop - retry until target is reached
ITERATION=1
CURRENT_PDFS=$(count_pdfs)

echo -e "${CYAN}Starting PDF collection...${NC}"
echo -e "${CYAN}Current PDFs: $CURRENT_PDFS / $TARGET_PDFS${NC}"
echo ""

while [ $CURRENT_PDFS -lt $TARGET_PDFS ] && [ $ITERATION -le $MAX_ITER ]; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}ITERATION $ITERATION / $MAX_ITER${NC}"
    echo -e "${BLUE}Current: $CURRENT_PDFS PDFs | Target: $TARGET_PDFS PDFs${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Generate queries with retry limit
    QUERY_RETRY=0
    MAX_QUERY_RETRIES=3
    while [ $QUERY_RETRY -lt $MAX_QUERY_RETRIES ]; do
        generate_queries $ITERATION
        if [ $? -eq 0 ]; then
            break
        fi

        QUERY_RETRY=$((QUERY_RETRY + 1))
        if [ $QUERY_RETRY -lt $MAX_QUERY_RETRIES ]; then
            echo -e "${RED}Failed to generate queries (attempt $QUERY_RETRY/$MAX_QUERY_RETRIES), retrying in 5 seconds...${NC}"
            sleep 5
        else
            echo -e "${RED}Failed to generate queries after $MAX_QUERY_RETRIES attempts. Skipping to next iteration.${NC}"
            ITERATION=$((ITERATION + 1))
            continue 2  # Continue outer loop
        fi
    done

    # Display generated queries
    echo ""
    echo -e "${CYAN}Search queries for this iteration:${NC}"
    cat "$QUERY_FILE" | nl -w2 -s'. '
    echo ""

    # Run PullR
    echo -e "${YELLOW}Searching and downloading papers...${NC}"
    python pullr.py "$QUERY_FILE" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --mode fuzzy \
        --max-papers "$PAPERS_PER_QUERY" \
        --parallel "$PARALLEL" \
        $VERBOSE

    # Count PDFs after this iteration
    NEW_PDF_COUNT=$(count_pdfs)
    PDFS_ADDED=$((NEW_PDF_COUNT - CURRENT_PDFS))
    CURRENT_PDFS=$NEW_PDF_COUNT

    echo ""
    echo -e "${GREEN}Iteration $ITERATION complete:${NC}"
    echo -e "  - PDFs added this iteration: $PDFS_ADDED"
    echo -e "  - Total PDFs collected: $CURRENT_PDFS / $TARGET_PDFS"
    echo ""

    # Check if we've reached the target
    if [ $CURRENT_PDFS -ge $TARGET_PDFS ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ TARGET REACHED!${NC}"
        echo -e "${GREEN}Successfully collected $CURRENT_PDFS PDFs${NC}"
        echo -e "${GREEN}========================================${NC}"
        break
    fi

    ITERATION=$((ITERATION + 1))

    # Small delay before next iteration
    if [ $ITERATION -le $MAX_ITER ]; then
        echo -e "${YELLOW}Need $((TARGET_PDFS - CURRENT_PDFS)) more PDFs. Starting next iteration...${NC}"
        sleep 2
    fi
done

# Final summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FINAL SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"

FINAL_PDF_COUNT=$(count_pdfs)
NUM_TXT=$(find "$OUTPUT_DIR" -name "*.txt" -type f ! -name "*references*.txt" ! -name "*summary*.txt" 2>/dev/null | wc -l)

echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Total PDFs collected:${NC} $FINAL_PDF_COUNT"
echo -e "${GREEN}Total abstracts:${NC} $NUM_TXT"
echo -e "${GREEN}Iterations used:${NC} $((ITERATION - 1)) / $MAX_ITER"
echo ""

if [ $FINAL_PDF_COUNT -ge $TARGET_PDFS ]; then
    echo -e "${GREEN}✓ SUCCESS: Target of $TARGET_PDFS PDFs reached!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ PARTIAL SUCCESS: Collected $FINAL_PDF_COUNT / $TARGET_PDFS PDFs${NC}"
    echo -e "${YELLOW}Maximum iterations reached. You may want to:${NC}"
    echo -e "${YELLOW}  - Run again with --max-iter $((MAX_ITER * 2))${NC}"
    echo -e "${YELLOW}  - Try different search parameters${NC}"
    echo -e "${YELLOW}  - Broaden your topic${NC}"
    exit 2
fi

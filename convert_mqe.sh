#!/bin/bash
# Batch convert MQE PDFs to markdown using Nougat with GPU
# This script activates the nougat conda environment and runs the conversion

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MQE PDF to Markdown Converter${NC}"
echo -e "${BLUE}Using Nougat with GPU Acceleration${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if in conda environment
if [[ "$CONDA_DEFAULT_ENV" != "nougat" ]]; then
    echo "Activating nougat conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate nougat
fi

# Run the conversion script
python convert_mqe_pdfs.py "$@"

echo ""
echo -e "${GREEN}Done!${NC}"

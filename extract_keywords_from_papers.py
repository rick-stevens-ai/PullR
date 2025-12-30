#!/usr/bin/env python3
"""
Extract new keywords from existing paper abstracts for expanded search.

This script analyzes a collection of paper abstracts and extracts related
research topics, methods, and concepts that can be used for further searches.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter
from openai import OpenAI
from tqdm import tqdm


def load_model_config(model_shortname, config_file="model_servers.yaml"):
    """Load model configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    for server in config.get('servers', []):
        if server.get('shortname') == model_shortname:
            # Expand environment variables in API key
            api_key = server.get('openai_api_key', '')
            if api_key.startswith('${') and api_key.endswith('}'):
                env_var = api_key[2:-1]
                api_key = os.environ.get(env_var, api_key)

            return {
                'openai_api_key': api_key,
                'openai_api_base': server.get('openai_api_base'),
                'openai_model': server.get('openai_model')
            }

    raise ValueError(f"Model '{model_shortname}' not found in {config_file}")


def get_openai_client(model_config):
    """Create OpenAI client from model config"""
    return OpenAI(
        api_key=model_config['openai_api_key'],
        base_url=model_config['openai_api_base']
    )


def read_abstracts(directory: str, limit: int = None) -> List[Dict[str, str]]:
    """Read all abstract files from directory

    Args:
        directory: Directory containing .txt abstract files
        limit: Maximum number of abstracts to read (None = all)

    Returns:
        List of dicts with 'filename' and 'content'
    """
    abstracts = []
    txt_files = list(Path(directory).glob('*.txt'))

    if limit:
        txt_files = txt_files[:limit]

    print(f"Reading {len(txt_files)} abstract files...")

    for txt_file in tqdm(txt_files, desc="Loading abstracts"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and len(content) > 100:  # Skip very short files
                    abstracts.append({
                        'filename': txt_file.name,
                        'content': content
                    })
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")

    return abstracts


def extract_keywords_from_batch(abstracts_batch: List[str], client: OpenAI,
                                  model: str, verbose: bool = False) -> List[str]:
    """Extract keywords from a batch of abstracts using LLM

    Args:
        abstracts_batch: List of abstract texts
        client: OpenAI client
        model: Model name to use
        verbose: Print detailed output

    Returns:
        List of extracted keywords/topics
    """
    # Combine abstracts for batch processing
    combined_text = "\n\n---\n\n".join(abstracts_batch)

    prompt = f"""Analyze these research paper abstracts and extract SPECIFIC, NICHE research topics that would make excellent search keywords for finding related papers.

FOCUS ON:
1. Specific technical methods and algorithms (e.g., "variational quantum eigensolver optimization")
2. Novel combinations of concepts (e.g., "DNA origami photonic crystal assembly")
3. Emerging techniques (e.g., "machine learning force field parameterization")
4. Specific applications (e.g., "enzyme active site electrostatic engineering")
5. Unique material systems or approaches (e.g., "peptide-based quantum dot scaffolds")

AVOID:
- Generic terms like "quantum mechanics", "machine learning", "nanotechnology"
- Single-word keywords
- Very broad topics that would return millions of papers

Extract 15-25 SPECIFIC, NICHE keywords/phrases (4-8 words each) that represent:
- Specialized methods
- Unique technical approaches
- Specific applications or systems
- Novel interdisciplinary combinations

Format: One keyword per line, no numbering or bullets. Be specific and technical.

Abstracts:
{combined_text}

SPECIFIC Technical Keywords:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research librarian expert at identifying key topics and search terms from scientific literature."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        keywords_text = response.choices[0].message.content.strip()
        keywords = [line.strip() for line in keywords_text.split('\n') if line.strip() and len(line.strip()) > 3]

        if verbose:
            print(f"  Extracted {len(keywords)} keywords from batch")

        return keywords

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


def extract_keywords_from_papers(directory: str, output_file: str,
                                   model_name: str = "llama70",
                                   batch_size: int = 10,
                                   max_papers: int = None,
                                   min_frequency: int = 2,
                                   max_keywords: int = 50,
                                   exclude_keywords_file: str = None,
                                   verbose: bool = False):
    """Main function to extract keywords from paper collection

    Args:
        directory: Directory containing abstract .txt files
        output_file: Output file for new keywords
        model_name: LLM model to use for extraction
        batch_size: Number of abstracts to process per LLM call
        max_papers: Maximum papers to analyze (None = all)
        min_frequency: Minimum times a keyword must appear to be included
        max_keywords: Maximum number of keywords to output
        exclude_keywords_file: File with keywords to exclude (already searched)
        verbose: Print detailed progress
    """
    print(f"=== Extracting Keywords from Paper Collection ===")
    print(f"Directory: {directory}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size} abstracts per LLM call")

    # Load model config
    print("\nLoading model configuration...")
    model_config = load_model_config(model_name)
    client = get_openai_client(model_config)
    openai_model = model_config['openai_model']

    # Read abstracts
    abstracts = read_abstracts(directory, limit=max_papers)
    print(f"Loaded {len(abstracts)} abstracts")

    if len(abstracts) == 0:
        print("Error: No abstracts found!")
        return

    # Load existing keywords to exclude
    exclude_keywords = set()
    if exclude_keywords_file and os.path.exists(exclude_keywords_file):
        with open(exclude_keywords_file, 'r', encoding='utf-8') as f:
            exclude_keywords = set(line.strip().lower() for line in f if line.strip())
        print(f"Excluding {len(exclude_keywords)} already-searched keywords")

    # Extract keywords in batches
    all_keywords = []
    num_batches = (len(abstracts) + batch_size - 1) // batch_size

    print(f"\nExtracting keywords from {num_batches} batches...")

    for i in tqdm(range(0, len(abstracts), batch_size), desc="Processing batches"):
        batch = abstracts[i:i+batch_size]
        batch_texts = [a['content'] for a in batch]

        keywords = extract_keywords_from_batch(batch_texts, client, openai_model, verbose)
        all_keywords.extend(keywords)

        if verbose:
            print(f"  Batch {i//batch_size + 1}/{num_batches}: {len(keywords)} keywords")

    # Count keyword frequencies and filter
    print(f"\nProcessing {len(all_keywords)} total extracted keywords...")

    # Normalize keywords (lowercase, strip)
    normalized_keywords = [k.lower().strip() for k in all_keywords]
    keyword_counts = Counter(normalized_keywords)

    # Filter by frequency and exclusions
    filtered_keywords = []
    for keyword, count in keyword_counts.most_common():
        # Skip if below minimum frequency
        if count < min_frequency:
            continue

        # Skip if in exclusion list
        if keyword in exclude_keywords:
            if verbose:
                print(f"  Excluding (already searched): {keyword}")
            continue

        # Skip if very short or very long
        if len(keyword) < 5 or len(keyword) > 100:
            continue

        filtered_keywords.append((keyword, count))

        if len(filtered_keywords) >= max_keywords:
            break

    # Write to output file
    print(f"\nWriting {len(filtered_keywords)} new keywords to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword, count in filtered_keywords:
            f.write(f"{keyword}\n")
            if verbose:
                print(f"  {keyword} (appeared {count} times)")

    # Print summary
    print(f"\n=== Extraction Complete ===")
    print(f"Abstracts analyzed: {len(abstracts)}")
    print(f"Total keywords extracted: {len(all_keywords)}")
    print(f"Unique keywords: {len(keyword_counts)}")
    print(f"Keywords above frequency {min_frequency}: {len([k for k, c in keyword_counts.items() if c >= min_frequency])}")
    print(f"New keywords (after exclusions): {len(filtered_keywords)}")
    print(f"Output file: {output_file}")

    if len(filtered_keywords) > 0:
        print(f"\nTop 10 new keywords:")
        for i, (keyword, count) in enumerate(filtered_keywords[:10], 1):
            print(f"  {i}. {keyword} (frequency: {count})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract new keywords from existing paper abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract keywords from fuzzy_papers directory
  python extract_keywords_from_papers.py fuzzy_papers --output new_keywords.txt

  # Use specific model and exclude original keywords
  python extract_keywords_from_papers.py fuzzy_papers -o new_keywords.txt \\
      --model llama70 --exclude keywords.txt

  # Process only 500 papers with larger batches
  python extract_keywords_from_papers.py fuzzy_papers -o new_keywords.txt \\
      --max-papers 500 --batch-size 20

  # Get more keywords with lower frequency threshold
  python extract_keywords_from_papers.py fuzzy_papers -o new_keywords.txt \\
      --min-frequency 1 --max-keywords 100
        """
    )

    parser.add_argument('directory', help='Directory containing abstract .txt files')
    parser.add_argument('--output', '-o', default='new_keywords.txt',
                        help='Output file for extracted keywords (default: new_keywords.txt)')
    parser.add_argument('--model', default='llama70',
                        help='Model to use for keyword extraction (default: llama70)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of abstracts per LLM call (default: 10)')
    parser.add_argument('--max-papers', type=int, default=None,
                        help='Maximum number of papers to analyze (default: all)')
    parser.add_argument('--min-frequency', type=int, default=2,
                        help='Minimum keyword frequency to include (default: 2)')
    parser.add_argument('--max-keywords', type=int, default=50,
                        help='Maximum number of keywords to output (default: 50)')
    parser.add_argument('--exclude', dest='exclude_file', default=None,
                        help='File with keywords to exclude (already searched)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress information')

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' not found")
        sys.exit(1)

    # Run extraction
    extract_keywords_from_papers(
        directory=args.directory,
        output_file=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_papers=args.max_papers,
        min_frequency=args.min_frequency,
        max_keywords=args.max_keywords,
        exclude_keywords_file=args.exclude_file,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detailed PubMed adapter debugging"""

import yaml
import sys
sys.path.insert(0, '.')

# Load config
with open('api_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pubmed_config = config['apis']['pubmed']
pubmed_config['enabled'] = True

print("=" * 70)
print("PubMed Adapter Detailed Debug")
print("=" * 70)

# Import adapter with modifications for debugging
from api_adapters import PubMedAdapter

# Create adapter with verbose mode
adapter = PubMedAdapter(pubmed_config, verbose=True)

print(f"\nAdapter config:")
print(f"  Endpoint: {adapter.config['endpoint']}")
print(f"  Rate limit: {adapter.config['rate_limit']}s")
print(f"  API key: {adapter.config.get('api_key', 'None')}")

# Monkey-patch the _make_request method to add extra logging
original_make_request = adapter._make_request

def debug_make_request(url, params=None, headers=None, max_retries=3, timeout=30):
    print(f"\n>>> _make_request called:")
    print(f"    URL: {url}")
    print(f"    Params: {params}")
    print(f"    Headers: {headers}")

    response = original_make_request(url, params, headers, max_retries, timeout)

    if response:
        print(f"<<< _make_request returned:")
        print(f"    Status: {response.status_code}")
        print(f"    Content length: {len(response.content)} bytes")
        print(f"    Content preview: {response.text[:200]}")
    else:
        print(f"<<< _make_request returned: None")

    return response

adapter._make_request = debug_make_request

# Run search
print(f"\n{'=' * 70}")
print(f"Executing adapter.search('quantum computing', limit=5)")
print(f"{'=' * 70}")

try:
    papers = adapter.search("quantum computing", limit=5)

    print(f"\n{'=' * 70}")
    print(f"Search Results:")
    print(f"{'=' * 70}")
    print(f"Papers returned: {len(papers)}")

    if papers:
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.get('title', 'N/A')[:60]}")
            print(f"   Year: {paper.get('year', 'N/A')}")
            print(f"   PMID: {paper.get('paperId', 'N/A')}")
    else:
        print("No papers returned")

except Exception as e:
    print(f"\n❌ Exception during search:")
    print(f"   Type: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

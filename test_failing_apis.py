#!/usr/bin/env python3
"""Test the failing APIs directly to diagnose issues"""

import yaml
import sys
sys.path.insert(0, '.')
from api_adapters import get_api_adapter

# Load config
with open('api_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

test_query = "quantum computing"

print("=" * 70)
print("Testing Non-Working APIs in Detail")
print("=" * 70)

failing_apis = ['semantic_scholar', 'pubmed', 'europepmc']

for api_name in failing_apis:
    print(f"\n{'=' * 70}")
    print(f"Testing: {api_name.upper()}")
    print('=' * 70)
    
    api_config = config['apis'][api_name]
    
    # Temporarily enable
    api_config['enabled'] = True
    
    print(f"Endpoint: {api_config.get('endpoint', 'N/A')}")
    print(f"Requires Key: {api_config.get('requires_key', False)}")
    print(f"API Key Set: {'Yes' if api_config.get('api_key') and not api_config['api_key'].startswith('${') else 'No'}")
    print(f"Rate Limit: {api_config.get('rate_limit', 'N/A')}s")
    print()
    
    try:
        print(f"Creating adapter...")
        adapter = get_api_adapter(api_name, api_config, verbose=True)
        
        if not adapter:
            print(f"❌ Failed to create adapter")
            continue
            
        print(f"✓ Adapter created successfully")
        print(f"\nSearching for: '{test_query}'...")
        
        papers = adapter.search(test_query, limit=5)
        
        if papers and len(papers) > 0:
            print(f"✅ SUCCESS: Found {len(papers)} papers")
            print(f"\nSample results:")
            for i, paper in enumerate(papers[:3], 1):
                title = paper.get('title', 'No title')[:60]
                year = paper.get('year', 'N/A')
                print(f"  {i}. [{year}] {title}...")
        else:
            print(f"⚠️  NO RESULTS: Search completed but returned 0 papers")
            print(f"   This could mean:")
            print(f"   - Query format is wrong for this API")
            print(f"   - API is rate limiting (returns empty instead of 429)")
            print(f"   - API requires authentication")
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        
        # Check for specific error types
        error_str = str(e).lower()
        if '429' in error_str or 'rate limit' in error_str:
            print(f"   → This is RATE LIMITING from our previous heavy usage")
            print(f"   → Solution: Wait 1-2 hours for rate limits to reset")
        elif '403' in error_str or 'forbidden' in error_str:
            print(f"   → This is FORBIDDEN - likely invalid/missing API key")
            print(f"   → Solution: Add valid API key to environment variables")
        elif '500' in error_str or 'internal server' in error_str:
            print(f"   → This is SERVER ERROR - the API is having issues")
            print(f"   → Solution: Wait for API to recover, or disable temporarily")
        elif '401' in error_str or 'unauthorized' in error_str:
            print(f"   → This is UNAUTHORIZED - API key required")
            print(f"   → Solution: Get API key from provider's website")
        elif 'timeout' in error_str:
            print(f"   → This is TIMEOUT - API is slow or unresponsive")
            print(f"   → Solution: Increase timeout or try later")

print(f"\n{'=' * 70}")
print("Summary & Recommendations")
print('=' * 70)
print()
print("Common causes of API failures:")
print("1. Rate Limiting - We downloaded 1,786 papers, APIs need time to reset")
print("2. Missing API Keys - Some APIs require authentication")
print("3. Server Issues - Free academic APIs often have reliability problems")
print("4. Query Format - Each API expects queries in different formats")
print()
print("Recommended actions:")
print("✓ Keep using the 3 working APIs (ArXiv, OpenAlex, CrossRef)")
print("✓ Wait 2-4 hours before retrying failed APIs")
print("✓ Consider getting API keys for better reliability:")
print("  - Semantic Scholar: https://www.semanticscholar.org/product/api")
print("  - PubMed (NCBI): https://www.ncbi.nlm.nih.gov/account/")
print("  - CORE: https://core.ac.uk/services/api")

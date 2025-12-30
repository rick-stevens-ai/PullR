#!/usr/bin/env python3
"""
Detailed debugging for failing APIs.

This script performs deep inspection of API calls to diagnose issues:
- Shows exact request URLs and parameters
- Displays response headers and status codes
- Shows response content (truncated if large)
- Tests different query formats
- Checks authentication headers
"""

import yaml
import sys
import requests
import time
import json
from urllib.parse import urlencode, quote_plus

sys.path.insert(0, '.')
from api_adapters import get_api_adapter


def debug_semantic_scholar(config):
    """Debug Semantic Scholar API in detail"""
    print("=" * 70)
    print("SEMANTIC SCHOLAR DEBUGGING")
    print("=" * 70)

    endpoint = config.get('endpoint', 'https://api.semanticscholar.org/graph/v1/paper/search')
    api_key = config.get('api_key', '')

    # Check if API key is set
    if api_key and not api_key.startswith('${'):
        print(f"✓ API Key provided: {api_key[:10]}...")
        has_key = True
    else:
        print("✗ No API key provided")
        has_key = False

    # Test query
    query = "quantum computing"
    params = {
        'query': query,
        'limit': 5,
        'fields': 'paperId,title,abstract,year,authors,venue,openAccessPdf,externalIds'
    }

    url = f"{endpoint}?{urlencode(params)}"
    print(f"\nRequest URL:\n  {url}")

    # Prepare headers
    headers = {}
    if has_key:
        headers['x-api-key'] = api_key

    print(f"\nRequest Headers:")
    for k, v in headers.items():
        if k.lower() == 'x-api-key':
            print(f"  {k}: {v[:10]}...")
        else:
            print(f"  {k}: {v}")

    # Make request
    print("\nMaking request...")
    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)

        print(f"\nResponse Status: {response.status_code} {response.reason}")
        print(f"Response Headers:")
        for k, v in response.headers.items():
            print(f"  {k}: {v}")

        print(f"\nResponse Content (first 500 chars):")
        content = response.text[:500]
        print(f"  {content}")

        if response.status_code == 200:
            try:
                data = response.json()
                if 'data' in data:
                    print(f"\n✅ SUCCESS: Found {len(data['data'])} papers")
                    if data['data']:
                        print(f"Sample: {data['data'][0].get('title', 'N/A')[:60]}")
                else:
                    print(f"\n⚠️ Response missing 'data' field")
            except:
                print(f"\n❌ Failed to parse JSON response")
        elif response.status_code == 403:
            print(f"\n❌ FORBIDDEN (403)")
            print(f"Possible causes:")
            print(f"  1. Rate limiting - too many requests recently")
            print(f"  2. Invalid API key")
            print(f"  3. IP blocked")
            print(f"  4. Service requires authentication")
        elif response.status_code == 429:
            print(f"\n❌ RATE LIMITED (429)")
            print(f"You've exceeded the rate limit. Wait before retrying.")
            if 'Retry-After' in response.headers:
                print(f"Retry after: {response.headers['Retry-After']} seconds")

    except Exception as e:
        print(f"\n❌ Exception: {type(e).__name__}")
        print(f"   {str(e)}")


def debug_pubmed(config):
    """Debug PubMed API in detail"""
    print("\n" + "=" * 70)
    print("PUBMED DEBUGGING")
    print("=" * 70)

    base_url = config.get('endpoint', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/')
    api_key = config.get('api_key', '')

    # Check if API key is set
    if api_key and not api_key.startswith('${'):
        print(f"✓ API Key provided: {api_key[:10]}...")
        has_key = True
    else:
        print("✗ No API key provided")
        has_key = False

    # PubMed uses two-step process: ESearch then EFetch
    query = "quantum computing"

    # Step 1: ESearch
    print("\n--- Step 1: ESearch (find paper IDs) ---")
    esearch_url = f"{base_url}esearch.fcgi"
    esearch_params = {
        'db': 'pubmed',
        'term': query,
        'retmax': 5,
        'retmode': 'json'
    }
    if has_key:
        esearch_params['api_key'] = api_key

    full_url = f"{esearch_url}?{urlencode(esearch_params)}"
    print(f"Request URL:\n  {full_url}")

    try:
        print("\nMaking ESearch request...")
        response = requests.get(esearch_url, params=esearch_params, timeout=10)

        print(f"\nResponse Status: {response.status_code} {response.reason}")
        print(f"Response Headers:")
        for k, v in list(response.headers.items())[:5]:
            print(f"  {k}: {v}")

        print(f"\nResponse Content (first 500 chars):")
        content = response.text[:500]
        print(f"  {content}")

        if response.status_code == 200:
            try:
                data = response.json()
                if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                    ids = data['esearchresult']['idlist']
                    count = data['esearchresult'].get('count', 0)
                    print(f"\n✅ ESearch SUCCESS: Found {count} total papers")
                    print(f"Retrieved {len(ids)} IDs: {ids[:5]}")

                    if ids:
                        # Step 2: EFetch
                        print("\n--- Step 2: EFetch (get paper details) ---")
                        efetch_url = f"{base_url}efetch.fcgi"
                        efetch_params = {
                            'db': 'pubmed',
                            'id': ','.join(ids[:2]),  # Just get 2 for testing
                            'retmode': 'xml'
                        }
                        if has_key:
                            efetch_params['api_key'] = api_key

                        full_url = f"{efetch_url}?{urlencode(efetch_params)}"
                        print(f"Request URL:\n  {full_url}")

                        time.sleep(0.4)  # Rate limiting
                        response2 = requests.get(efetch_url, params=efetch_params, timeout=10)

                        print(f"\nResponse Status: {response2.status_code} {response2.reason}")
                        print(f"\nResponse Content (first 800 chars):")
                        print(f"  {response2.text[:800]}")

                        if response2.status_code == 200:
                            print(f"\n✅ EFetch SUCCESS")
                        else:
                            print(f"\n❌ EFetch FAILED")
                else:
                    print(f"\n⚠️ Response missing expected fields")
            except Exception as e:
                print(f"\n❌ Failed to parse response: {e}")
        elif response.status_code == 400:
            print(f"\n❌ BAD REQUEST (400)")
            print(f"Possible causes:")
            print(f"  1. Invalid query parameters")
            print(f"  2. Malformed query term")
            print(f"  3. Missing required parameters")
        elif response.status_code == 429:
            print(f"\n❌ RATE LIMITED (429)")

    except Exception as e:
        print(f"\n❌ Exception: {type(e).__name__}")
        print(f"   {str(e)}")


def debug_europepmc(config):
    """Debug Europe PMC API in detail"""
    print("\n" + "=" * 70)
    print("EUROPE PMC DEBUGGING")
    print("=" * 70)

    endpoint = config.get('endpoint', 'https://www.ebi.ac.uk/europepmc/webservices/rest/search')

    query = "quantum computing"
    params = {
        'query': query,
        'format': 'json',
        'pageSize': 5,
        'resultType': 'core'
    }

    full_url = f"{endpoint}?{urlencode(params)}"
    print(f"Request URL:\n  {full_url}")

    try:
        print("\nMaking request...")
        response = requests.get(endpoint, params=params, timeout=10)

        print(f"\nResponse Status: {response.status_code} {response.reason}")
        print(f"Response Headers:")
        for k, v in list(response.headers.items())[:5]:
            print(f"  {k}: {v}")

        print(f"\nResponse Content (first 800 chars):")
        content = response.text[:800]
        print(f"  {content}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"\n✓ Valid JSON response")
                print(f"Response keys: {list(data.keys())}")

                if 'resultList' in data and 'result' in data['resultList']:
                    results = data['resultList']['result']
                    hit_count = data.get('hitCount', 0)
                    print(f"\n✅ SUCCESS: {hit_count} total hits, {len(results)} returned")
                    if results:
                        print(f"Sample: {results[0].get('title', 'N/A')[:60]}")
                    else:
                        print(f"\n⚠️ hitCount shows results but result list is empty")
                        print(f"This might be a query format issue")
                else:
                    print(f"\n⚠️ Response missing expected 'resultList' field")
                    print(f"Full response structure:")
                    print(json.dumps(data, indent=2)[:500])
            except Exception as e:
                print(f"\n❌ Failed to parse JSON: {e}")
        else:
            print(f"\n❌ Non-200 status code")

    except Exception as e:
        print(f"\n❌ Exception: {type(e).__name__}")
        print(f"   {str(e)}")


def test_with_adapter(api_name, config):
    """Test using the actual adapter code"""
    print("\n" + "=" * 70)
    print(f"TESTING WITH ACTUAL ADAPTER: {api_name.upper()}")
    print("=" * 70)

    try:
        # Enable temporarily
        config['enabled'] = True

        adapter = get_api_adapter(api_name, config, verbose=True)
        if not adapter:
            print(f"❌ Failed to create adapter")
            return

        print(f"✓ Adapter created")
        print(f"\nSearching with adapter...")

        papers = adapter.search("quantum computing", limit=5)

        if papers and len(papers) > 0:
            print(f"\n✅ SUCCESS: Found {len(papers)} papers via adapter")
            print(f"Sample: {papers[0].get('title', 'N/A')[:60]}")
        else:
            print(f"\n⚠️ Adapter returned 0 papers")

    except Exception as e:
        print(f"\n❌ Adapter Exception: {type(e).__name__}")
        print(f"   {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


def main():
    # Load config
    print("Loading API configuration...")
    try:
        with open('api_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    apis_to_debug = ['semantic_scholar', 'pubmed', 'europepmc']

    for api_name in apis_to_debug:
        api_config = config['apis'].get(api_name, {})

        # Direct API testing
        if api_name == 'semantic_scholar':
            debug_semantic_scholar(api_config)
        elif api_name == 'pubmed':
            debug_pubmed(api_config)
        elif api_name == 'europepmc':
            debug_europepmc(api_config)

        # Test with adapter
        test_with_adapter(api_name, api_config)

        print("\n")


if __name__ == "__main__":
    main()

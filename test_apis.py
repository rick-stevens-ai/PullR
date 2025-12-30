#!/usr/bin/env python3
"""
Test script for all API adapters.

Tests each of the 7 academic paper search APIs to verify they're working
and identify any connectivity or implementation issues.
"""

import sys
import time
import yaml
from datetime import datetime
from api_adapters import get_api_adapter


def load_api_config(config_file="api_config.yaml"):
    """Load API configuration"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_api(api_name, api_config, test_query="quantum computing"):
    """Test a single API adapter

    Args:
        api_name: Name of the API
        api_config: Configuration dict for the API
        test_query: Query to test with

    Returns:
        Dict with test results
    """
    result = {
        'api': api_name,
        'enabled': api_config.get('enabled', True),
        'requires_key': api_config.get('requires_key', False),
        'status': 'NOT_TESTED',
        'error': None,
        'papers_found': 0,
        'response_time': 0,
        'sample_title': None
    }

    # Skip if disabled
    if not result['enabled']:
        result['status'] = 'DISABLED'
        return result

    # Check for required API key
    if result['requires_key']:
        api_key = api_config.get('api_key')
        if not api_key or api_key.startswith('${'):
            result['status'] = 'MISSING_API_KEY'
            result['error'] = 'API key required but not provided'
            return result

    # Test the API
    try:
        print(f"\n  Testing {api_name}...", end=' ', flush=True)

        # Create adapter
        start_time = time.time()
        adapter = get_api_adapter(api_name, api_config, verbose=False)

        if not adapter:
            result['status'] = 'ADAPTER_FAILED'
            result['error'] = 'Failed to create adapter'
            print("❌ FAILED (adapter creation)")
            return result

        # Try search
        papers = adapter.search(test_query, limit=5)
        elapsed = time.time() - start_time
        result['response_time'] = round(elapsed, 2)

        if papers and len(papers) > 0:
            result['status'] = 'SUCCESS'
            result['papers_found'] = len(papers)
            result['sample_title'] = papers[0].get('title', 'N/A')[:60]
            print(f"✅ SUCCESS ({len(papers)} papers, {elapsed:.2f}s)")
        else:
            result['status'] = 'NO_RESULTS'
            result['error'] = 'Search returned no papers'
            print(f"⚠️  NO RESULTS ({elapsed:.2f}s)")

    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)
        print(f"❌ ERROR: {str(e)[:50]}")

    return result


def test_all_apis(config_file="api_config.yaml", test_query="quantum computing", verbose=False):
    """Test all configured APIs

    Args:
        config_file: Path to API configuration file
        test_query: Query to test with
        verbose: Print detailed information

    Returns:
        Dict with all test results
    """
    print("=" * 70)
    print("API Adapter Test Suite")
    print("=" * 70)
    print(f"Test Query: '{test_query}'")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load configuration
    try:
        config = load_api_config(config_file)
        apis = config.get('apis', {})
        print(f"Loaded configuration: {len(apis)} APIs configured")
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return {}

    # Test each API
    results = {}

    for api_name, api_config in apis.items():
        result = test_api(api_name, api_config, test_query)
        results[api_name] = result

        if verbose and result['status'] == 'SUCCESS':
            print(f"    Sample: {result['sample_title']}")

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    status_counts = {}
    for api_name, result in results.items():
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\nTotal APIs: {len(results)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Detailed results table
    print("\nDetailed Results:")
    print("-" * 70)
    print(f"{'API':<20} {'Status':<20} {'Papers':<10} {'Time':<10}")
    print("-" * 70)

    for api_name, result in sorted(results.items()):
        status_symbol = {
            'SUCCESS': '✅',
            'NO_RESULTS': '⚠️ ',
            'ERROR': '❌',
            'DISABLED': '⊘ ',
            'MISSING_API_KEY': '🔑',
            'NOT_TESTED': '?',
            'ADAPTER_FAILED': '❌'
        }.get(result['status'], '?')

        papers = str(result['papers_found']) if result['papers_found'] > 0 else '-'
        response_time = f"{result['response_time']}s" if result['response_time'] > 0 else '-'

        print(f"{api_name:<20} {status_symbol} {result['status']:<18} {papers:<10} {response_time:<10}")

        if result['error'] and verbose:
            print(f"  Error: {result['error']}")

    print("-" * 70)

    # Recommendations
    print("\nRecommendations:")
    working_apis = [name for name, r in results.items() if r['status'] == 'SUCCESS']
    failing_apis = [name for name, r in results.items() if r['status'] in ['ERROR', 'NO_RESULTS']]

    if working_apis:
        print(f"  ✅ Working APIs ({len(working_apis)}): {', '.join(working_apis)}")
    else:
        print(f"  ❌ No APIs are currently working!")

    if failing_apis:
        print(f"  ⚠️  Failing APIs ({len(failing_apis)}): {', '.join(failing_apis)}")
        print(f"     → These may need API keys, have rate limits, or be experiencing issues")

    # Check for rate limiting
    rate_limited = [name for name, r in results.items()
                    if r['error'] and ('429' in str(r['error']) or 'rate limit' in str(r['error']).lower())]
    if rate_limited:
        print(f"  🚫 Rate Limited ({len(rate_limited)}): {', '.join(rate_limited)}")
        print(f"     → Wait 30-60 minutes before retrying these APIs")

    # Check for missing keys
    missing_keys = [name for name, r in results.items() if r['status'] == 'MISSING_API_KEY']
    if missing_keys:
        print(f"  🔑 Need API Keys ({len(missing_keys)}): {', '.join(missing_keys)}")
        print(f"     → Add these API keys to environment variables")

    print("\n" + "=" * 70)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test all API adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all APIs with default query
  python test_apis.py

  # Test with custom query
  python test_apis.py --query "machine learning"

  # Verbose output with sample titles
  python test_apis.py --verbose

  # Test specific API
  python test_apis.py --api arxiv
        """
    )

    parser.add_argument('--config', default='api_config.yaml',
                        help='API configuration file (default: api_config.yaml)')
    parser.add_argument('--query', default='quantum computing',
                        help='Test query (default: "quantum computing")')
    parser.add_argument('--api', default=None,
                        help='Test only specific API (default: test all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output including sample titles')

    args = parser.parse_args()

    if args.api:
        # Test single API
        print(f"Testing single API: {args.api}")
        try:
            config = load_api_config(args.config)
            if args.api not in config['apis']:
                print(f"Error: API '{args.api}' not found in config")
                sys.exit(1)

            api_config = config['apis'][args.api]
            result = test_api(args.api, api_config, args.query)

            print(f"\nResult: {result['status']}")
            if result['papers_found'] > 0:
                print(f"Papers found: {result['papers_found']}")
                print(f"Sample: {result['sample_title']}")
            if result['error']:
                print(f"Error: {result['error']}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Test all APIs
        results = test_all_apis(args.config, args.query, args.verbose)

        # Exit with error if no APIs are working
        working = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        if working == 0:
            print("\n⚠️  WARNING: No APIs are currently working!")
            sys.exit(1)


if __name__ == "__main__":
    main()

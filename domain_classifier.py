"""
Domain Classification and API Routing for PullR

This module analyzes reference text to determine the subject domain (biomedical,
computer science, physics/math, general) and routes queries to the most appropriate
academic paper search APIs.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse


# Domain keywords for classification
DOMAIN_KEYWORDS = {
    'biomedical': [
        'protein', 'gene', 'clinical', 'disease', 'medicine', 'therapy',
        'patient', 'drug', 'molecular', 'biology', 'medical', 'health',
        'cancer', 'cell', 'dna', 'rna', 'pharmaceutical', 'diagnosis',
        'treatment', 'hospital', 'surgery', 'pathology', 'immunology',
        'genetics', 'biochemistry', 'pharmacology', 'epidemiology',
        'virus', 'bacteria', 'infection', 'vaccine', 'antibody', 'enzyme',
        'crispr', 'genome', 'chromosome', 'mutation', 'neurological'
    ],
    'computer_science': [
        'algorithm', 'neural', 'model', 'machine learning', 'deep learning',
        'ai', 'network', 'computing', 'software', 'data', 'artificial intelligence',
        'computer', 'programming', 'code', 'optimization', 'database',
        'security', 'encryption', 'compiler', 'architecture', 'processor',
        'gpu', 'cpu', 'memory', 'cache', 'parallel', 'distributed',
        'blockchain', 'cryptocurrency', 'nlp', 'computer vision', 'robotics',
        'reinforcement learning', 'supervised learning', 'unsupervised learning',
        'transformer', 'attention', 'bert', 'gpt', 'llm', 'language model',
        'convolutional', 'recurrent', 'backpropagation', 'gradient descent'
    ],
    'physics_math': [
        'quantum', 'theorem', 'equation', 'particle', 'physics',
        'mathematical', 'proof', 'topology', 'algebra', 'geometry',
        'calculus', 'differential', 'integral', 'matrix', 'tensor',
        'relativity', 'mechanics', 'thermodynamics', 'electromagnetism',
        'optics', 'astronomy', 'cosmology', 'astrophysics', 'photon',
        'electron', 'neutron', 'proton', 'atom', 'nuclear', 'radiation',
        'wave', 'frequency', 'amplitude', 'energy', 'momentum', 'force',
        'superconductor', 'semiconductor', 'qubit', 'entanglement',
        'superposition', 'wavefunct', 'hamiltonian', 'lagrangian'
    ]
}


def classify_reference_domain(reference_text: str, config: Optional[Dict] = None) -> str:
    """
    Classify a reference into a subject domain based on keyword analysis

    Args:
        reference_text: The reference text to classify
        config: Optional configuration with custom domain keywords

    Returns:
        Domain name: 'biomedical', 'computer_science', 'physics_math', or 'general'
    """
    if not reference_text:
        return 'general'

    # Normalize text for matching
    text_lower = reference_text.lower()

    # Use custom keywords if provided, otherwise use defaults
    keywords = DOMAIN_KEYWORDS
    if config and 'domain_routing' in config:
        # Override with config keywords if available
        custom_keywords = {}
        for domain, domain_config in config['domain_routing'].items():
            if 'keywords' in domain_config:
                custom_keywords[domain] = domain_config['keywords']
        if custom_keywords:
            keywords = custom_keywords

    # Count keyword matches for each domain
    domain_scores = {}
    for domain, domain_keywords in keywords.items():
        score = 0
        matched_keywords = []

        for keyword in domain_keywords:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                score += len(matches)
                matched_keywords.append(keyword)

        domain_scores[domain] = {
            'score': score,
            'keywords': matched_keywords
        }

    # Determine winner
    if not any(info['score'] > 0 for info in domain_scores.values()):
        return 'general'

    # Get domain with highest score
    best_domain = max(domain_scores.items(), key=lambda x: x[1]['score'])
    domain_name = best_domain[0]
    score = best_domain[1]['score']

    # Require minimum score threshold (at least 2 keyword matches)
    if score < 2:
        return 'general'

    return domain_name


def get_api_priority_for_domain(domain: str, config: Optional[Dict] = None) -> List[str]:
    """
    Get the ordered list of APIs to try for a given domain

    Args:
        domain: Domain name ('biomedical', 'computer_science', 'physics_math', 'general')
        config: Optional configuration with API routing rules

    Returns:
        Ordered list of API names to try
    """
    # Default API priorities by domain
    default_priorities = {
        'biomedical': [
            'pubmed',
            'europepmc',
            'openalex',
            'semantic_scholar'
        ],
        'computer_science': [
            'arxiv',
            'semantic_scholar',
            'openalex',
            'core'
        ],
        'physics_math': [
            'arxiv',
            'openalex',
            'semantic_scholar'
        ],
        'general': [
            'semantic_scholar',
            'openalex',
            'crossref',
            'core'
        ]
    }

    # Use config priorities if provided
    if config and 'domain_routing' in config:
        domain_config = config['domain_routing'].get(domain, {})
        primary_apis = domain_config.get('primary_apis', [])
        fallback_apis = domain_config.get('fallback_apis', [])
        if primary_apis or fallback_apis:
            return primary_apis + fallback_apis

    # Use defaults
    return default_priorities.get(domain, default_priorities['general'])


def detect_paper_type_from_url(url: str) -> Optional[str]:
    """
    Detect paper type/source from URL

    Args:
        url: Paper URL

    Returns:
        Paper type ('arxiv', 'pubmed', 'pmc', 'doi', etc.) or None
    """
    if not url or url == 'N/A':
        return None

    url_lower = url.lower()

    # ArXiv
    if 'arxiv.org' in url_lower:
        return 'arxiv'

    # PubMed
    if 'pubmed.ncbi.nlm.nih.gov' in url_lower or 'ncbi.nlm.nih.gov/pubmed' in url_lower:
        return 'pubmed'

    # PMC (PubMed Central)
    if 'pmc.ncbi.nlm.nih.gov' in url_lower or 'ncbi.nlm.nih.gov/pmc' in url_lower:
        return 'pmc'

    # Europe PMC
    if 'europepmc.org' in url_lower:
        return 'europepmc'

    # DOI
    if 'doi.org' in url_lower or url_lower.startswith('doi:'):
        return 'doi'

    # Semantic Scholar
    if 'semanticscholar.org' in url_lower:
        return 'semantic_scholar'

    # OpenAlex
    if 'openalex.org' in url_lower:
        return 'openalex'

    # CORE
    if 'core.ac.uk' in url_lower:
        return 'core'

    # CrossRef
    if 'crossref.org' in url_lower:
        return 'crossref'

    return None


def extract_arxiv_id(text: str) -> Optional[str]:
    """
    Extract arXiv ID from text

    Args:
        text: Text containing potential arXiv ID

    Returns:
        ArXiv ID if found, None otherwise
    """
    # New format: YYMM.NNNNN (e.g., 1706.03762)
    pattern1 = r'\b(\d{4}\.\d{4,5})\b'
    match = re.search(pattern1, text)
    if match:
        return match.group(1)

    # Old format: archive/YYMMNNN (e.g., cs/0703001)
    pattern2 = r'\b([a-z\-]+/\d{7})\b'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1)

    # arXiv: prefix format
    pattern3 = r'arXiv:(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})'
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def extract_pmid(text: str) -> Optional[str]:
    """
    Extract PubMed ID (PMID) from text

    Args:
        text: Text containing potential PMID

    Returns:
        PMID if found, None otherwise
    """
    # PMID: followed by digits
    pattern1 = r'PMID:?\s*(\d+)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1)

    # PubMed URL
    pattern2 = r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def extract_doi(text: str) -> Optional[str]:
    """
    Extract DOI from text

    Args:
        text: Text containing potential DOI

    Returns:
        DOI if found, None otherwise
    """
    # DOI: prefix format
    pattern1 = r'DOI:?\s*(10\.\d{4,}/[^\s]+)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1).rstrip('.,;')

    # DOI URL format
    pattern2 = r'doi\.org/(10\.\d{4,}/[^\s]+)'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1).rstrip('.,;')

    # Direct DOI format (10.xxxx/...)
    pattern3 = r'\b(10\.\d{4,}/[^\s]+)\b'
    match = re.search(pattern3, text)
    if match:
        doi = match.group(1).rstrip('.,;')
        # Validate it looks like a real DOI (has publisher and article parts)
        if '/' in doi and len(doi) > 8:
            return doi

    return None


def should_try_direct_fetch(reference_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Determine if we should try direct fetch based on identifiers in the text

    Args:
        reference_text: Reference text to analyze

    Returns:
        Tuple of (should_fetch, identifier_type, identifier_value)
    """
    # Check for arXiv ID
    arxiv_id = extract_arxiv_id(reference_text)
    if arxiv_id:
        return (True, 'arxiv', arxiv_id)

    # Check for PMID
    pmid = extract_pmid(reference_text)
    if pmid:
        return (True, 'pmid', pmid)

    # Check for DOI
    doi = extract_doi(reference_text)
    if doi:
        return (True, 'doi', doi)

    return (False, None, None)


def get_recommended_search_strategy(reference_text: str, domain: str) -> Dict[str, Any]:
    """
    Get recommended search strategy based on reference text and domain

    Args:
        reference_text: Reference text to analyze
        domain: Classified domain

    Returns:
        Dictionary with search strategy recommendations
    """
    # Check for direct identifiers
    has_direct, id_type, id_value = should_try_direct_fetch(reference_text)

    # Detect paper type from URL if present
    paper_type = detect_paper_type_from_url(reference_text)

    return {
        'domain': domain,
        'has_direct_identifier': has_direct,
        'identifier_type': id_type,
        'identifier_value': id_value,
        'detected_paper_type': paper_type,
        'recommended_approach': 'direct_fetch' if has_direct else 'search'
    }

"""
API Adapters for PullR - Multi-Source Academic Paper Search

This module provides a unified interface for querying multiple academic paper
search APIs (ArXiv, PubMed, OpenAlex, CrossRef, CORE, Europe PMC, Semantic Scholar).

Each API adapter:
- Implements the BaseAPIAdapter interface
- Handles API-specific request formatting
- Normalizes responses to a common format
- Manages rate limiting and error handling
"""

import time
import requests
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class RateLimiter:
    """Thread-safe rate limiter for API calls"""

    def __init__(self, api_name: str, delay_seconds: float):
        """
        Initialize rate limiter

        Args:
            api_name: Name of the API
            delay_seconds: Minimum seconds between calls
        """
        self.api_name = api_name
        self.delay = delay_seconds
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        """Wait for rate limit if necessary"""
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                time.sleep(sleep_time)
            self.last_call = time.time()


class BaseAPIAdapter(ABC):
    """Abstract base class for all API adapters"""

    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize API adapter

        Args:
            config: API configuration dictionary
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self.api_name = self.__class__.__name__.replace('Adapter', '').lower()
        self.rate_limiter = RateLimiter(
            self.api_name,
            config.get('rate_limit', 1.0)
        )

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for papers by keyword/query

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of normalized paper dictionaries
        """
        pass

    def search_exact(self, ref_info: Dict, limit: int = 10) -> List[Dict]:
        """
        Search for papers using extracted reference information

        Args:
            ref_info: Dictionary with keys: title, author, year
            limit: Maximum number of results

        Returns:
            List of normalized paper dictionaries
        """
        # Default implementation: construct query from ref_info
        query_parts = []
        if ref_info.get('title'):
            query_parts.append(ref_info['title'])
        if ref_info.get('author'):
            query_parts.append(ref_info['author'])
        if ref_info.get('year'):
            query_parts.append(str(ref_info['year']))

        query = ' '.join(query_parts)
        return self.search(query, limit)

    @abstractmethod
    def normalize_paper(self, raw_paper: Any) -> Dict:
        """
        Normalize API-specific paper format to common format

        Args:
            raw_paper: API-specific paper data

        Returns:
            Normalized paper dictionary with standard fields
        """
        pass

    def get_rate_limit_delay(self) -> float:
        """Get the rate limit delay for this API"""
        return self.config.get('rate_limit', 1.0)

    def requires_api_key(self) -> bool:
        """Check if this API requires an API key"""
        return self.config.get('requires_key', False)

    def is_enabled(self) -> bool:
        """Check if this API is enabled"""
        return self.config.get('enabled', True)

    def _make_request(self, url: str, params: Dict = None, headers: Dict = None,
                     max_retries: int = 3, timeout: int = 30) -> Optional[requests.Response]:
        """
        Make HTTP request with rate limiting and retry logic

        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()

                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout
                )

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited
                    wait_time = (2 ** attempt) * 2
                    if self.verbose:
                        print(f"  {self.api_name}: Rate limited (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [500, 502, 503, 504]:
                    # Server error
                    wait_time = (2 ** attempt) * 1
                    if self.verbose:
                        print(f"  {self.api_name}: Server error ({response.status_code}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 403:
                    if self.verbose:
                        print(f"  {self.api_name}: Access forbidden (403)")
                    return None
                elif response.status_code == 404:
                    if self.verbose:
                        print(f"  {self.api_name}: Not found (404)")
                    return None
                else:
                    if self.verbose:
                        print(f"  {self.api_name}: HTTP {response.status_code}")
                    return None

            except requests.exceptions.Timeout:
                if self.verbose:
                    print(f"  {self.api_name}: Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                if self.verbose:
                    print(f"  {self.api_name}: Error - {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

        return None

    def validate_config(self) -> bool:
        """
        Validate API configuration

        Returns:
            True if configuration is valid, False otherwise
        """
        if self.requires_api_key() and not self.config.get('api_key'):
            print(f"Warning: {self.api_name} requires API key but none provided")
            return False

        if not self.config.get('endpoint'):
            print(f"Error: {self.api_name} missing endpoint URL")
            return False

        return True


class SemanticScholarAdapter(BaseAPIAdapter):
    """Adapter for Semantic Scholar API"""

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Semantic Scholar by keyword"""
        url = self.config['endpoint']
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,authors,year,externalIds,url,venue,openAccessPdf,abstract,paperId'
        }

        headers = {}
        if self.config.get('api_key'):
            headers['x-api-key'] = self.config['api_key']

        response = self._make_request(url, params, headers)
        if not response:
            return []

        try:
            data = response.json()
            papers = data.get('data', [])
            return [self.normalize_paper(p) for p in papers]
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Failed to parse response - {e}")
            return []

    def normalize_paper(self, raw_paper: Dict) -> Dict:
        """Normalize Semantic Scholar paper format"""
        return {
            'paperId': raw_paper.get('paperId', 'unknown'),
            'title': raw_paper.get('title', 'No Title'),
            'abstract': raw_paper.get('abstract', ''),
            'year': raw_paper.get('year', 'N/A'),
            'authors': raw_paper.get('authors', []),
            'url': raw_paper.get('url', 'N/A'),
            'venue': raw_paper.get('venue', 'N/A'),
            'openAccessPdf': raw_paper.get('openAccessPdf'),
            'externalIds': raw_paper.get('externalIds', {}),
            'api_source': 'semantic_scholar'
        }


# Placeholder adapters - to be implemented in subsequent phases
class ArXivAdapter(BaseAPIAdapter):
    """Adapter for ArXiv API - CS/Physics/Math preprints"""

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search ArXiv by keyword"""
        try:
            import feedparser
        except ImportError:
            if self.verbose:
                print(f"  {self.api_name}: feedparser not installed, skipping")
            return []

        # Build query URL
        from urllib.parse import urlencode
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': limit,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        url = f"{self.config['endpoint']}?{urlencode(params)}"

        response = self._make_request(url)
        if not response:
            return []

        try:
            # Parse Atom feed with feedparser
            feed = feedparser.parse(response.content)

            if not feed.entries:
                if self.verbose:
                    print(f"  {self.api_name}: No results found")
                return []

            papers = []
            for entry in feed.entries:
                try:
                    paper = self.normalize_paper(entry)
                    papers.append(paper)
                except Exception as e:
                    if self.verbose:
                        print(f"  {self.api_name}: Error normalizing entry - {e}")
                    continue

            return papers
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Failed to parse response - {e}")
            return []

    def search_exact(self, ref_info: Dict, limit: int = 10) -> List[Dict]:
        """Search ArXiv using reference info"""
        # For ArXiv, construct a more targeted query
        query_parts = []

        if ref_info.get('title'):
            # Search in title field specifically
            title = ref_info['title'].replace('"', '')
            query_parts.append(f'ti:"{title}"')

        if ref_info.get('author'):
            # Search in author field
            author = ref_info['author'].replace('"', '')
            query_parts.append(f'au:{author}')

        if query_parts:
            try:
                import feedparser
            except ImportError:
                if self.verbose:
                    print(f"  {self.api_name}: feedparser not installed, skipping")
                return []

            from urllib.parse import urlencode

            # Use AND for more precise matching
            search_query = ' AND '.join(query_parts)

            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': limit,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            url = f"{self.config['endpoint']}?{urlencode(params)}"

            response = self._make_request(url)
            if not response:
                return []

            try:
                feed = feedparser.parse(response.content)
                papers = []
                for entry in feed.entries:
                    try:
                        paper = self.normalize_paper(entry)
                        papers.append(paper)
                    except Exception as e:
                        if self.verbose:
                            print(f"  {self.api_name}: Error normalizing entry - {e}")
                        continue
                return papers
            except Exception as e:
                if self.verbose:
                    print(f"  {self.api_name}: Failed to parse response - {e}")
                return []

        # Fallback to general search
        return self.search(' '.join([str(v) for v in ref_info.values() if v]), limit)

    def normalize_paper(self, raw_paper: Any) -> Dict:
        """Normalize ArXiv entry to standard format"""
        # Extract arXiv ID from entry.id (format: http://arxiv.org/abs/XXXX.XXXXX)
        arxiv_id = raw_paper.id.split('/abs/')[-1] if hasattr(raw_paper, 'id') else 'unknown'

        # Extract title
        title = raw_paper.title.replace('\n', ' ').strip() if hasattr(raw_paper, 'title') else 'No Title'

        # Extract abstract
        abstract = raw_paper.summary.replace('\n', ' ').strip() if hasattr(raw_paper, 'summary') else ''

        # Extract year from published date
        year = 'N/A'
        if hasattr(raw_paper, 'published'):
            try:
                year = int(raw_paper.published[:4])
            except (ValueError, TypeError):
                year = 'N/A'

        # Extract authors
        authors = []
        if hasattr(raw_paper, 'authors'):
            for author in raw_paper.authors:
                if hasattr(author, 'name'):
                    authors.append({'name': author.name})

        # Get PDF URL (always available for arXiv)
        pdf_url = None
        if hasattr(raw_paper, 'links'):
            for link in raw_paper.links:
                if hasattr(link, 'type') and link.type == 'application/pdf':
                    pdf_url = link.href
                    break

        # If no PDF link found, construct it from arXiv ID
        if not pdf_url and arxiv_id != 'unknown':
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Get primary category
        venue = 'arXiv'
        if hasattr(raw_paper, 'arxiv_primary_category'):
            if hasattr(raw_paper.arxiv_primary_category, 'term'):
                venue = f"arXiv {raw_paper.arxiv_primary_category.term}"
        elif hasattr(raw_paper, 'tags'):
            # Fallback: use first category tag
            for tag in raw_paper.tags:
                if hasattr(tag, 'term'):
                    venue = f"arXiv {tag.term}"
                    break

        # Get canonical URL
        url = raw_paper.link if hasattr(raw_paper, 'link') else f"https://arxiv.org/abs/{arxiv_id}"

        return {
            'paperId': f"arXiv:{arxiv_id}",
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'url': url,
            'venue': venue,
            'openAccessPdf': {'url': pdf_url} if pdf_url else None,
            'externalIds': {'ArXiv': arxiv_id},
            'api_source': 'arxiv'
        }


class PubMedAdapter(BaseAPIAdapter):
    """Adapter for PubMed API - Biomedical literature (42M+ abstracts)"""

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search PubMed using E-utilities (ESearch + EFetch)"""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            if self.verbose:
                print(f"  {self.api_name}: XML parsing not available")
            return []

        # Step 1: ESearch - Get PMIDs
        esearch_url = f"{self.config['endpoint']}esearch.fcgi"
        esearch_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': limit,
            'retmode': 'json',
            'sort': 'relevance'
        }

        if self.config.get('api_key'):
            esearch_params['api_key'] = self.config['api_key']

        search_response = self._make_request(esearch_url, params=esearch_params)
        if not search_response:
            return []

        try:
            search_data = search_response.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])

            if not pmids:
                if self.verbose:
                    print(f"  {self.api_name}: No results found")
                return []

            # Step 2: EFetch - Get full records
            efetch_url = f"{self.config['endpoint']}efetch.fcgi"
            efetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }

            if self.config.get('api_key'):
                efetch_params['api_key'] = self.config['api_key']

            fetch_response = self._make_request(efetch_url, params=efetch_params)
            if not fetch_response:
                return []

            # Parse XML response
            return self._parse_pubmed_xml(fetch_response.content)

        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Search failed - {e}")
            return []

    def _parse_pubmed_xml(self, xml_content: bytes) -> List[Dict]:
        """Parse PubMed XML response"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            papers = []

            for article in root.findall('.//PubmedArticle'):
                try:
                    paper = self.normalize_paper(article)
                    papers.append(paper)
                except Exception as e:
                    if self.verbose:
                        print(f"  {self.api_name}: Error parsing article - {e}")
                    continue

            return papers
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: XML parsing failed - {e}")
            return []

    def normalize_paper(self, article_element) -> Dict:
        """Normalize PubMed XML article to standard format"""
        import xml.etree.ElementTree as ET

        medline = article_element.find('.//MedlineCitation')
        if medline is None:
            raise ValueError("Invalid article structure")

        # Extract PMID
        pmid_elem = medline.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else 'unknown'

        # Extract article metadata
        article_node = medline.find('.//Article')
        if article_node is None:
            raise ValueError("No Article node")

        # Title
        title_elem = article_node.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else 'No Title'

        # Abstract
        abstract = ''
        abstract_node = article_node.find('.//Abstract')
        if abstract_node is not None:
            abstract_texts = abstract_node.findall('.//AbstractText')
            abstract_parts = []
            for text_elem in abstract_texts:
                if text_elem.text:
                    # Check for labeled sections
                    label = text_elem.get('Label', '')
                    if label:
                        abstract_parts.append(f"{label}: {text_elem.text}")
                    else:
                        abstract_parts.append(text_elem.text)
            abstract = ' '.join(abstract_parts)

        # Authors
        authors = []
        author_list = article_node.find('.//AuthorList')
        if author_list is not None:
            for author_elem in author_list.findall('.//Author'):
                lastname = author_elem.find('.//LastName')
                forename = author_elem.find('.//ForeName')
                if lastname is not None:
                    name_parts = []
                    if forename is not None and forename.text:
                        name_parts.append(forename.text)
                    if lastname.text:
                        name_parts.append(lastname.text)
                    if name_parts:
                        authors.append({'name': ' '.join(name_parts)})

        # Year
        year = 'N/A'
        pub_date = article_node.find('.//PubDate')
        if pub_date is not None:
            year_elem = pub_date.find('.//Year')
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except (ValueError, TypeError):
                    year = 'N/A'

        # Journal/Venue
        venue = 'N/A'
        journal = article_node.find('.//Journal/Title')
        if journal is not None and journal.text:
            venue = journal.text

        # Extract external IDs (PMC, DOI)
        pmc_id = None
        doi = None
        article_ids = article_element.findall('.//ArticleId')
        for aid in article_ids:
            id_type = aid.get('IdType', '')
            if id_type == 'pmc' and aid.text:
                pmc_id = aid.text
            elif id_type == 'doi' and aid.text:
                doi = aid.text

        # Check for open access PDF via PMC
        pdf_url = None
        if pmc_id:
            # PMC articles may have PDFs available
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"

        return {
            'paperId': f"PMID:{pmid}",
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            'venue': venue,
            'openAccessPdf': {'url': pdf_url} if pdf_url else None,
            'externalIds': {
                'PubMed': pmid,
                'PMC': pmc_id,
                'DOI': doi
            },
            'api_source': 'pubmed'
        }


class OpenAlexAdapter(BaseAPIAdapter):
    """Adapter for OpenAlex API - Universal open access database (200M+ works)"""

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search OpenAlex by keyword"""
        # Build query
        headers = {}
        email = self.config.get('email')
        if email:
            # Polite pool access (faster)
            headers['User-Agent'] = f"PullR (mailto:{email})"

        params = {
            'search': query,
            'per_page': min(limit, 25),  # OpenAlex max is 25
            'sort': 'relevance_score:desc'
        }

        response = self._make_request(
            self.config['endpoint'],
            params=params,
            headers=headers
        )

        if not response:
            return []

        try:
            data = response.json()
            results = data.get('results', [])

            if not results:
                if self.verbose:
                    print(f"  {self.api_name}: No results found")
                return []

            papers = []
            for work in results:
                try:
                    paper = self.normalize_paper(work)
                    papers.append(paper)
                except Exception as e:
                    if self.verbose:
                        print(f"  {self.api_name}: Error normalizing work - {e}")
                    continue

            return papers
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Failed to parse response - {e}")
            return []

    def normalize_paper(self, raw_paper: Dict) -> Dict:
        """Normalize OpenAlex work to standard format"""
        # Extract OpenAlex ID
        openalex_id = raw_paper.get('id', '').split('/')[-1]

        # Extract title
        title = raw_paper.get('title', 'No Title')

        # Reconstruct abstract from inverted index
        abstract = ''
        if raw_paper.get('abstract_inverted_index'):
            abstract = self._reconstruct_abstract(raw_paper['abstract_inverted_index'])

        # Extract year
        year = raw_paper.get('publication_year', 'N/A')

        # Extract authors
        authors = []
        for authorship in raw_paper.get('authorships', []):
            author = authorship.get('author', {})
            author_name = author.get('display_name')
            if author_name:
                authors.append({'name': author_name})

        # Get venue/source
        venue = 'N/A'
        primary_location = raw_paper.get('primary_location', {})
        if primary_location and primary_location.get('source'):
            venue = primary_location['source'].get('display_name', 'N/A')

        # Get URL - prefer DOI, fallback to OpenAlex URL
        url = raw_paper.get('doi', raw_paper.get('id', 'N/A'))

        # Get open access PDF
        pdf_url = None
        open_access = raw_paper.get('open_access', {})
        if open_access.get('is_oa'):
            pdf_url = open_access.get('oa_url')

        # Get DOI
        doi = raw_paper.get('doi', '').replace('https://doi.org/', '') if raw_paper.get('doi') else None

        return {
            'paperId': openalex_id,
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'url': url,
            'venue': venue,
            'openAccessPdf': {'url': pdf_url} if pdf_url else None,
            'externalIds': {
                'DOI': doi,
                'OpenAlex': openalex_id
            },
            'api_source': 'openalex'
        }

    def _reconstruct_abstract(self, inverted_index: Dict) -> str:
        """
        Reconstruct abstract text from OpenAlex inverted index format

        Inverted index format: {"word": [position1, position2, ...], ...}
        """
        if not inverted_index:
            return ''

        try:
            # Find the maximum position to determine array size
            max_pos = 0
            for positions in inverted_index.values():
                if positions:
                    max_pos = max(max_pos, max(positions))

            # Create array of words
            words = [''] * (max_pos + 1)

            # Fill in the words at their positions
            for word, positions in inverted_index.items():
                for pos in positions:
                    if 0 <= pos <= max_pos:
                        words[pos] = word

            # Join words and clean up
            abstract = ' '.join(words)
            return abstract.strip()
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Error reconstructing abstract - {e}")
            return ''


class EuropePMCAdapter(BaseAPIAdapter):
    """Adapter for Europe PMC API - Biomedical literature (42M+ abstracts)"""

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Europe PMC by keyword"""
        params = {
            'query': query,
            'format': 'json',
            'pageSize': min(limit, 100),  # Max 100
            'sort': 'relevance'
        }

        response = self._make_request(self.config['endpoint'], params=params)
        if not response:
            return []

        try:
            data = response.json()
            results = data.get('resultList', {}).get('result', [])

            if not results:
                if self.verbose:
                    print(f"  {self.api_name}: No results found")
                return []

            papers = []
            for result in results:
                try:
                    paper = self.normalize_paper(result)
                    papers.append(paper)
                except Exception as e:
                    if self.verbose:
                        print(f"  {self.api_name}: Error normalizing result - {e}")
                    continue

            return papers
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Failed to parse response - {e}")
            return []

    def normalize_paper(self, raw_paper: Dict) -> Dict:
        """Normalize Europe PMC result to standard format"""
        # Extract identifiers
        pmid = raw_paper.get('pmid')
        pmcid = raw_paper.get('pmcid')
        doi = raw_paper.get('doi')

        # Use best available ID
        paper_id = pmid or pmcid or doi or 'unknown'

        # Title
        title = raw_paper.get('title', 'No Title')

        # Abstract
        abstract = raw_paper.get('abstractText', '')

        # Year
        year = raw_paper.get('pubYear', 'N/A')

        # Authors
        authors = []
        author_list = raw_paper.get('authorList', {}).get('author', [])
        for author in author_list:
            full_name = author.get('fullName')
            if full_name:
                authors.append({'name': full_name})

        # Venue/Journal
        venue = raw_paper.get('journalTitle', 'N/A')

        # Check for open access PDF
        pdf_url = None
        full_text_urls = raw_paper.get('fullTextUrlList', {}).get('fullTextUrl', [])
        for ft_url in full_text_urls:
            if ft_url.get('documentStyle') == 'pdf':
                pdf_url = ft_url.get('url')
                break

        # URL
        source = raw_paper.get('source', 'MED')
        url = f"https://europepmc.org/article/{source}/{paper_id}"

        return {
            'paperId': paper_id,
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'url': url,
            'venue': venue,
            'openAccessPdf': {'url': pdf_url} if pdf_url else None,
            'externalIds': {
                'PubMed': pmid,
                'PMC': pmcid,
                'DOI': doi
            },
            'api_source': 'europepmc'
        }


class CrossRefAdapter(BaseAPIAdapter):
    """Adapter for CrossRef API - Metadata for 150M+ works"""

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search CrossRef by keyword"""
        params = {
            'query': query,
            'rows': min(limit, 100),
            'sort': 'relevance'
        }

        response = self._make_request(self.config['endpoint'], params=params)
        if not response:
            return []

        try:
            data = response.json()
            items = data.get('message', {}).get('items', [])

            if not items:
                if self.verbose:
                    print(f"  {self.api_name}: No results found")
                return []

            papers = []
            for item in items:
                try:
                    paper = self.normalize_paper(item)
                    papers.append(paper)
                except Exception as e:
                    if self.verbose:
                        print(f"  {self.api_name}: Error normalizing item - {e}")
                    continue

            return papers
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Failed to parse response - {e}")
            return []

    def normalize_paper(self, raw_paper: Dict) -> Dict:
        """Normalize CrossRef work to standard format"""
        # Extract DOI
        doi = raw_paper.get('DOI', 'unknown')

        # Extract title (can be array)
        title = raw_paper.get('title', ['No Title'])
        if isinstance(title, list):
            title = title[0] if title else 'No Title'

        # Abstract (often missing in CrossRef)
        abstract = raw_paper.get('abstract', '')

        # Extract year
        year = 'N/A'
        published = raw_paper.get('published', {}).get('date-parts', [[]])
        if published and published[0]:
            year = published[0][0]

        # Extract authors
        authors = []
        for author in raw_paper.get('author', []):
            given = author.get('given', '')
            family = author.get('family', '')
            name = f"{given} {family}".strip()
            if name:
                authors.append({'name': name})

        # Extract venue
        venue = raw_paper.get('container-title', ['N/A'])
        if isinstance(venue, list):
            venue = venue[0] if venue else 'N/A'

        # URL
        url = raw_paper.get('URL', f"https://doi.org/{doi}")

        return {
            'paperId': doi,
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'url': url,
            'venue': venue,
            'openAccessPdf': None,  # CrossRef doesn't provide PDFs
            'externalIds': {'DOI': doi},
            'api_source': 'crossref'
        }


class COREAdapter(BaseAPIAdapter):
    """Adapter for CORE API - 32M+ full-text open access papers"""

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search CORE by keyword"""
        if not self.config.get('api_key'):
            if self.verbose:
                print(f"  {self.api_name}: API key required")
            return []

        headers = {
            'Authorization': f"Bearer {self.config['api_key']}"
        }

        params = {
            'q': query,
            'limit': min(limit, 100)
        }

        response = self._make_request(
            self.config['endpoint'],
            params=params,
            headers=headers
        )

        if not response:
            return []

        try:
            data = response.json()
            results = data.get('results', [])

            if not results:
                if self.verbose:
                    print(f"  {self.api_name}: No results found")
                return []

            papers = []
            for result in results:
                try:
                    paper = self.normalize_paper(result)
                    papers.append(paper)
                except Exception as e:
                    if self.verbose:
                        print(f"  {self.api_name}: Error normalizing result - {e}")
                    continue

            return papers
        except Exception as e:
            if self.verbose:
                print(f"  {self.api_name}: Failed to parse response - {e}")
            return []

    def normalize_paper(self, raw_paper: Dict) -> Dict:
        """Normalize CORE article to standard format"""
        # Extract CORE ID
        core_id = raw_paper.get('id', 'unknown')

        # Title
        title = raw_paper.get('title', 'No Title')

        # Abstract
        abstract = raw_paper.get('abstract') or raw_paper.get('description', '')

        # Year
        year = raw_paper.get('yearPublished', 'N/A')

        # Authors
        authors = []
        for author in raw_paper.get('authors', []):
            if isinstance(author, str):
                authors.append({'name': author})
            elif isinstance(author, dict):
                name = author.get('name', '')
                if name:
                    authors.append({'name': name})

        # Venue/Publisher
        venue = raw_paper.get('publisher', 'N/A')

        # PDF URL
        pdf_url = raw_paper.get('downloadUrl')

        # URL
        url = pdf_url or (raw_paper.get('links', [{}])[0].get('url', 'N/A'))

        # DOI
        doi = raw_paper.get('doi')

        return {
            'paperId': f"CORE:{core_id}",
            'title': title,
            'abstract': abstract,
            'year': year,
            'authors': authors,
            'url': url,
            'venue': venue,
            'openAccessPdf': {'url': pdf_url} if pdf_url else None,
            'externalIds': {
                'CORE': core_id,
                'DOI': doi
            },
            'api_source': 'core'
        }


def get_api_adapter(api_name: str, config: Dict, verbose: bool = False) -> Optional[BaseAPIAdapter]:
    """
    Factory function to get API adapter by name

    Args:
        api_name: Name of the API (e.g., 'arxiv', 'pubmed')
        config: API configuration dictionary
        verbose: Enable verbose logging

    Returns:
        API adapter instance or None if not found
    """
    adapters = {
        'semantic_scholar': SemanticScholarAdapter,
        'arxiv': ArXivAdapter,
        'pubmed': PubMedAdapter,
        'openalex': OpenAlexAdapter,
        'europepmc': EuropePMCAdapter,
        'crossref': CrossRefAdapter,
        'core': COREAdapter
    }

    adapter_class = adapters.get(api_name.lower())
    if not adapter_class:
        if verbose:
            print(f"Warning: Unknown API '{api_name}'")
        return None

    try:
        adapter = adapter_class(config, verbose)
        if not adapter.is_enabled():
            if verbose:
                print(f"Info: API '{api_name}' is disabled")
            return None
        return adapter
    except NotImplementedError:
        if verbose:
            print(f"Info: API '{api_name}' not yet implemented")
        return None
    except Exception as e:
        if verbose:
            print(f"Error initializing '{api_name}' adapter: {e}")
        return None

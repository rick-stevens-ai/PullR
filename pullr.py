#!/usr/bin/env python3
"""
PullR - Research Paper Processor
Processes academic references using OpenAI LLM to extract paper information
and downloads abstracts and papers from academic databases.

Dependencies:
- pip install requests pyyaml openai PyPDF2 tqdm beautifulsoup4
"""

import argparse
import os
import sys
import time
import yaml
import re
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, OpenAIError
import requests

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from tqdm import tqdm
    PROGRESS_BAR_SUPPORT = True
except ImportError:
    PROGRESS_BAR_SUPPORT = False
    # Fallback progress bar class
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
            if desc:
                print(f"{desc}: 0/{total}")
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update()
        
        def update(self, n=1):
            self.n += n
            if self.desc and self.total:
                print(f"\r{self.desc}: {self.n}/{self.total}", end="", flush=True)
        
        def close(self):
            if self.desc:
                print()  # New line

try:
    from bs4 import BeautifulSoup
    WEB_SCRAPING_SUPPORT = True
except ImportError:
    WEB_SCRAPING_SUPPORT = False

MODEL_CONFIG_FILE = 'model_servers.yaml'
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = 'title,authors,year,externalIds,url,venue,openAccessPdf,abstract,paperId'

# Global rate limiter to prevent overwhelming the API
import threading
_last_api_call_time = 0
_api_call_lock = threading.Lock()

def wait_for_rate_limit():
    """Ensure minimum time between API calls globally"""
    global _last_api_call_time
    with _api_call_lock:
        current_time = time.time()
        time_since_last = current_time - _last_api_call_time
        min_interval = 0.5  # Minimum 0.5 seconds between any API calls
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        _last_api_call_time = time.time()

def sanitize_filename(name, max_length=50):
    """Create a safe file-system friendly name."""
    if not name:
        return "unknown"
    
    # Remove or replace problematic characters
    import unicodedata
    # Normalize unicode characters (e.g., é -> e)
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    
    # Keep only safe characters
    safe_chars = []
    for c in name:
        if c.isalnum():
            safe_chars.append(c)
        elif c in (' ', '.', '_', '-'):
            safe_chars.append(c)
        else:
            safe_chars.append('_')  # Replace unsafe chars with underscore
    
    result = "".join(safe_chars)
    
    # Clean up multiple spaces/underscores
    import re
    result = re.sub(r'[_\s]+', '_', result)
    result = result.strip('_. ')
    
    # Truncate to safe length
    if len(result) > max_length:
        result = result[:max_length].rstrip('_. ')
    
    # Ensure we have something
    if not result:
        result = "unknown"
        
    return result

def extract_text_from_pdf(pdf_path, verbose=False):
    """Extract text from PDF file with better structure preservation"""
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            total_pages = len(pdf_reader.pages)
            if verbose:
                print(f"Extracting text from {total_pages} pages...")
            
            for page_num in range(total_pages):
                if verbose and page_num % 10 == 0:
                    print(f"  Processing page {page_num + 1}/{total_pages}")
                
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # Clean up the text a bit
                page_text = re.sub(r'\n\s*\n', '\n', page_text)  # Remove excessive blank lines
                page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace but preserve structure
                
                # Add page break markers for reference section detection
                text_parts.append(f"\n--- PAGE {page_num + 1} ---\n{page_text}")
            
            full_text = '\n'.join(text_parts)
            
            if verbose:
                print(f"Extracted {len(full_text)} characters total")
                # Look for potential reference indicators
                ref_indicators = ['references', 'bibliography', 'works cited', 'literature cited']
                for indicator in ref_indicators:
                    count = len(re.findall(rf'(?i){indicator}', full_text))
                    if count > 0:
                        print(f"  Found {count} instances of '{indicator}'")
            
            return full_text
            
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")


def extract_references_from_text(text, client, openai_model, verbose=False):
    """Use OpenAI to extract individual references from paper text using intelligent chunk scanning"""
    
    if verbose:
        print("Starting intelligent reference extraction with chunk scanning...")
    
    # Step 1: Split the document into manageable chunks
    chunk_size = 8000  # Size that works well with LLM context
    chunks = []
    chunk_start = 0
    
    while chunk_start < len(text):
        chunk_end = min(chunk_start + chunk_size, len(text))
        # Try to break at natural boundaries (paragraphs)
        if chunk_end < len(text):
            last_paragraph = text.rfind('\n\n', chunk_start, chunk_end)
            if last_paragraph > chunk_start:
                chunk_end = last_paragraph + 2
        
        chunk = text[chunk_start:chunk_end]
        chunks.append({
            'text': chunk,
            'start': chunk_start,
            'end': chunk_end,
            'contains_references': False,
            'chunk_index': len(chunks)
        })
        chunk_start = chunk_end
    
    if verbose:
        print(f"Split document into {len(chunks)} chunks for analysis")
    
    # Step 2: Scan each chunk to identify which contain references
    reference_chunks = []
    
    for i, chunk_data in enumerate(chunks):
        if verbose:
            print(f"  Scanning chunk {i+1}/{len(chunks)} for references...")
        
        try:
            system_prompt_scan = """You are analyzing a chunk of an academic paper to determine if it contains references or bibliography.

Look for:
- Numbered reference lists (1., 2., 3., etc.)
- Author names followed by publication details
- Journal names, years, page numbers
- DOIs, URLs, or other citation elements
- Section headers like "References", "Bibliography", "Works Cited"

Respond with ONLY "YES" if this chunk appears to contain references/bibliography, or "NO" if it does not."""
            
            user_prompt_scan = f"""Does this text chunk contain references or bibliography entries?

{chunk_data['text'][:3000]}"""  # Use first 3000 chars to stay within limits
            
            response_scan = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_prompt_scan},
                    {"role": "user", "content": user_prompt_scan},
                ],
                temperature=0.1,
                max_tokens=10,
            )
            
            response_text = response_scan.choices[0].message.content.strip().upper()
            
            if "YES" in response_text:
                chunk_data['contains_references'] = True
                reference_chunks.append(chunk_data)
                if verbose:
                    print(f"    ✅ Chunk {i+1} contains references")
            elif verbose:
                print(f"    ❌ Chunk {i+1} does not contain references")
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            if verbose:
                print(f"    Error scanning chunk {i+1}: {e}")
            continue
    
    if verbose:
        print(f"Found {len(reference_chunks)} chunks containing references")
    
    if not reference_chunks:
        if verbose:
            print("No reference chunks found, falling back to pattern-based extraction")
        return extract_references_by_pattern(text, verbose)
    
    # Step 3: Extract references from reference-containing chunks, processing in groups of 10
    all_references = []
    
    for chunk_data in reference_chunks:
        if verbose:
            print(f"Extracting references from chunk {chunk_data['chunk_index']+1}...")
        
        try:
            system_prompt_extract = """You are extracting individual references from a chunk of an academic paper's reference section.

Extract EVERY complete reference you find. Each reference should be a complete bibliographic entry.

Guidelines:
- Look for complete citations (author, title, journal/conference, year)
- Some references may span multiple lines - combine them
- Remove page headers, footers, or other non-reference text
- Maintain the original order
- Each reference should be substantial (not just fragments)

Return each complete reference on its own line without numbering."""
            
            user_prompt_extract = f"""Extract all complete references from this text:

{chunk_data['text']}

Return each complete reference on a separate line."""
            
            response_extract = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_prompt_extract},
                    {"role": "user", "content": user_prompt_extract},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            
            extracted_text = response_extract.choices[0].message.content
            if extracted_text:
                chunk_refs = []
                for line in extracted_text.split('\n'):
                    line = line.strip()
                    if line and len(line) > 30:  # Filter very short lines
                        # Clean up any numbering the LLM might have added
                        cleaned_ref = re.sub(r'^\d+\.\s*', '', line)
                        cleaned_ref = re.sub(r'^\W+', '', cleaned_ref)
                        if cleaned_ref:
                            chunk_refs.append(cleaned_ref)
                
                if verbose:
                    print(f"  Extracted {len(chunk_refs)} references from chunk {chunk_data['chunk_index']+1}")
                
                all_references.extend(chunk_refs)
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            if verbose:
                print(f"Error extracting from chunk {chunk_data['chunk_index']+1}: {e}")
            continue
    
    if verbose:
        print(f"Total references extracted from all chunks: {len(all_references)}")
    
    # Step 4: Clean and process references in groups of 10 (like exact mode)
    if all_references:
        if verbose:
            print("Processing extracted references in groups of 10 for cleaning...")
        
        cleaned_references = preprocess_references(all_references, client, openai_model, verbose, source="pdf")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_references = []
        for ref in cleaned_references:
            ref_normalized = re.sub(r'\s+', ' ', ref.lower().strip())
            if ref_normalized not in seen and len(ref) > 30:
                seen.add(ref_normalized)
                unique_references.append(ref)
        
        if verbose:
            print(f"Final cleaned and deduplicated references: {len(unique_references)}")
        
        return unique_references
    
    # Fallback if no references found
    if verbose:
        print("No references extracted, trying pattern-based fallback...")
    return extract_references_by_pattern(text, verbose)


def extract_references_by_pattern(text, verbose=False):
    """Extract references using regex patterns as a fallback method"""
    # Look for common reference patterns
    reference_patterns = [
        # Pattern 1: Numbered references like "1. Author, Title..."
        r'^\s*(\d+)\.\s*(.{20,500}?)(?=^\s*\d+\.|$)',
        # Pattern 2: References with years in parentheses
        r'([A-Z][^.]*?\(\d{4}\)[^.]*?\.)',
        # Pattern 3: DOI-based references
        r'([^.]+?doi:\s*\d+\.\d+/[^\s]+)',
        # Pattern 4: Journal references with volume/page numbers
        r'([A-Z][^.]+?\d{4}[^.]*?\d+[^.]*?\d+[^.]*?\.)'
    ]
    
    all_pattern_refs = []
    
    for i, pattern in enumerate(reference_patterns):
        if verbose:
            print(f"  Trying pattern {i+1}...")
        
        try:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            pattern_refs = []
            
            for match in matches:
                if isinstance(match, tuple):
                    # Take the longest part of the tuple
                    ref_text = max(match, key=len) if match else ""
                else:
                    ref_text = match
                
                # Clean up the reference
                ref_text = re.sub(r'\s+', ' ', ref_text.strip())
                
                # Filter out very short or very long references
                if 30 < len(ref_text) < 1000:
                    pattern_refs.append(ref_text)
            
            if verbose:
                print(f"    Pattern {i+1} found {len(pattern_refs)} potential references")
            
            all_pattern_refs.extend(pattern_refs)
            
        except Exception as e:
            if verbose:
                print(f"    Pattern {i+1} failed: {e}")
    
    # Remove duplicates
    seen = set()
    unique_pattern_refs = []
    for ref in all_pattern_refs:
        ref_normalized = re.sub(r'\s+', ' ', ref.lower().strip())
        if ref_normalized not in seen:
            seen.add(ref_normalized)
            unique_pattern_refs.append(ref)
    
    if verbose:
        print(f"  Pattern-based extraction found {len(unique_pattern_refs)} unique references")
    
    return unique_pattern_refs

def preprocess_references(references, client, openai_model, verbose=False, source="general"):
    """Use LLM to clean and normalize references for better matching
    
    Args:
        references: List of reference strings to clean
        client: OpenAI client
        openai_model: Model name to use
        verbose: Whether to print detailed progress
        source: Source type ("txt", "pdf", "general") for context-specific prompts
    """
    if verbose:
        print(f"Preprocessing {len(references)} references with LLM...")
    
    # Enhanced system prompt based on source
    if source == "pdf":
        system_prompt = """You are an AI assistant that cleans and normalizes academic references extracted from PDF files.

PDF-extracted references often have issues like:
- Text split across lines inappropriately
- OCR errors and character misrecognition
- Merged references on single lines
- Missing or garbled text

Clean each reference by:
1. Ensuring each reference is on one line
2. Standardizing format (Author. Title. Journal/Conference. Year.)
3. Removing extra spaces, line breaks, and formatting issues
4. Fixing common OCR errors (e.g., "1" instead of "l", "0" instead of "O")
5. Ensuring completeness of bibliographic information
6. Fixing any truncated or incomplete references

Return one cleaned reference per line, maintaining the same order as input."""
    else:
        system_prompt = """You are an AI assistant that cleans and normalizes academic references for better database searching.

Given a list of academic references, clean each one by:
1. Ensuring each reference is on one line
2. Standardizing format (Author. Title. Journal/Conference. Year.)
3. Removing extra spaces, line breaks, and formatting issues
4. Fixing common formatting inconsistencies
5. Ensuring completeness of bibliographic information

Return one cleaned reference per line, maintaining the same order as input."""
    
    # Process references in batches to avoid token limits
    batch_size = 10
    cleaned_references = []
    
    progress_desc = "Preprocessing refs" if not verbose else f"Preprocessing {source} references"
    with tqdm(total=len(references), desc=progress_desc, disable=verbose) as pbar:
        for i in range(0, len(references), batch_size):
            batch = references[i:i+batch_size]
            batch_text = "\n".join([f"{j+1}. {ref}" for j, ref in enumerate(batch)])
            
            user_prompt = f"Clean and normalize these academic references:\n\n{batch_text}"
            
            try:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                
                cleaned_text = response.choices[0].message.content
                if cleaned_text:
                    # Parse cleaned references
                    for line in cleaned_text.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Remove numbering at the beginning
                            cleaned_ref = re.sub(r'^\d+\.\s*', '', line)
                            if cleaned_ref:
                                cleaned_references.append(cleaned_ref)
                
                pbar.update(len(batch))
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                if verbose:
                    print(f"Error preprocessing batch {i//batch_size + 1}: {e}")
                # If preprocessing fails, use original references
                cleaned_references.extend(batch)
                pbar.update(len(batch))
    
    if verbose:
        print(f"Preprocessed {len(references)} -> {len(cleaned_references)} cleaned references")
    
    return cleaned_references

def load_model_config(model_shortname):
    """Load model configuration from model_servers.yaml"""
    try:
        with open(MODEL_CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Model configuration file '{MODEL_CONFIG_FILE}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{MODEL_CONFIG_FILE}': {e}")

    model_config = None
    for server in config.get('servers', []):
        if server.get('shortname') == model_shortname:
            model_config = server
            break

    if not model_config:
        raise ValueError(f"Error: Model shortname '{model_shortname}' not found in '{MODEL_CONFIG_FILE}'.")
    
    return model_config

def get_openai_client(model_config):
    """Initialize OpenAI client with proper API key handling"""
    openai_api_key_config = model_config.get('openai_api_key')
    openai_api_key = None

    if openai_api_key_config == "${OPENAI_API_KEY}":
        openai_api_key = os.environ.get('OPENAI-API-KEY') or os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("Error: OpenAI API key is configured to use environment variable "
                             "'OPENAI-API-KEY' or 'OPENAI_API_KEY', but neither is set.")
    elif openai_api_key_config:
        openai_api_key = openai_api_key_config
    else:
        raise ValueError(f"Error: 'openai_api_key' not specified for model.")

    openai_api_base = model_config.get('openai_api_base')
    if not openai_api_base:
        raise ValueError(f"Error: 'openai_api_base' missing for model.")

    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        return client
    except Exception as e:
        raise ValueError(f"Error initializing OpenAI client: {e}")

def extract_reference_info(reference_text, client, openai_model, mode='fuzzy'):
    """Use OpenAI to extract information from academic references"""
    
    if mode == 'exact':
        system_prompt = """You are an AI assistant that extracts bibliographic information from academic references.
        Given a reference citation, extract the title, first author's last name, and publication year.
        Return the information in JSON format: {"title": "extracted title", "author": "last name", "year": "year"}
        If any information is unclear or missing, use null for that field."""
        
        user_prompt = f"Extract bibliographic information from this reference: {reference_text}"
        
        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            
            result_text = response.choices[0].message.content
            if result_text:
                # Try to parse JSON response
                import json
                try:
                    info = json.loads(result_text)
                    return info
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response: {result_text}")
                    return None
            return None
        except Exception as e:
            print(f"Error extracting reference info with OpenAI: {e}")
            return None
    
    else:  # fuzzy mode
        system_prompt = """You are an AI assistant that extracts key search terms from academic references. 
        Given a reference citation, extract the most important keywords that would be useful for searching academic databases.
        Focus on the main topic, methods, and key concepts. Return only 3-5 keywords separated by commas."""
        
        user_prompt = f"Extract search keywords from this reference: {reference_text}"
        
        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=100,
            )
            
            keywords_text = response.choices[0].message.content
            if keywords_text:
                keywords = [kw.strip() for kw in keywords_text.strip().split(',') if kw.strip()]
                return keywords[:5]  # Limit to 5 keywords max
            return []
        except Exception as e:
            print(f"Error processing reference with OpenAI: {e}")
            return []

def search_papers_fuzzy(keyword, api_key=None, limit=10, max_retries=3):
    """Search papers using Semantic Scholar API with keyword search and retry logic"""
    url = BASE_URL
    params = {
        'query': keyword,
        'limit': limit,
        'fields': FIELDS
    }
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key

    for attempt in range(max_retries):
        try:
            wait_for_rate_limit()  # Global rate limiting
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                print(f"Rate limited (429) on attempt {attempt + 1}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif response.status_code in [500, 502, 503, 504]:
                # Server errors - retry with backoff
                wait_time = (2 ** attempt) * 1  # 1, 2, 4 seconds
                print(f"Server error ({response.status_code}) on attempt {attempt + 1}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error searching for '{keyword}': Status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            wait_time = (2 ** attempt) * 1
            print(f"Timeout on attempt {attempt + 1}, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"Error searching for '{keyword}': {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    
    print(f"Failed to search for '{keyword}' after {max_retries} attempts")
    return None

def search_papers_exact(ref_info, api_key=None, limit=10, max_retries=3):
    """Search for specific paper using bibliographic information with retry logic"""
    if not ref_info:
        return None
    
    # Construct search query from extracted information
    query_parts = []
    
    if ref_info.get('title'):
        # Use title in quotes for exact matching
        query_parts.append(f'"{ref_info["title"]}"')
    
    if ref_info.get('author'):
        query_parts.append(f"author:{ref_info['author']}")
    
    if ref_info.get('year'):
        query_parts.append(f"year:{ref_info['year']}")
    
    if not query_parts:
        print("No valid search terms extracted from reference")
        return None
    
    query = ' '.join(query_parts)
    print(f"    Exact search query: {query}")
    
    url = BASE_URL
    params = {
        'query': query,
        'limit': limit,
        'fields': FIELDS
    }
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key

    for attempt in range(max_retries):
        try:
            wait_for_rate_limit()  # Global rate limiting
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                print(f"Rate limited (429) on exact search attempt {attempt + 1}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif response.status_code in [500, 502, 503, 504]:
                # Server errors - retry with backoff
                wait_time = (2 ** attempt) * 1  # 1, 2, 4 seconds
                print(f"Server error ({response.status_code}) on exact search attempt {attempt + 1}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Error in exact search: Status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            wait_time = (2 ** attempt) * 1
            print(f"Timeout on exact search attempt {attempt + 1}, waiting {wait_time}s...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"Error in exact search: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    
    print(f"Failed exact search after {max_retries} attempts")
    return None

def download_pdf(pdf_url, save_path, max_retries=2):
    """Download a PDF from URL with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(pdf_url, stream=True, timeout=60)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif response.status_code == 403:
                print(f"PDF access forbidden (403) - likely requires subscription or login")
                return False  # Don't retry 403 errors
            elif response.status_code == 404:
                print(f"PDF not found (404)")
                return False  # Don't retry 404 errors
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3  # 3, 6 seconds
                    print(f"PDF download rate limited (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"PDF download rate limited after {max_retries} attempts")
                    return False
            else:
                print(f"Failed to download PDF - status code: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
        except Exception as e:
            print(f"Exception while downloading PDF (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False
    
    return False

def search_with_fallbacks(reference_text, client, openai_model, ss_api_key=None, verbose=False, output_dir=None):
    """Try multiple search strategies to find a paper for a reference with 3 attempts each"""
    if verbose:
        print(f"    Attempting multiple search strategies...")
    
    # Strategy 0: Check for URLs and try web scraping first
    urls = detect_and_extract_urls(reference_text)
    if urls and output_dir:
        if verbose:
            print(f"    Found URLs in reference: {urls}")
        scraped_paper = process_url_reference(reference_text, output_dir, verbose, client, openai_model)
        if scraped_paper:
            if verbose:
                print(f"    ✅ Success with web scraping: {scraped_paper['title'][:60]}...")
            return [scraped_paper], "web_scraping"
    
    strategies = []
    
    # Extract bibliographic info once
    ref_info = extract_reference_info(reference_text, client, openai_model, mode='exact')
    
    # Strategy 1: Exact search using bibliographic info
    if ref_info:
        strategies.append(("exact", lambda: search_papers_exact(ref_info, ss_api_key, limit=5)))
    
    # Strategy 2: Title-only search (if we have a title)
    if ref_info and ref_info.get('title'):
        title = ref_info['title']
        strategies.append(("title", lambda: search_papers_fuzzy(title, ss_api_key, limit=5)))
        # More relaxed title search (remove quotes, punctuation)
        clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
        if clean_title != title:
            strategies.append(("title_clean", lambda: search_papers_fuzzy(clean_title, ss_api_key, limit=5)))
    
    # Strategy 3: First author + year search
    if ref_info and ref_info.get('author') and ref_info.get('year'):
        author_year_query = f"author:{ref_info['author']} year:{ref_info['year']}"
        strategies.append(("author_year", lambda: search_papers_fuzzy(author_year_query, ss_api_key, limit=5)))
    
    # Strategy 4: Author only (more relaxed)
    if ref_info and ref_info.get('author'):
        author_query = f"author:{ref_info['author']}"
        strategies.append(("author", lambda: search_papers_fuzzy(author_query, ss_api_key, limit=5)))
    
    # Strategy 5: Year only (if specific year)
    if ref_info and ref_info.get('year'):
        try:
            year = int(ref_info['year'])
            if 1900 <= year <= 2030:  # Reasonable year range
                year_query = f"year:{year}"
                strategies.append(("year", lambda: search_papers_fuzzy(year_query, ss_api_key, limit=10)))
        except (ValueError, TypeError):
            pass
    
    # Strategy 6: Keyword extraction and search
    keywords = extract_reference_info(reference_text, client, openai_model, mode='fuzzy')
    if keywords:
        for i, keyword in enumerate(keywords[:3]):  # Try top 3 keywords
            strategies.append((f"keyword_{i+1}", lambda k=keyword: search_papers_fuzzy(k, ss_api_key, limit=5)))
    
    # Strategy 7: Simple text searches with varying lengths
    words = reference_text.split()
    for word_count in [15, 10, 5]:  # Try different lengths
        if len(words) >= word_count:
            simple_query = ' '.join(words[:word_count])
            strategies.append((f"text_{word_count}w", lambda q=simple_query: search_papers_fuzzy(q, ss_api_key, limit=5)))
    
    # Strategy 8: Just first author's last name + key terms
    if ref_info and ref_info.get('author'):
        author_last = ref_info['author'].split()[-1] if ref_info['author'] else ""
        if author_last and keywords:
            combined_query = f"{author_last} {keywords[0]}"
            strategies.append(("author_keyword", lambda: search_papers_fuzzy(combined_query, ss_api_key, limit=5)))
    
    # Try each strategy with up to 3 attempts
    for strategy_name, search_func in strategies:
        for attempt in range(3):
            try:
                if verbose:
                    attempt_str = f" (attempt {attempt+1})" if attempt > 0 else ""
                    print(f"    Strategy: {strategy_name}{attempt_str}")
                
                papers_data = search_func()
                if papers_data and papers_data.get('data') and len(papers_data['data']) > 0:
                    if verbose:
                        print(f"    ✅ Success with {strategy_name}: {len(papers_data['data'])} papers found")
                    return papers_data['data'], strategy_name
                
                if attempt < 2:  # Don't sleep after last attempt
                    time.sleep(1)  # Rate limiting between attempts
                    
            except Exception as e:
                if verbose:
                    print(f"    Error in {strategy_name} attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2)  # Longer delay on error
    
    if verbose:
        print(f"    ❌ All strategies failed")
    return [], "failed"

class ThreadSafeCounter:
    """Thread-safe counter for tracking progress"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def process_single_reference(args):
    """Process a single reference - designed for parallel execution"""
    (reference, index, total_refs, client, openai_model, ss_api_key, 
     output_dir, mode, max_papers_per_ref, verbose, counters) = args
    
    result = {
        'index': index,
        'reference': reference[:100] + '...' if len(reference) > 100 else reference,
        'success': False,
        'strategy': None,
        'papers_found': 0,
        'abstracts_saved': 0,
        'pdfs_downloaded': 0,
        'error': None
    }
    
    try:
        if verbose:
            print(f"[{index+1}/{total_refs}] Processing reference: {result['reference']}")
        
        if mode == 'exact':
            # Use fallback strategies for exact mode
            papers, strategy = search_with_fallbacks(reference, client, openai_model, ss_api_key, verbose, output_dir)
            if papers:
                result['success'] = True
                result['strategy'] = strategy
                result['papers_found'] = len(papers)
                papers = papers[:1]  # Limit to 1 paper in exact mode
            else:
                papers = []
                
        else:  # fuzzy mode
            # Extract keywords for fuzzy matching
            keywords = extract_reference_info(reference, client, openai_model, mode='fuzzy')
            if not keywords:
                result['error'] = "No keywords extracted"
                return result
            
            # Search for papers using the first keyword
            main_keyword = keywords[0]
            papers_data = search_papers_fuzzy(main_keyword, ss_api_key, limit=max_papers_per_ref)
            if papers_data and papers_data.get('data'):
                papers = papers_data['data']
                result['success'] = True
                result['strategy'] = 'keyword'
                result['papers_found'] = len(papers)
            else:
                papers = []
        
        # Process papers
        for j, paper in enumerate(papers):
            if verbose:
                print(f"  Processing paper {j+1}/{len(papers)}: {paper.get('title', 'No title')[:60]}...")
            
            # Save abstract
            abstract_file = save_abstract(paper, output_dir)
            if abstract_file:
                result['abstracts_saved'] += 1
                counters['abstracts'].increment()
                if verbose:
                    print(f"    Saved abstract: {os.path.basename(abstract_file)}")
            
            # Try to download PDF if available
            if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
                pdf_url = paper['openAccessPdf']['url']
                paper_id = paper.get('paperId', 'unknown')
                pdf_filename = f"{paper_id}.pdf"
                pdf_path = os.path.join(output_dir, pdf_filename)
                
                if not os.path.exists(pdf_path):
                    if verbose:
                        print(f"    Downloading PDF...")
                    if download_pdf(pdf_url, pdf_path):
                        result['pdfs_downloaded'] += 1
                        counters['pdfs'].increment()
                        if verbose:
                            print(f"    Downloaded PDF: {pdf_filename}")
                    elif verbose:
                        print(f"    Failed to download PDF")
                else:
                    result['pdfs_downloaded'] += 1
                    counters['pdfs'].increment()
                    if verbose:
                        print(f"    PDF already exists: {pdf_filename}")
            elif verbose:
                print(f"    No open access PDF available")
            
            # Rate limiting per paper
            time.sleep(1.0)  # Increased from 0.5 to 1.0 seconds
        
        if result['success']:
            counters['successful_searches'].increment()
            
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"  Error processing reference: {e}")
    
    # Additional rate limiting at the end of each reference processing
    time.sleep(0.5)
    
    return result

def save_abstract(paper, output_dir):
    """Save paper abstract to file with improved content detection"""
    paper_id = paper.get('paperId', 'unknown')
    title = paper.get('title', 'No Title')
    abstract = paper.get('abstract', '') or ''
    year = paper.get('year', 'N/A')
    authors = paper.get('authors', [])
    url = paper.get('url', 'N/A')
    
    # Create safe filename
    safe_title = sanitize_filename(title, max_length=50)
    abstract_filename = os.path.join(output_dir, f"{paper_id}_{safe_title}.txt")
    
    # Format author names - handle both cleaned and original format
    if paper.get('authors_cleaned'):
        author_str = paper['authors_cleaned']
    elif authors and isinstance(authors, list) and len(authors) > 0:
        author_names = [author.get('name', 'Unknown') if isinstance(author, dict) else str(author) for author in authors[:5]]
        author_str = ', '.join(author_names)
        if len(authors) > 5:
            author_str += f" et al. (and {len(authors) - 5} more)"
    else:
        author_str = "Unknown"
    
    # Detect content issues
    content_status = []
    if not abstract.strip():
        content_status.append("NO_ABSTRACT")
    elif len(abstract.strip()) < 50:
        content_status.append("SHORT_ABSTRACT")
    
    if paper_id == 'unknown':
        content_status.append("NO_PAPER_ID")
    
    # Detect if this might be a URL-based reference
    if url != 'N/A' and any(domain in url.lower() for domain in ['arxiv.org', 'github.com', 'doi.org', 'researchgate.net']):
        content_status.append("EXTERNAL_URL")
    
    # Check if this is LLM-processed web content
    if paper.get('llm_processed'):
        content_status.append("LLM_PROCESSED")
    if paper.get('content_source') == 'web_scraped':
        content_status.append("WEB_SCRAPED")
    if paper.get('document_cleaned'):
        content_status.append("DOCUMENT_CLEANED")
    
    try:
        with open(abstract_filename, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\n")
            f.write(f"Authors: {author_str}\n")
            f.write(f"Year: {year}\n")
            f.write(f"Paper ID: {paper_id}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Venue: {paper.get('venue', 'N/A')}\n")
            
            if content_status:
                f.write(f"Content Status: {', '.join(content_status)}\n")
            
            f.write(f"\nAbstract:\n")
            if abstract.strip():
                f.write(f"{abstract}\n")
            else:
                f.write("*** NO ABSTRACT AVAILABLE ***\n")
                f.write("This paper may require manual lookup or web scraping.\n")
                
                # Suggest alternative sources
                if url != 'N/A':
                    f.write(f"Try accessing directly: {url}\n")
                    
        return abstract_filename
    except Exception as e:
        print(f"Error saving abstract: {e}")
        return None

def detect_and_extract_urls(reference_text):
    """Detect URLs in reference text and extract relevant information"""
    import re
    
    # Common URL patterns in academic references
    url_patterns = [
        r'https?://[^\s\]]+',  # Standard URLs
        r'doi:\s*10\.\d+/[^\s\]]+',  # DOI patterns
        r'arXiv:\d+\.\d+',  # arXiv IDs
        r'www\.[^\s\]]+',  # www URLs without protocol
    ]
    
    urls = []
    for pattern in url_patterns:
        matches = re.findall(pattern, reference_text, re.IGNORECASE)
        urls.extend(matches)
    
    # Clean and normalize URLs
    cleaned_urls = []
    for url in urls:
        url = url.strip('.,;])')  # Remove trailing punctuation
        if url.startswith('doi:'):
            url = f"https://doi.org/{url[4:].strip()}"
        elif url.startswith('www.'):
            url = f"https://{url}"
        cleaned_urls.append(url)
    
    return cleaned_urls

def try_web_scraping(url, verbose=False):
    """Attempt to scrape comprehensive information from a URL"""
    if not WEB_SCRAPING_SUPPORT:
        if verbose:
            print(f"    Web scraping not available (BeautifulSoup not installed)")
        return None
        
    if verbose:
        print(f"    Attempting to scrape: {url}")
    
    try:
        import requests
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to extract title and abstract
            title = None
            abstract = None
            full_content = None
            
            # Extended title selectors
            title_selectors = [
                'title', 'h1', '.title', '#title', '.article-title', 
                '.paper-title', '.entry-title', '.post-title',
                'h1.title', 'h2.title', '.main-title'
            ]
            for selector in title_selectors:
                element = soup.select_one(selector)
                if element and element.get_text().strip():
                    title = element.get_text().strip()
                    # Clean up title (remove site name suffixes)
                    if ' | ' in title:
                        title = title.split(' | ')[0]
                    if ' - ' in title and len(title.split(' - ')) > 1:
                        # Keep the longest part, likely the actual title
                        parts = title.split(' - ')
                        title = max(parts, key=len)
                    break
            
            # Extended abstract selectors
            abstract_selectors = [
                '.abstract', '#abstract', '.summary', '.description',
                '.paper-abstract', '.article-abstract', '.entry-summary',
                '.excerpt', '.intro', '.introduction', '.overview',
                '[class*="abstract"]', '[id*="abstract"]',
                '.lead', '.subtitle'
            ]
            for selector in abstract_selectors:
                element = soup.select_one(selector)
                if element and element.get_text().strip():
                    abstract = element.get_text().strip()
                    break
            
            # If no abstract found, try to capture comprehensive content
            if not abstract or len(abstract.strip()) < 100:
                if verbose:
                    print(f"    Abstract too short or missing, capturing full content...")
                
                # Try to find main content areas
                content_selectors = [
                    'main', '.main', '#main', '.content', '#content',
                    '.article', '.post', '.paper', '.entry',
                    '.main-content', '.article-content', '.post-content',
                    'article', '[role="main"]'
                ]
                
                content_element = None
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        content_element = element
                        break
                
                # If no main content area found, use body
                if not content_element:
                    content_element = soup.find('body')
                
                if content_element:
                    # Extract all text content
                    full_content = content_element.get_text(separator=' ', strip=True)
                    
                    # Clean up the content
                    import re
                    # Remove excessive whitespace
                    full_content = re.sub(r'\s+', ' ', full_content)
                    # Remove very short lines (likely navigation/footer text)
                    lines = full_content.split('\n')
                    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 20]
                    full_content = ' '.join(cleaned_lines)
                    
                    if verbose:
                        print(f"    Captured {len(full_content)} characters of content")
            
            # Return structured data
            result = {
                'title': title or 'Web-scraped content',
                'abstract': abstract or None,
                'full_content': full_content,
                'url': url,
                'source': 'web_scraped',
                'needs_llm_processing': not abstract or len(abstract.strip()) < 100
            }
            
            return result
                
    except Exception as e:
        if verbose:
            print(f"    Web scraping failed: {e}")
    
    return None

def process_scraped_content_with_llm(scraped_data, client, openai_model, verbose=False):
    """Use LLM to process scraped content and extract useful information"""
    if not scraped_data.get('needs_llm_processing') or not scraped_data.get('full_content'):
        return scraped_data
    
    if verbose:
        print(f"    Processing scraped content with LLM...")
    
    try:
        # Truncate content if too long for LLM context
        content = scraped_data['full_content']
        max_content_length = 12000  # Leave room for prompts
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"
        
        system_prompt = """You are an AI assistant that processes web-scraped academic content to extract key information.

Given text scraped from a webpage (likely an academic paper, article, or research document), extract and summarize:
1. The main topic/subject matter
2. Key findings or main points
3. Research methods or approach (if applicable)
4. Authors or contributors (if mentioned)
5. Publication details (if available)
6. A concise abstract/summary of the content

Provide your response in this format:
**Topic:** [Main subject]
**Key Points:** [2-3 main findings or points]
**Methods:** [Research approach if applicable, or "Not specified"]
**Authors:** [If mentioned, or "Not specified"]
**Summary:** [2-3 sentence abstract of the content]

Focus on academic and research-relevant information. Ignore navigation, advertisements, and boilerplate text."""
        
        user_prompt = f"""Extract key information from this web-scraped content:

Title: {scraped_data.get('title', 'No title')}
URL: {scraped_data.get('url', 'No URL')}

Content:
{content}"""
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        
        llm_processed_content = response.choices[0].message.content
        
        if llm_processed_content:
            # Update the scraped data with LLM-processed content
            scraped_data['abstract'] = llm_processed_content
            scraped_data['llm_processed'] = True
            
            if verbose:
                print(f"    LLM processing completed, generated {len(llm_processed_content)} characters")
        
        return scraped_data
        
    except Exception as e:
        if verbose:
            print(f"    LLM processing failed: {e}")
        # Fallback: use first part of full content as abstract
        if scraped_data.get('full_content'):
            fallback_abstract = scraped_data['full_content'][:1000] + "..."
            scraped_data['abstract'] = f"Content Preview: {fallback_abstract}"
        return scraped_data

def cleanup_document_with_llm(paper_data, client, openai_model, verbose=False):
    """Second-pass LLM cleanup to format and improve document quality"""
    if not client or not openai_model:
        return paper_data
    
    if verbose:
        print(f"    Performing document cleanup and formatting...")
    
    try:
        title = paper_data.get('title', 'No Title')
        abstract = paper_data.get('abstract', 'No abstract available')
        url = paper_data.get('url', 'No URL')
        authors = paper_data.get('authors', [])
        year = paper_data.get('year', 'N/A')
        venue = paper_data.get('venue', 'N/A')
        
        # Format author information
        if authors and isinstance(authors, list) and len(authors) > 0:
            author_str = ', '.join([author.get('name', 'Unknown') if isinstance(author, dict) else str(author) for author in authors[:5]])
            if len(authors) > 5:
                author_str += f" et al. (and {len(authors) - 5} more)"
        else:
            author_str = 'Not specified'
        
        system_prompt = """You are an expert document editor that cleans up and formats academic content for maximum readability and professionalism.

Your task is to:
1. **Title Optimization**: Choose the best title between the current title and any title mentioned in the abstract/content. Prefer more specific, informative titles over generic ones.
2. **Content Consistency**: Ensure title and abstract are consistent and complement each other
3. **Format Cleaning**: Remove redundant headings, fix formatting issues, improve structure
4. **Quality Enhancement**: Make the content more readable while preserving academic accuracy
5. **Information Integration**: If authors/venue info is scattered in the content, consolidate it properly

Return your response in this exact format:
**TITLE:** [Optimized title - choose the most informative and specific one]
**AUTHORS:** [Clean, formatted author list or "Not specified"]
**VENUE:** [Publication venue if identifiable, or "Not specified"]
**YEAR:** [Publication year if identifiable, or "Not specified"]
**ABSTRACT:**
[Clean, well-formatted abstract with proper structure. Remove redundant headings, fix formatting issues, ensure coherent flow. If the original abstract is just extracted keywords or bullet points, rewrite it as a proper abstract paragraph.]

Guidelines:
- Prioritize informativeness over brevity for titles
- Ensure abstract flows naturally as coherent paragraphs
- Remove web-specific formatting artifacts
- Maintain academic tone and accuracy
- If multiple title candidates exist, choose the most descriptive one"""
        
        user_prompt = f"""Clean up and format this academic document:

CURRENT TITLE: {title}
CURRENT AUTHORS: {author_str}
CURRENT VENUE: {venue}
CURRENT YEAR: {year}
SOURCE URL: {url}

CURRENT ABSTRACT/CONTENT:
{abstract}

Please optimize the title, clean up the formatting, and ensure everything is consistent and professional."""
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=1500,
        )
        
        cleaned_content = response.choices[0].message.content
        
        if cleaned_content:
            # Parse the cleaned content back into structured data
            cleaned_data = parse_cleaned_document(cleaned_content, paper_data, verbose)
            
            if verbose:
                print(f"    Document cleanup completed - improved title and format")
            
            return cleaned_data
        
        return paper_data
        
    except Exception as e:
        if verbose:
            print(f"    Document cleanup failed: {e}")
        return paper_data

def parse_cleaned_document(cleaned_content, original_paper_data, verbose=False):
    """Parse the LLM-cleaned document back into structured paper data"""
    import re
    
    try:
        # Initialize with original data
        cleaned_paper = original_paper_data.copy()
        
        # Extract sections using regex
        title_match = re.search(r'\*\*TITLE:\*\*\s*(.+?)(?=\n|\*\*|$)', cleaned_content, re.IGNORECASE | re.DOTALL)
        authors_match = re.search(r'\*\*AUTHORS:\*\*\s*(.+?)(?=\n|\*\*|$)', cleaned_content, re.IGNORECASE | re.DOTALL)
        venue_match = re.search(r'\*\*VENUE:\*\*\s*(.+?)(?=\n|\*\*|$)', cleaned_content, re.IGNORECASE | re.DOTALL)
        year_match = re.search(r'\*\*YEAR:\*\*\s*(.+?)(?=\n|\*\*|$)', cleaned_content, re.IGNORECASE | re.DOTALL)
        abstract_match = re.search(r'\*\*ABSTRACT:\*\*\s*(.+?)$', cleaned_content, re.IGNORECASE | re.DOTALL)
        
        # Update fields if found
        if title_match:
            new_title = title_match.group(1).strip()
            if new_title and new_title.lower() not in ['not specified', 'no title', 'unknown']:
                cleaned_paper['title'] = new_title
        
        if authors_match:
            authors_text = authors_match.group(1).strip()
            if authors_text and authors_text.lower() != 'not specified':
                # Keep as string for display, but mark as cleaned
                cleaned_paper['authors_cleaned'] = authors_text
        
        if venue_match:
            venue_text = venue_match.group(1).strip()
            if venue_text and venue_text.lower() != 'not specified':
                cleaned_paper['venue'] = venue_text
        
        if year_match:
            year_text = year_match.group(1).strip()
            if year_text and year_text.lower() != 'not specified':
                cleaned_paper['year'] = year_text
        
        if abstract_match:
            new_abstract = abstract_match.group(1).strip()
            if new_abstract:
                cleaned_paper['abstract'] = new_abstract
        
        # Mark as cleaned
        cleaned_paper['document_cleaned'] = True
        
        return cleaned_paper
        
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not parse cleaned document structure: {e}")
        # Return original data with just the cleaned content as abstract
        original_paper_data['abstract'] = cleaned_content
        original_paper_data['document_cleaned'] = True
        return original_paper_data

def process_url_reference(reference_text, output_dir, verbose=False, client=None, openai_model=None):
    """Process a reference that contains URLs with enhanced content extraction"""
    urls = detect_and_extract_urls(reference_text)
    
    if not urls:
        return None
    
    for url in urls:
        scraped_data = try_web_scraping(url, verbose)
        if scraped_data:
            # Use LLM to process content if needed and available
            if client and openai_model and scraped_data.get('needs_llm_processing'):
                scraped_data = process_scraped_content_with_llm(scraped_data, client, openai_model, verbose)
            
            # Create a paper-like object for saving
            paper = {
                'paperId': f"web_{hash(url) % 1000000}",
                'title': scraped_data['title'],
                'abstract': scraped_data.get('abstract') or 'No abstract available',
                'url': scraped_data['url'],
                'authors': [],
                'year': 'N/A',
                'venue': 'Web Source',
                'content_source': 'web_scraped',
                'llm_processed': scraped_data.get('llm_processed', False)
            }
            
            # Second pass: cleanup and format the document with LLM
            if client and openai_model:
                paper = cleanup_document_with_llm(paper, client, openai_model, verbose)
            
            # Save the scraped content
            saved_file = save_abstract(paper, output_dir)
            if saved_file and verbose:
                processing_note = " (LLM processed)" if scraped_data.get('llm_processed') else ""
                print(f"    Saved web-scraped content{processing_note}: {saved_file}")
            
            return paper
    
    return None

def process_pdf_references(pdf_file, model_shortname, output_dir, ss_api_key=None, verbose=False, parallel_threads=1):
    """Extract references from PDF and download them"""
    
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    # Load model configuration
    model_config = load_model_config(model_shortname)
    client = get_openai_client(model_config)
    openai_model = model_config.get('openai_model')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Extracting text from PDF: {pdf_file}")
    
    # Extract text from PDF
    try:
        pdf_text = extract_text_from_pdf(pdf_file, verbose)
        if verbose:
            print(f"Extracted {len(pdf_text)} characters from PDF")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")
    
    # Extract individual references
    if verbose:
        print("Extracting references using LLM...")
    references = extract_references_from_text(pdf_text, client, openai_model, verbose)
    
    if not references:
        print("No references found in the PDF")
        return
    
    print(f"Found {len(references)} references in PDF")
    
    # Preprocess references with LLM
    references = preprocess_references(references, client, openai_model, verbose, source="txt")
    
    # Save extracted references to a file for review
    refs_file = os.path.join(output_dir, "extracted_references.txt")
    with open(refs_file, 'w', encoding='utf-8') as f:
        for i, ref in enumerate(references, 1):
            f.write(f"{i}. {ref}\n")
    if verbose:
        print(f"Saved extracted references to: {refs_file}")
    
    # Initialize thread-safe counters
    counters = {
        'successful_searches': ThreadSafeCounter(),
        'abstracts': ThreadSafeCounter(),
        'pdfs': ThreadSafeCounter()
    }
    
    if parallel_threads > 1:
        # Parallel processing
        if verbose:
            print(f"Processing {len(references)} references using {parallel_threads} parallel threads...")
        
        # Prepare arguments for parallel processing
        args_list = []
        for i, reference in enumerate(references):
            args_list.append((
                reference, i, len(references), client, openai_model, ss_api_key,
                output_dir, 'exact', 1, verbose, counters  # PDF mode always uses exact mode
            ))
        
        # Progress bars for parallel processing
        abstract_pbar = tqdm(total=len(references), desc="Abstracts", position=0, leave=True)
        pdf_pbar = tqdm(total=len(references), desc="PDFs", position=1, leave=True)
        
        # Process references in parallel
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_reference, args): args[1] for args in args_list}
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    
                    # Update progress bars
                    abstract_pbar.set_description(f"Abstracts ({counters['abstracts'].value}/{len(references)})")
                    pdf_pbar.set_description(f"PDFs ({counters['pdfs'].value}/{len(references)})")
                    abstract_pbar.update(1)
                    pdf_pbar.update(1)
                    
                    if verbose and result.get('error'):
                        print(f"Error in reference {result['index']+1}: {result['error']}")
                        
                except Exception as e:
                    if verbose:
                        print(f"Unexpected error: {e}")
        
        abstract_pbar.close()
        pdf_pbar.close()
        
        # Get final counts
        successful_searches = counters['successful_searches'].value
        total_abstracts = counters['abstracts'].value
        total_pdfs = counters['pdfs'].value
        
    else:
        # Sequential processing (original logic)
        total_abstracts = 0
        total_pdfs = 0
        successful_searches = 0
        
        # Process each reference with progress bars
        abstract_pbar = tqdm(total=len(references), desc="Abstracts", position=0, leave=True)
        pdf_pbar = tqdm(total=len(references), desc="PDFs", position=1, leave=True)
        
        for i, reference in enumerate(references, 1):
            if verbose:
                print(f"\n[{i}/{len(references)}] Processing reference:")
                print(f"  {reference[:100]}..." if len(reference) > 100 else f"  {reference}")
            
            # Try multiple search strategies
            papers, strategy = search_with_fallbacks(reference, client, openai_model, ss_api_key, verbose, output_dir)
            
            if not papers:
                if verbose:
                    print(f"  ❌ Failed to find any papers for this reference")
                abstract_pbar.update(1)
                pdf_pbar.update(1)
                continue
            
            successful_searches += 1
            if verbose:
                print(f"  ✅ Found {len(papers)} papers using strategy: {strategy}")
            
            # Process the best match (first paper)
            paper = papers[0]
            if verbose:
                print(f"    Processing best match: {paper.get('title', 'No title')[:60]}...")
            
            # Save abstract
            abstract_file = save_abstract(paper, output_dir)
            if abstract_file:
                total_abstracts += 1
                if verbose:
                    print(f"      Saved abstract: {os.path.basename(abstract_file)}")
            
            abstract_pbar.set_description(f"Abstracts ({total_abstracts}/{len(references)})")
            abstract_pbar.update(1)
            
            # Try to download PDF if available
            pdf_downloaded = False
            if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
                pdf_url = paper['openAccessPdf']['url']
                paper_id = paper.get('paperId', 'unknown')
                pdf_filename = f"{paper_id}.pdf"
                pdf_path = os.path.join(output_dir, pdf_filename)
                
                if not os.path.exists(pdf_path):
                    if verbose:
                        print(f"      Downloading PDF...")
                    if download_pdf(pdf_url, pdf_path):
                        total_pdfs += 1
                        pdf_downloaded = True
                        if verbose:
                            print(f"      Downloaded PDF: {pdf_filename}")
                    elif verbose:
                        print(f"      Failed to download PDF")
                else:
                    total_pdfs += 1
                    pdf_downloaded = True
                    if verbose:
                        print(f"      PDF already exists: {pdf_filename}")
            elif verbose:
                print(f"      No open access PDF available")
            
            pdf_pbar.set_description(f"PDFs ({total_pdfs}/{len(references)})")
            pdf_pbar.update(1)
            
            # Small delay to be respectful to APIs
            if not verbose:
                time.sleep(1)  # Shorter delay in non-verbose mode
            else:
                time.sleep(2)
        
        abstract_pbar.close()
        pdf_pbar.close()
    
    print(f"\n=== PDF Processing Complete ===")
    print(f"References extracted from PDF: {len(references)}")
    print(f"Successful searches: {successful_searches}/{len(references)} ({successful_searches/len(references)*100:.1f}%)")
    print(f"Total abstracts saved: {total_abstracts}/{len(references)} ({total_abstracts/len(references)*100:.1f}%)")
    print(f"Total PDFs downloaded: {total_pdfs}/{len(references)} ({total_pdfs/len(references)*100:.1f}%)")
    print(f"Output directory: {output_dir}")
    if verbose:
        print(f"Extracted references saved to: {refs_file}")

def extract_only_mode(input_file, model_shortname, output_dir, verbose=False):
    """Extract and clean references without downloading papers"""
    
    # Load model configuration
    model_config = load_model_config(model_shortname)
    client = get_openai_client(model_config)
    openai_model = model_config.get('openai_model')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if input_file.lower().endswith('.pdf'):
        # PDF mode - extract references from PDF
        if not PDF_SUPPORT:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        if verbose:
            print(f"Extracting references from PDF: {input_file}")
        
        # Extract text from PDF
        try:
            pdf_text = extract_text_from_pdf(input_file, verbose)
            if verbose:
                print(f"Extracted {len(pdf_text)} characters from PDF")
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
        
        # Extract individual references
        if verbose:
            print("Extracting references using LLM...")
        references = extract_references_from_text(pdf_text, client, openai_model, verbose)
        
        if not references:
            print("No references found in the PDF")
            return
        
        print(f"Found {len(references)} references in PDF")
        source_type = "pdf"
        
    else:
        # Text file mode - read existing references
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                references = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        except FileNotFoundError:
            raise FileNotFoundError(f"References file '{input_file}' not found.")
        
        if verbose:
            print(f"Read {len(references)} references from file")
        source_type = "txt"
    
    # Preprocess references with LLM for cleaning
    if verbose:
        print("Cleaning and processing references with LLM...")
    
    cleaned_references = preprocess_references(references, client, openai_model, verbose, source=source_type)
    
    # Save original references
    original_refs_file = os.path.join(output_dir, "original_references.txt")
    with open(original_refs_file, 'w', encoding='utf-8') as f:
        for i, ref in enumerate(references, 1):
            f.write(f"{i}. {ref}\n")
    
    # Save cleaned references
    cleaned_refs_file = os.path.join(output_dir, "cleaned_references.txt")
    with open(cleaned_refs_file, 'w', encoding='utf-8') as f:
        f.write("# Cleaned and formatted references processed by PullR\n")
        f.write(f"# Original file: {input_file}\n")
        f.write(f"# Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Model used: {openai_model}\n")
        f.write(f"# Total references: {len(cleaned_references)}\n\n")
        for i, ref in enumerate(cleaned_references, 1):
            f.write(f"{i}. {ref}\n")
    
    # Save summary report
    summary_file = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PullR Extract-Only Mode Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Input type: {source_type.upper()}\n")
        f.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model used: {openai_model}\n")
        f.write(f"Original references found: {len(references)}\n")
        f.write(f"Cleaned references output: {len(cleaned_references)}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Output Files:\n")
        f.write(f"- Original references: {os.path.basename(original_refs_file)}\n")
        f.write(f"- Cleaned references: {os.path.basename(cleaned_refs_file)}\n")
        f.write(f"- This summary: {os.path.basename(summary_file)}\n\n")
        
        f.write("Processing Notes:\n")
        f.write("- References have been cleaned and standardized using AI\n")
        f.write("- No papers were downloaded (extract-only mode)\n")
        f.write("- Use cleaned references for further processing or manual review\n")
    
    print(f"\n=== Extract-Only Mode Complete ===")
    print(f"Input: {input_file}")
    print(f"Original references: {len(references)}")
    print(f"Cleaned references: {len(cleaned_references)}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - {os.path.basename(original_refs_file)} (original)")
    print(f"  - {os.path.basename(cleaned_refs_file)} (cleaned)")
    print(f"  - {os.path.basename(summary_file)} (summary)")
    
    if verbose:
        print(f"\nUse the cleaned references for:")
        print(f"  - Manual review and editing")
        print(f"  - Input to other reference managers")
        print(f"  - Further processing with PullR (without --extract-only)")
        print(f"  - Citation formatting in documents")

def find_pdf_files(directory_path, verbose=False, sample_size=None):
    """Find all PDF files in a directory, optionally sampling a subset"""
    pdf_files = []
    
    if not os.path.isdir(directory_path):
        return pdf_files
    
    # Find all PDF files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)
    
    # Sort for consistent ordering
    pdf_files = sorted(pdf_files)
    total_pdfs = len(pdf_files)
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < total_pdfs:
        import random
        # Set seed for reproducible sampling (based on directory path)
        random.seed(hash(directory_path) % 2**32)
        pdf_files = random.sample(pdf_files, sample_size)
        # Re-sort sampled files for consistent processing order
        pdf_files = sorted(pdf_files)
        
        if verbose:
            print(f"Found {total_pdfs} PDF files in {directory_path}")
            print(f"Randomly sampled {sample_size} PDFs for processing:")
            for pdf in pdf_files[:5]:  # Show first 5 sampled
                print(f"  - {os.path.basename(pdf)}")
            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more sampled files")
    else:
        if verbose:
            print(f"Found {len(pdf_files)} PDF files in {directory_path}")
            for pdf in pdf_files[:5]:  # Show first 5
                print(f"  - {os.path.basename(pdf)}")
            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more")
    
    return pdf_files

def process_pdf_directory(directory_path, model_shortname, output_dir, ss_api_key=None, verbose=False, parallel_threads=1, extract_only=False, sample_size=None):
    """Process all PDFs in a directory, optionally sampling a subset"""
    
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    # Find all PDF files (with optional sampling)
    pdf_files = find_pdf_files(directory_path, verbose, sample_size)
    total_pdfs_found = len(find_pdf_files(directory_path, verbose=False))  # Get total count for reporting
    
    if not pdf_files:
        print(f"No PDF files found in directory: {directory_path}")
        return
    
    if sample_size and sample_size < total_pdfs_found:
        print(f"Processing {len(pdf_files)} randomly sampled PDF files from {total_pdfs_found} total PDFs in directory: {directory_path}")
    else:
        print(f"Processing {len(pdf_files)} PDF files from directory: {directory_path}")
    
    # Load model configuration
    model_config = load_model_config(model_shortname)
    client = get_openai_client(model_config)
    openai_model = model_config.get('openai_model')
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics tracking
    total_pdfs = len(pdf_files)
    successful_pdfs = 0
    total_references = 0
    total_papers_found = 0
    total_abstracts = 0
    total_pdfs_downloaded = 0
    processing_errors = []
    
    # Progress tracking
    pdf_progress = tqdm(total=total_pdfs, desc="Processing PDFs", position=0, leave=True)
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_name = os.path.basename(pdf_file)
        pdf_progress.set_description(f"Processing PDF {i}/{total_pdfs}: {pdf_name[:30]}...")
        
        try:
            if verbose:
                print(f"\n[{i}/{total_pdfs}] Processing: {pdf_name}")
            
            # Create subdirectory for this PDF
            pdf_safe_name = sanitize_filename(os.path.splitext(pdf_name)[0], max_length=50)
            pdf_output_dir = os.path.join(output_dir, f"pdf_{i:03d}_{pdf_safe_name}")
            os.makedirs(pdf_output_dir, exist_ok=True)
            
            if extract_only:
                # Extract-only mode for this PDF
                if verbose:
                    print(f"  Extract-only mode for: {pdf_name}")
                
                # Use the existing extract_only_mode function for this PDF
                extract_only_mode(pdf_file, model_shortname, pdf_output_dir, verbose)
                successful_pdfs += 1
                
                # Count references from the cleaned file
                cleaned_refs_file = os.path.join(pdf_output_dir, "cleaned_references.txt")
                if os.path.exists(cleaned_refs_file):
                    with open(cleaned_refs_file, 'r', encoding='utf-8') as f:
                        ref_count = len([line for line in f if line.strip() and not line.strip().startswith('#') and re.match(r'^\d+\.', line.strip())])
                        total_references += ref_count
                
            else:
                # Full processing mode for this PDF
                if verbose:
                    print(f"  Full processing mode for: {pdf_name}")
                
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(pdf_file, verbose)
                
                # Extract references
                references = extract_references_from_text(pdf_text, client, openai_model, verbose)
                
                if not references:
                    if verbose:
                        print(f"  No references found in {pdf_name}")
                    processing_errors.append(f"{pdf_name}: No references found")
                    pdf_progress.update(1)
                    continue
                
                pdf_ref_count = len(references)
                total_references += pdf_ref_count
                
                if verbose:
                    print(f"  Found {pdf_ref_count} references")
                
                # Save extracted references for this PDF
                refs_file = os.path.join(pdf_output_dir, f"extracted_references_{pdf_safe_name}.txt")
                with open(refs_file, 'w', encoding='utf-8') as f:
                    f.write(f"# References extracted from: {pdf_name}\n")
                    f.write(f"# Extraction date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Total references: {pdf_ref_count}\n\n")
                    for j, ref in enumerate(references, 1):
                        f.write(f"{j}. {ref}\n")
                
                # Preprocess references
                cleaned_references = preprocess_references(references, client, openai_model, verbose, source="pdf")
                
                # Process references with search and download
                pdf_abstracts = 0
                pdf_downloads = 0
                pdf_papers_found = 0
                
                # Initialize counters for this PDF
                counters = {
                    'successful_searches': ThreadSafeCounter(),
                    'abstracts': ThreadSafeCounter(),
                    'pdfs': ThreadSafeCounter()
                }
                
                # Process each reference (simplified version of the main processing)
                for j, reference in enumerate(cleaned_references):
                    try:
                        papers, strategy = search_with_fallbacks(reference, client, openai_model, ss_api_key, verbose, pdf_output_dir)
                        
                        if papers:
                            pdf_papers_found += len(papers)
                            # Take the best match
                            paper = papers[0]
                            
                            # Save abstract
                            abstract_file = save_abstract(paper, pdf_output_dir)
                            if abstract_file:
                                pdf_abstracts += 1
                            
                            # Try to download PDF
                            if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
                                pdf_url = paper['openAccessPdf']['url']
                                paper_id = paper.get('paperId', 'unknown')
                                pdf_filename = f"{paper_id}.pdf"
                                pdf_path = os.path.join(pdf_output_dir, pdf_filename)
                                
                                if not os.path.exists(pdf_path):
                                    if download_pdf(pdf_url, pdf_path):
                                        pdf_downloads += 1
                                else:
                                    pdf_downloads += 1
                            
                            # Rate limiting
                            time.sleep(1)
                        
                    except Exception as e:
                        if verbose:
                            print(f"    Error processing reference {j+1}: {e}")
                        continue
                
                total_papers_found += pdf_papers_found
                total_abstracts += pdf_abstracts
                total_pdfs_downloaded += pdf_downloads
                
                if verbose:
                    print(f"  Results: {pdf_papers_found} papers found, {pdf_abstracts} abstracts, {pdf_downloads} PDFs")
                
                successful_pdfs += 1
            
        except Exception as e:
            error_msg = f"{pdf_name}: {str(e)}"
            processing_errors.append(error_msg)
            if verbose:
                print(f"  Error processing {pdf_name}: {e}")
        
        pdf_progress.update(1)
    
    pdf_progress.close()
    
    # Create summary report
    summary_file = os.path.join(output_dir, "batch_processing_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("PullR Batch PDF Processing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input directory: {directory_path}\n")
        f.write(f"Processing mode: {'Extract-only' if extract_only else 'Full processing'}\n")
        f.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model used: {openai_model}\n")
        f.write(f"Parallel threads: {parallel_threads}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Processing Statistics:\n")
        f.write(f"Total PDFs found in directory: {total_pdfs_found}\n")
        if sample_size and sample_size < total_pdfs_found:
            f.write(f"PDFs sampled for processing: {total_pdfs}\n")
            f.write(f"Sampling method: Random sampling\n")
        f.write(f"Successfully processed: {successful_pdfs}\n")
        f.write(f"Processing errors: {len(processing_errors)}\n")
        f.write(f"Success rate: {successful_pdfs/total_pdfs*100:.1f}%\n\n")
        
        f.write("Reference Statistics:\n")
        f.write(f"Total references extracted: {total_references}\n")
        f.write(f"Average references per PDF: {total_references/max(successful_pdfs,1):.1f}\n\n")
        
        if not extract_only:
            f.write("Paper Discovery Statistics:\n")
            f.write(f"Total papers found: {total_papers_found}\n")
            f.write(f"Total abstracts saved: {total_abstracts}\n")
            f.write(f"Total PDFs downloaded: {total_pdfs_downloaded}\n")
            f.write(f"Average papers per PDF: {total_papers_found/max(successful_pdfs,1):.1f}\n\n")
        
        if processing_errors:
            f.write("Processing Errors:\n")
            for error in processing_errors:
                f.write(f"- {error}\n")
        
        f.write(f"\nOutput Structure:\n")
        f.write(f"Each PDF created a subdirectory: pdf_XXX_[filename]/\n")
        if extract_only:
            f.write(f"Each subdirectory contains: original_references.txt, cleaned_references.txt, processing_summary.txt\n")
        else:
            f.write(f"Each subdirectory contains: extracted references, abstracts, and downloaded PDFs\n")
    
    # Print final summary
    print(f"\n=== Batch PDF Processing Complete ===")
    print(f"Directory: {directory_path}")
    print(f"Mode: {'Extract-only' if extract_only else 'Full processing'}")
    if sample_size and sample_size < total_pdfs_found:
        print(f"Sampling: {total_pdfs} randomly selected from {total_pdfs_found} total PDFs")
    print(f"PDFs processed: {successful_pdfs}/{total_pdfs} ({successful_pdfs/total_pdfs*100:.1f}%)")
    print(f"Total references: {total_references}")
    if not extract_only:
        print(f"Papers found: {total_papers_found}")
        print(f"Abstracts saved: {total_abstracts}")
        print(f"PDFs downloaded: {total_pdfs_downloaded}")
    print(f"Output directory: {output_dir}")
    print(f"Summary report: {os.path.basename(summary_file)}")
    
    if processing_errors:
        print(f"\nProcessing errors ({len(processing_errors)}):")
        for error in processing_errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(processing_errors) > 5:
            print(f"  ... and {len(processing_errors) - 5} more (see summary file)")
    
    return successful_pdfs, total_references

def process_references(references_file, model_shortname, output_dir, mode='exact', ss_api_key=None, max_papers_per_ref=5, verbose=False, parallel_threads=1):
    """Process references file and download papers"""
    
    # Load model configuration
    model_config = load_model_config(model_shortname)
    client = get_openai_client(model_config)
    openai_model = model_config.get('openai_model')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read references
    try:
        with open(references_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    except FileNotFoundError:
        raise FileNotFoundError(f"References file '{references_file}' not found.")
    
    if verbose:
        print(f"Processing {len(references)} references in {mode} mode...")
    
    # Preprocess references if in exact mode for better matching
    if mode == 'exact':
        references = preprocess_references(references, client, openai_model, verbose, source="txt")
        
        # Save cleaned references to file for debugging
        cleaned_refs_file = os.path.join(output_dir, "cleaned_references.txt")
        with open(cleaned_refs_file, 'w', encoding='utf-8') as f:
            for i, ref in enumerate(references, 1):
                f.write(f"{i}. {ref}\n")
        if verbose:
            print(f"Saved cleaned references to: {cleaned_refs_file}")
    
    # Initialize thread-safe counters
    counters = {
        'successful_searches': ThreadSafeCounter(),
        'abstracts': ThreadSafeCounter(),
        'pdfs': ThreadSafeCounter()
    }
    
    if parallel_threads > 1:
        # Parallel processing
        if verbose:
            print(f"Processing {len(references)} references using {parallel_threads} parallel threads...")
        
        # Prepare arguments for parallel processing
        args_list = []
        for i, reference in enumerate(references):
            args_list.append((
                reference, i, len(references), client, openai_model, ss_api_key,
                output_dir, mode, max_papers_per_ref, verbose, counters
            ))
        
        # Progress bars for parallel processing
        abstract_pbar = tqdm(total=len(references), desc="Abstracts", position=0, leave=True)
        pdf_pbar = tqdm(total=len(references), desc="PDFs", position=1, leave=True)
        
        # Process references in parallel
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_reference, args): args[1] for args in args_list}
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    
                    # Update progress bars
                    abstract_pbar.set_description(f"Abstracts ({counters['abstracts'].value}/{len(references)})")
                    pdf_pbar.set_description(f"PDFs ({counters['pdfs'].value}/{len(references)})")
                    abstract_pbar.update(1)
                    pdf_pbar.update(1)
                    
                    if verbose and result.get('error'):
                        print(f"Error in reference {result['index']+1}: {result['error']}")
                        
                except Exception as e:
                    if verbose:
                        print(f"Unexpected error: {e}")
        
        abstract_pbar.close()
        pdf_pbar.close()
        
        # Get final counts
        successful_searches = counters['successful_searches'].value
        total_abstracts = counters['abstracts'].value
        total_pdfs = counters['pdfs'].value
        
    else:
        # Sequential processing (original logic)
        total_abstracts = 0
        total_pdfs = 0
        successful_searches = 0
        
        # Progress bars
        abstract_pbar = tqdm(total=len(references), desc="Abstracts", position=0, leave=True)
        pdf_pbar = tqdm(total=len(references), desc="PDFs", position=1, leave=True)
        
        for i, reference in enumerate(references, 1):
            if verbose:
                print(f"\n[{i}/{len(references)}] Processing reference:")
                print(f"  {reference[:100]}..." if len(reference) > 100 else f"  {reference}")
            
            if mode == 'exact':
                # Use fallback strategies for exact mode
                papers, strategy = search_with_fallbacks(reference, client, openai_model, ss_api_key, verbose, output_dir)
                if papers:
                    papers_data = {'data': papers}
                    successful_searches += 1
                    if verbose:
                        print(f"  ✅ Found papers using strategy: {strategy}")
                else:
                    papers_data = None
                    if verbose:
                        print(f"  ❌ No papers found")
                
            else:  # fuzzy mode
                # Extract keywords for fuzzy matching
                keywords = extract_reference_info(reference, client, openai_model, mode='fuzzy')
                if not keywords:
                    if verbose:
                        print("  No keywords extracted, skipping...")
                    abstract_pbar.update(1)
                    pdf_pbar.update(1)
                    continue
                    
                if verbose:
                    print(f"  Extracted keywords: {', '.join(keywords)}")
                
                # Search for papers using the first keyword (most relevant)
                main_keyword = keywords[0]
                papers_data = search_papers_fuzzy(main_keyword, ss_api_key, limit=max_papers_per_ref)
                if papers_data and papers_data.get('data'):
                    successful_searches += 1
            
            if not papers_data or 'data' not in papers_data:
                if verbose:
                    print(f"  No papers found")
                abstract_pbar.update(1)
                pdf_pbar.update(1)
                continue
                
            papers = papers_data['data']
            if verbose:
                if mode == 'exact':
                    print(f"  Found {len(papers)} matching papers")
                else:
                    print(f"  Found {len(papers)} similar papers")
            
            # Process each paper
            papers_processed = 0
            for j, paper in enumerate(papers):
                # In exact mode, limit to 1 paper unless there are multiple exact matches
                if mode == 'exact' and papers_processed >= 1:
                    break
                    
                if verbose:
                    print(f"    Processing paper {j+1}/{len(papers)}: {paper.get('title', 'No title')[:60]}...")
                
                # Save abstract
                abstract_file = save_abstract(paper, output_dir)
                if abstract_file:
                    total_abstracts += 1
                    papers_processed += 1
                    if verbose:
                        print(f"      Saved abstract: {os.path.basename(abstract_file)}")
                
                abstract_pbar.set_description(f"Abstracts ({total_abstracts}/{len(references)})")
                
                # Try to download PDF if available
                pdf_downloaded = False
                if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
                    pdf_url = paper['openAccessPdf']['url']
                    paper_id = paper.get('paperId', 'unknown')
                    pdf_filename = f"{paper_id}.pdf"
                    pdf_path = os.path.join(output_dir, pdf_filename)
                    
                    if not os.path.exists(pdf_path):
                        if verbose:
                            print(f"      Downloading PDF...")
                        if download_pdf(pdf_url, pdf_path):
                            total_pdfs += 1
                            pdf_downloaded = True
                            if verbose:
                                print(f"      Downloaded PDF: {pdf_filename}")
                        elif verbose:
                            print(f"      Failed to download PDF")
                    else:
                        total_pdfs += 1
                        pdf_downloaded = True
                        if verbose:
                            print(f"      PDF already exists: {pdf_filename}")
                elif verbose:
                    print(f"      No open access PDF available")
                
                pdf_pbar.set_description(f"PDFs ({total_pdfs}/{len(references)})")
                
                # Small delay to be respectful to APIs
                if not verbose:
                    time.sleep(0.5)  # Shorter delay in non-verbose mode
                else:
                    time.sleep(1)
            
            abstract_pbar.update(1)
            pdf_pbar.update(1)
            
            # Longer delay between references in verbose mode
            if verbose:
                time.sleep(1)
        
        abstract_pbar.close()
        pdf_pbar.close()
    
    print(f"\n=== Processing Complete ===")
    print(f"Mode: {mode}")
    print(f"Successful searches: {successful_searches}/{len(references)} ({successful_searches/len(references)*100:.1f}%)")
    print(f"Total abstracts saved: {total_abstracts}/{len(references)} ({total_abstracts/len(references)*100:.1f}%)")
    print(f"Total PDFs downloaded: {total_pdfs}/{len(references)} ({total_pdfs/len(references)*100:.1f}%)")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="PullR - Research Paper Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PDF mode - extract references from PDF and download them with fallbacks
  python pullr.py paper.pdf --model gpt41 --output-dir ./papers --mode pdf
  
  # Directory mode - process all PDFs in a directory
  python pullr.py ./pdf_directory --model gpt41 --output-dir ./papers --mode pdf
  
  # Exact mode - download papers that exactly match each reference
  python pullr.py mtb-refs.txt --model gpt41 --output-dir ./papers --mode exact
  
  # Fuzzy mode - download N similar papers for each reference
  python pullr.py mtb-refs.txt --model gpt41 --output-dir ./papers --mode fuzzy --max-papers 5
  
  # Parallel processing - use 4 threads for faster downloading
  python pullr.py paper.pdf --model gpt41 --output-dir ./papers --mode pdf --parallel 4
  
  # Verbose mode with parallel processing
  python pullr.py refs.txt --model gpt41 --output-dir ./papers --parallel 3 --verbose
  
  # Extract and clean references only (no downloading)
  python pullr.py paper.pdf --model gpt41 --output-dir ./refs --mode pdf --extract-only
  python pullr.py ./pdf_directory --model gpt41 --output-dir ./refs --mode pdf --extract-only
  python pullr.py refs.txt --model gpt41 --output-dir ./cleaned --extract-only
  
  # Sample processing - randomly select N PDFs from directory
  python pullr.py ./pdf_directory --model gpt41 --output-dir ./papers --mode pdf --sample 5
        """)
    
    parser.add_argument('input_file', help='Input: PDF file, directory of PDFs, or text file with references (one per line)')
    parser.add_argument('--model', required=True, help='Model shortname from model_servers.yaml')
    parser.add_argument('--output-dir', required=True, help='Directory to save abstracts and papers')
    parser.add_argument('--mode', choices=['pdf', 'exact', 'fuzzy'], default='exact',
                        help='Processing mode: "pdf" for PDF reference extraction, "exact" for specific papers, "fuzzy" for similar papers (default: exact)')
    parser.add_argument('--ss-api-key', help='Semantic Scholar API key (optional)')
    parser.add_argument('--max-papers', type=int, default=5, 
                        help='Maximum papers to process per reference in fuzzy mode (default: 5, ignored in exact/pdf modes)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show detailed progress and debugging information')
    parser.add_argument('--parallel', type=int, default=1, metavar='N',
                        help='Number of parallel threads for downloading (default: 1, max recommended: 10)')
    parser.add_argument('--extract-only', action='store_true',
                        help='Extract and clean references only, do not download papers or search databases')
    parser.add_argument('--sample', type=int, metavar='N',
                        help='When processing a directory, randomly sample N PDFs instead of processing all')
    
    args = parser.parse_args()
    
    # Validate inputs - can be file or directory
    if not os.path.exists(args.input_file):
        print(f"Error: Input path '{args.input_file}' not found.")
        sys.exit(1)
    
    if not os.path.exists(MODEL_CONFIG_FILE):
        print(f"Error: Model configuration file '{MODEL_CONFIG_FILE}' not found.")
        sys.exit(1)
    
    # Validate parallel parameter
    if args.parallel < 1:
        print("Error: --parallel must be at least 1")
        sys.exit(1)
    elif args.parallel > 10:
        print("Warning: Using more than 10 parallel threads may overwhelm APIs and cause rate limiting issues")
        print(f"Recommended maximum is 10, but you specified {args.parallel}")
    
    # Validate sample parameter
    if args.sample is not None:
        if args.sample < 1:
            print("Error: --sample must be at least 1")
            sys.exit(1)
        if not os.path.isdir(args.input_file):
            print("Error: --sample can only be used with directory input")
            sys.exit(1)
    
    # Check PDF mode requirements
    if args.mode == 'pdf':
        if not PDF_SUPPORT:
            print("Error: PDF mode requires PyPDF2. Install with: pip install PyPDF2")
            sys.exit(1)
        # PDF mode can accept either a PDF file or a directory
        if os.path.isfile(args.input_file) and not args.input_file.lower().endswith('.pdf'):
            print("Error: PDF mode requires a PDF file or directory containing PDFs as input.")
            sys.exit(1)
        elif os.path.isdir(args.input_file):
            # Check if directory contains any PDFs
            pdf_files = find_pdf_files(args.input_file)
            if not pdf_files:
                print(f"Error: No PDF files found in directory '{args.input_file}'.")
                sys.exit(1)
            # Additional validation for sampling
            if args.sample is not None and args.sample > len(pdf_files):
                print(f"Warning: Requested sample size ({args.sample}) is larger than available PDFs ({len(pdf_files)}). Processing all PDFs.")
                args.sample = None
    
    # Get Semantic Scholar API key from environment if not provided
    ss_api_key = args.ss_api_key or os.environ.get('SS-API-KEY') or os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
    if not ss_api_key:
        print("Warning: No Semantic Scholar API key provided. Rate limits may apply.")
    
    # Check for extract-only mode
    if args.extract_only:
        print(f"Running in EXTRACT-ONLY mode - will extract and clean references without downloading")
        try:
            if os.path.isdir(args.input_file):
                # Directory processing in extract-only mode
                process_pdf_directory(
                    args.input_file,
                    args.model,
                    args.output_dir,
                    ss_api_key,
                    args.verbose,
                    args.parallel,
                    extract_only=True,
                    sample_size=args.sample
                )
            else:
                # Single file processing in extract-only mode
                extract_only_mode(
                    args.input_file,
                    args.model,
                    args.output_dir,
                    args.verbose
                )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    # Show mode information for normal processing
    if args.mode == 'pdf':
        print(f"Running in PDF mode - will extract references from PDF and download them with fallback strategies")
    elif args.mode == 'exact':
        print(f"Running in EXACT mode - will find papers that exactly match each reference")
    else:
        print(f"Running in FUZZY mode - will find up to {args.max_papers} similar papers per reference")
    
    if args.parallel > 1:
        print(f"Using {args.parallel} parallel threads for faster processing")
    
    try:
        if args.mode == 'pdf':
            if os.path.isdir(args.input_file):
                # Directory processing in full mode
                process_pdf_directory(
                    args.input_file,
                    args.model,
                    args.output_dir,
                    ss_api_key,
                    args.verbose,
                    args.parallel,
                    extract_only=False,
                    sample_size=args.sample
                )
            else:
                # Single PDF processing
                process_pdf_references(
                    args.input_file,
                    args.model,
                    args.output_dir,
                    ss_api_key,
                    args.verbose,
                    args.parallel
                )
        else:
            # Text file processing (exact/fuzzy modes)
            process_references(
                args.input_file,
                args.model,
                args.output_dir,
                args.mode,
                ss_api_key,
                args.max_papers,
                args.verbose,
                args.parallel
            )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from typing import List, Dict, Any
from core.logger import logger
from urllib.parse import urlparse

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain.split('/')[0]
    except Exception:
        return "unknown"

def run_curator_agent(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Curator Agent - Deduplicate and clean URLs from search results"""
    logger.info(f"Curator Agent: Processing {len(search_results)} raw search results")
    
    unique_results = []
    seen_urls = set()
    
    BLOCKED_DOMAINS = {"facebook.com", "instagram.com", "tiktok.com", "pinterest.com", "threads.net"}
    
    for result in search_results:
        url = result.get("url", "")
        if not url or url in seen_urls:
            continue
            
        domain = extract_domain(url)
        if any(b in domain for b in BLOCKED_DOMAINS):
            continue
            
        seen_urls.add(url)
        
        unique_results.append({
            "url": url,
            "title": result.get("title", ""),
            "content": result.get("content", ""),
            "domain": extract_domain(url)
        })
        
    logger.info(f"Curator Agent: Kept {len(unique_results)} unique URLs")
    return unique_results

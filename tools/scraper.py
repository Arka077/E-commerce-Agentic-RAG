import json
import re
from typing import Dict, Any, Optional
import aiohttp
from bs4 import BeautifulSoup
from core.logger import logger

# Lazy / optional import of crawl4ai
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        excluded_tags=[
            "nav",
            "footer",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe"
        ],
        word_count_threshold=10
    )
except ImportError:
    CRAWL4AI_AVAILABLE = False
    browser_config = None
    run_config = None


def extract_json_ld(html_content: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse Schema.org / JSON-LD product structured metadata.
    Finds product name, brand, price, currency, availability, and ratings.
    """
    if not html_content:
        return None

    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        scripts = soup.find_all('script', type='application/ld+json')
        
        for script in scripts:
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
            except Exception:
                continue
                
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if "@graph" in data and isinstance(data["@graph"], list):
                    items = data["@graph"]
                else:
                    items = [data]
                    
            for item in items:
                if not isinstance(item, dict):
                    continue
                type_val = item.get("@type", "")
                is_product = False
                if isinstance(type_val, str) and "Product" in type_val:
                    is_product = True
                elif isinstance(type_val, list) and any("Product" in str(t) for t in type_val):
                    is_product = True
                    
                if is_product:
                    name = item.get("name")
                    brand = item.get("brand")
                    if isinstance(brand, dict):
                        brand = brand.get("name")
                        
                    price = None
                    currency = "INR"
                    availability = None
                    offers = item.get("offers")
                    if isinstance(offers, list) and offers:
                        offers = offers[0]
                    if isinstance(offers, dict):
                        price = offers.get("price") or offers.get("lowPrice") or offers.get("highPrice")
                        currency = offers.get("priceCurrency") or currency
                        avail_raw = str(offers.get("availability", ""))
                        if "InStock" in avail_raw:
                            availability = "In Stock"
                        elif "OutOfStock" in avail_raw:
                            availability = "Out of Stock"
                        else:
                            availability = avail_raw.split("/")[-1] if "/" in avail_raw else avail_raw

                    # Extract ratings
                    rating = None
                    reviews_count = None
                    agg_rating = item.get("aggregateRating")
                    if isinstance(agg_rating, dict):
                        rating = agg_rating.get("ratingValue")
                        reviews_count = agg_rating.get("reviewCount") or agg_rating.get("ratingCount")
                        
                    structured = {
                        "name": name,
                        "brand": brand,
                        "price": price,
                        "currency": currency,
                        "availability": availability,
                        "rating": rating,
                        "reviews_count": reviews_count,
                        "sku": item.get("sku")
                    }
                    # Filter out None values
                    return {k: v for k, v in structured.items() if v is not None}
    except Exception as e:
        logger.debug(f"JSON-LD extraction error: {e}")
        
    return None


def format_structured_data(data: Optional[Dict[str, Any]]) -> str:
    """Format structured metadata into readable markdown header."""
    if not data:
        return ""
    lines = ["### [Verified Product Specifications (JSON-LD)]"]
    if "name" in data:
        lines.append(f"- **Product:** {data['name']}")
    if "brand" in data:
        lines.append(f"- **Brand:** {data['brand']}")
    if "price" in data:
        lines.append(f"- **Price:** {data.get('currency', 'INR')} {data['price']}")
    if "availability" in data:
        lines.append(f"- **Availability:** {data['availability']}")
    if "rating" in data:
        rev = f" ({data['reviews_count']} reviews)" if "reviews_count" in data else ""
        lines.append(f"- **Rating:** {data['rating']}/5{rev}")
    if "sku" in data:
        lines.append(f"- **SKU:** {data['sku']}")
    lines.append("")
    return "\n".join(lines)


async def _scrape_with_crawl4ai(url: str) -> Optional[Dict[str, Any]]:
    """Scrape using Crawl4AI headless browser."""
    if not CRAWL4AI_AVAILABLE or browser_config is None:
        return None

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if not result.success:
            logger.warning(f"Crawl4AI failed for {url}: {result.error_message}")
            return None

        text = result.markdown or ""
        html = getattr(result, "html", "") or ""
        structured = extract_json_ld(html)
        
        return {
            "text": text,
            "html": html,
            "structured": structured
        }


async def _scrape_with_aiohttp_fallback(url: str) -> Dict[str, Any]:
    """Fallback scraper using aiohttp and BeautifulSoup."""
    logger.info(f"Using aiohttp fallback scraper for: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            html = await response.text()
            
            structured = extract_json_ld(html)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove junk elements
            for element in soup(["script", "style", "meta", "link", "nav", "footer", "header", "noscript", "iframe", "svg"]):
                element.decompose()
                
            text = soup.get_text(separator='\n', strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return {
                "text": clean_text,
                "html": html,
                "structured": structured
            }


async def scrape_product_content(url: str, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
    """
    Scrape and extract clean content and JSON-LD structured specifications from a URL.
    Attempts Crawl4AI headless browser first, falling back to aiohttp/BeautifulSoup.
    Returns:
        {
            "text": str,
            "metadata": {
                "url": url,
                "structured_data": dict or None
            }
        }
    """
    try:
        logger.info(f"Scraping: {url}")
        crawl_res = None
        
        if CRAWL4AI_AVAILABLE:
            try:
                crawl_res = await _scrape_with_crawl4ai(url)
            except Exception as e:
                logger.warning(f"Crawl4AI execution error on {url}: {e}. Triggering fallback.")
                crawl_res = None
                
        if not crawl_res or not crawl_res.get("text"):
            crawl_res = await _scrape_with_aiohttp_fallback(url)
            
        raw_text = crawl_res.get("text", "")
        structured = crawl_res.get("structured")
        
        if not raw_text or len(raw_text.strip()) < 150:
            logger.warning(f"Insufficient content: {url}")
            return {
                "text": "Insufficient content",
                "metadata": {
                    "url": url,
                    "structured_data": structured
                }
            }
            
        structured_header = format_structured_data(structured)
        final_text = f"{structured_header}\n{raw_text}" if structured_header else raw_text
        final_text = final_text[:15000]
        
        logger.info(f"Successfully scraped {len(final_text)} characters from {url} (JSON-LD: {bool(structured)})")
        return {
            "text": final_text,
            "metadata": {
                "url": url,
                "structured_data": structured
            }
        }
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return {
            "text": f"Error scraping: {str(e)}",
            "metadata": {
                "url": url,
                "structured_data": None
            }
        }

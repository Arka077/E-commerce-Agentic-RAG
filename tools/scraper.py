import aiohttp
from bs4 import BeautifulSoup
from core.logger import logger

async def scrape_product_content(url: str, session: aiohttp.ClientSession) -> str:
    """Scrape and clean content from a URL asynchronously"""
    try:
        logger.info(f"Asynchronously scraping: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        async with session.get(url, timeout=15, headers=headers) as response:
            response.raise_for_status()
            html = await response.text()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "meta", "link", "nav", "footer", "header", "noscript", "iframe"]):
                element.decompose()
            
            # Extract text with newline separator to preserve structure
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit content
            text = text[:8000]
            
            if len(text) > 200:
                logger.info(f"Successfully scraped {len(text)} characters from {url}")
                return text
            else:
                return "Insufficient content"
                
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return f"Error scraping: {str(e)}"

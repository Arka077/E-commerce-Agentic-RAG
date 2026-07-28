import asyncio
from datetime import timedelta
from typing import List, Dict, Any

from aiobreaker import CircuitBreaker
from langchain_community.tools.tavily_search import TavilySearchResults

from core.config import settings
from core.logger import logger

# -------------------------
# Circuit Breaker Configuration
# -------------------------
tavily_breaker = CircuitBreaker(
    fail_max=settings.CIRCUIT_BREAKER_FAIL_THRESHOLD,
    timeout_duration=timedelta(
        seconds=settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS
    ),
)

# -------------------------
# API Key Rotation
# -------------------------
_tavily_key_idx = 0

def _get_next_tavily_key() -> str:
    global _tavily_key_idx
    keys = settings.TAVILY_API_KEYS

    if not keys:
        raise ValueError("No Tavily API keys configured")

    key = keys[_tavily_key_idx]
    _tavily_key_idx = (_tavily_key_idx + 1) % len(keys)
    return key

# -------------------------
# Blocking Search Execution
# -------------------------
def _search_sync(query: str) -> List[Dict[str, Any]]:
    api_key = _get_next_tavily_key()
    logger.info(f"Tavily search query: '{query}'")

    search = TavilySearchResults(
        max_results=settings.TAVILY_MAX_RESULTS,
        api_key=api_key,
        search_depth="advanced",
    )
    return search.invoke(query)

# -------------------------
# Async Wrapper (Protected)
# -------------------------
async def search_ecommerce_products(query: str) -> List[Dict[str, Any]]:
    """
    Performs a Tavily search protected by an aiobreaker circuit breaker.
    """
    try:
        return await tavily_breaker.call_async(
            asyncio.to_thread, _search_sync, query
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        raise

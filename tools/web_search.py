import asyncio
from datetime import timedelta
from typing import List, Dict, Any

from aiobreaker import CircuitBreaker
from tavily import TavilyClient

from core.config import settings
from core.logger import logger

tavily_breaker = CircuitBreaker(
    fail_max=settings.CIRCUIT_BREAKER_FAIL_THRESHOLD,
    timeout_duration=timedelta(
        seconds=settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS
    ),
)

_tavily_key_idx = 0

def _get_next_tavily_key() -> str:
    global _tavily_key_idx
    keys = settings.TAVILY_API_KEYS

    if not keys:
        raise ValueError("No Tavily API keys configured")

    key = keys[_tavily_key_idx]
    _tavily_key_idx = (_tavily_key_idx + 1) % len(keys)
    return key

def _search_sync(query: str) -> List[Dict[str, Any]]:
    api_key = _get_next_tavily_key()
    logger.info(f"Tavily search query: '{query}'")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=settings.TAVILY_MAX_RESULTS,
        search_depth="advanced"
    )
    return response.get("results", [])

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

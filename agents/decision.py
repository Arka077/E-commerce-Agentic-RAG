from typing import List, Tuple
from core.llm import llm_router, llm_breaker
from core.logger import logger

async def run_decision_agent(query: str, chat_history: List = None) -> Tuple[bool, str]:
    """
    Decision Agent - Decides if web search is needed.
    Returns: (needs_search: bool, reasoning: str)
    """
    logger.info("Running Decision Agent")
    
    if chat_history is None:
        chat_history = []
        
    # Check if we have previous answers
    if not chat_history or len(chat_history) <= 1:
        logger.info("First query: Search is required")
        return True, "First query - need fresh data"
        
    # Build conversation context
    context = "Previous Conversation:\n"
    for msg in chat_history[-6:]:  # Last 3 exchanges
        if isinstance(msg, dict):
            role = msg.get("role", "").upper()
            content = msg.get("content", "")[:200]
        else:
            try:
                role = "USER" if msg[0] else "ASSISTANT"
                content = (msg[0] or msg[1] or "")[:200]
            except Exception:
                continue
        context += f"{role}: {content}\n"
        
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a decision agent for e-commerce queries.\n\n"
                "Analyze if the current query can be answered from PREVIOUS conversation context or if it needs NEW web search.\n\n"
                "Rules:\n"
                "1. If query asks about SPECIFIC products/prices/comparisons from before -> NO search needed, use previous context\n"
                "2. If query asks for DIFFERENT products or NEW information -> YES search needed\n"
                "3. If query is a follow-up question about something already discussed -> NO search needed\n"
                "4. If unclear or asking for LATEST data -> YES search needed\n\n"
                "Respond with ONLY: \"SEARCH\" or \"CONTEXT\""
            )
        },
        {
            "role": "user",
            "content": f"{context}\n\nCurrent query: {query}\n\nDecision: (respond with ONLY \"SEARCH\" or \"CONTEXT\")"
        }
    ]
    
    if not llm_breaker.can_execute():
        logger.warning("LLM Circuit Breaker is OPEN. Skipping decision LLM and defaulting to SEARCH.")
        return True, "Error in decision (LLM breaker OPEN): defaulting to search"
        
    try:
        response = await llm_router.acompletion(
            model="mistral-small",
            messages=messages,
            temperature=0.0
        )
        decision = response.choices[0].message.content.strip().upper()
        llm_breaker.record_success()
        
        needs_search = "SEARCH" in decision
        reasoning = f"Decision: {'Will search web' if needs_search else 'Will use previous context'}"
        logger.info(f"Decision Agent reasoning: {reasoning}")
        return needs_search, reasoning
    except Exception as e:
        llm_breaker.record_failure()
        logger.error(f"Decision Agent error: {str(e)}")
        return True, f"Error in decision (default to search): {str(e)[:30]}"

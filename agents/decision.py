from typing import List, Tuple
from core.llm import llm_router, llm_breaker, PRIMARY_MODEL
from core.logger import logger

async def run_decision_agent(query: str, chat_history: List = None) -> Tuple[bool, str]:
    """
    Determines whether a new web search is required or existing context is sufficient.
    """
    logger.info("Running Decision Agent")
    
    if chat_history is None:
        chat_history = []
        
    if not chat_history or len(chat_history) <= 1:
        logger.info("First query: Search is required")
        return True, "First query - need fresh data"
        
    context = "Previous Conversation:\n"
    for msg in chat_history[-6:]:
        if isinstance(msg, dict):
            role = msg.get("role", "").upper()
            content = msg.get("content", "")[:1200]
        else:
            try:
                role = "USER" if msg[0] else "ASSISTANT"
                content = (msg[0] or msg[1] or "")[:1200]
            except Exception:
                continue
        context += f"{role}: {content}\n"
        
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent decision agent for an e-commerce assistant.\n\n"
                "Your task: Decide if the current user query requires a NEW WEB SEARCH or if it can be answered using the EXISTING CONVERSATION and indexed knowledge base.\n\n"
                "CRITICAL ROUTING RULES:\n"
                "1. If the user asks a follow-up question referencing previously discussed products, options, or recommendations (e.g., 'tell me the specs of the above products', 'give details on those laptops', 'compare them', 'which one should I buy', 'how is the battery of the first one?') -> Output 'CONTEXT'. DO NOT search the web because the full specifications are already available in the knowledge base and conversation.\n"
                "2. If the user asks for clarifications, price comparisons, or pros/cons of products already mentioned -> Output 'CONTEXT'.\n"
                "3. If the user asks about an entirely DIFFERENT product category, brand, or new item NOT mentioned in the conversation -> Output 'SEARCH'.\n"
                "4. If the user explicitly asks for 'new/unrelated products' or wants to restart search -> Output 'SEARCH'.\n\n"
                "Output ONLY one word: either 'CONTEXT' or 'SEARCH'."
            )
        },
        {
            "role": "user",
            "content": f"{context}\n\nCurrent User Query: {query}\n\nDecision (Respond with ONLY 'CONTEXT' or 'SEARCH'):"
        }
    ]
    
    if not llm_breaker.can_execute():
        logger.warning("LLM Circuit Breaker is OPEN. Skipping decision LLM and defaulting to SEARCH.")
        return True, "Error in decision (LLM breaker OPEN): defaulting to search"
        
    try:
        response = await llm_router.acompletion(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=0.0
        )
        decision = response.choices[0].message.content.strip().upper()
        llm_breaker.record_success()
        
        # Robust check: prioritize CONTEXT if mentioned or if it starts with CONTEXT
        if "CONTEXT" in decision:
            needs_search = False
        elif "SEARCH" in decision:
            needs_search = True
        else:
            needs_search = False
            
        reasoning = f"Decision: {'Will search web' if needs_search else 'Will use previous context'}"
        logger.info(f"Decision Agent: {decision} -> {reasoning}")
        return needs_search, reasoning
    except Exception as e:
        llm_breaker.record_failure()
        logger.error(f"Decision Agent error: {str(e)}")
        return True, f"Error in decision (default to search): {str(e)[:30]}"

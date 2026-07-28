import time
from typing import List, Dict, Any, AsyncGenerator
from core.llm import llm_router, llm_breaker
from core.logger import logger

async def run_synthesis_agent(
    query: str, 
    retrieved_chunks: List[Dict[str, Any]], 
    chat_history: List = None
) -> AsyncGenerator[str, None]:
    """
    Synthesis Agent - Streams the generated answer asynchronously using LiteLLM.
    """
    logger.info("Running Synthesis Agent (Streaming)")
    
    if chat_history is None:
        chat_history = []
        
    # Prepare context with temporal scraped_timestamp metadata
    if retrieved_chunks:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks[:5]):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            scraped_at = meta.get("scraped_timestamp", "N/A")
            context_parts.append(
                f"[{i+1}] Source URL: {source} | Scraped Time: {scraped_at}\n"
                f"Content: {chunk.get('document', '')}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "[No relevant information found in knowledge base]"
        
    # Build chat history context string
    chat_context = ""
    if chat_history and len(chat_history) > 1:
        chat_context = "Previous Conversation:\n"
        for msg in chat_history[:-1]:  # All except current query
            if isinstance(msg, dict):
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
            else:
                try:
                    role = "USER" if msg[0] else "ASSISTANT"
                    content = msg[0] or msg[1] or ""
                except Exception:
                    continue
            chat_context += f"{role}: {content[:300]}\n\n"
            
    # System prompt emphasizing temporal context to prevent pricing hallucinations
    system_prompt = (
        "You are an expert e-commerce assistant. The default region is INDIA and the current year is 2026.\n"
        "Your task is to answer the user's query using the provided context and previous conversation.\n\n"
        "CRITICAL TEMPORAL RULES:\n"
        "1. Prioritize and heavily weight the most recent data (check the 'Scraped Time' in the context) for pricing, specifications, and availability.\n"
        "2. If you see conflicting prices, assume the one with the latest 'Scraped Time' is the current active price. Mention this explicitly.\n"
        "3. Provide clean, concise markdown answers and always cite the source URLs.\n"
        "4. If answering a follow-up query, combine previous insights with new information."
    )
    
    user_prompt = f"""
{chat_context}

Current Query: {query}

Context Chunks:
{context}

Please provide your recommendation and answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Check circuit breaker
    if not llm_breaker.can_execute():
        logger.error("LLM Circuit Breaker is OPEN. Aborting synthesis.")
        yield "⚠️ Synthesis is temporarily unavailable due to upstream API issues. Please try again shortly."
        return
        
    try:
        response = await llm_router.acompletion(
            model="mistral-large",
            messages=messages,
            temperature=0.2,
            stream=True
        )
        
        # Stream the chunks
        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
                
        llm_breaker.record_success()
    except Exception as e:
        llm_breaker.record_failure()
        logger.error(f"Synthesis Agent failed: {str(e)}")
        yield f"⚠️ Error generating answer: {str(e)[:100]}"

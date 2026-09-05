from typing import List, Dict, Any, AsyncGenerator
from core.llm import llm_router, llm_breaker, PRIMARY_MODEL
from core.logger import logger

async def run_synthesis_agent(
    query: str, 
    retrieved_chunks: List[Dict[str, Any]], 
    chat_history: List = None
) -> AsyncGenerator[str, None]:
    """
    Stream response synthesis using retrieved product context and conversation history.
    """
    logger.info("Running Synthesis Agent (Streaming)")
    
    if chat_history is None:
        chat_history = []
        
    # Format retrieved chunks and metadata
    if retrieved_chunks:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks[:5]):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown")
            scraped_at = meta.get("scraped_timestamp", "N/A")
            structured = meta.get("structured_data", {})
            
            spec_summary = ""
            if structured:
                spec_summary = f"Verified Specs: {structured}\n"
                
            context_parts.append(
                f"[{i+1}] Source URL: {source} | Scraped Time: {scraped_at}\n"
                f"{spec_summary}"
                f"Content: {chunk.get('document', '')}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "[No relevant information found in knowledge base]"
        
    chat_context = ""
    if chat_history and len(chat_history) > 1:
        chat_context = "Previous Conversation:\n"
        for msg in chat_history[:-1]:
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
            
    system_prompt = (
        "You are an elite, objective e-commerce shopping consultant and product analyst. "
        "The default region is INDIA and the current year is 2026. All prices must be quoted in Indian Rupees (₹ INR).\n\n"
        "CORE RESPONSIBILITIES ACROSS ANY PRODUCT CATEGORY (Electronics, Fashion, Home, Beauty, Fitness, Appliances, etc.):\n"
        "1. PRICING & TEMPORAL ACCURACY:\n"
        "   - Prioritize recent prices and specifications from the provided context (check 'Scraped Time' and 'Verified Specs').\n"
        "   - Always quote the exact price in ₹ INR and mention the retailer/store (e.g., Amazon, Flipkart, Official Store) when available.\n"
        "   - If discounts, card offers, or price variants exist in context, highlight them clearly.\n\n"
        "2. STRUCTURED PRODUCT COMPARISON TABLE:\n"
        "   - Whenever comparing, recommending, or discussing 2 or more products, you MUST include a clean Markdown comparison table:\n"
        "     | Product / Model | Price (INR) | Key Specs & Highlights | Best For | Top Advantage | Watch Out For | Source |\n"
        "   - Ensure all columns are populated accurately from the context.\n\n"
        "3. DEEP SPECIFICATION & FEATURE BREAKDOWNS:\n"
        "   - For each recommended item, provide a structured breakdown adapted to its specific category:\n"
        "     * Tech/Electronics: CPU/Processor, GPU/Graphics (TGP if gaming), RAM & Storage, Display specs (refresh rate, color gamut), Battery, Weight, Ports.\n"
        "     * Apparel/Footwear: Material/Fabric, Fit & Sizing, Cushioning/Sole, Weather Resistance, Care.\n"
        "     * Appliances/Home: Capacity/Dimensions, Power/Wattage, Energy Rating, Key Modes, Warranty.\n"
        "     * Beauty/Personal Care: Key Active Ingredients, Skin/Hair Type, Formula, Volume/Size.\n"
        "     * Other categories: Primary build materials, durability, dimensions, warranty, and intended use.\n\n"
        "4. BALANCED PROS & CONS:\n"
        "   - Provide 2-3 genuine pros and at least 1-2 realistic cons or trade-offs for each product based on user reviews or expert tests in the context.\n\n"
        "5. BUYING ADVICE & FINAL VERDICT:\n"
        "   - Give clear, decisive recommendations based on user priorities (e.g., 'Best Overall', 'Best Value for Money', 'Best Premium Choice').\n"
        "   - If answering a follow-up query, seamlessly build upon previously discussed context without repeating the entire conversation."
    )
    
    user_prompt = f"""
{chat_context}

Current User Query: {query}

Retrieved Knowledge Base Context:
{context}

Please provide a detailed, beautifully structured e-commerce recommendation including Markdown comparison tables, detailed specifications, current ₹ INR pricing, pros/cons, and final buying advice:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    if not llm_breaker.can_execute():
        logger.error("LLM Circuit Breaker is OPEN. Aborting synthesis.")
        yield "⚠️ Synthesis is temporarily unavailable due to upstream API issues. Please try again shortly."
        return
        
    try:
        response = await llm_router.acompletion(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=0.2,
            stream=True
        )
        
        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content
                
        llm_breaker.record_success()
    except Exception as e:
        llm_breaker.record_failure()
        logger.error(f"Synthesis Agent failed: {str(e)}")
        yield f"⚠️ Error generating answer: {str(e)[:100]}"

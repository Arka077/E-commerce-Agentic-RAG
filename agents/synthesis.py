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
        "The default region is INDIA. All prices must be quoted in Indian Rupees (₹ INR).\n\n"
        "CORE RESPONSIBILITIES ACROSS ANY PRODUCT CATEGORY (Electronics, Fashion, Home, Beauty, Fitness, Appliances, etc.):\n"
        "1. STRICT FACTUAL GROUNDING (CRITICAL MANDATE):\n"
        "   - You MUST recommend, discuss, and table ONLY products and exact models that are explicitly present in the retrieved context.\n"
        "   - NEVER extrapolate, invent, or bring in external/unreleased product models (e.g. do not guess future releases or substitute unlisted models) from your parametric memory.\n"
        "   - If only 2 or 3 products exist in the context, focus solely on those. If a specification or price is not mentioned in the context, write 'Not specified' rather than guessing.\n\n"
        "2. PRICING & TEMPORAL ACCURACY:\n"
        "   - Prioritize verified prices and specifications from the provided context (check 'Scraped Time' and 'Verified Specs').\n"
        "   - Quote the exact price in ₹ INR and mention the retailer/store (e.g., Amazon, Flipkart, 91mobiles, Official Store) when available.\n"
        "   - If discounts, card offers, or price variants exist in context, highlight them clearly.\n\n"
        "3. STRUCTURED PRODUCT COMPARISON TABLE:\n"
        "   - When comparing or recommending 2 or more products from the context, provide a clean Markdown comparison table:\n"
        "     | Product / Model | Price (INR) | Key Specs & Highlights | Best For | Top Advantage | Watch Out For | Source |\n"
        "   - Ensure all columns are populated strictly with facts from the context.\n\n"
        "4. DEEP SPECIFICATION & FEATURE BREAKDOWNS:\n"
        "   - For each recommended item from the context, provide a structured breakdown adapted to its category:\n"
        "     * Tech/Electronics: Processor, Display, RAM/Storage, Battery/Charging, Camera/Build, OS.\n"
        "     * Apparel/Footwear: Material/Fabric, Fit & Sizing, Cushioning, Weather Resistance, Care.\n"
        "     * Appliances/Home: Capacity/Dimensions, Power/Wattage, Energy Rating, Key Modes, Warranty.\n"
        "     * Beauty/Personal Care: Key Active Ingredients, Skin/Hair Type, Formula, Volume/Size.\n"
        "     * Other categories: Primary build materials, durability, dimensions, warranty, and intended use.\n\n"
        "5. BALANCED PROS & CONS:\n"
        "   - Provide 2-3 genuine pros and at least 1-2 realistic cons or trade-offs for each product based on information in the context.\n\n"
        "6. BUYING ADVICE & FINAL VERDICT:\n"
        "   - Give clear, decisive recommendations based on user priorities (e.g., 'Best Overall', 'Best Value for Money', 'Best Premium Choice') selecting from the context items."
    )
    
    user_prompt = f"""
{chat_context}

Current User Query: {query}

Retrieved Knowledge Base Context:
{context}

Please provide a detailed, beautifully structured e-commerce recommendation based strictly on the retrieved knowledge:"""

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

from typing import Tuple, List, Any
import json
from core.llm import llm_router, GUARDRAIL_MODEL, llm_breaker
from core.logger import logger

REJECTION_MESSAGE = (
    "👋 I am your **E-commerce & Shopping Assistant**! 🛍️\n\n"
    "I specialize in:\n"
    "- 🔍 Searching and comparing products across retailers\n"
    "- 💰 Finding current market prices, deals, and discounts\n"
    "- 📊 Analyzing product specifications, materials, and features\n"
    "- ⚖️ Providing unbiased buying recommendations\n\n"
    "Please ask me a question about products, shopping, or buying advice (e.g., *'Best running shoes under ₹4,000'* or *'Compare iPhone 15 vs Galaxy S24'*)."
)

async def run_ecommerce_guardrail(query: str, chat_history: List[Any] = None) -> Tuple[bool, str]:
    """
    Ultra-fast guardrail using Gemini Flash Lite to ensure queries are strictly e-commerce/shopping related.
    Returns:
        (is_allowed: bool, message_or_reason: str)
    """
    logger.info(f"Guardrail: Evaluating query for e-commerce intent: '{query[:80]}'")
    
    # Fast-path for common greetings
    clean_q = query.strip().lower()
    if clean_q in ["hi", "hello", "hey", "help", "who are you", "what can you do"]:
        return False, REJECTION_MESSAGE
        
    if not llm_breaker.can_execute():
        logger.warning("LLM Circuit Breaker is OPEN. Permitting query through guardrail as fallback.")
        return True, "Circuit breaker bypass"

    has_prior_shopping = bool(chat_history and len(chat_history) > 0)

    system_prompt = (
        "You are an ultra-fast query classification guardrail for an e-commerce shopping platform.\n"
        "Your sole task is to determine whether the user query is related to shopping, purchasing, products, or e-commerce.\n\n"
        "ALLOWED INTENTS (ANY category of physical or digital consumer products: electronics, clothing, shoes, beauty, home, groceries, appliances, fitness, etc.):\n"
        "- Product search, recommendations, reviews, or buying advice.\n"
        "- Price inquiries, discounts, deals, availability, and retailer comparison.\n"
        "- Product specs, features, dimensions, ingredients, sizing, compatibility, or durability.\n"
        "- Product comparisons (X vs Y).\n"
        "- Follow-up questions about previously discussed products (e.g., 'which is lighter?', 'does it have warranty?').\n\n"
        "DISALLOWED INTENTS:\n"
        "- Writing code, debugging software, technical programming algorithms.\n"
        "- General knowledge trivia, history, politics, philosophy, geography not tied to shopping.\n"
        "- Creative writing (poems, fiction), math problem solving, academic essays.\n"
        "- Medical diagnoses or non-commercial queries.\n\n"
        "Return ONLY a JSON object: {\"is_ecommerce\": true/false, \"reason\": \"<short explanation>\"}"
    )

    user_prompt = f"User Query: {query}\nHas Prior Shopping Conversation: {has_prior_shopping}"

    try:
        response = await llm_router.acompletion(
            model=GUARDRAIL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        is_ecommerce = bool(data.get("is_ecommerce", False))
        reason = data.get("reason", "")
        
        logger.info(f"Guardrail result: is_ecommerce={is_ecommerce} (Reason: {reason})")
        
        if is_ecommerce:
            return True, reason
        else:
            return False, REJECTION_MESSAGE
            
    except Exception as e:
        logger.warning(f"Guardrail evaluation encountered error: {e}. Defaulting to ALLOW.")
        return True, "Fallback allow on error"

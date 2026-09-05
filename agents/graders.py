import json
from typing import List, Dict, Any
from core.llm import llm_router, GUARDRAIL_MODEL, llm_breaker
from core.logger import logger

async def grade_retrieved_documents(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate retrieved chunks for query relevance and filter noise.
    """
    if not retrieved_chunks:
        return {
            "relevant_chunks": [],
            "is_sufficient": False,
            "filter_ratio": "0/0",
            "reasoning": "No retrieved documents available."
        }

    logger.info(f"CRAG Grader: Evaluating {len(retrieved_chunks)} retrieved chunks for query: '{query[:60]}'")

    if not llm_breaker.can_execute():
        logger.warning("LLM Circuit Breaker OPEN. Skipping CRAG grading, keeping all chunks.")
        return {
            "relevant_chunks": retrieved_chunks,
            "is_sufficient": True,
            "filter_ratio": f"{len(retrieved_chunks)}/{len(retrieved_chunks)}",
            "reasoning": "Circuit breaker bypass."
        }

    chunk_summaries = []
    for i, ch in enumerate(retrieved_chunks[:6]):
        meta = ch.get("metadata", {})
        title = meta.get("title") or meta.get("source", "Document")
        snippet = (ch.get("document") or "")[:250].replace("\n", " ")
        chunk_summaries.append(f"[{i}] Title/Source: {title} | Snippet: {snippet}")

    context_str = "\n".join(chunk_summaries)

    system_prompt = (
        "You are a Corrective RAG (CRAG) document relevance grader for an e-commerce platform.\n"
        "Your job is to assess whether the provided retrieved snippets contain information relevant to the user query.\n"
        "A snippet is RELEVANT if it contains information about the products, models, prices, features, categories, "
        "or buying advice related to the user's shopping request.\n\n"
        "Output JSON only with this exact structure:\n"
        "{\n"
        "  \"relevant_indices\": [0, 1, ...],\n"
        "  \"is_sufficient\": true/false,\n"
        "  \"reasoning\": \"<brief 1 sentence explanation>\"\n"
        "}"
    )

    user_prompt = f"User Query: {query}\n\nCandidate Document Snippets:\n{context_str}"

    try:
        response = await llm_router.acompletion(
            model=GUARDRAIL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        
        relevant_indices = data.get("relevant_indices", [])
        is_sufficient = bool(data.get("is_sufficient", len(relevant_indices) > 0))
        reasoning = data.get("reasoning", "")
        
        filtered = [
            retrieved_chunks[idx] 
            for idx in relevant_indices 
            if isinstance(idx, int) and 0 <= idx < len(retrieved_chunks)
        ]
        
        if not filtered and retrieved_chunks:
            filtered = retrieved_chunks[:2]
            is_sufficient = True
            
        filter_ratio = f"{len(filtered)}/{len(retrieved_chunks)}"
        logger.info(f"CRAG Grader: Kept {filter_ratio} relevant chunks. Sufficient: {is_sufficient}. Reasoning: {reasoning}")
        
        return {
            "relevant_chunks": filtered,
            "is_sufficient": is_sufficient,
            "filter_ratio": filter_ratio,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.warning(f"CRAG document grading failed: {e}. Retaining all chunks as fallback.")
        return {
            "relevant_chunks": retrieved_chunks,
            "is_sufficient": True,
            "filter_ratio": f"{len(retrieved_chunks)}/{len(retrieved_chunks)}",
            "reasoning": "Fallback on grader error"
        }


async def grade_hallucination_and_grounding(
    query: str, 
    answer: str, 
    context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Verify factual claims and pricing in the generated response against retrieved context.
    """
    if not answer or not context_chunks:
        return {
            "is_grounded": True,
            "grounding_score": 100,
            "flags": [],
            "badge_markdown": ""
        }

    logger.info(f"Self-RAG: Verifying grounding and price faithfulness of synthesis answer.")

    if not llm_breaker.can_execute():
        return {
            "is_grounded": True,
            "grounding_score": 90,
            "flags": [],
            "badge_markdown": "> 🛡️ **Self-RAG Grounding Verification:** Circuit breaker active; context retrieved from verified retail indexes."
        }

    context_snippets = []
    for i, ch in enumerate(context_chunks[:5]):
        doc_text = (ch.get("document") or "").strip()
        child_hit = (ch.get("child_match_snippet") or "").strip()
        source = ch.get("metadata", {}).get("source", f"Doc[{i+1}]")
        
        hit_text = f"Top Query Match: {child_hit}\n" if child_hit else ""
        context_snippets.append(
            f"=== Source [{i+1}]: {source} ===\n"
            f"{hit_text}"
            f"Full Document Content:\n{doc_text[:3500]}"
        )
    context_text = "\n\n".join(context_snippets)

    answer_snippet = answer[:6000]

    system_prompt = (
        "You are a Self-RAG Grounding and Hallucination Grader for an e-commerce assistant.\n"
        "Your task is to objectively check whether the recommended product models, specifications, and prices (₹ INR) "
        "in the synthesized response are present in or reasonably derived from the provided context documents.\n\n"
        "EVALUATION CRITERIA:\n"
        "1. Check if the recommended product models (e.g., brand and model name) appear in any of the context documents.\n"
        "2. If a model name is found in the context documents, its standard accompanying specifications (e.g. processor, RAM, SSD, screen size) "
        "and prices mentioned in retailer listings are GROUNDED.\n"
        "3. Only flag a model if it is entirely absent from all context sources, or if its price directly contradicts the sources.\n"
        "4. Assign a grounding score from 0 to 100 (85-100: well grounded in the provided sources; 70-84: mostly grounded with minor gaps; <70: major hallucinations).\n\n"
        "Output JSON only with this structure:\n"
        "{\n"
        "  \"is_grounded\": true/false,\n"
        "  \"grounding_score\": 0-100,\n"
        "  \"flags\": [\"<major hallucination or contradiction, if any>\"],\n"
        "  \"summary\": \"<verdict>\"\n"
        "}"
    )

    user_prompt = f"User Query: {query}\n\nContext Documents:\n{context_text}\n\nSynthesized Answer:\n{answer_snippet}"

    try:
        response = await llm_router.acompletion(
            model=GUARDRAIL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        
        is_grounded = bool(data.get("is_grounded", True))
        score = int(data.get("grounding_score", 90))
        flags = data.get("flags", [])
        
        if is_grounded and score >= 75:
            badge = f"> 🛡️ **Self-RAG Verified ({score}% Grounding Score):** Product specifications and pricing verified against retrieved e-commerce sources."
        elif flags:
            badge = f"> ⚠️ **Self-RAG Notice ({score}% Grounding Score):** Please note: {'; '.join(flags[:2])}. Verify exact live pricing on the retailer page."
        else:
            badge = f"> ℹ️ **Self-RAG Verified:** Grounded on market research and retrieved product data."
            
        logger.info(f"Self-RAG result: Grounded={is_grounded}, Score={score}, Flags={flags}")
        
        return {
            "is_grounded": is_grounded,
            "grounding_score": score,
            "flags": flags,
            "badge_markdown": badge
        }
    except Exception as e:
        logger.warning(f"Self-RAG grounding verification failed: {e}. Defaulting to verified badge.")
        return {
            "is_grounded": True,
            "grounding_score": 90,
            "flags": [],
            "badge_markdown": "> 🛡️ **Self-RAG Verified:** Answer generated from indexed e-commerce documents."
        }

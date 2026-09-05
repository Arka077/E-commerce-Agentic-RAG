import asyncio
from typing import AsyncGenerator, Dict, Any, List, Literal, TypedDict
from langgraph.graph import StateGraph, START, END

from core.logger import logger, set_session_id
from agents.guardrail import run_ecommerce_guardrail
from agents.decision import run_decision_agent
from agents.graders import grade_retrieved_documents, grade_hallucination_and_grounding
from agents.synthesis import run_synthesis_agent
from workflow.services import SearchService, IndexService, RetrievalService

class GraphState(TypedDict):
    query: str
    chat_history: List[Any]
    session_id: str
    is_ecommerce: bool
    needs_search: bool
    cleaned_documents: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    is_sufficient: bool
    search_retry_count: int
    answer: str
    citations: List[Dict[str, Any]]
    grounding_report: Dict[str, Any]
    progress_log: List[str]

async def guardrail_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing guardrail_node")
    progress_log = list(state.get("progress_log", []))
    
    is_ecommerce, message_or_reason = await run_ecommerce_guardrail(
        state["query"], 
        state.get("chat_history")
    )
    
    if is_ecommerce:
        progress_log.append("Guardrail: Shopping intent verified")
        return {
            "is_ecommerce": True,
            "progress_log": progress_log
        }
    else:
        progress_log.append("Guardrail: Non-ecommerce intent flagged")
        return {
            "is_ecommerce": False,
            "answer": message_or_reason,
            "progress_log": progress_log
        }

async def decision_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing decision_node")
    needs_search, reasoning = await run_decision_agent(state["query"], state["chat_history"])
    progress_log = list(state.get("progress_log", []))
    progress_log.append("Decision completed")
    return {
        "needs_search": needs_search,
        "progress_log": progress_log
    }

async def search_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing search_node")
    progress_log = list(state.get("progress_log", []))
    progress_log.append("Searching web")
    progress_log.append("Scraping sources")
    
    search_service = SearchService()
    result = await search_service.execute(
        state["query"], 
        chat_history=state.get("chat_history")
    )
    
    return {
        "search_results": result["search_results"],
        "cleaned_documents": result["cleaned_documents"],
        "progress_log": progress_log
    }

async def index_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing index_node")
    progress_log = list(state.get("progress_log", []))
    progress_log.append("Indexing documents")
    
    index_service = IndexService()
    await index_service.index_documents(
        state.get("cleaned_documents", []), 
        session_id=state.get("session_id")
    )
    
    return {
        "progress_log": progress_log
    }

async def retrieval_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing retrieval_node")
    progress_log = list(state.get("progress_log", []))
    progress_log.append("Retrieving knowledge")
    
    retrieval_service = RetrievalService()
    result = await retrieval_service.retrieve(
        state["query"], 
        session_id=state.get("session_id")
    )
    
    return {
        "retrieved_chunks": result["retrieved_chunks"],
        "citations": result["citations"],
        "progress_log": progress_log
    }

async def crag_grader_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing crag_grader_node (Corrective RAG)")
    progress_log = list(state.get("progress_log", []))
    
    retrieved = state.get("retrieved_chunks", [])
    grade_res = await grade_retrieved_documents(state["query"], retrieved)
    
    relevant = grade_res.get("relevant_chunks", retrieved)
    is_sufficient = grade_res.get("is_sufficient", True)
    ratio = grade_res.get("filter_ratio", f"{len(relevant)}/{len(retrieved)}")
    
    progress_log.append(f"CRAG: Kept {ratio} relevant document chunks")
    
    return {
        "relevant_chunks": relevant,
        "is_sufficient": is_sufficient,
        "progress_log": progress_log
    }

async def synthesis_node(state: GraphState) -> AsyncGenerator[Dict[str, Any], None]:
    logger.info("Executing synthesis_node")
    progress_log = list(state.get("progress_log", []))
    progress_log.append("Synthesizing answer")
    yield {
        "progress_log": progress_log,
        "answer": ""
    }
    
    chunks_to_use = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    partial_answer = ""
    async for chunk in run_synthesis_agent(state["query"], chunks_to_use, state["chat_history"]):
        partial_answer += chunk
        yield {
            "answer": partial_answer
        }

async def self_rag_grader_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing self_rag_grader_node (Grounding & Fact Verification)")
    progress_log = list(state.get("progress_log", []))
    
    answer = state.get("answer", "")
    chunks_to_use = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
    
    report = await grade_hallucination_and_grounding(state["query"], answer, chunks_to_use)
    badge = report.get("badge_markdown", "")
    
    final_answer = answer
    if badge and badge not in final_answer:
        final_answer = f"{answer}\n\n{badge}"
        
    progress_log.append(f"Self-RAG: Verified with score {report.get('grounding_score', 90)}%")
    
    return {
        "answer": final_answer,
        "grounding_report": report,
        "progress_log": progress_log
    }

def route_guardrail(state: GraphState) -> Literal["decision", "end"]:
    if state.get("is_ecommerce", False):
        return "decision"
    return "end"

def route_decision(state: GraphState) -> Literal["search", "retrieve"]:
    if state.get("needs_search", False):
        return "search"
    return "retrieve"

def route_crag(state: GraphState) -> Literal["synthesis", "search"]:
    retry_count = state.get("search_retry_count", 0)
    if not state.get("is_sufficient", True) and retry_count < 1:
        logger.info("CRAG: Chunks insufficient, triggering fallback search")
        state["search_retry_count"] = retry_count + 1
        return "search"
    return "synthesis"

workflow = StateGraph(GraphState)

workflow.add_node("guardrail", guardrail_node)
workflow.add_node("decision", decision_node)
workflow.add_node("search", search_node)
workflow.add_node("index", index_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("crag_grader", crag_grader_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("self_rag_grader", self_rag_grader_node)

workflow.add_edge(START, "guardrail")

workflow.add_conditional_edges(
    "guardrail",
    route_guardrail,
    {
        "decision": "decision",
        "end": END
    }
)

workflow.add_conditional_edges(
    "decision",
    route_decision,
    {
        "search": "search",
        "retrieve": "retrieve"
    }
)

workflow.add_edge("search", "index")
workflow.add_edge("index", "retrieve")
workflow.add_edge("retrieve", "crag_grader")

workflow.add_conditional_edges(
    "crag_grader",
    route_crag,
    {
        "synthesis": "synthesis",
        "search": "search"
    }
)

workflow.add_edge("synthesis", "self_rag_grader")
workflow.add_edge("self_rag_grader", END)

graph = workflow.compile()


# 5. Graph Orchestrator runner
async def run_ecommerce_workflow(
    query: str, 
    chat_history: List, 
    session_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Executes the LangGraph workflow and streams progress updates and tokens.
    """
    set_session_id(session_id)
    logger.info(f"Starting LangGraph workflow for query: '{query}' (Session: {session_id})")
    
    initial_state: GraphState = {
        "query": query,
        "chat_history": chat_history,
        "session_id": session_id,
        "is_ecommerce": True,
        "needs_search": False,
        "cleaned_documents": [],
        "search_results": [],
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "is_sufficient": True,
        "search_retry_count": 0,
        "answer": "",
        "citations": [],
        "grounding_report": {},
        "progress_log": []
    }
    
    last_state = initial_state
    
    run_config = {
        "configurable": {"session_id": session_id},
        "tags": ["ecommerce-rag", f"session:{session_id}"],
        "metadata": {"session_id": session_id, "query": query}
    }
    
    try:
        async for event in graph.astream_events(initial_state, version="v2", config=run_config):
            event_type = event["event"]
            name = event["name"]
            data = event.get("data", {})
            
            if event_type == "on_chain_end" and name in [
                "guardrail", "decision", "search", "index", "retrieve", "crag_grader", "self_rag_grader"
            ]:
                state_updates = data.get("output", {})
                last_state = {**last_state, **state_updates}
                
                status = f"Executing: {name}..."
                if name == "guardrail":
                    if not last_state.get("is_ecommerce", True):
                        status = "Guardrail: Non-ecommerce intent"
                    else:
                        status = "Shopping intent verified..."
                elif name == "decision":
                    status = "Deciding path..."
                elif name == "search":
                    status = "Searching web..."
                elif name == "index":
                    status = "Indexing documents..."
                elif name == "retrieve":
                    status = "Retrieving context..."
                elif name == "crag_grader":
                    status = "Evaluating relevance (CRAG)..."
                elif name == "self_rag_grader":
                    status = "Verifying facts (Self-RAG)..."
                    
                yield {
                    "status": status,
                    "answer": last_state["answer"],
                    "progress_log": last_state["progress_log"],
                    "citations": last_state["citations"]
                }
                
            elif event_type == "on_chain_stream" and name == "synthesis":
                chunk = data.get("chunk", {})
                last_state = {**last_state, **chunk}
                
                yield {
                    "status": "Generating answer...",
                    "answer": last_state["answer"],
                    "progress_log": last_state["progress_log"],
                    "citations": last_state["citations"]
                }
                
        logger.info("LangGraph workflow execution completed successfully")
        yield {
            "status": "Complete",
            "answer": last_state["answer"],
            "progress_log": last_state["progress_log"],
            "citations": last_state["citations"]
        }
        
    except Exception as e:
        logger.exception("An unhandled exception occurred in the LangGraph orchestrator pipeline:")
        yield {
            "status": "Error",
            "error_message": "An unexpected error occurred in our systems. Please check inputs or try again later.",
            "answer": f"⚠️ System Error: {str(e)[:150]}",
            "progress_log": last_state.get("progress_log", []) + [f"❌ Pipeline failed: {str(e)[:50]}"],
            "citations": last_state.get("citations", [])
        }

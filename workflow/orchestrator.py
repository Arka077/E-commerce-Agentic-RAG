import asyncio
from typing import AsyncGenerator, Dict, Any, List, Literal, TypedDict
from langgraph.graph import StateGraph, START, END

from core.logger import logger, set_session_id
from agents.decision import run_decision_agent
from agents.synthesis import run_synthesis_agent
from workflow.services import SearchService, IndexService, RetrievalService

# 1. Define State structure
class GraphState(TypedDict):
    query: str
    chat_history: List[Any]
    session_id: str
    needs_search: bool
    cleaned_documents: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    progress_log: List[str]

# 2. Define Node Functions
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
    result = await search_service.execute(state["query"])
    
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
    await index_service.index_documents(state.get("cleaned_documents", []))
    
    return {
        "progress_log": progress_log
    }

async def retrieval_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing retrieval_node")
    progress_log = list(state.get("progress_log", []))
    progress_log.append("Retrieving knowledge")
    
    retrieval_service = RetrievalService()
    result = await retrieval_service.retrieve(state["query"])
    
    return {
        "retrieved_chunks": result["retrieved_chunks"],
        "citations": result["citations"],
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
    
    partial_answer = ""
    async for chunk in run_synthesis_agent(state["query"], state["retrieved_chunks"], state["chat_history"]):
        partial_answer += chunk
        yield {
            "answer": partial_answer
        }

# 3. Define Conditional Routing Edge Logic
def route_decision(state: GraphState) -> Literal["search", "retrieve"]:
    if state["needs_search"]:
        return "search"
    return "retrieve"

# 4. Construct Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("decision", decision_node)
workflow.add_node("search", search_node)
workflow.add_node("index", index_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("synthesis", synthesis_node)

# Set Graph entrypoint
workflow.add_edge(START, "decision")

# Set Conditional Edge
workflow.add_conditional_edges(
    "decision",
    route_decision,
    {
        "search": "search",
        "retrieve": "retrieve"
    }
)

# Connect Sibling Nodes
workflow.add_edge("search", "index")
workflow.add_edge("index", "retrieve")
workflow.add_edge("retrieve", "synthesis")
workflow.add_edge("synthesis", END)

# Compile Graph
graph = workflow.compile()


# 5. Graph Orchestrator runner
async def run_ecommerce_workflow(
    query: str, 
    chat_history: List, 
    session_id: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Runs the compiled LangGraph state machine.
    Yields intermediate state logs, and streams the final synthesis output.
    """
    set_session_id(session_id)
    logger.info(f"Starting LangGraph workflow for query: '{query}'")
    
    # Initialize baseline state
    initial_state: GraphState = {
        "query": query,
        "chat_history": chat_history,
        "session_id": session_id,
        "needs_search": False,
        "cleaned_documents": [],
        "search_results": [],
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "progress_log": []
    }
    
    last_state = initial_state
    
    try:
        # Step 1: Stream nodes execution using astream_events
        async for event in graph.astream_events(initial_state, version="v2"):
            event_type = event["event"]
            name = event["name"]
            data = event.get("data", {})
            
            # Handle completion of standard (non-streaming) nodes
            if event_type == "on_chain_end" and name in ["decision", "search", "index", "retrieve"]:
                state_updates = data.get("output", {})
                last_state = {**last_state, **state_updates}
                
                # Format status message for UI
                status = f"Executing: {name}..."
                if name == "decision":
                    status = "Deciding path..."
                elif name == "search":
                    status = "Searching web..."
                elif name == "index":
                    status = "Indexing documents..."
                elif name == "retrieve":
                    status = "Retrieving context..."
                    
                yield {
                    "status": status,
                    "answer": last_state["answer"],
                    "progress_log": last_state["progress_log"],
                    "citations": last_state["citations"]
                }
                
            # Handle intermediate chunks from the streaming synthesis node
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

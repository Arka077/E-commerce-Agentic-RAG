import uuid
from typing import List, Tuple
import gradio as gr
from core.logger import logger
from workflow.orchestrator import run_ecommerce_workflow

async def chat_interface(message: str, chat_history: List, session_id: str):
    """Async chat handler that yields streaming updates to the Gradio chatbot UI."""
    if not message or message.strip() == "":
        yield "", chat_history, "🟢 **Status:** Ready"
        return
        
    # Append the user message to history
    chat_history = list(chat_history)
    chat_history.append({"role": "user", "content": message})
    
    # Pre-append the assistant placeholder message
    chat_history.append({"role": "assistant", "content": "⏳ Initializing RAG pipeline..."})
    yield "", chat_history, "⏳ **Status:** Initializing RAG pipeline..."
    
    try:
        # Run workflow and consume async generator
        async for step in run_ecommerce_workflow(message, chat_history[:-1], session_id):
            status = step["status"]
            answer = step["answer"]
            progress_log = step.get("progress_log", [])
            citations = step.get("citations", [])
            
            # 1. Format simplified, user-friendly status outside of chatbot
            status_text = f"⚙️ **Status:** {status}"
            
            # 2. Filter process log to keep only high-level user-meaningful events
            filtered_logs = []
            for log in progress_log:
                # Do not display internal implementation details
                if any(x in log for x in [
                    "parent &", "child chunks", 
                    "Indexing", "Indexed", 
                    "Retrieving", "Retrieved", 
                    "Curator kept", "unique URLs",
                    "raw results", "Qdrant"
                ]):
                    continue
                
                # Format to user-friendly messages
                cleaned_log = log
                if "Decision:" in log:
                    cleaned_log = "🤔 Deciding whether fresh information is needed"
                elif "Generating search" in log or "search queries" in log:
                    cleaned_log = "🔍 Formulating search queries"
                elif "Searching" in log or "Tavily" in log:
                    cleaned_log = "🔍 Searching the web for the latest product details"
                elif "Scraping" in log or "Scraped:" in log:
                    cleaned_log = "📥 Scraping product specs and pricing sources"
                elif "Synthesizing" in log or "Generating answer" in log:
                    cleaned_log = "💡 Generating the final answer"
                    
                if cleaned_log not in filtered_logs:
                    filtered_logs.append(cleaned_log)
            
            logs_markdown = "\n".join(f"✓ {log}" for log in filtered_logs) if filtered_logs else "*No logs yet.*"
            
            # 3. Stream ONLY the answer in chatbot. Citations are appended only after Complete.
            if status == "Complete":
                status_text = "🟢 **Status:** Complete"
                formatted_response = answer
                if citations:
                    formatted_response += "\n\n---\n\n**Sources:**\n"
                    for cite in citations[:3]:
                        formatted_response += f"- [{cite['source']}]({cite['url']})\n"
            elif status == "Error":
                status_text = "🔴 **Status:** Error"
                error_msg = step.get("error_message", "Unknown pipeline error")
                formatted_response = f"{answer}\n\n⚠️ **Pipeline Error:** {error_msg}"
            else:
                formatted_response = answer
                
            chat_history[-1] = {"role": "assistant", "content": formatted_response}
            
            # Yield in real-time (logs stream immediately)
            yield "", chat_history, status_text
                
    except Exception as e:
        logger.exception("Exception in UI chat interface stream:")
        chat_history[-1] = {
            "role": "assistant",
            "content": f"⚠️ **System Error:** An unexpected UI error occurred. Details: {str(e)[:150]}"
        }
        yield "", chat_history, "🔴 **Status:** Error"

# Build Gradio interface
with gr.Blocks(title="🛍️ E-commerce Chat RAG") as demo:
    # Session-specific state storage
    session_id_state = gr.State(lambda: str(uuid.uuid4())[:8])
    
    gr.Markdown("""
    # 🛍️ E-commerce Chat RAG System
    
    Ask any question about e-commerce products and prices!
    
    *Agents:* 🔍 Search → 📄 Chunk → 🗂️ Index → 🔎 Retrieve → 💡 Synthesize
    """)
    
    # Status display outside of chatbot
    status_display = gr.Markdown(value="🟢 **Status:** Ready")
    
    chatbot = gr.Chatbot(
        label="💬 Chat History",
        height=600
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What is the current price of iPhone 14 in India?",
            lines=2,
            scale=4
        )
        submit_btn = gr.Button("Send 📤", variant="primary", scale=1)
        
   
        
    # Wire submit events
    submit_btn.click(
        fn=chat_interface,
        inputs=[msg, chatbot, session_id_state],
        outputs=[msg, chatbot, status_display],
        queue=True
    )
    
    msg.submit(
        fn=chat_interface,
        inputs=[msg, chatbot, session_id_state],
        outputs=[msg, chatbot, status_display],
        queue=True
    )

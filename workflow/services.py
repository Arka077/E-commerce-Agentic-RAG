import asyncio
import re
import json
from typing import List, Dict, Any, Optional

from core.logger import logger
from core.llm import llm_router, PRIMARY_MODEL
from agents.curator import run_curator_agent
from tools.web_search import search_ecommerce_products
from tools.scraper import scrape_product_content
from tools.text_splitter import chunk_content
from database.vector_store import VectorStore

class SearchService:
    async def rewrite_query_for_search(self, query: str, chat_history: Optional[List] = None) -> str:
        """
        Resolves coreferences and anaphoric pronouns ('the above products', 'these laptops')
        into concrete product search queries using conversation context.
        """
        if not chat_history or len(chat_history) <= 1:
            return query
            
        context = ""
        for msg in chat_history[-4:]:
            role = "User" if (isinstance(msg, dict) and msg.get("role") == "user") else "Assistant"
            text = msg.get("content", "") if isinstance(msg, dict) else (msg[0] or msg[1] or "")
            context += f"{role}: {text[:800]}\n"
            
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query rewriter for e-commerce search. "
                    "Analyze the conversation context and the user's latest query. "
                    "If the query contains pronouns or relative terms like 'above products', 'these', 'them', 'the first laptop', "
                    "rewrite it into an unambiguous, self-contained search query that explicitly includes the specific product names and models from the conversation. "
                    "If the query is already specific and standalone, return it as is. "
                    "Return ONLY the rewritten search query string."
                )
            },
            {
                "role": "user",
                "content": f"Conversation:\n{context}\nUser Query: {query}\n\nRewritten search query:"
            }
        ]
        try:
            res = await llm_router.acompletion(
                model=PRIMARY_MODEL,
                messages=messages,
                temperature=0.0
            )
            rewritten = res.choices[0].message.content.strip().strip('"').strip("'")
            if rewritten and len(rewritten) > 3:
                logger.info(f"Context-Aware Query Rewriting: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Failed to rewrite query: {e}")
        return query

    async def execute(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Search, curate, and scrape product details from web sources.
        """
        effective_query = await self.rewrite_query_for_search(query, chat_history) if chat_history else query
        logger.info(f"SearchService: Executing search for query: '{effective_query}' (Original: '{query}')")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate 3-4 search variations for an e-commerce query. "
                    "Focus on specific product names, prices, availability. Default region is INDIA, year is 2026. "
                    "Return ONLY a JSON array of strings."
                )
            },
            {
                "role": "user",
                "content": f"Query: {effective_query}\n\nReturn search variations as JSON array:"
            }
        ]
        
        search_queries = [effective_query]
        try:
            response = await llm_router.acompletion(
                model=PRIMARY_MODEL,
                messages=messages,
                temperature=0.2
            )
            raw_content = response.choices[0].message.content
            
            json_match = re.search(r'\[.*\]', raw_content, re.DOTALL)
            if json_match:
                search_queries.extend(json.loads(json_match.group()))
        except Exception as e:
            logger.error(f"SearchService: Failed to generate search variations: {str(e)}")
            search_queries.extend([f"{effective_query} price", f"{effective_query} buy online"])
            
        search_queries = list(dict.fromkeys(search_queries))[:3]
        logger.info(f"SearchService: Executing search queries: {search_queries}")
        
        search_tasks = [search_ecommerce_products(sq) for sq in search_queries]
        search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        raw_results = []
        for res in search_results_list:
            if isinstance(res, Exception):
                logger.error(f"SearchService: Search task encountered error: {str(res)}")
            elif isinstance(res, list):
                raw_results.extend(res)
                
        curated_docs = run_curator_agent(raw_results)
        curated_docs = curated_docs[:10]
        
        cleaned_documents = []
        scrape_tasks = [scrape_product_content(doc["url"]) for doc in curated_docs]
        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
            
        for doc, res in zip(curated_docs, scrape_results):
            if isinstance(res, Exception):
                logger.error(f"SearchService: Scraper task for {doc['url']} failed: {str(res)}")
            elif isinstance(res, dict):
                text = res.get("text", "")
                if text and not text.startswith("Error") and text != "Insufficient content":
                    cleaned_documents.append({
                        "url": doc["url"],
                        "content": text,
                        "source": doc.get("domain", ""),
                        "structured_data": res.get("metadata", {}).get("structured_data")
                    })
            elif isinstance(res, str) and not res.startswith("Error") and res != "Insufficient content":
                cleaned_documents.append({
                    "url": doc["url"],
                    "content": res,
                    "source": doc.get("domain", ""),
                    "structured_data": None
                })
            
        return {
            "search_results": curated_docs,
            "cleaned_documents": cleaned_documents
        }

class IndexService:
    async def index_documents(self, documents: List[Dict[str, Any]], session_id: Optional[str] = None) -> None:
        """
        Chunks the cleaned documents using True Parent-Document hierarchy and indexes in Qdrant with session isolation.
        """
        active_session = session_id or "global"
        logger.info(f"IndexService: Chunking and indexing {len(documents)} documents for session '{active_session}'")
        
        parent_chunks = []
        child_chunks = []
        for doc in documents:
            try:
                chunks_data = chunk_content(
                    content=doc["content"], 
                    url=doc["url"], 
                    session_id=active_session,
                    structured_data=doc.get("structured_data")
                )
                parent = chunks_data.get("parent")
                children = chunks_data.get("children", [])
                
                if parent:
                    parent_chunks.append(parent)
                child_chunks.extend(children)
            except Exception as e:
                logger.error(f"IndexService: Error chunking document {doc['url']}: {str(e)}")
                
        all_chunks = parent_chunks + child_chunks
        if all_chunks:
            try:
                db = VectorStore()
                await asyncio.to_thread(db.add_chunks, all_chunks, active_session)
            except Exception as e:
                logger.error(f"IndexService: Error indexing chunks to Qdrant: {str(e)}")

class RetrievalService:
    async def retrieve(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve relevant context chunks and citations from the vector store.
        """
        active_session = session_id or "global"
        logger.info(f"RetrievalService: Retrieving context for query: '{query}' (Session: {active_session})")
        try:
            db = VectorStore()
            retrieved_hits = await asyncio.to_thread(db.search, query, k=5, session_id=active_session)
            
            citations = []
            seen_urls = set()
            for hit in retrieved_hits:
                meta = hit.get("metadata", {})
                url = meta.get("source", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    citations.append({
                        "source": meta.get("source", "Unknown"),
                        "url": url
                    })
                
            return {
                "retrieved_chunks": retrieved_hits,
                "citations": citations[:4]
            }
        except Exception as e:
            logger.error(f"RetrievalService: Error during retrieval: {str(e)}")
            return {
                "retrieved_chunks": [],
                "citations": []
            }

import asyncio
import re
import json
from typing import List, Dict, Any
import aiohttp

from core.logger import logger
from core.llm import llm_router
from agents.curator import run_curator_agent
from tools.web_search import search_ecommerce_products
from tools.scraper import scrape_product_content
from tools.text_splitter import chunk_content
from database.vector_store import VectorStore

class SearchService:
    async def execute(self, query: str) -> Dict[str, Any]:
        """
        Executes the search and scraping workflow:
        1. Generates search variations using the LLM.
        2. Performs web search using Tavily.
        3. Curates/deduplicates URLs.
        4. Scrapes and cleans product page content.
        """
        logger.info(f"SearchService: Executing search for query: '{query}'")
        
        # 1. Search Query Generation using the LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate 3-4 search variations for an e-commerce query. "
                    "Focus on product names, prices, availability. Default region is INDIA, year is 2026. "
                    "Return ONLY a JSON array of strings."
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nReturn search variations as JSON array:"
            }
        ]
        
        search_queries = [query]
        try:
            response = await llm_router.acompletion(
                model="mistral-small",
                messages=messages,
                temperature=0.2
            )
            raw_content = response.choices[0].message.content
            
            # JSON parsing & extraction
            json_match = re.search(r'\[.*\]', raw_content, re.DOTALL)
            if json_match:
                search_queries.extend(json.loads(json_match.group()))
        except Exception as e:
            logger.error(f"SearchService: Failed to generate search variations: {str(e)}")
            # Search variation fallback logic
            search_queries.extend([f"{query} price", f"{query} buy online"])
            
        search_queries = list(dict.fromkeys(search_queries))[:3]
        logger.info(f"SearchService: Executing search queries: {search_queries}")
        
        # 2. Web searching (Tavily search concurrently)
        search_tasks = [search_ecommerce_products(sq) for sq in search_queries]
        search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        raw_results = []
        for res in search_results_list:
            if isinstance(res, Exception):
                logger.error(f"SearchService: Search task encountered error: {str(res)}")
            elif isinstance(res, list):
                raw_results.extend(res)
                
        # 3. URL curation / deduplication (Curator Agent)
        curated_docs = run_curator_agent(raw_results)
        curated_docs = curated_docs[:10]
        
        # 4. Web scraping (aiohttp session management & concurrency)
        cleaned_documents = []
        try:
            async with aiohttp.ClientSession() as session:
                scrape_tasks = [scrape_product_content(doc["url"], session) for doc in curated_docs]
                scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
                
            # Content cleaning / filtering
            for doc, content in zip(curated_docs, scrape_results):
                if isinstance(content, Exception):
                    logger.error(f"SearchService: Scraper task for {doc['url']} failed: {str(content)}")
                elif isinstance(content, str) and not content.startswith("Error") and content != "Insufficient content":
                    cleaned_documents.append({
                        "url": doc["url"],
                        "content": content,
                        "source": doc["domain"]
                    })
        except Exception as e:
            logger.error(f"SearchService: Scrape session failed: {str(e)}")
            
        return {
            "search_results": curated_docs,
            "cleaned_documents": cleaned_documents
        }

class IndexService:
    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Chunks the cleaned documents and indexes them in Qdrant.
        """
        logger.info(f"IndexService: Chunking and indexing {len(documents)} documents")
        
        # Document chunking & parent/child chunk generation
        parent_chunks = []
        child_chunks = []
        for doc in documents:
            try:
                chunks_data = chunk_content(doc["content"], doc["url"])
                parent = chunks_data.get("parent")
                children = chunks_data.get("children", [])
                
                if parent:
                    parent_chunks.append(parent)
                child_chunks.extend(children)
            except Exception as e:
                logger.error(f"IndexService: Error chunking document {doc['url']}: {str(e)}")
                
        all_chunks = parent_chunks + child_chunks
        
        # Qdrant indexing & Error handling
        if all_chunks:
            try:
                db = VectorStore()
                # Run the blocking database operation in a background thread to preserve async behavior
                await asyncio.to_thread(db.add_chunks, all_chunks)
            except Exception as e:
                logger.error(f"IndexService: Error indexing chunks to Qdrant: {str(e)}")
                # Handled internally to preserve system stability

class RetrievalService:
    async def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Performs semantic search in Qdrant, constructs citations, and returns the results.
        """
        logger.info(f"RetrievalService: Retrieving context for query: '{query}'")
        try:
            db = VectorStore()
            # Run the blocking database search in a background thread
            retrieved_hits = await asyncio.to_thread(db.search, query, k=5)
            
            # Citation construction
            citations = []
            for hit in retrieved_hits[:3]:
                meta = hit.get("metadata", {})
                citations.append({
                    "source": meta.get("source", "Unknown"),
                    "url": meta.get("source", "")
                })
                
            return {
                "retrieved_chunks": retrieved_hits,
                "citations": citations
            }
        except Exception as e:
            logger.error(f"RetrievalService: Error during retrieval: {str(e)}")
            return {
                "retrieved_chunks": [],
                "citations": []
            }

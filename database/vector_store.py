import os
import re
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    PayloadSchemaType
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from core.config import settings
from core.logger import logger

class VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            
            try:
                cls._instance.client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=5.0
                )
                cls._instance.client.get_collections()
                logger.info(f"Connected to online Qdrant cloud database at {settings.QDRANT_URL}")
            except Exception as e:
                logger.warning(
                    f"Remote Qdrant Cloud cluster unreachable ({e}). "
                    f"Falling back to local in-memory Qdrant instance."
                )
                cls._instance.client = QdrantClient(location=":memory:")
                
            cls._instance.collection_name = "ecommerce_chunks"
            cls._instance._ensure_collection()
            
            logger.info(f"Initializing SentenceTransformer embedding model on CPU ({settings.EMBEDDING_MODEL_NAME})")
            cls._instance.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            cls._instance.embedding_model.to('cpu')
            
            logger.info(f"Initializing CrossEncoder reranker model on CPU ({settings.RERANKER_MODEL_NAME})")
            cls._instance.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
        return cls._instance

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            
            # Ensure payload index on session_id for fast isolated filtering
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="session_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass  # Index might already exist
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {str(e)}")
            raise

    def add_chunks(self, chunks: List[Dict[str, Any]], session_id: Optional[str] = None):
        """
        Add chunks to Qdrant collection with session isolation and parent-document linkage.
        """
        if not chunks:
            return
            
        active_session = session_id or "global"
        logger.info(f"Generating embeddings and indexing {len(chunks)} chunks in Qdrant (Session: {active_session})")
        texts = [chunk["document"] for chunk in chunks]
        
        # Generate dense embeddings
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        points = []
        for i, chunk in enumerate(chunks):
            point_id = chunk["id"]
            if len(point_id) < 36:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
                
            chunk_session = chunk.get("session_id") or active_session
            payload = {
                "document": chunk["document"],
                "raw_text": chunk.get("raw_text", chunk["document"]),
                "source": chunk.get("source", ""),
                "chunk_type": chunk.get("chunk_type", "child"),
                "created_at": chunk.get("created_at", ""),
                "scraped_timestamp": chunk.get("scraped_timestamp", ""),
                "sequence": chunk.get("sequence", None),
                "parent_id": chunk.get("parent_id", None),
                "parent_content": chunk.get("parent_content", chunk["document"]),
                "session_id": chunk_session,
                "structured_data": chunk.get("structured_data", {})
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Successfully indexed {len(points)} chunks in Qdrant for session '{active_session}'")

    def search(
        self, 
        query_text: str, 
        k: int = 5, 
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with dense vectors, BM25, Cross-Encoder reranking, and parent document resolution.
        """
        logger.info(f"Hybrid SOTA Search for: '{query_text[:50]}' (Session: {session_id or 'all'})")
        
        try:
            query_vector = self.embedding_model.encode(
                query_text, 
                convert_to_numpy=True, 
                show_progress_bar=False
            ).tolist()
            
            candidate_pool_size = max(k * 4, 15)
            
            query_filter = None
            if session_id and session_id != "global":
                query_filter = Filter(must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                ])
            
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=candidate_pool_size
            )
            
            hits = response.points
            
            if not hits and query_filter is not None:
                logger.info("No session-specific chunks found; falling back to broader corpus")
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=candidate_pool_size
                )
                hits = response.points
                
            if not hits:
                return []
                
            candidates = []
            for hit in hits:
                payload = hit.payload or {}
                candidates.append({
                    "id": hit.id,
                    "document": payload.get("document", ""),
                    "raw_text": payload.get("raw_text", payload.get("document", "")),
                    "parent_content": payload.get("parent_content", payload.get("document", "")),
                    "parent_id": payload.get("parent_id") or hit.id,
                    "metadata": {
                        "source": payload.get("source", ""),
                        "chunk_type": payload.get("chunk_type", ""),
                        "created_at": payload.get("created_at", ""),
                        "scraped_timestamp": payload.get("scraped_timestamp", ""),
                        "sequence": payload.get("sequence", None),
                        "parent_id": payload.get("parent_id", None),
                        "session_id": payload.get("session_id", ""),
                        "structured_data": payload.get("structured_data", {})
                    },
                    "dense_score": float(hit.score)
                })
                
            tokenized_corpus = [
                re.findall(r'\w+', (c["document"] + " " + str(c["metadata"].get("structured_data", ""))).lower())
                for c in candidates
            ]
            tokenized_query = re.findall(r'\w+', query_text.lower())
            
            if tokenized_corpus and tokenized_query:
                bm25 = BM25Okapi(tokenized_corpus)
                bm25_scores = bm25.get_scores(tokenized_query)
                
                bm25_ranked_indices = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)
                bm25_ranks = {candidates[idx]["id"]: rank for rank, idx in enumerate(bm25_ranked_indices)}
                
                dense_ranked_indices = sorted(range(len(candidates)), key=lambda i: candidates[i]["dense_score"], reverse=True)
                dense_ranks = {candidates[idx]["id"]: rank for rank, idx in enumerate(dense_ranked_indices)}
                
                for c in candidates:
                    cid = c["id"]
                    d_rank = dense_ranks.get(cid, len(candidates))
                    s_rank = bm25_ranks.get(cid, len(candidates))
                    c["rrf_score"] = (1.0 / (60.0 + d_rank)) + (1.0 / (60.0 + s_rank))
                    
                candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
                logger.info(f"Applied BM25 + Dense RRF fusion over {len(candidates)} candidates")

            if self.reranker:
                try:
                    pairs = [(query_text, c["document"][:800]) for c in candidates]
                    rerank_scores = self.reranker.predict(pairs)
                    for i, score in enumerate(rerank_scores):
                        candidates[i]["rerank_score"] = float(score)
                    candidates.sort(key=lambda x: x.get("rerank_score", -999.0), reverse=True)
                    logger.info("Successfully re-ranked candidates using Cross-Encoder")
                except Exception as e:
                    logger.error(f"Cross-encoder reranking error: {e}")

            seen_parent_ids = set()
            parent_results = []
            
            for c in candidates:
                parent_id = c["parent_id"]
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)
                
                full_parent_doc = c["parent_content"] or c["document"]
                
                parent_results.append({
                    "id": c["id"],
                    "parent_id": parent_id,
                    "document": full_parent_doc,
                    "child_match_snippet": c["document"][:300],
                    "metadata": c["metadata"],
                    "score": c.get("rerank_score", c.get("rrf_score", c["dense_score"]))
                })
                
                if len(parent_results) >= k:
                    break
                    
            logger.info(f"Retrieved and resolved {len(parent_results)} parent documents after SOTA re-ranking")
            return parent_results

        except Exception as e:
            logger.error(f"Qdrant SOTA search error: {str(e)}")
            return []

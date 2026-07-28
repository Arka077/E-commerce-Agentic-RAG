import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from core.config import settings
from core.logger import logger

class VectorStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            
            # Connect only to the online Qdrant cloud database
            cls._instance.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
            logger.info(f"Connected to online Qdrant database at {settings.QDRANT_URL}")
                
            cls._instance.collection_name = "ecommerce_chunks"
            cls._instance._ensure_collection()
            logger.info("Initializing SentenceTransformer embedding model on CPU")
            cls._instance.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            cls._instance.embedding_model.to('cpu')
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
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {str(e)}")
            raise

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add chunks to Qdrant collection"""
        if not chunks:
            return
            
        logger.info(f"Generating embeddings and indexing {len(chunks)} chunks in Qdrant")
        texts = [chunk["document"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        points = []
        for i, chunk in enumerate(chunks):
            # Ensure ID is a valid UUID
            point_id = chunk["id"]
            if len(point_id) < 36:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
                
            payload = {
                "document": chunk["document"],
                "source": chunk.get("source", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "created_at": chunk.get("created_at", ""),
                "scraped_timestamp": chunk.get("scraped_timestamp", ""),
                "sequence": chunk.get("sequence", None),
                "parent_id": chunk.get("parent_id", None)
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
        logger.info(f"Successfully indexed {len(points)} chunks in Qdrant")

    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top k similar chunks from Qdrant Cloud"""
        logger.info(f"Searching Qdrant Cloud for: '{query_text[:50]}'")
        
        try:
            query_vector = self.embedding_model.encode(query_text, convert_to_numpy=True, show_progress_bar=False).tolist()
            
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k
            )
            
            results = []
            for hit in response.points:
                results.append({
                    "id": hit.id,
                    "document": hit.payload.get("document", ""),
                    "metadata": {
                        "source": hit.payload.get("source", ""),
                        "chunk_type": hit.payload.get("chunk_type", ""),
                        "created_at": hit.payload.get("created_at", ""),
                        "scraped_timestamp": hit.payload.get("scraped_timestamp", ""),
                        "sequence": hit.payload.get("sequence", None),
                        "parent_id": hit.payload.get("parent_id", None),
                    },
                    "distance": float(hit.score)
                })
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {str(e)}")
            return []

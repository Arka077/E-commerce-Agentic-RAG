import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Embedding & Reranker config
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM Models
    GEMINI_MODEL: str = "gemini/gemini-3.6-flash"
    GEMINI_LITE_MODEL: str = "gemini/gemini-3.5-flash-lite"
    
    # LangSmith Tracing & Observability
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "ecommerce-agentic-rag"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    
    # Qdrant Vector DB cloud connection (required)
    QDRANT_URL: str
    QDRANT_API_KEY: str
    
    # Tavily Config
    TAVILY_MAX_RESULTS: int = 15
    
    # Resilience Config
    TIMEOUT_SECONDS: float = 120.0
    MAX_RETRIES: int = 5
    CIRCUIT_BREAKER_FAIL_THRESHOLD: int = 5
    CIRCUIT_BREAKER_COOLDOWN_SECONDS: int = 60
    
    # API key lists populated dynamically by model validator
    GEMINI_API_KEYS: List[str] = Field(default_factory=list)
    TAVILY_API_KEYS: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def assemble_api_keys(cls, values: dict) -> dict:
        gemini_keys = []
        tavily_keys = []
        
        # Helper to fetch values from case-insensitive dict or os.environ
        def get_value(key_name: str) -> str:
            if key_name in values:
                return values[key_name]
            if key_name.lower() in values:
                return values[key_name.lower()]
            return os.environ.get(key_name) or os.environ.get(key_name.lower()) or ""
            
        # Check standard keys first
        primary_gemini = get_value("GEMINI_API_KEY")
        if primary_gemini:
            gemini_keys.append(primary_gemini)
            
        primary_tavily = get_value("TAVILY_API_KEY")
        if primary_tavily:
            tavily_keys.append(primary_tavily)
            
        # Scan for rotated Gemini keys starting from index 2
        i = 2
        while True:
            key = f"GEMINI_API_KEY_{i}"
            val = get_value(key)
            if not val:
                break
            gemini_keys.append(val)
            i += 1
            
        i = 2
        while True:
            key = f"TAVILY_API_KEY_{i}"
            val = get_value(key)
            if not val:
                break
            tavily_keys.append(val)
            i += 1
            
        values["GEMINI_API_KEYS"] = gemini_keys
        values["TAVILY_API_KEYS"] = tavily_keys
        
        # Configure HuggingFace token if present
        hf_token = get_value("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Configure LangSmith environment if key is available
        langchain_api_key = get_value("LANGCHAIN_API_KEY")
        if langchain_api_key:
            values["LANGCHAIN_API_KEY"] = langchain_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = get_value("LANGCHAIN_TRACING_V2") or "true"
            os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = get_value("LANGCHAIN_PROJECT") or "ecommerce-agentic-rag"
            os.environ["LANGCHAIN_ENDPOINT"] = get_value("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
        
        # Fail fast: validate key existence
        if not gemini_keys:
            raise ValueError("❌ Missing required LLM configuration: Please supply GEMINI_API_KEY")
        if not tavily_keys:
            raise ValueError("❌ Missing required configuration: TAVILY_API_KEY (or TAVILY_API_KEYS)")
            
        return values

settings = Settings()

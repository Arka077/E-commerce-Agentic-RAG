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
    
    # Embedding config
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM Models
    MISTRAL_MODEL_SMALL: str = "mistral-small-latest"
    MISTRAL_MODEL_LARGE: str = "mistral-large-latest"
    
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
    MISTRAL_API_KEYS: List[str] = Field(default_factory=list)
    TAVILY_API_KEYS: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def assemble_api_keys(cls, values: dict) -> dict:
        mistral_keys = []
        tavily_keys = []
        
        # Helper to fetch values from case-insensitive dict or os.environ
        def get_value(key_name: str) -> str:
            if key_name in values:
                return values[key_name]
            if key_name.lower() in values:
                return values[key_name.lower()]
            return os.environ.get(key_name) or os.environ.get(key_name.lower()) or ""
            
        # Check standard keys first
        primary_mistral = get_value("MISTRAL_API_KEY")
        if primary_mistral:
            mistral_keys.append(primary_mistral)
            
        primary_tavily = get_value("TAVILY_API_KEY")
        if primary_tavily:
            tavily_keys.append(primary_tavily)
            
        # Scan for rotated keys starting from index 2
        i = 2
        while True:
            key = f"MISTRAL_API_KEY_{i}"
            val = get_value(key)
            if not val:
                break
            mistral_keys.append(val)
            i += 1
            
        i = 2
        while True:
            key = f"TAVILY_API_KEY_{i}"
            val = get_value(key)
            if not val:
                break
            tavily_keys.append(val)
            i += 1
            
        values["MISTRAL_API_KEYS"] = mistral_keys
        values["TAVILY_API_KEYS"] = tavily_keys
        
        # Fail fast: validate key existence
        if not mistral_keys:
            raise ValueError("❌ Missing required configuration: MISTRAL_API_KEY (or MISTRAL_API_KEYS)")
        if not tavily_keys:
            raise ValueError("❌ Missing required configuration: TAVILY_API_KEY (or TAVILY_API_KEYS)")
            
        return values

settings = Settings()

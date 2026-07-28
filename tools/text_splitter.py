import uuid
from datetime import datetime
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_uuid() -> str:
    """Generate unique ID (full UUID format for Qdrant compatibility)"""
    return str(uuid.uuid4())

def chunk_content(content: str, url: str) -> Dict[str, Any]:
    """Split content into parent and child chunks with temporal context"""
    parent_id = generate_uuid()
    scraped_timestamp = datetime.now().isoformat()
    
    parent_chunk = {
        "id": parent_id,
        "document": content[:4000],
        "source": url,
        "chunk_type": "parent",
        "created_at": datetime.now().isoformat(),
        "scraped_timestamp": scraped_timestamp
    }
    
    # Create child chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    child_texts = splitter.split_text(content)
    child_chunks = []
    
    for i, child_text in enumerate(child_texts[:5]):  # Limit to 5 children
        child_id = generate_uuid()
        child_chunks.append({
            "id": child_id,
            "document": child_text,
            "source": url,
            "chunk_type": "child",
            "parent_id": parent_id,
            "sequence": i,
            "created_at": datetime.now().isoformat(),
            "scraped_timestamp": scraped_timestamp
        })
        
    return {
        "parent": parent_chunk,
        "children": child_chunks,
        "count": len(child_chunks)
    }

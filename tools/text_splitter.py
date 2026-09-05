import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_uuid() -> str:
    """Generate unique UUID string."""
    return str(uuid.uuid4())

def chunk_content(
    content: str, 
    url: str, 
    session_id: Optional[str] = None, 
    structured_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Split text into parent and child chunks for hierarchical retrieval.
    """
    parent_id = generate_uuid()
    scraped_timestamp = datetime.now().isoformat()
    domain = urlparse(url).netloc or "web"
    
    parent_text = content[:4500]
    parent_chunk = {
        "id": parent_id,
        "document": parent_text,
        "source": url,
        "chunk_type": "parent",
        "session_id": session_id or "global",
        "created_at": datetime.now().isoformat(),
        "scraped_timestamp": scraped_timestamp,
        "structured_data": structured_data or {}
    }
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    raw_child_texts = splitter.split_text(content)
    child_chunks = []
    
    prod_name = structured_data.get("name") if structured_data else None
    context_prefix = f"[{domain}{f' - {prod_name[:40]}' if prod_name else ''}]: "
    
    for i, child_text in enumerate(raw_child_texts[:8]):
        child_id = generate_uuid()
        contextualized_text = f"{context_prefix}{child_text}"
        
        child_chunks.append({
            "id": child_id,
            "document": contextualized_text,
            "raw_text": child_text,
            "source": url,
            "chunk_type": "child",
            "parent_id": parent_id,
            "parent_content": parent_text,
            "sequence": i,
            "session_id": session_id or "global",
            "created_at": datetime.now().isoformat(),
            "scraped_timestamp": scraped_timestamp,
            "structured_data": structured_data or {}
        })
        
    return {
        "parent": parent_chunk,
        "children": child_chunks,
        "count": len(child_chunks)
    }

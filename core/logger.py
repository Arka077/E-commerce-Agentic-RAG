import logging
import sys
from contextvars import ContextVar
from typing import Optional

# Thread-safe/Async-safe Context variable to hold session/chat ID
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)

class SessionIDFilter(logging.Filter):
    """Filter that injects the active session_id into logging records."""
    def filter(self, record):
        record.session_id = session_id_var.get() or "system"
        return True

# Reconfigure stdout/stderr on Windows to avoid UnicodeEncodeError for ₹, emojis, etc.
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def setup_logger(name: str = "ecommerce_chatbot") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Stdout stream handler with utf-8 fallback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Custom format to ensure structured logs
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [Session:%(session_id)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
    )
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SessionIDFilter())
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

def set_session_id(session_id: str) -> None:
    """Set the session ID for structured logging in the current task/context."""
    session_id_var.set(session_id)

def clear_session_id() -> None:
    """Clear the session ID for structured logging in the current context."""
    session_id_var.set(None)

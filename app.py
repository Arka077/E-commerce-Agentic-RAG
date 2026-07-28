import os
import asyncio
import warnings
from dotenv import load_dotenv

# Suppress torch FutureWarning on HF Spaces
warnings.filterwarnings('ignore', category=FutureWarning, module='.*torch.*')

# Load environment variables
load_dotenv()

# Import config and logger to verify environment variables early
from core.config import settings
from core.logger import logger
from ui.app import demo

if __name__ == "__main__":
    logger.info("Starting E-commerce RAG Chatbot...")
    try:
        demo.launch(share=False, show_error=True)
    finally:
        # Clean up event loop on shutdown
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
            loop.close()
        except Exception:
            pass

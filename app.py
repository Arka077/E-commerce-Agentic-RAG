import os
import sys
import subprocess
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore', category=FutureWarning, module='.*torch.*')

load_dotenv()

from core.config import settings
from core.logger import logger
from ui.app import demo

def ensure_playwright_browsers():
    try:
        logger.info("Ensuring Playwright browser binaries are ready...")
        res = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=180
        )
        if res.returncode == 0:
            logger.info("Playwright Chromium is installed and ready.")
        else:
            logger.warning(f"Playwright installation warning: {res.stderr[:200]}")
    except Exception as e:
        logger.warning(f"Playwright auto-install check encountered: {e}")

if __name__ == "__main__":
    ensure_playwright_browsers()
    logger.info("Starting E-commerce RAG Chatbot...")
    demo.launch(share=False, show_error=True)

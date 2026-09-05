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
            [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
            capture_output=True,
            text=True,
            timeout=180
        )
        if res.returncode != 0:
            res = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=180
            )
        if res.returncode == 0:
            logger.info("Playwright Chromium is installed and ready.")
        else:
            logger.warning(f"Playwright installation note: {res.stderr[:200]}")
    except Exception as e:
        logger.warning(f"Playwright auto-install check encountered: {e}")

# Workaround for Python 3.13 asyncio BaseEventLoop.__del__ invalid fd warning
try:
    import asyncio.selector_events
    _orig_close_self_pipe = asyncio.selector_events.BaseSelectorEventLoop._close_self_pipe
    def _safe_close_self_pipe(self):
        try:
            if getattr(self, "_ssock", None) is not None and self._ssock.fileno() != -1:
                _orig_close_self_pipe(self)
            elif getattr(self, "_ssock", None) is not None:
                self._ssock.close()
                self._ssock = None
        except Exception:
            pass
    asyncio.selector_events.BaseSelectorEventLoop._close_self_pipe = _safe_close_self_pipe
except Exception:
    pass

if __name__ == "__main__":
    ensure_playwright_browsers()
    logger.info("Starting E-commerce RAG Chatbot...")
    demo.launch(share=False, show_error=True, ssr_mode=False)

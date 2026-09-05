import time
from litellm import Router
from core.config import settings
from core.logger import logger

model_list = []

for key in settings.GEMINI_API_KEYS:
    model_list.append({
        "model_name": "gemini-flash",
        "litellm_params": {
            "model": settings.GEMINI_MODEL,
            "api_key": key
        }
    })
    model_list.append({
        "model_name": "gemini-lite",
        "litellm_params": {
            "model": settings.GEMINI_LITE_MODEL,
            "api_key": key
        }
    })

logger.info(f"Initialized LiteLLM Router with {len(settings.GEMINI_API_KEYS)} Gemini deployments ({settings.GEMINI_MODEL} & {settings.GEMINI_LITE_MODEL})")

PRIMARY_MODEL = "gemini-flash"
GUARDRAIL_MODEL = "gemini-lite"

llm_router = Router(
    model_list=model_list,
    num_retries=settings.MAX_RETRIES,
    timeout=settings.TIMEOUT_SECONDS,
    cooldown_time=settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS
)

class LLMCircuitBreaker:
    def __init__(self, name: str, fail_threshold: int = 5, cooldown_seconds: int = 60):
        self.name = name
        self.fail_threshold = fail_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning(f"LLM Circuit Breaker '{self.name}' failure: {self.failure_count}/{self.fail_threshold}")
        if self.failure_count >= self.fail_threshold:
            self.state = "OPEN"
            logger.error(f"LLM Circuit Breaker '{self.name}' is OPEN (cooldown: {self.cooldown_seconds}s)")

    def can_execute(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                logger.info(f"LLM Circuit Breaker '{self.name}' cooldown expired, entering HALF-OPEN state")
                return True
            return False
        return True

llm_breaker = LLMCircuitBreaker(
    name="GeminiLLM",
    fail_threshold=settings.CIRCUIT_BREAKER_FAIL_THRESHOLD,
    cooldown_seconds=settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS
)

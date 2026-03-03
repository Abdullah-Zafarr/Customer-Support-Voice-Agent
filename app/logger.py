import logging
import time
from fastapi import Request

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("voice_agent")

async def log_latency_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    logger.info(f"Method={request.method} Path={request.url.path} StatusCode={response.status_code} Latency={formatted_process_time}ms")
    return response

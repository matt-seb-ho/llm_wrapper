import hashlib
import logging

import orjson
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def stable_hash(obj: object) -> str:
    """md5 of a json‑serialisable object"""
    # dumped = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    dumped = orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)
    return hashlib.md5(dumped).hexdigest()


def transient_retry(*, attempts: int = 5, wait_min: int = 1, wait_max: int = 120):
    def _is_transient(exc: Exception):
        s = str(exc)
        return any(key in s for key in ("Rate limit", "Bad gateway", "JSONDecodeError"))

    return retry(
        wait=wait_exponential(min=wait_min, max=wait_max),
        stop=stop_after_attempt(attempts),
        retry=retry_if_exception_type(Exception) & retry_if_exception(_is_transient),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
        reraise=True,
    )

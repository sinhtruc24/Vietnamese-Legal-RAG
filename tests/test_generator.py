import time

from legalrag.generator import RateLimiter, _retry_delay


def test_retry_delay_parses_provider_hint():
    assert _retry_delay("Quota exceeded ... Please retry in 33.5558s.") == 33.5558
    assert _retry_delay("no hint here", default=7) == 7


def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(requests_per_minute=600)  # one call every 0.1 s
    start = time.monotonic()
    for _ in range(4):
        limiter.wait()
    assert time.monotonic() - start >= 0.29


def test_rate_limiter_disabled_is_free():
    limiter = RateLimiter(None)
    start = time.monotonic()
    for _ in range(100):
        limiter.wait()
    assert time.monotonic() - start < 0.05

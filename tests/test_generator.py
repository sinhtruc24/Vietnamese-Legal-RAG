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


def _fake_generator(monkeypatch, errors):
    """Generator whose API call raises the given errors in order, then succeeds."""
    openai = __import__("pytest").importorskip("openai")
    try:  # openai >= 3 is built on httpx2
        import httpx2 as httpx
    except ImportError:
        import httpx

    from legalrag import generator as gen

    monkeypatch.setattr(gen.time, "sleep", lambda s: None)
    g = gen.OpenAICompatibleGenerator("http://localhost:1/v1", "m")
    queue = list(errors)

    def create(**kwargs):
        if queue:
            status, message = queue.pop(0)
            response = httpx.Response(status, request=httpx.Request("POST", "http://localhost:1"))
            cls = openai.RateLimitError if status == 429 else openai.InternalServerError
            raise cls(message, response=response, body=None)

        class Msg:
            content = "Trả lời [1]"

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()

    monkeypatch.setattr(g.client.chat.completions, "create", create)
    return g, gen


def test_generator_retries_overload_and_rate_limit(monkeypatch):
    g, _ = _fake_generator(monkeypatch, [(503, "overloaded"), (429, "Please retry in 3s. PerMinute")])
    assert g.generate([{"role": "user", "content": "hi"}]) == "Trả lời [1]"


def test_generator_stops_on_daily_quota(monkeypatch):
    import pytest

    g, gen = _fake_generator(monkeypatch, [(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    with pytest.raises(gen.QuotaExhausted):
        g.generate([{"role": "user", "content": "hi"}])

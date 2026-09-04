"""src/core/neo4j_client.py is the minimal legacy-compat shim forked from
ai-knowledge-graph-platform's 1553-line neo4j_client.py (only __init__/run/close/
get_neo4j — the surface the 8 ported src/graph/*.py modules actually call).
Verifies the singleton wiring and the with_retry-backed retry behavior on `run()`.
"""

import pytest
from neo4j.exceptions import TransientError

import src.core.neo4j_client as neo4j_client_module
from src.core.neo4j_client import get_neo4j


@pytest.fixture(autouse=True)
def _reset_singleton():
    neo4j_client_module._client = None
    yield
    neo4j_client_module._client = None


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff sleeps for real seconds by default — not needed to prove
    the retry logic fires and gives up, so collapse it to keep the test fast."""
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr("src.core.retry.asyncio.sleep", _instant_sleep)


class _FakeSession:
    def __init__(self, fail_times, succeed_rows):
        self._fail_times = fail_times
        self._succeed_rows = succeed_rows
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def run(self, cypher, parameters=None):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise TransientError("simulated transient failure")
        return _FakeResult(self._succeed_rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for row in self._rows:
            yield _FakeRecord(row)


class _FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


def test_get_neo4j_returns_a_singleton():
    client_a = get_neo4j()
    client_b = get_neo4j()
    assert client_a is client_b


@pytest.mark.asyncio
async def test_run_retries_on_transient_error_then_succeeds():
    client = get_neo4j()
    fake_session = _FakeSession(fail_times=2, succeed_rows=[{"n": 1}])
    client._driver = _FakeDriver(fake_session)

    rows = await client.run("RETURN 1 AS n")

    assert rows == [{"n": 1}]
    assert fake_session.calls == 3  # 2 failures + 1 success, within max_attempts=3...


@pytest.mark.asyncio
async def test_run_gives_up_after_max_attempts():
    client = get_neo4j()
    fake_session = _FakeSession(fail_times=10, succeed_rows=[])
    client._driver = _FakeDriver(fake_session)

    with pytest.raises(TransientError):
        await client.run("RETURN 1")

    assert fake_session.calls == 3  # with_retry default max_attempts=3


@pytest.mark.asyncio
async def test_run_does_not_retry_non_transient_exceptions():
    client = get_neo4j()

    class _RaisingSession(_FakeSession):
        async def run(self, cypher, parameters=None):
            self.calls += 1
            raise ValueError("not a retryable error")

    fake_session = _RaisingSession(fail_times=0, succeed_rows=[])
    client._driver = _FakeDriver(fake_session)

    with pytest.raises(ValueError):
        await client.run("RETURN 1")

    assert fake_session.calls == 1

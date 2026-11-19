"""Token usage ledger and its report sites (offline: mocked providers, no model, no database)."""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from graphrag.agents.ingestion_agent import IngestionAgent  # noqa: E402
from graphrag.core import llm_client  # noqa: E402
from graphrag.core.models import Document, Entity, IngestMessage, Relation  # noqa: E402
from graphrag.evaluation.ragas_evaluator import judge_usage_callbacks  # noqa: E402
from graphrag.observability.usage import (  # noqa: E402
    UsageLedger, current_ledger, kind_scope, ledger_scope, record_usage)
from scripts import benchmark_retrieval_ablation as ablation  # noqa: E402

DEEPSEEK = dict(provider="deepseek", model="deepseek-v4-flash")


def test_totals_sum_measured_calls_and_price_them_with_the_price_table():
    ledger = UsageLedger()
    ledger.add("generation", "deepseek", "deepseek-v4-flash", 1_000_000, 1_000_000)
    totals = ledger.totals("generation")
    assert (totals["prompt_tokens"], totals["completion_tokens"], totals["total_tokens"]) == (1_000_000, 1_000_000, 2_000_000)
    assert totals["n_calls"] == 1 and totals["n_calls_without_usage"] == 0
    assert totals["cost_usd"] == pytest.approx(0.75)                    # 0.15 + 0.60 per million, off-peak table


def test_unknown_usage_is_never_turned_into_zero():
    ledger = UsageLedger()
    ledger.add("generation", "deepseek", "deepseek-v4-flash", 100, 20)
    ledger.add("generation", "deepseek", "deepseek-v4-flash", None, None)      # the provider reported nothing
    ledger.add("rewrite", "deepseek", "deepseek-v4-flash", 50, None)           # half known is unknown
    generation = ledger.totals("generation")
    assert generation["n_calls"] == 2 and generation["n_calls_without_usage"] == 1
    assert generation["prompt_tokens"] is None and generation["total_tokens"] is None and generation["cost_usd"] is None
    assert ledger.totals("rewrite")["prompt_tokens"] is None
    empty = ledger.totals("extraction")                                       # no call seen: not a measured zero either
    assert empty["n_calls"] == 0 and empty["prompt_tokens"] is None and empty["cost_usd"] is None


def test_an_unpriced_model_gives_tokens_but_no_cost():
    ledger = UsageLedger()
    ledger.add("embedding", "openai", "text-embedding-3-large", 900, 0)
    totals = ledger.totals("embedding")
    assert totals["total_tokens"] == 900 and totals["cost_usd"] is None


def test_the_ledger_is_thread_safe():
    ledger = UsageLedger()

    def work():
        for _ in range(500):
            ledger.add("judge", "deepseek", "deepseek-v4-flash", 2, 1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    totals = ledger.totals("judge")
    assert totals["n_calls"] == 4000 and totals["prompt_tokens"] == 8000 and totals["completion_tokens"] == 4000


def test_recording_is_a_no_op_outside_a_scope_and_scopes_reset():
    record_usage(**DEEPSEEK, prompt_tokens=5, completion_tokens=5)               # nobody is measuring
    assert current_ledger() is None
    ledger = UsageLedger()
    with ledger_scope(ledger, "extraction"):
        assert current_ledger() is ledger
        record_usage(**DEEPSEEK, prompt_tokens=10, completion_tokens=2)
        with kind_scope("rewrite"):
            record_usage(**DEEPSEEK, prompt_tokens=1, completion_tokens=1)
        record_usage(**DEEPSEEK, prompt_tokens=10, completion_tokens=2, kind="embedding")
        record_usage(**DEEPSEEK, prompt_tokens=10, completion_tokens=2)          # back to the scope's kind
    assert current_ledger() is None
    assert ledger.totals("extraction")["prompt_tokens"] == 20 and ledger.totals("rewrite")["prompt_tokens"] == 1
    assert ledger.totals("embedding")["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_child_tasks_inherit_the_scope_and_a_concurrent_scope_does_not_leak():
    first, second = UsageLedger(), UsageLedger()

    async def call(tokens):
        await asyncio.sleep(0)
        record_usage(**DEEPSEEK, prompt_tokens=tokens, completion_tokens=0)

    async def run(ledger, tokens):
        with ledger_scope(ledger, "generation"):
            await asyncio.gather(*(call(tokens) for _ in range(5)))

    await asyncio.gather(run(first, 1), run(second, 10))
    assert first.totals("generation")["prompt_tokens"] == 5 and second.totals("generation")["prompt_tokens"] == 50


def test_openai_compatible_responses_are_reported_with_their_provider():
    ledger = UsageLedger()
    response = SimpleNamespace(model="deepseek-v4-flash", usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30))
    with ledger_scope(ledger, "generation"):
        llm_client._report_openai_compatible_usage(response, "deepseek")
        llm_client._report_openai_compatible_usage(SimpleNamespace(model="deepseek-v4-flash", usage=None), "deepseek")
    totals = ledger.totals("generation")
    assert totals["n_calls"] == 2 and totals["n_calls_without_usage"] == 1 and totals["prompt_tokens"] is None
    assert ledger.snapshot()[0]["prompt_tokens"] == 120 and ledger.snapshot()[0]["provider"] == "deepseek"
    llm_client._report_openai_compatible_usage(response, "deepseek")             # outside a scope: harmless


@pytest.mark.asyncio
async def test_the_embedder_reports_total_tokens_or_counts_the_call_as_unmeasured():
    embedder = llm_client.OpenAIEmbedder(api_key="test-key")
    data = [SimpleNamespace(embedding=[0.1, 0.2])]
    embedder._client = MagicMock()
    ledger = UsageLedger()
    with ledger_scope(ledger):
        embedder._client.embeddings.create.return_value = SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=7))
        assert await embedder.embed(["alpha"]) == [[0.1, 0.2]]
        embedder._client.embeddings.create.return_value = SimpleNamespace(data=data, usage=None)
        await embedder.embed(["beta"])
    totals = ledger.totals("embedding")
    assert totals["n_calls"] == 2 and totals["n_calls_without_usage"] == 1 and totals["prompt_tokens"] is None
    assert ledger.snapshot()[0]["prompt_tokens"] == 7 and ledger.snapshot()[0]["model"] == "text-embedding-3-large"


def test_the_judge_callback_reads_langchain_usage_and_does_not_touch_scores():
    ledger = UsageLedger()
    handler = judge_usage_callbacks(ledger, {"provider": "deepseek", "model": "deepseek-v4-flash"})[0]
    handler.on_llm_end(SimpleNamespace(llm_output={"token_usage": {"prompt_tokens": 400, "completion_tokens": 60}},
                                       generations=[]))
    message = SimpleNamespace(usage_metadata={"input_tokens": 300, "output_tokens": 40})
    handler.on_llm_end(SimpleNamespace(llm_output=None, generations=[[SimpleNamespace(message=message)]]))
    totals = ledger.totals("judge")
    assert totals["prompt_tokens"] == 700 and totals["completion_tokens"] == 100 and totals["n_calls"] == 2
    assert totals["cost_usd"] is not None
    handler.on_llm_end(SimpleNamespace(llm_output=None, generations=[]))             # nothing reported
    assert ledger.totals("judge")["n_calls_without_usage"] == 1 and ledger.totals("judge")["prompt_tokens"] is None


def _agent(extract, artifacts=None, embed_chunks_usage=False):
    from unittest.mock import AsyncMock

    agent = IngestionAgent.__new__(IngestionAgent)
    agent._embedder = AsyncMock()

    async def embed_chunks(chunks):
        if embed_chunks_usage:
            record_usage(provider="openai", model="text-embedding-3-large", kind="embedding",
                         prompt_tokens=11, completion_tokens=0)
        return chunks

    agent._embedder.embed_chunks = AsyncMock(side_effect=embed_chunks)
    agent._embedder.embed_texts = AsyncMock(side_effect=lambda texts: [[0.0] for _ in texts])
    agent._extractor = AsyncMock()
    agent._extractor.extract = AsyncMock(side_effect=extract)
    agent._artifact_extractor = AsyncMock() if artifacts else None
    if artifacts:
        agent._artifact_extractor.extract = AsyncMock(side_effect=artifacts)
    return agent


def _message() -> IngestMessage:
    text = "\n\n".join(f"Paragraph {i} with enough content to form its own chunk." * 20 for i in range(3))
    return IngestMessage(document=Document(filename="f.txt", source_path="f.txt", raw_text=text, tenant="automotive"))


@pytest.mark.asyncio
async def test_the_manifest_carries_measured_totals_per_stage_and_none_when_nothing_was_measured():
    entity = Entity(name="Acme", type="SUPPLIER", description="d")
    relation = Relation(source_entity_id=entity.id, target_entity_id=entity.id, relation="RELATES_TO")

    async def extract(chunk):
        record_usage(**DEEPSEEK, prompt_tokens=100, completion_tokens=20)            # kind comes from the scope
        return [entity], [relation]

    async def artifacts(chunk, names):
        record_usage(**DEEPSEEK, prompt_tokens=40, completion_tokens=10)
        return []

    result = await _agent(extract, artifacts, embed_chunks_usage=True).extract(_message())
    n = len(result["chunks"])
    stages = result["manifest"].stage_metrics
    assert stages["extraction"]["prompt_tokens"] == 100 * n and stages["extraction"]["completion_tokens"] == 20 * n
    assert stages["extraction"]["n_calls"] == n and stages["extraction"]["n_calls_without_usage"] == 0
    assert stages["extraction"]["cost_usd"] == pytest.approx((100 * n * 0.15 + 20 * n * 0.60) / 1_000_000)
    assert stages["artifacts"]["prompt_tokens"] == 40 * n and stages["artifacts"]["items"] == 0
    assert stages["embedding"]["prompt_tokens"] == 11 and stages["embedding"]["cost_usd"] is None   # unpriced model
    assert "manifest_never_invents_missing_cost" in stages["cost_status"]["value"]

    async def silent(chunk):
        return [entity], [relation]                                                   # no provider usage at all

    quiet = (await _agent(silent).extract(_message()))["manifest"].stage_metrics
    assert quiet["extraction"]["prompt_tokens"] is None and quiet["extraction"]["cost_usd"] is None
    assert quiet["extraction"]["n_calls"] == 0 and quiet["artifacts"]["total_tokens"] is None


def test_the_ablation_usage_summary_separates_priced_and_unpriced_kinds():
    ledger = UsageLedger()
    ledger.add("generation", "deepseek", "deepseek-v4-flash", 1_000_000, 0)
    ledger.add("embedding", "openai", "text-embedding-3-large", 500, 0)
    summary = ablation.usage_summary(ledger)
    assert summary["by_kind"]["generation"]["cost_usd"] == pytest.approx(0.15)
    assert summary["cost_usd_of_priced_kinds"] == pytest.approx(0.15) and summary["unpriced_kinds"] == ["embedding"]
    assert summary["total"]["cost_usd"] is None and len(summary["buckets"]) == 2
    nothing = ablation.usage_summary(UsageLedger())
    assert nothing["by_kind"] == {} and nothing["total"]["prompt_tokens"] is None
    assert nothing["cost_usd_of_priced_kinds"] is None and "off-peak" in nothing["basis"]

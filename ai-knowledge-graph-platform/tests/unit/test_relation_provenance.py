"""Per-document relation provenance: shape guards (live behaviour: tests/e2e/test_live_relation_provenance.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock

from graphrag.core.models import Relation
from graphrag.graph.neo4j_client import Neo4jClient, _doc_provenance_sets


def _client() -> Neo4jClient:
    c = Neo4jClient.__new__(Neo4jClient)
    c.run = AsyncMock(return_value=[])
    return c


def test_fragment_gives_each_document_its_own_slot():
    f = _doc_provenance_sets("$d", "$c", "$m", "$t")
    for prop in ("doc_chunk_ids", "doc_extraction_models", "doc_observed_at"):
        assert f"r.{prop} = [" in f
    # replaces only the matching document's slot, appends when new
    assert "prior_docs[i] <> $d" in f and "$d IN prior_docs THEN 0 ELSE 1" in f
    assert "$c" in f and "$m" in f and "$t" in f


async def test_single_and_batch_merges_both_write_the_provenance_slots():
    c = _client()
    rel = Relation(source_entity_id="a", target_entity_id="b", relation="OWNS", source_doc_id="d1",
                   source_chunk_id="c1", extraction_model="m1")
    await c.merge_relation(rel, "A", "ORG", "B", "ORG", tenant="t")
    q, kw = c.run.await_args_list[0].args[0], c.run.await_args_list[0].kwargs
    assert "r.doc_chunk_ids" in q and kw["chunk_id"] == "c1" and kw["extraction_model"] == "m1"

    c.run.reset_mock()
    await c.merge_relations_batch([{"src_name": "A", "src_type": "ORG", "tgt_name": "B", "tgt_type": "ORG",
                                    "relation": "OWNS", "source_doc_id": "d1", "chunk_id": "c1"}], tenant="t")
    assert "r.doc_chunk_ids" in c.run.await_args.args[0] and "row.chunk_id" in c.run.await_args.args[0]


async def test_reconcile_keeps_provenance_aligned_with_remaining_documents():
    c = _client()
    await c.reconcile_document_evidence("d1", tenant="t")
    joined = "\n".join(call.args[0] for call in c.run.await_args_list)
    assert "r.doc_chunk_ids = [i IN keep_idx" in joined
    assert "r.doc_extraction_models = [i IN keep_idx" in joined
    assert "r.doc_observed_at = [i IN keep_idx" in joined
    # undirected match => each edge is visited twice; the lists must come from a snapshot, not from r
    assert "all_chunks[i]" in joined and "r.doc_chunk_ids[i]" not in joined


async def test_extraction_records_the_model_that_actually_answered():
    """extraction_model is the provider-reported model for the call, not the configured label."""
    import json
    from types import SimpleNamespace
    from unittest.mock import patch

    from graphrag.core import llm_client
    from graphrag.core.models import Chunk
    from graphrag.ingestion.extractor import Extractor

    payload = json.dumps({"entities": [{"name": "Acme", "type": "ORG"}, {"name": "Bolt", "type": "ORG"}],
                          "relations": [{"source": "Acme", "target": "Bolt", "relation": "OWNS"}]})

    class FakeLLM:
        async def generate(self, prompt, **kw):
            llm_client._report_openai_compatible_usage(SimpleNamespace(model="served-model-x", usage=None))
            return payload

    ex = Extractor.__new__(Extractor)
    ex._model_name = "configured-label"
    ex._entity_types = ["ORG"]
    chunk = Chunk(id="c1", document_id="d1", text="Acme owns Bolt.", tenant="t", chunk_index=0)
    with patch("graphrag.ingestion.extractor.get_llm", return_value=FakeLLM()), \
         patch("graphrag.ingestion.extractor.get_settings") as gs:
        gs.return_value.llm_cache_enabled = False
        entities, relations = await ex.extract(chunk)
    assert {e.extraction_model for e in entities} == {"served-model-x"}
    assert relations and relations[0].extraction_model == "served-model-x"


async def test_extraction_falls_back_to_the_configured_label_when_no_model_is_reported():
    import json
    from unittest.mock import patch

    from graphrag.core.models import Chunk
    from graphrag.ingestion.extractor import Extractor

    class QuietLLM:
        async def generate(self, prompt, **kw):
            return json.dumps({"entities": [{"name": "Acme", "type": "ORG"}], "relations": []})

    ex = Extractor.__new__(Extractor)
    ex._model_name = "configured-label"
    ex._entity_types = ["ORG"]
    chunk = Chunk(id="c1", document_id="d1", text="Acme.", tenant="t", chunk_index=0)
    with patch("graphrag.ingestion.extractor.get_llm", return_value=QuietLLM()), \
         patch("graphrag.ingestion.extractor.get_settings") as gs:
        gs.return_value.llm_cache_enabled = False
        entities, _ = await ex.extract(chunk)
    assert entities[0].extraction_model == "configured-label"

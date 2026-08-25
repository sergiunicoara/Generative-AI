"""Global search over hierarchical community summaries.

The default path is deliberately retrieval-only: it sends the most relevant
community summaries directly to the final answer synthesis rather than making
one LLM map call per community and a second reduce call.  The legacy
``map_reduce`` path remains available behind configuration for controlled
ablations and rollback.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from graphrag.core.config import get_settings, resolve_tenant_config
from graphrag.core.llm_client import get_llm
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.ingestion.embedder import Embedder
from graphrag.enterprise.models import AccessContext

log = structlog.get_logger(__name__)

_MAP_PROMPT = """\
You are answering a question using the following community summary.
Extract any relevant information, or respond with "Not relevant."

Question: {question}
Community summary: {summary}

Relevant information:"""

_REDUCE_PROMPT = """\
Merge the partial answers below into a compact factual summary answering the question.
Keep every substantive fact; drop redundancy across partials. No preamble, no
markdown formatting, no restating the question.

Question: {question}

Partial answers:
{partial_answers}

Merged facts:"""


def _direct_context(communities: list[dict], max_chars: int) -> str:
    """Return a deterministic, bounded high-level retrieval context."""
    parts: list[str] = []
    remaining = max(max_chars, 0)
    for community in communities:
        summary = str(community.get("summary") or "").strip()
        if not summary or remaining <= 0:
            continue
        header = (
            f"[Community {community.get('community_id', 'unknown')} | "
            f"Level {community.get('level', 'unknown')} | "
            f"Similarity {float(community.get('score') or 0.0):.3f}]\n"
        )
        available = max(remaining - len(header), 0)
        if available <= 0:
            break
        text = summary[:available]
        parts.append(header + text)
        remaining -= len(header) + len(text)
    return "\n\n".join(parts)


class GlobalSearch:
    def __init__(self):
        cfg = get_settings()
        self._cfg = cfg.retrieval
        self._neo4j = get_neo4j()
        self._embedder = Embedder()

    async def search(
        self,
        question: str,
        tenant: str = "default",
        valid_at: str | None = None,
        transaction_at: str | None = None,
        config_overrides: dict | None = None,
        access_context: AccessContext | None = None,
    ) -> dict:
        # Per-tenant config: merge this tenant's overrides over the global
        # retrieval defaults (mirrors LocalSearch.search — resolved from
        # self._cfg). Empty tenant_overrides ⇒ global dict unchanged.
        cfg = {**resolve_tenant_config(self._cfg, tenant), **(config_overrides or {})}
        if get_settings().access_control.get("enabled", False):
            # Community summaries are derived artifacts and are not yet
            # re-materialised per ACL. Returning none is safer than allowing a
            # summary to disclose protected evidence through a public member.
            log.info("global_search.acl_denied", tenant=tenant)
            return {"communities": [], "synthesized_answer": ""}

        # Skip global search when vector_search_enabled=false (e.g. OpenAI quota exhausted)
        if not cfg.get("vector_search_enabled", True):
            log.info("global_search.vector_skipped", reason="vector_search_enabled=false")
            return {"communities": [], "synthesized_answer": ""}

        _t0 = time.monotonic()
        embedding = await self._embedder.embed_text(question)
        log.info(
            "global_search.embed.done",
            elapsed_ms=round((time.monotonic() - _t0) * 1000, 1),
        )

        top_k = cfg.get("global_top_communities", 5)
        _t0 = time.monotonic()
        communities = await self._neo4j.vector_search_communities(
            embedding,
            top_k=top_k,
            tenant=tenant,
            valid_at=valid_at,
            transaction_at=transaction_at,
        )
        log.info(
            "global_search.community_vector_search.done",
            elapsed_ms=round((time.monotonic() - _t0) * 1000, 1),
            communities=len(communities),
        )

        if not communities:
            log.warning(
                "global_search.no_communities",
                tenant=tenant,
                hint=(
                    "No Community nodes found. Run scripts/community_rebuild.py "
                    "or enable graph.auto_rebuild_communities in settings."
                ),
            )
            return {"communities": [], "synthesized_answer": ""}

        # Attach a representative source-document set per community so
        # citations survive into the final answer — see
        # Neo4jClient.get_community_source_documents for why this exists.
        community_ids = [c["community_id"] for c in communities if c.get("community_id")]
        doc_map: dict[str, list[str]] = {}
        if community_ids:
            try:
                doc_map = await self._neo4j.get_community_source_documents(
                    community_ids, tenant=tenant,
                )
            except Exception as exc:  # noqa: BLE001 — citations are best-effort, never fatal
                log.warning("global_search.source_documents_failed", error=str(exc)[:160])
        for c in communities:
            c["source_documents"] = doc_map.get(c.get("community_id"), [])

        # Warn if the top communities are connected-components fallbacks —
        # this signals graspologic is missing and global quality is degraded.
        fallback_communities = [
            c for c in communities
            if str(c.get("summary", "")).startswith("[fallback:")
        ]
        if fallback_communities:
            log.error(
                "global_search.degraded_communities",
                fallback_count=len(fallback_communities),
                total=len(communities),
                tenant=tenant,
                impact="answers based on connected-components, not Leiden hierarchy",
            )

        strategy = cfg.get("global_search_strategy", "direct_context")
        if strategy == "direct_context":
            synthesized = _direct_context(
                communities,
                int(cfg.get("global_direct_context_max_chars", 6000)),
            )
            log.info(
                "global_search.done",
                communities=len(communities),
                strategy=strategy,
                llm_calls=0,
                context_chars=len(synthesized),
            )
            return {"communities": communities, "synthesized_answer": synthesized}

        if strategy != "map_reduce":
            raise ValueError(f"Unsupported global_search_strategy: {strategy}")

        # Legacy map-reduce: retained solely for comparison and rollback.
        llm = get_llm()
        _t0 = time.monotonic()
        map_tasks = [
            llm.generate(_MAP_PROMPT.format(question=question, summary=c["summary"]))
            for c in communities
        ]
        map_texts = await asyncio.gather(*map_tasks)
        log.info(
            "global_search.map.done",
            elapsed_ms=round((time.monotonic() - _t0) * 1000, 1),
            map_calls=len(map_tasks),
        )

        partial_answers = []
        for community, text in zip(communities, map_texts):
            if text and "not relevant" not in text.lower():
                partial_answers.append(f"[Level {community['level']}] {text}")

        if not partial_answers:
            log.info(
                "global_search.done",
                communities=len(communities),
                partial_answers=0,
                reason="all_map_results_not_relevant",
            )
            return {"communities": communities, "synthesized_answer": ""}

        if len(partial_answers) == 1:
            # Nothing to synthesize — reduce would spend a full LLM call
            # (measured live: 9-17.6s) reformatting one extraction into
            # prose that is only ever consumed as context by the final
            # synthesis call (context_builder.py), never shown to the user.
            # Strip the "[Level N] " prefix — it exists only to label
            # sources for the reduce prompt above; downstream wants facts.
            log.info("global_search.reduce.skipped", reason="single_partial_answer")
            only = partial_answers[0]
            synthesized = only.split("] ", 1)[1] if only.startswith("[Level ") else only
            log.info(
                "global_search.done",
                communities=len(communities),
                partial_answers=1,
            )
            return {"communities": communities, "synthesized_answer": synthesized}

        # Reduce: synthesize all partial answers
        max_tokens = cfg.get("global_reduce_max_tokens", 300)
        _t0 = time.monotonic()
        synthesized = await llm.generate(
            _REDUCE_PROMPT.format(
                question=question,
                partial_answers="\n\n".join(partial_answers),
            ),
            max_tokens=max_tokens,
        )
        log.info(
            "global_search.reduce.done",
            elapsed_ms=round((time.monotonic() - _t0) * 1000, 1),
        )

        log.info(
            "global_search.done",
            communities=len(communities),
            partial_answers=len(partial_answers),
            strategy=strategy,
        )
        return {
            "communities": communities,
            "synthesized_answer": synthesized or "",
        }

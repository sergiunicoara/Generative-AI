"""``full_context`` ablation profile: no retrieval, the whole tenant corpus goes into the prompt.

Same generator, same answer prompt (``answer_prompt``), same ``normalize_dashes`` and the same
golden / faithfulness scoring as the retrieval profiles. Evaluation-only: it is not a production
retrieval mode and is not registered in ``HybridRetriever``'s profile table.

Deliberate choices (each is recorded in the result file under ``run.full_context``):

* Superseded documents are excluded, as in every retrieval profile (``include_superseded`` is False by
  default there), so all profiles see the same set of documents; the excluded names are recorded.
* The answer prompt carries the same ``[DOC: <id>]`` citation rule as every other ablation profile
  (``eval_citation_instruction``, built by ``citations.citation_instruction``). The answer is returned as
  generated; the benchmark parses and strips the tags identically for all profiles. This profile has no
  retrieval, so it has no context citations.
* The aerospace post-generation repair (``ground_regulatory_identifiers``) is NOT applied. It appends
  every AD identifier found in the context; with the whole corpus as context it would append all of them.
* The faithfulness judge receives the full corpus (one string per document) as its contexts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from graphrag.core.config import get_settings, resolve_tenant_config
from graphrag.core.llm_client import get_generation_route, get_llm
from graphrag.core.llm_utils import normalize_dashes
from graphrag.core.models import QueryResult
from graphrag.core.prompt_security import escape_prompt_data
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.observability.correlation import tenant_context
from graphrag.retrieval.answer_policy import answer_prompt

PROFILE = "full_context"
# gpt-oss-120b on Groq: 131,072-token window (Groq model page; re-check before relying on it).
DEFAULT_CONTEXT_WINDOW = 131_072
# Head-room for the generator's output, reasoning tokens included.
DEFAULT_OUTPUT_RESERVE = 16_384

_CORPUS_QUERY = """
MATCH (d:Document {tenant: $tenant})
WHERE coalesce(d.is_deleted, false) = false
MATCH (c:Chunk {document_id: d.id})
RETURN d.filename AS filename, d.superseded_by AS superseded_by, c.chunk_index AS idx, c.text AS text
ORDER BY filename, idx
"""


@dataclass(frozen=True)
class Corpus:
    tenant: str
    documents: list[dict]   # {name, filename, superseded, n_chunks, text}
    context: str            # the whole corpus as the single prompt context
    contexts: list[str]     # one string per document: what the faithfulness judge receives
    excluded_superseded: tuple[str, ...] = ()  # superseded documents left out, as retrieval leaves them out

    @property
    def names(self) -> list[str]:
        return [doc["name"] for doc in self.documents]


def _stem(filename: str) -> str:
    return filename[:-4] if filename.lower().endswith(".txt") else filename


def build_corpus(tenant: str, rows: list[dict], *, include_superseded: bool = False) -> Corpus:
    """Group chunk rows by document (chunk order kept) into one labelled context."""
    by_file: dict[str, dict] = {}
    excluded: list[str] = []
    for row in rows:
        if row.get("superseded_by") and not include_superseded:
            if _stem(row["filename"]) not in excluded:
                excluded.append(_stem(row["filename"]))
            continue
        entry = by_file.setdefault(row["filename"], {
            "filename": row["filename"], "name": _stem(row["filename"]),
            "superseded": bool(row.get("superseded_by")), "chunks": [],
        })
        entry["chunks"].append(row["text"] or "")
    documents = []
    for entry in by_file.values():
        text = "\n\n".join(entry["chunks"])
        documents.append({"name": entry["name"], "filename": entry["filename"],
                          "superseded": entry["superseded"], "n_chunks": len(entry["chunks"]), "text": text})
    contexts = [f"[Document: {doc['name']}]\n{doc['text']}" for doc in documents]
    return Corpus(tenant=tenant, documents=documents, context="\n\n".join(contexts), contexts=contexts,
                  excluded_superseded=tuple(excluded))


async def load_corpus(tenant: str) -> Corpus:
    with tenant_context(tenant):
        rows = await get_neo4j().run(_CORPUS_QUERY, tenant=tenant)
    corpus = build_corpus(tenant, rows)
    if not corpus.documents:
        raise ValueError(f"no documents found for tenant {tenant!r}")
    return corpus


def count_tokens(text: str) -> tuple[int, str]:
    """Token count and the tokenizer used (tiktoken o200k_base when installed, else chars/4)."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text)), "tiktoken:o200k_base"
    except Exception:  # noqa: BLE001 - tiktoken missing or its data not downloadable
        return -(-len(text) // 4), "estimate:chars/4"


def corpus_report(corpus: Corpus, cfg: dict[str, Any], *, window: int = DEFAULT_CONTEXT_WINDOW,
                  output_reserve: int = DEFAULT_OUTPUT_RESERVE) -> dict:
    """Size of the corpus and of the full prompt, and whether it fits ``window`` minus the output reserve."""
    template = answer_prompt(cfg).format(context="", question="")
    corpus_tokens, tokenizer = count_tokens(corpus.context)
    prompt_tokens = corpus_tokens + count_tokens(template)[0] + 200  # +200: a typical question
    budget = window - output_reserve
    return {
        "n_documents": len(corpus.documents),
        "n_chunks": sum(doc["n_chunks"] for doc in corpus.documents),
        "words": len(corpus.context.split()),
        "characters": len(corpus.context),
        "corpus_tokens": corpus_tokens,
        "prompt_tokens_estimate": prompt_tokens,
        "tokenizer": tokenizer,
        "context_window": window,
        "output_reserve": output_reserve,
        "fits": prompt_tokens <= budget,
        "context_sha256": hashlib.sha256(corpus.context.encode("utf-8")).hexdigest(),
        "documents": [{"name": d["name"], "chunks": d["n_chunks"], "superseded": d["superseded"]}
                      for d in corpus.documents],
        "superseded_documents_included": [d["name"] for d in corpus.documents if d["superseded"]],
        "superseded_documents_excluded": list(corpus.excluded_superseded),
        "post_generation_repair": "not applied (see module docstring)",
        "prompt": "shared answer prompt + the shared [DOC: id] citation instruction (same text for all profiles)",
        "citations": "answer citations only (parsed from [DOC: id] tags by the shared parser); no context citations",
    }


def require_fit(report: dict) -> None:
    if not report["fits"]:
        raise ValueError(
            f"full_context prompt (~{report['prompt_tokens_estimate']} tokens) does not fit the "
            f"{report['context_window']}-token window with {report['output_reserve']} reserved for output")


async def answer_full_context(question: str, corpus: Corpus, *, tenant: str,
                              citation_instruction: str | None = None) -> QueryResult:
    """Answer ``question`` with the whole corpus as context (one generator call, no retrieval)."""
    cfg = resolve_tenant_config(get_settings().retrieval, tenant)
    if citation_instruction:
        cfg = {**cfg, "eval_citation_instruction": citation_instruction}
    prompt = answer_prompt(cfg).format(context=escape_prompt_data(corpus.context), question=question)
    answer = normalize_dashes(await get_llm().generate(prompt) or "Insufficient context to answer this question.")
    return QueryResult(
        question=question, answer=answer, contexts=list(corpus.contexts),
        citations=[],
        retrieval_mode=PROFILE, routing_reason="full_context_profile",
        model_version=get_generation_route()["primary"],
    )

"""``full_context`` ablation profile: no retrieval, the whole tenant corpus goes into the prompt.

Same generator, same answer prompt (``answer_prompt``), same ``normalize_dashes`` and the same
golden / faithfulness scoring as the retrieval profiles. Evaluation-only: it is not a production
retrieval mode and is not registered in ``HybridRetriever``'s profile table.

Deliberate choices (each is recorded in the result file under ``run.full_context``):

* All documents are included, superseded ones too ("all 12 documents"), plainly labelled with the
  document name only. Normal retrieval excludes superseded documents, so this is a real difference.
* Citations are the documents whose name appears in the answer: there is no retrieval to cite from.
* The aerospace post-generation repair (``ground_regulatory_identifiers``) is NOT applied. It appends
  every AD identifier found in the context; with the whole corpus as context it would append all of them.
* The faithfulness judge receives the full corpus (one string per document) as its contexts.
"""
from __future__ import annotations

import hashlib
import re
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

    @property
    def names(self) -> list[str]:
        return [doc["name"] for doc in self.documents]


def _stem(filename: str) -> str:
    return filename[:-4] if filename.lower().endswith(".txt") else filename


def build_corpus(tenant: str, rows: list[dict]) -> Corpus:
    """Group chunk rows by document (chunk order kept) into one labelled context."""
    by_file: dict[str, dict] = {}
    for row in rows:
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
    return Corpus(tenant=tenant, documents=documents, context="\n\n".join(contexts), contexts=contexts)


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
        "post_generation_repair": "not applied (see module docstring)",
        "citations": "documents named in the answer",
    }


def require_fit(report: dict) -> None:
    if not report["fits"]:
        raise ValueError(
            f"full_context prompt (~{report['prompt_tokens_estimate']} tokens) does not fit the "
            f"{report['context_window']}-token window with {report['output_reserve']} reserved for output")


def cited_documents(answer: str, names: list[str]) -> list[str]:
    """Documents named in the answer, case-insensitive and hyphen-tolerant, in corpus order."""
    def norm(text: str) -> str:
        return re.sub(r"[\s_‐-―-]+", "-", text.lower())

    haystack = norm(answer)
    return [name for name in names if norm(name) in haystack]


async def answer_full_context(question: str, corpus: Corpus, *, tenant: str) -> QueryResult:
    """Answer ``question`` with the whole corpus as context (one generator call, no retrieval)."""
    cfg = resolve_tenant_config(get_settings().retrieval, tenant)
    prompt = answer_prompt(cfg).format(context=escape_prompt_data(corpus.context), question=question)
    answer = await get_llm().generate(prompt) or "Insufficient context to answer this question."
    answer = normalize_dashes(answer)
    return QueryResult(
        question=question, answer=answer, contexts=list(corpus.contexts),
        citations=cited_documents(answer, corpus.names),
        retrieval_mode=PROFILE, routing_reason="full_context_profile",
        model_version=get_generation_route()["primary"],
    )

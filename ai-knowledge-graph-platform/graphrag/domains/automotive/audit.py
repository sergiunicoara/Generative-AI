"""IATF 16949 clause evidence view: deterministic, offline, no LLM.

For each clause in ``config/iatf_clauses.yml`` this reports which *current* documents evidence it, the exact passages
(file and line number), superseded revisions that also mention it, contradiction tags shared with other documents, and
how old the evidence is. The result is the evidence an auditor reviews; it never states that a clause is "compliant".

Document state comes from the corpus files and their metadata:
  * superseded: the document is the predecessor in a configured supersession chain (ontology yml) or sits in the
    corpus ``archive`` folder;
  * recorded at: ``saved_at`` in the metadata. The corpus has no effective dates, so an ``as_of`` view treats a document
    as existing from the time it was recorded. This is a stated approximation, not a revision-in-force date.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[3]
CATALOGUE = ROOT / "config" / "iatf_clauses.yml"
ONTOLOGY = ROOT / "config" / "ontologies" / "automotive_iatf.yml"
CORPUS = ROOT / "data" / "automotive"
METADATA = ROOT / "data" / "metadata" / "automotive"

STATUSES = ("evidenced", "stale", "only_superseded", "missing")


@dataclass(frozen=True)
class Clause:
    id: str
    heading: str
    doc_prefixes: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass
class CorpusDocument:
    doc_id: str
    folder: str
    recorded_at: datetime | None
    contradictions: tuple[str, ...]
    lines: list[str] = field(repr=False, default_factory=list)
    superseded_by: str | None = None

    @property
    def current(self) -> bool:
        return self.superseded_by is None and self.folder != "archive"


def fold(text: str) -> str:
    """Lower-case and strip diacritics, so 'reevaluării' matches 'reevaluarii' and 'ș'/'ş' match 's'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_catalogue(path: Path = CATALOGUE) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    clauses = [Clause(str(c["id"]), str(c["heading"]), tuple(c.get("doc_prefixes") or ()),
                      tuple(fold(k) for k in c.get("keywords") or ()))
               for c in data.get("clauses") or []]
    ids = [c.id for c in clauses]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clause id in catalogue")
    if any(not c.keywords for c in clauses):
        raise ValueError("every clause needs at least one keyword")
    return {"standard": data.get("standard", ""), "mapping_status": data.get("mapping_status", "proposed"),
            "max_document_age_days": int(data.get("max_document_age_days", 365)), "clauses": clauses}


def _supersession(ontology_path: Path) -> dict[str, str]:
    data = yaml.safe_load(ontology_path.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for chain in data.get("supersession_chains") or []:
        out[Path(str(chain["predecessor"])).stem] = Path(str(chain["successor"])).stem
    return out


def load_corpus(corpus_dir: Path = CORPUS, metadata_dir: Path = METADATA,
                ontology_path: Path = ONTOLOGY) -> list[CorpusDocument]:
    supersedes = _supersession(ontology_path)
    docs = []
    for path in sorted(corpus_dir.glob("*/*.txt")):
        meta_path = metadata_dir / f"{path.stem}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        docs.append(CorpusDocument(
            doc_id=path.stem, folder=path.parent.name, recorded_at=_parse_time(meta.get("saved_at")),
            contradictions=tuple(meta.get("detected_contradictions") or ()),
            lines=path.read_text(encoding="utf-8").splitlines(), superseded_by=supersedes.get(path.stem)))
    return docs


def matches_prefix(doc_id: str, prefixes: Iterable[str]) -> bool:
    return any(doc_id == p or doc_id.startswith(p + "-") for p in prefixes)


def _passages(doc: CorpusDocument, keywords: tuple[str, ...], limit: int) -> tuple[int, list[dict[str, Any]]]:
    patterns = [re.compile(r"(?<!\w)" + re.escape(k)) for k in keywords]
    hits = []
    for number, line in enumerate(doc.lines, start=1):
        folded = fold(line)
        found = [k for k, p in zip(keywords, patterns) if p.search(folded)]
        if found:
            hits.append({"line": number, "text": line.strip()[:240], "keywords": found})
    return len(hits), hits[:limit]


def _as_of_view(docs: list[CorpusDocument], as_of: datetime | None) -> list[CorpusDocument]:
    """Documents recorded by ``as_of``; a successor recorded later does not yet supersede its predecessor."""
    if as_of is None:
        return docs
    present = [d for d in docs if d.recorded_at is None or d.recorded_at <= as_of]
    ids = {d.doc_id for d in present}
    return [CorpusDocument(d.doc_id, d.folder, d.recorded_at, d.contradictions, d.lines,
                           d.superseded_by if d.superseded_by in ids else None) for d in present]


def clause_evidence(clause: Clause, docs: list[CorpusDocument], *, max_age_days: int, now: datetime,
                    max_passages: int = 3) -> dict[str, Any]:
    current, superseded = [], []
    for doc in docs:
        if not matches_prefix(doc.doc_id, clause.doc_prefixes):
            continue
        count, passages = _passages(doc, clause.keywords, max_passages)
        if not count:
            continue
        age = (now - doc.recorded_at).days if doc.recorded_at else None
        entry = {"doc_id": doc.doc_id, "folder": doc.folder,
                 "recorded_at": doc.recorded_at.isoformat() if doc.recorded_at else None, "age_days": age,
                 "matching_lines": count, "passages": passages, "contradiction_tags": list(doc.contradictions)}
        if doc.current:
            current.append(entry)
        else:
            superseded.append({**entry, "superseded_by": doc.superseded_by or "(archived)"})
    tags = sorted({t for e in current for t in e["contradiction_tags"]})
    conflicts = [{"tag": t, "documents": sorted(d.doc_id for d in docs if t in d.contradictions)} for t in tags]
    fresh = [e for e in current if e["age_days"] is not None and e["age_days"] <= max_age_days]
    if fresh:
        status = "evidenced"
    elif current:
        status = "stale"
    elif superseded:
        status = "only_superseded"
    else:
        status = "missing"
    return {"clause": clause.id, "heading": clause.heading, "status": status,
            "evidence": current, "superseded_evidence": superseded, "conflicts": conflicts,
            "needs_attention": status != "evidenced" or bool(conflicts)}


def evidence_digest(clause_view: dict[str, Any]) -> str:
    """Stable fingerprint of the evidence an auditor saw: documents, recorded times and line numbers, not wording."""
    material = {
        "clause": clause_view["clause"],
        "evidence": sorted((e["doc_id"], e["recorded_at"], tuple(p["line"] for p in e["passages"]))
                           for e in clause_view["evidence"]),
        "superseded": sorted(e["doc_id"] for e in clause_view["superseded_evidence"]),
        "conflicts": [(c["tag"], tuple(c["documents"])) for c in clause_view["conflicts"]],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def audit_view(*, catalogue: dict[str, Any] | None = None, docs: list[CorpusDocument] | None = None,
               as_of: datetime | str | None = None, now: datetime | None = None,
               clause_ids: Iterable[str] | None = None) -> dict[str, Any]:
    catalogue = catalogue or load_catalogue()
    as_of_dt = _parse_time(as_of) if isinstance(as_of, str) else as_of
    now = now or as_of_dt or datetime.now(timezone.utc)
    view_docs = _as_of_view(docs if docs is not None else load_corpus(), as_of_dt)
    wanted = set(clause_ids) if clause_ids else None
    clauses = []
    for clause in catalogue["clauses"]:
        if wanted is not None and clause.id not in wanted:
            continue
        item = clause_evidence(clause, view_docs, max_age_days=catalogue["max_document_age_days"], now=now)
        item["evidence_digest"] = evidence_digest(item)
        clauses.append(item)
    counts = {s: sum(1 for c in clauses if c["status"] == s) for s in STATUSES}
    return {
        "standard": catalogue["standard"], "mapping_status": catalogue["mapping_status"],
        "as_of": as_of_dt.isoformat() if as_of_dt else None, "evaluated_at": now.isoformat(),
        "documents_considered": len(view_docs), "counts": counts,
        "coverage": round(counts["evidenced"] / len(clauses), 4) if clauses else None,
        "clauses": clauses,
        "disclaimer": "Evidence located for review, not a compliance determination. The clause mapping is "
                      f"'{catalogue['mapping_status']}' and must be confirmed by a quality engineer.",
    }


def freshness_problems(view: dict[str, Any]) -> list[dict[str, Any]]:
    """Clauses an operator should hear about: no current evidence, only stale evidence, or open conflicts."""
    problems = []
    for c in view["clauses"]:
        if c["status"] != "evidenced":
            problems.append({"clause": c["clause"], "problem": c["status"]})
        for conflict in c["conflicts"]:
            problems.append({"clause": c["clause"], "problem": "conflict", "tag": conflict["tag"],
                             "documents": conflict["documents"]})
    return problems


CSV_FIELDS = ("clause", "heading", "status", "doc_id", "current", "recorded_at", "age_days", "line", "passage",
              "contradiction_tags", "signoff_decision", "signoff_by", "signoff_at", "signoff_digest_matches")


def export_rows(view: dict[str, Any], signoffs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One row per evidence passage (or one row for a clause without evidence), with the latest sign-off attached."""
    signoffs = signoffs or {}
    rows = []
    for c in view["clauses"]:
        s = signoffs.get(c["clause"]) or {}
        base = {"clause": c["clause"], "heading": c["heading"], "status": c["status"],
                "signoff_decision": s.get("decision", ""), "signoff_by": s.get("reviewer", ""),
                "signoff_at": s.get("signed_at", ""),
                "signoff_digest_matches": (s.get("evidence_digest") == c["evidence_digest"]) if s else ""}
        entries = [(e, True) for e in c["evidence"]] + [(e, False) for e in c["superseded_evidence"]]
        if not entries:
            rows.append({**base, "doc_id": "", "current": "", "recorded_at": "", "age_days": "", "line": "",
                         "passage": "", "contradiction_tags": ""})
        for entry, is_current in entries:
            for p in entry["passages"] or [{"line": "", "text": ""}]:
                rows.append({**base, "doc_id": entry["doc_id"], "current": is_current,
                             "recorded_at": entry["recorded_at"], "age_days": entry["age_days"], "line": p["line"],
                             "passage": p["text"], "contradiction_tags": " ".join(entry["contradiction_tags"])})
    return rows


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


__all__ = ["CSV_FIELDS", "STATUSES", "Clause", "CorpusDocument", "audit_view", "clause_evidence", "evidence_digest",
           "export_rows", "fold", "freshness_problems", "load_catalogue", "load_corpus", "matches_prefix", "to_csv"]

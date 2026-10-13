"""IATF clause evidence view: matching, statuses, as-of, digest, export, catalogue, real-corpus findings."""
from __future__ import annotations

import csv
import io
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from graphrag.domains.automotive import audit
from graphrag.domains.automotive.audit import Clause, CorpusDocument

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)
SUPPLIERS = Clause("8.4.2.4", "Supplier monitoring", ("RFA", "CSR-CLIENT"), ("reevaluarii", "kpi"))


def _doc(doc_id, lines, *, days_old=10, folder="medium", tags=(), superseded_by=None):
    return CorpusDocument(doc_id, folder, NOW - timedelta(days=days_old), tuple(tags), list(lines), superseded_by)


def _catalogue(*clauses, max_age=365):
    return {"standard": "IATF 16949:2016", "mapping_status": "proposed", "max_document_age_days": max_age,
            "clauses": list(clauses)}


def test_fold_strips_romanian_diacritics_both_comma_and_cedilla_forms():
    assert audit.fold("Reevaluării ŞI reevaluării Țintă ţintă") == "reevaluarii si reevaluarii tinta tinta"


@pytest.mark.parametrize("doc_id,prefix,expected", [
    ("PC-COMP-07-rev3", "PC", True), ("PCAL-PROC-10", "PC", False), ("MC-01-S8-rev6", "MC-01", True),
    ("CSR-CLIENT-2023", "CSR-CLIENT", True), ("IL-NC-06-rev2", "IL-NC", True), ("IL-NCX-1", "IL-NC", False),
])
def test_prefix_matching_respects_the_dash_boundary(doc_id, prefix, expected):
    assert audit.matches_prefix(doc_id, [prefix]) is expected


def test_evidenced_with_passages_line_numbers_and_accent_insensitive_keywords():
    doc = _doc("RFA-REG-01-rev5", ["intro", "Furnizorii critici: reevaluării SEMESTRIALE", "x", "KPI lunar"])
    item = audit.clause_evidence(SUPPLIERS, [doc], max_age_days=365, now=NOW)
    assert item["status"] == "evidenced"
    # matching is by word start, not stemming: "reevaluare" would NOT match "reevaluării"; the catalogue lists forms
    assert [p["line"] for p in item["evidence"][0]["passages"]] == [2, 4]
    assert item["evidence"][0]["matching_lines"] == 2


def test_keyword_must_start_a_word_and_other_families_are_ignored():
    unrelated_family = _doc("PFMEA-01", ["reevaluare kpi"])
    inside_word = _doc("RFA-REG-01-rev5", ["preevaluarii", "skpi"])
    item = audit.clause_evidence(SUPPLIERS, [unrelated_family, inside_word], max_age_days=365, now=NOW)
    assert item["status"] == "missing" and item["evidence"] == []


def test_status_order_stale_only_superseded_and_missing():
    old = _doc("RFA-REG-01-rev5", ["kpi"], days_old=400)
    assert audit.clause_evidence(SUPPLIERS, [old], max_age_days=365, now=NOW)["status"] == "stale"
    retired = _doc("CSR-CLIENT-2021", ["kpi"], superseded_by="CSR-CLIENT-2023")
    archived = _doc("RFA-REG-00", ["kpi"], folder="archive")
    item = audit.clause_evidence(SUPPLIERS, [retired, archived], max_age_days=365, now=NOW)
    assert item["status"] == "only_superseded"
    assert {e["superseded_by"] for e in item["superseded_evidence"]} == {"CSR-CLIENT-2023", "(archived)"}
    assert audit.clause_evidence(SUPPLIERS, [], max_age_days=365, now=NOW)["status"] == "missing"


def test_conflicts_list_every_document_sharing_a_tag_of_the_current_evidence():
    a = _doc("RFA-REG-01-rev5", ["kpi"], tags=["C05"])
    b = _doc("PFMEA-01", ["nothing"], tags=["C05"])
    item = audit.clause_evidence(SUPPLIERS, [a, b], max_age_days=365, now=NOW)
    assert item["conflicts"] == [{"tag": "C05", "documents": ["PFMEA-01", "RFA-REG-01-rev5"]}]
    assert item["needs_attention"] is True and item["status"] == "evidenced"


def test_as_of_drops_later_documents_and_a_later_successor_does_not_yet_supersede():
    old = _doc("CSR-CLIENT-2021", ["kpi"], days_old=300, superseded_by="CSR-CLIENT-2023")
    new = _doc("CSR-CLIENT-2023", ["kpi"], days_old=10)
    as_of = NOW - timedelta(days=100)
    view = audit.audit_view(catalogue=_catalogue(SUPPLIERS), docs=[old, new], as_of=as_of)
    item = view["clauses"][0]
    assert view["documents_considered"] == 1
    assert [e["doc_id"] for e in item["evidence"]] == ["CSR-CLIENT-2021"] and item["superseded_evidence"] == []
    now_view = audit.audit_view(catalogue=_catalogue(SUPPLIERS), docs=[old, new], now=NOW)
    assert [e["doc_id"] for e in now_view["clauses"][0]["evidence"]] == ["CSR-CLIENT-2023"]


def test_digest_is_stable_and_changes_when_the_evidence_changes():
    docs = [_doc("RFA-REG-01-rev5", ["kpi", "x"])]
    first = audit.audit_view(catalogue=_catalogue(SUPPLIERS), docs=docs, now=NOW)["clauses"][0]
    again = audit.audit_view(catalogue=_catalogue(SUPPLIERS), docs=docs, now=NOW)["clauses"][0]
    assert first["evidence_digest"] == again["evidence_digest"]
    moved = audit.audit_view(catalogue=_catalogue(SUPPLIERS), docs=[_doc("RFA-REG-01-rev5", ["x", "kpi"])],
                             now=NOW)["clauses"][0]
    assert moved["evidence_digest"] != first["evidence_digest"]


def test_export_rows_carry_signoff_and_flag_a_digest_that_no_longer_matches():
    view = audit.audit_view(catalogue=_catalogue(SUPPLIERS, Clause("8.7", "NC", ("IL-NC",), ("neconform",))),
                            docs=[_doc("RFA-REG-01-rev5", ["kpi"])], now=NOW)
    signoffs = {"8.4.2.4": {"decision": "accepted", "reviewer": "auditor-1", "signed_at": "t",
                            "evidence_digest": "stale-digest"}}
    rows = audit.export_rows(view, signoffs)
    by_clause = {r["clause"]: r for r in rows}
    assert by_clause["8.4.2.4"]["signoff_digest_matches"] is False and by_clause["8.4.2.4"]["line"] == 1
    assert by_clause["8.7"]["status"] == "missing" and by_clause["8.7"]["doc_id"] == ""
    parsed = list(csv.DictReader(io.StringIO(audit.to_csv(rows))))
    assert tuple(parsed[0]) == audit.CSV_FIELDS and len(parsed) == 2


def test_catalogue_rejects_duplicates_and_clauses_without_keywords(tmp_path):
    bad = tmp_path / "c.yml"
    bad.write_text("clauses:\n  - {id: '1', heading: a, keywords: [x]}\n  - {id: '1', heading: b, keywords: [y]}\n")
    with pytest.raises(ValueError, match="duplicate"):
        audit.load_catalogue(bad)
    bad.write_text("clauses:\n  - {id: '1', heading: a, keywords: []}\n")
    with pytest.raises(ValueError, match="keyword"):
        audit.load_catalogue(bad)


def test_shipped_catalogue_is_marked_proposed_and_holds_no_standard_text():
    cat = audit.load_catalogue()
    assert cat["mapping_status"] == "proposed" and len(cat["clauses"]) >= 10
    assert all(len(c.heading) <= 80 for c in cat["clauses"])


def test_real_corpus_findings_are_reproducible():
    """Recorded findings on the synthetic corpus (docs/iatf-audit.md). A change here is a real change to report."""
    view = audit.audit_view(now=NOW)
    status = {c["clause"]: c["status"] for c in view["clauses"]}
    assert {c for c, s in status.items() if s == "missing"} == {"4.4.1.2", "8.5.1.1", "8.5.4"}
    supplier = next(c for c in view["clauses"] if c["clause"] == "8.4.2.4")
    assert "CSR-CLIENT-2021" in [e["doc_id"] for e in supplier["superseded_evidence"]]
    assert "CSR-CLIENT-2021" not in [e["doc_id"] for e in supplier["evidence"]]
    assert view["documents_considered"] == 30


def test_freshness_cli_exits_nonzero_on_missing_clauses_and_zero_when_not_failing_on_them():
    script = str(ROOT / "scripts" / "iatf_audit.py")
    failing = subprocess.run([sys.executable, script, "freshness"], capture_output=True)
    assert failing.returncode == 1
    tolerant = subprocess.run([sys.executable, script, "freshness", "--fail-on", "stale"], capture_output=True)
    assert tolerant.returncode == 0

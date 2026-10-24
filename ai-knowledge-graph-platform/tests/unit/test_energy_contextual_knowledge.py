"""The shared contextual-knowledge evaluation on the Energy maintenance path.

Regression: the shared evaluator must agree with Energy's own temporal
guidance selection (``select_bulletin``) in every state it distinguishes, and
adding it must not change the answer, its status or its citations.
"""

from __future__ import annotations

import pytest

from graphrag.domains.energy import answers as energy_answers
from graphrag.domains.energy.answers import _knowledge_evaluation, select_bulletin
from graphrag.domains.energy.vocabulary import parse_instant
from tests.unit.test_energy_temporal_correctness import _bulletin_graph

R1 = {"id": "R1", "threshold": 90.0, "valid_from": "2026-01-01T00:00:00Z", "recorded_at": "2026-01-01T00:00:00Z"}
R2_SUPERSEDING = {"id": "R2", "threshold": 85.0, "valid_from": "2026-06-01T00:00:00Z",
                  "recorded_at": "2026-05-01T00:00:00Z", "supersedes": "R1"}
R2_UNORDERED = {k: v for k, v in R2_SUPERSEDING.items() if k != "supersedes"}


@pytest.mark.parametrize("revisions, at, status, applicable, conflict, review", [
    ([R1], "2025-06-01T00:00:00Z", "missing", [], [], True),
    ([R1], "2026-03-01T00:00:00Z", "current", ["R1@v1"], [], False),
    ([{**R1, "valid_to": "2026-04-01T00:00:00Z"}], "2026-05-01T00:00:00Z", "expired", [], [], True),
    ([R1, R2_SUPERSEDING], "2026-07-01T00:00:00Z", "current", ["R2@v1"], [], False),
    ([R1, R2_UNORDERED], "2026-07-01T00:00:00Z", "conflicting", ["R1@v1", "R2@v1"], ["ambiguous_revision"], True),
], ids=["missing", "current", "expired", "superseded", "conflicting"])
def test_shared_evaluation_agrees_with_energy_guidance_selection(revisions, at, status, applicable, conflict,
                                                                 review):
    graph = _bulletin_graph(revisions)
    instant = parse_instant(at)
    selection = select_bulletin(graph, as_of=instant, known_as=instant)
    assert selection.status == status
    guidance = _knowledge_evaluation(graph, [], instant, instant, "energy-demo", None)["guidance"]
    assert guidance["applicable"] == applicable
    assert [c["kind"] for c in guidance["conflicts"]] == conflict
    assert guidance["human_review_required"] is review
    if selection.is_current:
        assert guidance["applicable"] == [f"{selection.bulletin['bulletinId']}@v1"]


def test_evaluation_is_additive_to_the_answer(monkeypatch):
    """Same answer, status, guidance and citations with or without the shared block."""
    graph = _bulletin_graph([R1])
    kwargs = dict(tenant="t", dataset_tenant="t", as_of="2026-03-01T00:00:00Z")
    with_block = energy_answers.answer(graph, "maintenance_review", **kwargs)
    monkeypatch.setattr(energy_answers, "_knowledge_evaluation", lambda *a, **k: {"disabled": True})
    baseline = energy_answers.answer(graph, "maintenance_review", **kwargs)
    for key in ("status", "answer", "evidence", "guidance", "authoritative_bulletin"):
        assert with_block[key] == baseline[key]
    assert with_block["knowledge_evaluation"]["guidance"]["applicable"] == ["R1@v1"]
    assert with_block["knowledge_evaluation"]["release"]["release_id"] == "unversioned-graph"


def test_flagged_asset_is_an_advisory_failure_with_cited_observation():
    graph = _bulletin_graph([R1])
    instant = parse_instant("2026-03-01T00:00:00Z")
    row = {"asset": "https://example.energy.demo/asset/WT-01", "componentType": "gearbox", "temperature": "96.0"}
    original = energy_answers._observation_evidence
    try:
        energy_answers._observation_evidence = lambda g, r, t: energy_answers.Evidence(
            source_id="OBS-WT-01", source_type="observation", source_document="", field_or_span="temperature_c",
            observed_at="", valid_from=None, valid_to=None, recorded_at=None, access_scope=t, value="96.0 C")
        block = _knowledge_evaluation(graph, [row], instant, instant, "t", None)
    finally:
        energy_answers._observation_evidence = original
    asset = block["assets"][0]
    result = asset["results"][0]
    assert asset["asset"] == "WT-01" and result["outcome"] == "fail" and result["modality"] == "recommended"
    assert {"R1", "OBS-WT-01"} <= set(result["citations"])
    # a recommendation fails without making the decision non-compliant
    assert asset["overall"] == "compliant" and asset["human_review_required"] is False


def test_answers_from_the_service_name_the_publication(tmp_path):
    from graphrag.domains.energy.demo import TENANT, EnergyDemoService
    from graphrag.domains.energy.fixtures import create_sap_fixture_sqlite

    database = tmp_path / "energy.sqlite"
    create_sap_fixture_sqlite(database)
    service = EnergyDemoService(source_db=database)
    result = service.answer("maintenance_review", tenant=TENANT)
    version = service.publication_report().version_id
    assert result["publication_version_id"] == version
    assert result["knowledge_evaluation"]["release"]["release_id"] == version
    assert service.answer("maintenance_review", tenant="someone-else")["status"] == "not_found"

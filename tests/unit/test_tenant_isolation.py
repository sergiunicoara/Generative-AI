"""Regression guards for the tenant-isolation and metric-correctness fixes.

Every test here corresponds to a defect that shipped and that the existing
suite could not catch, because no test ever inspected a tenant predicate and
no test ever executed Cypher.

The defects:
  1. `$tenant = 'default' OR x.tenant = $tenant` appeared 80 times, making
     tenant "default" — also the default value of every route's tenant
     parameter — a read-every-tenant wildcard.
  2. Routes took `tenant` from the request body, so any token holder could
     name any tenant.
  3. audit_trail matched entities on (name, type) with no tenant, attaching
     one tenant's ChangeLog to another tenant's node.
  4. apply_calibration's final bin was half-open, so confidence 1.0 — the
     extractor's clamp ceiling — could never be calibrated.
  5. graph_evaluator's `prev_orphans or orphans` reported zero orphan growth
     when the previous snapshot was healthy.
  6. community_manager counted every RELATES_TO edge in the database because
     a WHERE attached to an OPTIONAL MATCH does not filter rows.
"""

from __future__ import annotations

import pathlib
import re
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ── 1. The wildcard must not come back ────────────────────────────────────────

class TestNoTenantWildcard:
    def test_no_cypher_treats_default_as_a_wildcard(self):
        """No source file may reintroduce `$tenant = 'default' OR ...`.

        This is a text-level guard on purpose: the pattern is a Cypher string,
        so no type checker or import graph would catch its return, and it
        reads as innocuous at the call site.
        """
        offenders = []
        for path in (REPO_ROOT / "graphrag").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                if re.search(r"\$tenant\s*=\s*'default'", line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
        assert not offenders, (
            "tenant 'default' is being used as a read-everything wildcard at:\n  "
            + "\n  ".join(offenders)
        )

    def test_no_tenant_filter_is_conditional_on_truthiness(self):
        """`"...tenant..." if tenant else ""` silently widens on a falsy tenant."""
        offenders = []
        for path in (REPO_ROOT / "graphrag").rglob("*.py"):
            # tenancy.py documents the anti-pattern in its module docstring.
            if "__pycache__" in path.parts or path.name == "tenancy.py":
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "tenant" in line and re.search(r'if tenant else ""', line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
        assert not offenders, (
            "tenant filter is dropped rather than enforced when tenant is falsy at:\n  "
            + "\n  ".join(offenders)
        )


# ── 2. Tenant comes from the token, not the request ───────────────────────────

class TestTenantFromToken:
    async def test_get_tenant_rejects_a_token_with_no_tenant_claim(self):
        from fastapi import HTTPException

        from api.auth.dependencies import get_tenant

        with pytest.raises(HTTPException) as exc:
            await get_tenant(user={"sub": "u", "scope": "read"})
        assert exc.value.status_code == 403

    async def test_get_tenant_returns_the_claim(self):
        from api.auth.dependencies import get_tenant

        assert await get_tenant(user={"sub": "u", "tenant": "acme"}) == "acme"

    def test_issued_tokens_carry_a_tenant_claim(self):
        """Every create_access_token call site must stamp a tenant.

        AST-based rather than regex: a non-greedy `\\{(.*?)\\}` stops at the
        FIRST closing brace, which breaks the instant a payload dict contains
        any nested {...} literal of its own (e.g. a set literal building the
        scope string) — a real payload can still carry "tenant" past that
        point and the old regex would falsely fail. Walking the AST for
        dict-literal keys is correct regardless of what's nested inside.
        """
        import ast

        auth_src = (REPO_ROOT / "api" / "routes" / "auth.py").read_text(encoding="utf-8")
        tree = ast.parse(auth_src)

        payload_dicts = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "create_access_token"
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                payload_dicts.append(node.args[0])

        assert payload_dicts, "no create_access_token call sites found — test is stale"
        for payload in payload_dicts:
            keys = {k.value for k in payload.keys if isinstance(k, ast.Constant)}
            assert "tenant" in keys, (
                f"token payload has no tenant claim (line {payload.lineno}): {keys}"
            )

    def test_no_route_accepts_tenant_from_the_client(self):
        """Handlers must take tenant via Depends(get_tenant), never as a default."""
        offenders = []
        for path in (REPO_ROOT / "api" / "routes").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r'tenant:\s*str\s*(\|\s*None\s*)?=\s*("|\')', line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
        assert not offenders, (
            "tenant is client-supplied at:\n  " + "\n  ".join(offenders)
        )

    def test_route_signatures_resolve_tenant_from_the_token_ast(self):
        """AST version of the check above — the regex form cannot see the shapes
        that actually shipped.

        `tenant: str = "literal"` is only one way to take tenant from the
        client. The regex missed all three real holes found 2026-08-18:
          - a bare PATH parameter (`tenant: str`) on
            DELETE /kg/cache/flush/{tenant};
          - `tenant: str = Query(default=None)`;
          - a tenant nested inside a Pydantic request-body model
            (SourceSystem / SourceMapping on POST /kg/sources).

        Rule enforced here, per route handler:
          - a parameter named tenant/token_tenant must default to
            Depends(get_tenant), OR the handler must call the reject helper;
          - a parameter annotated with a model that declares a `tenant` field
            must be accompanied by a call to the reject helper.

        Dev-only handlers are exempt, but the exemption is derived from the
        code (the handler calls is_dev_env and refuses outside dev) rather than
        from a hand-maintained name list — so deleting a dev gate re-arms this
        test instead of silently widening the exemption.

        See docs/context_graph_gap_plan.md F12.
        """
        import ast

        _ASSERT_HELPERS = {"assert_request_tenant", "_assert_body_tenant"}
        _HTTP_METHODS = {"get", "post", "put", "patch", "delete"}

        def _tenant_bearing_models() -> set[str]:
            names: set[str] = set()
            for src in (REPO_ROOT / "graphrag", REPO_ROOT / "api"):
                for p in src.rglob("*.py"):
                    if "__pycache__" in p.parts:
                        continue
                    try:
                        tree = ast.parse(p.read_text(encoding="utf-8"))
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.ClassDef):
                            continue
                        for stmt in node.body:
                            if (isinstance(stmt, ast.AnnAssign)
                                    and isinstance(stmt.target, ast.Name)
                                    and stmt.target.id == "tenant"):
                                names.add(node.name)
            return names

        def _is_depends_get_tenant(default) -> bool:
            return (isinstance(default, ast.Call)
                    and getattr(default.func, "id", None) == "Depends"
                    and bool(default.args)
                    and getattr(default.args[0], "id", None) == "get_tenant")

        def _calls(fn, names: set[str]) -> bool:
            return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") in names
                       for n in ast.walk(fn))

        def _is_route(fn) -> bool:
            for dec in fn.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute) and target.attr in _HTTP_METHODS:
                    return True
            return False

        models = _tenant_bearing_models()
        assert models, "found no tenant-bearing models — the scan is broken, not the code"

        offenders: list[str] = []
        for path in (REPO_ROOT / "api" / "routes").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(REPO_ROOT)
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not _is_route(fn):
                    continue
                if _calls(fn, {"is_dev_env"}):
                    continue   # dev-gated; refuses outside development
                guarded = _calls(fn, _ASSERT_HELPERS)

                args = fn.args
                params = args.posonlyargs + args.args + args.kwonlyargs
                pad = len(args.posonlyargs) + len(args.args) - len(args.defaults)
                defaults = [None] * pad + list(args.defaults) + list(args.kw_defaults)

                for param, default in zip(params, defaults):
                    annotation = getattr(param.annotation, "id", None)
                    if param.arg in ("tenant", "token_tenant"):
                        if not _is_depends_get_tenant(default) and not guarded:
                            offenders.append(
                                f"{rel}:{fn.lineno} {fn.name}(): parameter "
                                f"{param.arg!r} is client-supplied and neither "
                                f"Depends(get_tenant) nor reject-checked"
                            )
                    elif annotation in models and not guarded:
                        offenders.append(
                            f"{rel}:{fn.lineno} {fn.name}(): body model "
                            f"{annotation!r} carries a tenant field but the "
                            f"handler never calls assert_request_tenant"
                        )

        assert not offenders, (
            "route(s) take tenant from the client:\n  " + "\n  ".join(offenders)
        )


class TestRequireTenant:
    @pytest.mark.parametrize("missing", [None, "", "   "])
    def test_falsy_tenant_raises(self, missing):
        from graphrag.core.tenancy import require_tenant

        with pytest.raises(ValueError, match="tenant is required"):
            require_tenant(missing)

    def test_valid_tenant_passes_through(self):
        from graphrag.core.tenancy import require_tenant

        assert require_tenant("acme") == "acme"


# ── 3. Audit trail is tenant-scoped ───────────────────────────────────────────

class TestAuditTrailTenantScoping:
    async def test_entity_batch_matches_within_the_tenant(self):
        from graphrag.graph.audit_trail import AuditTrail

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        trail = AuditTrail(neo4j)

        await trail.log_entities_batch(
            [{"name": "Boeing", "type": "ORG", "log_id": "1", "operation": "upsert",
              "old_values": "{}", "new_values": "{}", "changed_by": "t",
              "source_doc_id": "d"}],
            tenant="acme",
        )

        cypher = neo4j.run.call_args[0][0]
        assert "tenant: $tenant" in cypher, (
            "MATCH has no tenant in its key — `LIMIT 1` can bind another "
            "tenant's entity with the same (name, type)"
        )
        assert neo4j.run.call_args.kwargs["tenant"] == "acme"

    async def test_get_history_is_tenant_scoped(self):
        from graphrag.graph.audit_trail import AuditTrail

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        await AuditTrail(neo4j).get_history("Boeing", "ORG", tenant="acme")

        cypher = neo4j.run.call_args[0][0]
        assert "tenant: $tenant" in cypher
        assert neo4j.run.call_args.kwargs["tenant"] == "acme"


# ── 4. Calibration covers the closed upper bound ──────────────────────────────

class TestCalibrationUpperBound:
    async def test_confidence_of_exactly_one_is_calibrated(self):
        """1.0 is the extractor's clamp ceiling and the most over-confident value.

        With a half-open final bin it matched nothing and was returned raw —
        the one value most in need of correction was the one that bypassed it.
        """
        from graphrag.graph.confidence_calibration import CalibrationService

        svc = CalibrationService(AsyncMock())
        curve = [
            {"bin_start": 0.0, "bin_end": 0.9, "n": 5,  "mean_actual": 0.5},
            {"bin_start": 0.9, "bin_end": 1.0, "n": 20, "mean_actual": 0.62},
        ]
        svc.calibration_curve = AsyncMock(return_value=curve)

        assert await svc.apply_calibration(1.0, tenant="acme") == 0.62
        assert await svc.apply_calibration(0.95, tenant="acme") == 0.62
        assert await svc.apply_calibration(0.5, tenant="acme") == 0.5


# ── 5. Orphan growth from a healthy baseline is reported ──────────────────────

class TestOrphanDelta:
    async def test_growth_from_a_zero_baseline_is_not_silenced(self):
        from graphrag.graph.graph_evaluator import GraphEvaluator

        neo4j = AsyncMock()
        # 1st query returns total + current orphans; 2nd returns the previous
        # snapshot, which is healthy (0 orphans) — the case that was silenced.
        neo4j.run = AsyncMock(side_effect=[
            [{"total": 1000, "orphans": 500}],
            [{"prev_orphans": 0}],
        ])

        result = await GraphEvaluator(neo4j).orphan_growth_rate(tenant="acme")

        assert result["orphan_count"] == 500
        assert result["orphan_delta"] == 500, (
            "a clean previous snapshot (0 orphans) must not be treated as "
            "'no previous data' — that silences the alarm case"
        )


# ── 6. Community staleness counts only this tenant's edges ────────────────────

class TestCommunityStalenessEdgeCount:
    async def test_edge_count_is_scoped_to_the_tenant(self):
        """A WHERE on an OPTIONAL MATCH nulls the optional node, it does not
        drop the row — so `count(r)` counted every edge in the database."""
        from graphrag.graph.community_manager import CommunityManager

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[
            {"entity_count": 1, "edge_count": 1, "community_count": 1}
        ])

        await CommunityManager(neo4j).snapshot(tenant="acme")

        cypher = neo4j.run.call_args_list[0][0][0]
        assert "RELATES_TO {tenant: $tenant}" in cypher, (
            "RELATES_TO is not tenant-scoped in the staleness edge count"
        )
        assert "OPTIONAL MATCH (d:Document" not in cypher


# ── 7. GDPR erasure cannot touch another tenant's data ─────────────────────────
#
# Worse than a read leak: these were unscoped DELETEs, reachable via
# POST /kg/gdpr/forget-entity and /forget-document. A tenant's own legitimate
# erasure request could destroy another tenant's documents, chunks, and
# compliance audit trail.

class TestGDPRTenantScoping:
    def _svc(self, run_results=None):
        from graphrag.graph.gdpr import GDPRService

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=run_results) if run_results is not None \
            else AsyncMock(return_value=[])
        return GDPRService(neo4j), neo4j

    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_forget_entity_rejects_missing_tenant(self, missing):
        svc, _ = self._svc()
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.forget_entity("Boeing", "ORG", missing)

    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_forget_document_rejects_missing_tenant(self, missing):
        svc, _ = self._svc()
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.forget_document("doc-1", missing)

    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_deletion_audit_log_rejects_missing_tenant(self, missing):
        svc, _ = self._svc()
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.deletion_audit_log(missing)

    async def test_changelog_deletion_traverses_has_change_and_is_tenant_scoped(self):
        """Was `MATCH (cl:ChangeLog {target_label:'Entity'}) WHERE cl.target_id
        = $name` -- unscoped AND matched a field that never held the entity's
        name (audit_trail.py stores target_id: e.id). Every ChangeLog-touching
        statement must now filter by tenant and go through HAS_CHANGE."""
        svc, neo4j = self._svc()
        await svc.forget_entity("Boeing", "ORG", "acme")

        changelog_calls = [
            call for call in neo4j.run.call_args_list
            if "ChangeLog" in call[0][0]
        ]
        assert changelog_calls, "no ChangeLog-touching query was issued"
        for call in changelog_calls:
            cypher = call[0][0]
            assert "target_label: 'Entity'" not in cypher, (
                "the old wrong-predicate ChangeLog match must not reappear"
            )
            assert "HAS_CHANGE" in cypher
            assert "tenant: $tenant" in cypher
            assert call.kwargs.get("tenant") == "acme"

    async def test_forget_document_chunk_and_document_deletes_are_tenant_scoped(self):
        svc, neo4j = self._svc(run_results=[
            [],   # exclusive_rows (no entities exclusive to this doc)
            [{"n": 3}],  # chunk delete count
            [],   # document delete
        ])
        await svc.forget_document("doc-1", "acme")

        delete_calls = [
            call for call in neo4j.run.call_args_list
            if "DETACH DELETE c" in call[0][0] or "DETACH DELETE d" in call[0][0]
        ]
        assert len(delete_calls) == 2, "expected exactly one chunk-delete and one document-delete"
        for call in delete_calls:
            assert "tenant: $tenant" in call[0][0], (
                f"unscoped DETACH DELETE: {call[0][0]!r}"
            )
            assert call.kwargs.get("tenant") == "acme"

    async def test_forget_document_exclusivity_check_scopes_other_chunk(self):
        """The NOT EXISTS subquery's `other:Chunk` previously had no tenant
        filter -- a same-keyed chunk in a different tenant could mask a
        genuinely-exclusive entity from being erased at all."""
        svc, neo4j = self._svc(run_results=[
            [],   # exclusive_rows (no entities exclusive to this doc)
            [{"n": 0}],  # chunk delete count
            [],   # document delete
        ])
        await svc.forget_document("doc-1", "acme")

        exclusivity_call = neo4j.run.call_args_list[0]
        cypher = exclusivity_call[0][0]
        assert "NOT EXISTS" in cypher
        # The subquery's `other:Chunk` must carry the same tenant filter as
        # the outer match, not just the outer match alone.
        assert "other:Chunk {tenant: $tenant}" in cypher

    async def test_redact_mentions_write_is_tenant_scoped(self):
        """The read half of this helper was already tenant-scoped; the
        write (the actual redaction) was not -- matched by chunk_id alone."""
        svc, neo4j = self._svc(run_results=[
            [{"chunk_id": "c1", "text": "Boeing built this."}],  # the read
            [],  # the SET
        ])
        await svc._redact_mentions_in_chunks("Boeing", "ORG", "acme")

        set_call = neo4j.run.call_args_list[1]
        cypher = set_call[0][0]
        assert "SET c.text" in cypher
        assert "tenant: $tenant" in cypher
        assert set_call.kwargs.get("tenant") == "acme"


# ── 8. The remaining 7 graph/*.py modules that had zero or partial tenant
#        scoping — one targeted regression per module, not GDPR's full depth,
#        but each proves both the require_tenant guard AND that the fixed
#        Cypher actually carries the filter now. ────────────────────────────

class TestRemainingGraphModuleTenantScoping:
    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_cycle_detector_run_rejects_missing_tenant(self, missing):
        from graphrag.graph.cycle_detector import CycleDetector

        with pytest.raises(ValueError, match="tenant is required"):
            await CycleDetector(AsyncMock()).run(missing)

    async def test_cycle_detector_cypher_queries_scope_entity_by_tenant(self):
        from graphrag.graph.cycle_detector import CycleDetector

        neo4j = AsyncMock()
        # _check_apoc probes availability first; make it report unavailable
        # so _detect_cypher (not _detect_apoc) is the one exercised.
        neo4j.run = AsyncMock(side_effect=[Exception("no apoc"), []])
        await CycleDetector(neo4j).detect("acme")

        cypher = neo4j.run.call_args_list[1][0][0]
        assert "Entity {tenant: $tenant}" in cypher
        assert neo4j.run.call_args_list[1].kwargs.get("tenant") == "acme"

    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_propagation_service_rejects_missing_tenant(self, missing):
        from graphrag.graph.propagation import PropagationService

        svc = PropagationService(AsyncMock())
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.mark_dirty(missing, "Boeing")
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.batch_recompute_dirty(missing)

    async def test_propagation_mark_dirty_is_tenant_scoped(self):
        from graphrag.graph.propagation import PropagationService

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[{"flagged": 0}])
        await PropagationService(neo4j).mark_dirty("acme", "Boeing")

        cypher = neo4j.run.call_args[0][0]
        assert "ancestor:Entity {tenant: $tenant}" in cypher
        assert "changed:Entity {name: $name, tenant: $tenant}" in cypher
        assert neo4j.run.call_args.kwargs.get("tenant") == "acme"

    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_multimodal_set_embedding_rejects_missing_tenant(self, missing):
        from graphrag.graph.multimodal import MultiModalEntityService

        svc = MultiModalEntityService(AsyncMock())
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.set_embedding(missing, "attach-1", [0.1, 0.2])

    async def test_multimodal_set_embedding_is_tenant_scoped(self):
        from graphrag.graph.multimodal import MultiModalEntityService

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        await MultiModalEntityService(neo4j).set_embedding("acme", "attach-1", [0.1])

        cypher = neo4j.run.call_args[0][0]
        assert "MediaAttachment {id: $id, tenant: $tenant}" in cypher
        assert neo4j.run.call_args.kwargs.get("tenant") == "acme"

    async def test_multimodal_route_requires_tenant_dependency(self):
        """POST /kg/multimodal/set-embedding previously took no tenant
        dependency at all -- pin that it now does."""
        import inspect

        from api.routes.kg.knowledge import set_media_embedding

        sig = inspect.signature(set_media_embedding)
        assert "tenant" in sig.parameters

    async def test_community_manager_mark_rebuilt_scopes_snapshot_by_tenant(self):
        from graphrag.graph.community_manager import CommunityManager

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(side_effect=[
            [{"entity_count": 1, "edge_count": 1, "community_count": 1}],  # snapshot() stats query
            [],  # snapshot() CREATE
            [],  # the rebuild-milestone SET
        ])
        await CommunityManager(neo4j).mark_rebuilt(tenant="acme")

        set_call = neo4j.run.call_args_list[-1]
        assert "CommunitySnapshot {id: $id, tenant: $tenant}" in set_call[0][0]
        assert set_call.kwargs.get("tenant") == "acme"

    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_document_authority_rejects_missing_tenant(self, missing):
        from graphrag.graph.document_authority import DocumentAuthorityService

        svc = DocumentAuthorityService(AsyncMock())
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.register_supersession(missing, "new-doc", ["old-doc"])
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.get_authority(missing, "doc-1")
        with pytest.raises(ValueError, match="tenant is required"):
            await svc.apply_authority_weights(missing, [{"source_doc_id": "d1"}])

    async def test_document_authority_register_supersession_is_tenant_scoped(self):
        from graphrag.graph.document_authority import DocumentAuthorityService

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        await DocumentAuthorityService(neo4j).register_supersession("acme", "new", ["old"])

        cypher = neo4j.run.call_args[0][0]
        assert "new:Document {id: $new_id, tenant: $tenant}" in cypher
        assert "old:Document {id: $old_id, tenant: $tenant}" in cypher
        assert neo4j.run.call_args.kwargs.get("tenant") == "acme"

    async def test_document_authority_apply_weights_is_tenant_scoped(self):
        """This one feeds retrieval ranking directly -- an unscoped read
        here was a cross-tenant leak into every answer, not just hygiene."""
        from graphrag.graph.document_authority import DocumentAuthorityService

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[{"id": "d1", "level": 1, "superseded": False}])
        await DocumentAuthorityService(neo4j).apply_authority_weights(
            "acme", [{"source_doc_id": "d1", "confidence": 0.8}]
        )

        cypher = neo4j.run.call_args[0][0]
        assert "Document {id: doc_id, tenant: $tenant}" in cypher
        assert neo4j.run.call_args.kwargs.get("tenant") == "acme"

    async def test_community_summarizer_scopes_entity_and_document_by_tenant(self):
        from graphrag.core.models import Community
        from graphrag.graph.community_summarizer import CommunitySummarizer

        summarizer = CommunitySummarizer.__new__(CommunitySummarizer)
        summarizer._neo4j = AsyncMock()
        summarizer._neo4j.run = AsyncMock(return_value=[])

        with patch("graphrag.graph.community_summarizer.get_llm") as mock_llm, \
             patch("graphrag.graph.community_summarizer.get_embedder") as mock_emb:
            mock_llm.return_value.generate = AsyncMock(return_value="summary")
            mock_emb.return_value.embed = AsyncMock(return_value=[[0.1]])
            community = Community(level=0, member_entity_ids=["e1"], tenant="acme")
            await summarizer._summarize_one(community)

        cypher = summarizer._neo4j.run.call_args[0][0]
        assert "Entity {id: eid, tenant: $tenant}" in cypher
        assert summarizer._neo4j.run.call_args.kwargs.get("tenant") == "acme"

    @pytest.mark.parametrize("missing", [None, "", "   "])
    async def test_ingestion_validator_rejects_missing_tenant(self, missing):
        from graphrag.graph.ingestion_validator import IngestionValidator

        validator = IngestionValidator(AsyncMock())
        with pytest.raises(ValueError, match="tenant is required"):
            await validator.validate(missing)
        with pytest.raises(ValueError, match="tenant is required"):
            await validator.remove_self_loops(missing)

    async def test_ingestion_validator_remove_self_loops_is_tenant_scoped(self):
        """The DELETE -- highest-severity fix in this module."""
        from graphrag.graph.ingestion_validator import IngestionValidator

        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[{"removed": 0}])
        await IngestionValidator(neo4j).remove_self_loops("acme")

        cypher = neo4j.run.call_args[0][0]
        assert "Entity {tenant: $tenant}" in cypher
        assert "RELATES_TO {tenant: $tenant}" in cypher
        assert neo4j.run.call_args.kwargs.get("tenant") == "acme"


# ── 9. context_graph.py write routes -- tenant from token, body must agree ────

class TestContextGraphBodyTenantMismatch:
    def test_mismatch_helper_rejects_disagreement(self):
        from fastapi import HTTPException

        from api.routes.context_graph import _assert_body_tenant

        with pytest.raises(HTTPException) as exc:
            _assert_body_tenant("evil-tenant", "real-tenant")
        assert exc.value.status_code == 403

    def test_mismatch_helper_allows_agreement(self):
        from api.routes.context_graph import _assert_body_tenant

        _assert_body_tenant("acme", "acme")  # must not raise

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth.dependencies import get_current_user
        from api.routes import context_graph as cg_routes

        app = FastAPI()
        app.include_router(cg_routes.router)
        app.dependency_overrides[get_current_user] = lambda: {
            "scope": "write", "sub": "test", "tenant": "acme",
        }
        return TestClient(app)

    def test_record_action_rejects_body_tenant_mismatch(self):
        client = self._client()
        resp = client.post("/context-graph/actions", json={
            "tenant": "other-tenant", "decision_id": "d1",
            "action_type": "x", "actor_id": "a1", "reason_code": "test",
        })
        assert resp.status_code == 403

    def test_record_action_accepts_matching_tenant(self):
        from unittest.mock import AsyncMock as _AsyncMock

        client = self._client()
        with patch(
            "api.routes.context_graph.ContextGraphRepository.record_action",
            new=_AsyncMock(return_value="action-1"),
        ):
            resp = client.post("/context-graph/actions", json={
                "tenant": "acme", "decision_id": "d1",
                "action_type": "x", "actor_id": "a1", "reason_code": "test",
            })
        assert resp.status_code == 200
        assert resp.json()["action_id"] == "action-1"

    def test_wpp_campaign_trace_forwards_token_tenant_not_hardcoded_default(self):
        """record_wpp_campaign_placement defaults tenant="marketing" in its
        own signature; the route must override that with the real caller
        tenant, not silently let every trace land under "marketing"."""
        from unittest.mock import AsyncMock as _AsyncMock

        client = self._client()
        captured = {}

        async def _fake_record(self, **kwargs):
            captured.update(kwargs)
            return "decision-1"

        with patch(
            "api.routes.context_graph.ContextGraphTraceService.record_wpp_campaign_placement",
            new=_fake_record,
        ):
            resp = client.post("/context-graph/wpp/campaign-placement", json={
                "placement_id": "p1", "question": "q",
                "statement_ids": ["s1"], "statement_versions": ["v1"],
            })
        assert resp.status_code == 200
        assert captured.get("tenant") == "acme", (
            f"expected the token tenant 'acme' to be forwarded, got {captured.get('tenant')!r}"
        )


# ── 10. SPARQL export path is per-tenant, not one shared file ─────────────────

class TestSPARQLPerTenantExport:
    def _client(self, tenant: str = "acme"):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.auth.dependencies import get_current_user
        from api.routes.kg import knowledge as kg_knowledge

        app = FastAPI()
        app.include_router(kg_knowledge.router)
        app.dependency_overrides[get_current_user] = lambda: {
            "scope": "read", "sub": "test", "tenant": tenant,
        }
        return TestClient(app)

    def test_route_requires_tenant_dependency(self):
        """Previously took no tenant dependency at all -- pin the signature."""
        import inspect

        from api.routes.kg.knowledge import sparql_query

        assert "tenant" in inspect.signature(sparql_query).parameters

    def test_missing_export_reports_the_tenant_scoped_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
        client = self._client(tenant="acme")
        resp = client.post("/sparql", json={"query": "SELECT * WHERE { ?s ?p ?o }"})
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert "acme" in detail
        assert "graph_export.ttl" in detail

    def test_two_tenants_read_different_export_files(self, tmp_path, monkeypatch):
        """The core regression: tenant A's export must not be what tenant
        B's SPARQL query sees, even against the same export directory."""
        monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))

        acme_dir = tmp_path / "acme"
        acme_dir.mkdir()
        (acme_dir / "graph_export.ttl").write_text(
            "@prefix ex: <http://example.org/> .\nex:acme-only ex:name \"Acme Corp\" .\n"
        )
        # No export written for "other-tenant" at all.

        resp_acme = self._client(tenant="acme").post(
            "/sparql", json={"query": "SELECT ?s WHERE { ?s ?p ?o }"}
        )
        assert resp_acme.status_code == 200
        assert resp_acme.json()["count"] == 1

        resp_other = self._client(tenant="other-tenant").post(
            "/sparql", json={"query": "SELECT ?s WHERE { ?s ?p ?o }"}
        )
        assert resp_other.status_code == 404, (
            "other-tenant must not silently see acme's export"
        )

    @pytest.mark.parametrize("bad_tenant", ["../../etc", "acme/../../x", "UPPER", "a b", ""])
    async def test_path_traversal_and_malformed_tenant_rejected(self, bad_tenant):
        """Tenant now flows into a filesystem path -- must be validated
        against a strict pattern before it ever touches Path(...)."""
        from api.routes.kg.knowledge import _TENANT_PATH_RE

        assert _TENANT_PATH_RE.fullmatch(bad_tenant) is None


# ── 12. Body/path-supplied tenant is rejected, not honoured (F12) ─────────────

class TestClientSuppliedTenantRejected:
    """Three write routes took tenant from client-controlled input rather than
    the token, so a tenant-A token could act on tenant B:

      - DELETE /kg/cache/flush/{tenant}  (PATH param) — flush B's answer cache,
        forcing every later query there to re-run full retrieval at real LLM
        cost: a cross-tenant availability/billing attack.
      - POST /kg/sources                 (BODY field) — plant catalog entries
        in B's namespace (MERGE (s:KGSource {tenant: $tenant, id: $id})).
      - POST /kg/sources/{id}/mappings   (BODY field) — same.

    All three now reject a mismatch (403) rather than silently honouring it,
    matching the convention the Context Graph routes already used.
    See docs/context_graph_gap_plan.md F12.
    """

    def test_helper_rejects_mismatch_and_allows_match(self):
        from fastapi import HTTPException

        from api.auth.dependencies import assert_request_tenant

        with pytest.raises(HTTPException) as exc:
            assert_request_tenant("victim-tenant", "attacker-tenant")
        assert exc.value.status_code == 403

        assert_request_tenant("acme", "acme")  # must not raise

    def test_mismatch_message_does_not_invent_a_silent_overwrite(self):
        """Reject, don't overwrite: an overwrite turns both a client bug and a
        deliberate cross-tenant write into an unremarkable 200."""
        from fastapi import HTTPException

        from api.auth.dependencies import assert_request_tenant

        with pytest.raises(HTTPException) as exc:
            assert_request_tenant("evil", "real")
        assert "evil" in str(exc.value.detail) and "real" in str(exc.value.detail)

    @pytest.mark.parametrize(
        "module_name,func_name",
        [
            ("api.routes.kg.health", "cache_flush_tenant"),
            ("api.routes.kg.sources", "upsert_source"),
            ("api.routes.kg.sources", "add_source_mapping"),
        ],
    )
    def test_route_resolves_tenant_from_token(self, module_name, func_name):
        """Each previously-vulnerable handler must now depend on get_tenant."""
        import importlib
        import inspect

        mod = importlib.import_module(module_name)
        sig = inspect.signature(getattr(mod, func_name))
        depends_params = [
            p for p in sig.parameters.values()
            if getattr(p.default, "dependency", None) is not None
        ]
        assert depends_params, f"{func_name} has no Depends(...) tenant parameter"
        assert any(
            p.default.dependency.__name__ == "get_tenant" for p in depends_params
        ), f"{func_name} does not resolve tenant via get_tenant"

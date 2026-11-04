"""Export to data catalogs: Apache Atlas glossary terms and OpenLineage run events.

* Published business definitions (``graphrag/context_graph/definitions.py``)
  become Atlas glossary terms. Only definitions the reader may see are
  exported (tenant, latest version, published, access). A new definition
  version updates the existing term (matched by name in the glossary) instead
  of creating a second one; ``depends_on`` becomes ``seeAlso`` links once every
  term has a GUID.
* Relational builds (``graphrag/ingestion/build_ledger.py``) become OpenLineage
  run events: inputs are exactly the tables the build read, each with its
  expectation results as a ``dataQualityAssertions`` facet; the output graph
  dataset appears only when the build published.

Payload shapes follow the Atlas v2 glossary REST model and the OpenLineage 2.0
run-event specification as documented; the schemas in this module are a
hand-written subset of the required fields, not the published schema files.
Clients take an injectable ``httpx.AsyncClient`` and have only been exercised
against mocked transports, never a real Atlas, Collibra or Marquez instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Iterable

import httpx

from graphrag.context_graph.definitions import BusinessDefinition, visible_definitions
from graphrag.core.connector_url_safety import assert_safe_connector_url
from graphrag.enterprise.models import AccessContext
from graphrag.ingestion.build_ledger import BuildRecord

PRODUCER = "https://github.com/sergiunicoara/Generative-AI/tree/main/ai-knowledge-graph-platform"
OPENLINEAGE_SCHEMA_URL = "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
DQ_FACET_SCHEMA_URL = ("https://openlineage.io/spec/facets/1-0-1/DataQualityAssertionsDatasetFacet.json"
                       "#/$defs/DataQualityAssertionsDatasetFacet")


class CatalogExportError(RuntimeError):
    pass


# -- Atlas glossary --------------------------------------------------------------------------


def atlas_term(definition: BusinessDefinition, glossary_guid: str) -> dict[str, Any]:
    attributes = {
        "graphrag.definitionId": definition.id, "graphrag.version": str(definition.version),
        "graphrag.tenant": definition.tenant, "graphrag.owner": definition.owner,
        "graphrag.precision": definition.precision,
    }
    if definition.precision_note:
        attributes["graphrag.precisionNote"] = definition.precision_note
    if definition.review_due:
        attributes["graphrag.reviewDue"] = definition.review_due.isoformat()
    if definition.executable:
        attributes["graphrag.executableKind"] = definition.executable.kind
    if definition.depends_on:
        attributes["graphrag.dependsOn"] = ",".join(definition.depends_on)
    return {
        "name": definition.term,
        "anchor": {"glossaryGuid": glossary_guid},
        "shortDescription": definition.definition[:256],
        "longDescription": definition.definition,
        "additionalAttributes": attributes,
    }


def atlas_terms(definitions: Iterable[BusinessDefinition], *, tenant: str, access: AccessContext,
                glossary_guid: str) -> list[dict[str, Any]]:
    """Terms for the definitions ``access`` may read (published, latest version, same tenant)."""
    return [atlas_term(d, glossary_guid) for d in visible_definitions(definitions, tenant=tenant, access=access)]


@dataclass(frozen=True)
class AtlasConfig:
    base_url: str
    username: str
    password: str = field(repr=False)
    glossary_guid: str
    timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        assert_safe_connector_url(self.base_url, context="AtlasConfig base_url")

    @classmethod
    def from_env(cls) -> "AtlasConfig | None":
        values = {k: os.getenv(k, "") for k in ("ATLAS_BASE_URL", "ATLAS_USERNAME", "ATLAS_PASSWORD",
                                                 "ATLAS_GLOSSARY_GUID")}
        if not all(values.values()):
            return None
        return cls(values["ATLAS_BASE_URL"].rstrip("/"), values["ATLAS_USERNAME"], values["ATLAS_PASSWORD"],
                   values["ATLAS_GLOSSARY_GUID"])


class AtlasClient:
    def __init__(self, config: AtlasConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client

    async def _call(self, method: str, path: str, **kwargs) -> Any:
        owned = self._client is None
        client = self._client or httpx.AsyncClient(follow_redirects=False)
        try:
            response = await client.request(method, f"{self.config.base_url}{path}",
                                            auth=(self.config.username, self.config.password),
                                            timeout=self.config.timeout_seconds, **kwargs)
        except httpx.HTTPError as exc:
            raise CatalogExportError(f"Atlas unreachable ({type(exc).__name__})") from exc
        finally:
            if owned:
                await client.aclose()
        if response.status_code >= 400:
            raise CatalogExportError(f"Atlas {method} {path} returned HTTP {response.status_code}")
        return response.json() if response.content else {}

    async def existing_terms(self) -> dict[str, dict[str, Any]]:
        terms: dict[str, dict[str, Any]] = {}
        offset, limit = 0, 200
        while True:
            page = await self._call("GET", f"/api/atlas/v2/glossary/{self.config.glossary_guid}/terms",
                                    params={"limit": limit, "offset": offset})
            for term in page or []:
                terms[str(term.get("name"))] = term
            if len(page or []) < limit:
                return terms
            offset += limit

    async def upsert_terms(self, terms: list[dict[str, Any]]) -> dict[str, Any]:
        """Create new terms, update existing ones by name, then link ``dependsOn`` as ``seeAlso``."""
        existing = await self.existing_terms()
        guids: dict[str, str] = {}
        by_definition: dict[str, str] = {}
        created, updated = [], []
        for term in terms:
            current = existing.get(term["name"])
            if current is None:
                result = await self._call("POST", "/api/atlas/v2/glossary/term", json=term)
                created.append(term["name"])
            else:
                result = await self._call("PUT", f"/api/atlas/v2/glossary/term/{current['guid']}",
                                          json=term | {"guid": current["guid"]})
                updated.append(term["name"])
            guid = str(result.get("guid") or (current or {}).get("guid") or "")
            guids[term["name"]] = guid
            by_definition[term["additionalAttributes"]["graphrag.definitionId"]] = guid
        linked = []
        for term in terms:
            depends = term["additionalAttributes"].get("graphrag.dependsOn", "")
            related = [{"termGuid": by_definition[d]} for d in depends.split(",") if d and by_definition.get(d)]
            if related:
                guid = guids[term["name"]]
                await self._call("PUT", f"/api/atlas/v2/glossary/term/{guid}",
                                 json=term | {"guid": guid, "seeAlso": related})
                linked.append(term["name"])
        return {"created": created, "updated": updated, "linked": linked}


# -- OpenLineage -----------------------------------------------------------------------------


def _event_type(outcome: str) -> str:
    return {"published": "COMPLETE", "skipped": "COMPLETE", "blocked": "FAIL", "failed": "FAIL"}.get(outcome, "RUNNING")


def run_event(build: BuildRecord, *, namespace: str = "graphrag") -> dict[str, Any]:
    by_table: dict[str, list[dict[str, Any]]] = {}
    for result in build.expectations:
        by_table.setdefault(str(result["table"]), []).append(result)
    inputs = []
    for item in build.inputs:
        dataset: dict[str, Any] = {"namespace": build.source_uri or f"source://{build.source_id}",
                                   "name": item["table"]}
        assertions = by_table.get(item["table"])
        if assertions:
            dataset["inputFacets"] = {"dataQualityAssertions": {
                "_producer": PRODUCER, "_schemaURL": DQ_FACET_SCHEMA_URL,
                "assertions": [{"assertion": f"{a['kind']}:{a['id']}", "success": bool(a["passed"]),
                                **({"column": a["column"]} if a.get("column") else {})} for a in assertions],
            }}
        inputs.append(dataset)
    outputs = []
    if build.outcome == "published":
        outputs.append({"namespace": f"graphrag://{build.tenant}", "name": f"graph/{build.source_id}",
                        "outputFacets": {"outputStatistics": {
                            "_producer": PRODUCER,
                            "_schemaURL": "https://openlineage.io/spec/facets/1-0-2/OutputStatisticsOutputDatasetFacet.json"
                                          "#/$defs/OutputStatisticsOutputDatasetFacet",
                            "rowCount": int(sum(build.outputs.values()))}}})
    run_facets: dict[str, Any] = {}
    if build.error:
        run_facets["errorMessage"] = {
            "_producer": PRODUCER,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-1/ErrorMessageRunFacet.json#/$defs/ErrorMessageRunFacet",
            "message": build.error, "programmingLanguage": "python"}
    return {
        "eventType": _event_type(build.outcome),
        "eventTime": build.finished_at or build.started_at,
        "run": {"runId": build.build_id, "facets": run_facets},
        "job": {"namespace": f"{namespace}.{build.tenant}", "name": f"relational.{build.mapping_id}"},
        "inputs": inputs,
        "outputs": outputs,
        "producer": PRODUCER,
        "schemaURL": OPENLINEAGE_SCHEMA_URL,
    }


class OpenLineageClient:
    """POST run events to an OpenLineage HTTP endpoint (for example Marquez ``/api/v1/lineage``)."""

    def __init__(self, url: str, *, api_key: str = "", client: httpx.AsyncClient | None = None,
                 timeout_seconds: float = 15.0):
        assert_safe_connector_url(url, context="OpenLineage url")
        self._url, self._api_key, self._client, self._timeout = url, api_key, client, timeout_seconds

    @classmethod
    def from_env(cls) -> "OpenLineageClient | None":
        url = os.getenv("OPENLINEAGE_URL", "")
        return cls(url, api_key=os.getenv("OPENLINEAGE_API_KEY", "")) if url else None

    async def emit(self, event: dict[str, Any]) -> None:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        owned = self._client is None
        client = self._client or httpx.AsyncClient(follow_redirects=False)
        try:
            response = await client.post(self._url, json=event, headers=headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise CatalogExportError(f"OpenLineage endpoint unreachable ({type(exc).__name__})") from exc
        finally:
            if owned:
                await client.aclose()
        if response.status_code >= 400:
            raise CatalogExportError(f"OpenLineage endpoint returned HTTP {response.status_code}")


# Hand-written subsets of the required fields (see the module docstring).
ATLAS_TERM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "anchor", "longDescription", "additionalAttributes"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "anchor": {"type": "object", "required": ["glossaryGuid"],
                   "properties": {"glossaryGuid": {"type": "string", "minLength": 1}}},
        "shortDescription": {"type": "string", "maxLength": 256},
        "longDescription": {"type": "string"},
        "additionalAttributes": {"type": "object", "additionalProperties": {"type": "string"}},
    },
}
_DATASET = {"type": "object", "required": ["namespace", "name"],
            "properties": {"namespace": {"type": "string", "minLength": 1}, "name": {"type": "string", "minLength": 1}}}
OPENLINEAGE_RUN_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["eventTime", "run", "job", "producer", "schemaURL"],
    "properties": {
        "eventType": {"enum": ["START", "RUNNING", "COMPLETE", "ABORT", "FAIL", "OTHER"]},
        "eventTime": {"type": "string", "format": "date-time"},
        "run": {"type": "object", "required": ["runId"],
                "properties": {"runId": {"type": "string", "format": "uuid"}, "facets": {"type": "object"}}},
        "job": {"type": "object", "required": ["namespace", "name"],
                "properties": {"namespace": {"type": "string"}, "name": {"type": "string"}}},
        "inputs": {"type": "array", "items": _DATASET},
        "outputs": {"type": "array", "items": _DATASET},
        "producer": {"type": "string", "format": "uri"},
        "schemaURL": {"type": "string", "format": "uri"},
    },
}


__all__ = [
    "ATLAS_TERM_SCHEMA", "AtlasClient", "AtlasConfig", "CatalogExportError", "OPENLINEAGE_RUN_EVENT_SCHEMA",
    "OpenLineageClient", "atlas_term", "atlas_terms", "run_event",
]

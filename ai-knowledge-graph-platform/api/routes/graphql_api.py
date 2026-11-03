"""Read-only GraphQL consumer API over the governed MCP capability registry.

``POST /graphql`` is a stable query surface for consumers that prefer one
typed endpoint to several REST routes.  It adds no access path of its own:
every field resolves by calling ``CapabilityRegistry.call("<id>@<version>")``
with the identity built from the verified token, exactly like
``POST /knowledge/evaluate``.  Scopes, tenant binding, argument validation,
timeouts, size limits, provenance receipts and audit are therefore the ones the
MCP surface enforces; nothing here touches Neo4j, SPARQL, SHACL services or the
knowledge service directly.

Design limits, on purpose:

* **Query only.**  The schema has no mutation or subscription type.
* **No tenant argument anywhere.**  The tenant is the token's; the registry
  forces it onto every call.
* **Bounded requests.**  Query length, variables size, selection depth and the
  number of top-level fields (aliases included) are capped before execution.
* **Refusals are GraphQL errors** with ``extensions.code`` set to the
  registry's denial reason (or the capability's own error code).  GraphQL-level
  errors are HTTP 200; only auth, scope and malformed bodies are 4xx.
* Capability payloads are exposed as the ``JSON`` scalar so a capability's
  result shape can grow without breaking the GraphQL contract; ``graphStats``
  is typed because its shape is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from graphql import (
    DocumentNode, ExecutionResult, FieldNode, FragmentDefinitionNode, FragmentSpreadNode, GraphQLArgument,
    GraphQLBoolean, GraphQLError, GraphQLField, GraphQLInt, GraphQLList, GraphQLNonNull,
    GraphQLObjectType, GraphQLResolveInfo, GraphQLScalarType, GraphQLSchema, GraphQLString,
    InlineFragmentNode, OperationDefinitionNode, SelectionSetNode, ValidationContext, ValidationRule,
    execute, parse, specified_rules, validate, value_from_ast_untyped,
)
from graphql.error import GraphQLSyntaxError
from pydantic import BaseModel, ConfigDict, Field, field_validator
import structlog

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from mcp_server.capabilities import build_registry
from mcp_server.identity import CallerIdentity
from mcp_server.registry import CapabilityRegistry, DeniedCapabilityCall

log = structlog.get_logger(__name__)
router = APIRouter(tags=["GraphQL"])

AUDIT_TRANSPORT = "graphql"
MAX_QUERY_CHARS = 10_000
MAX_VARIABLES_CHARS = 100_000
MAX_OPERATION_NAME_CHARS = 128
MAX_DEPTH = 15  # the standard full introspection query (graphql-core 3.2) nests 15 deep
MAX_TOP_LEVEL_FIELDS = 10
MAX_ERROR_MESSAGE_CHARS = 500

# GraphQL field -> capability.  The only place a capability id/version is named.
CAP_GRAPH_STATS = "kg.graph.stats@1.0.0"
CAP_FACTS = "kg.facts.query@1.0.0"
CAP_ENTITY = "kg.entity.lookup@1.0.0"
CAP_PRECEDENTS = "cg.precedent.find@1.0.0"
CAP_EVALUATE = "kg.knowledge.evaluate@1.0.0"
CAP_DISCOVER = "platform.capabilities.discover@1.0.0"


def get_registry() -> CapabilityRegistry:
    """The capability registry whose audit events are labelled ``graphql``."""
    return build_registry(audit_transport=AUDIT_TRANSPORT)


# ---------------------------------------------------------------------------
# request body


class GraphQLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    variables: dict[str, Any] | None = None
    operationName: str | None = Field(default=None, max_length=MAX_OPERATION_NAME_CHARS)  # noqa: N815

    @field_validator("variables")
    @classmethod
    def _bounded_variables(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None and len(repr(value)) > MAX_VARIABLES_CHARS:
            raise ValueError(f"variables exceed {MAX_VARIABLES_CHARS} characters")
        return value


# ---------------------------------------------------------------------------
# schema


JSON = GraphQLScalarType(
    name="JSON",
    description="Arbitrary JSON value: a capability payload or structured input.",
    serialize=jsonable_encoder,
    parse_value=lambda value: value,
    parse_literal=lambda node, variables=None: value_from_ast_untyped(node, variables),
)

GRAPH_STATS = GraphQLObjectType("GraphStats", lambda: {
    "entities": GraphQLField(GraphQLInt),
    "edges": GraphQLField(GraphQLInt),
    "communities": GraphQLField(GraphQLInt),
    "documents": GraphQLField(GraphQLInt),
    "provenanceReceipt": GraphQLField(
        JSON, resolve=lambda source, _info: source.get("provenance_receipt") if isinstance(source, dict) else None),
})


@dataclass(frozen=True)
class RequestContext:
    registry: CapabilityRegistry
    identity: CallerIdentity


def _error_from_denial(denied: DeniedCapabilityCall) -> GraphQLError:
    return GraphQLError(
        (denied.detail or denied.reason)[:MAX_ERROR_MESSAGE_CHARS],
        extensions={"code": denied.reason, "capability": denied.capability, "operationId": denied.operation_id},
    )


def _error_from_result(result: dict[str, Any], capability: str) -> GraphQLError:
    error = result["error"]
    if not isinstance(error, dict):
        error = {"code": "capability_error", "detail": error}
    detail = error.get("message") or error.get("detail") or error.get("code") or "capability error"
    return GraphQLError(
        str(detail)[:MAX_ERROR_MESSAGE_CHARS],
        extensions={"code": str(error.get("code") or "capability_error"), "capability": capability},
    )


async def _invoke(info: GraphQLResolveInfo, capability: str, args: dict[str, Any]) -> Any:
    """Call one capability through the registry; map refusals to GraphQL errors."""
    context: RequestContext = info.context
    clean_args = {name: value for name, value in args.items() if value is not None}
    try:
        result = await context.registry.call(capability, clean_args, context.identity)
    except GraphQLError:
        raise
    except Exception as exc:  # noqa: BLE001 - never leak backend detail to the consumer
        log.error("graphql.capability_failed", capability=capability, error=str(exc)[:300])
        raise GraphQLError("internal error", extensions={"code": "internal_error", "capability": capability}) from None
    if isinstance(result, DeniedCapabilityCall):
        raise _error_from_denial(result)
    if isinstance(result, dict) and "error" in result:
        raise _error_from_result(result, capability)
    return result


def _resolver(capability: str, rename: dict[str, str] | None = None) -> Callable[..., Awaitable[Any]]:
    """Resolver for a field: its GraphQL arguments become the capability's arguments."""
    rename = rename or {}

    async def resolve(_source: Any, info: GraphQLResolveInfo, **kwargs: Any) -> Any:
        return await _invoke(info, capability, {rename.get(name, name): value for name, value in kwargs.items()})

    return resolve


def _arg(type_: Any) -> GraphQLArgument:
    return GraphQLArgument(type_)


def _build_schema() -> GraphQLSchema:
    required_string = GraphQLNonNull(GraphQLString)
    query = GraphQLObjectType("Query", {
        "graphStats": GraphQLField(
            GRAPH_STATS, description="Entity, edge, community and document counts for the caller's tenant.",
            resolve=_resolver(CAP_GRAPH_STATS)),
        "facts": GraphQLField(
            JSON, description="Governed fact lookup for a natural-language question.",
            args={"question": _arg(required_string), "limit": _arg(GraphQLInt)},
            resolve=_resolver(CAP_FACTS)),
        "entity": GraphQLField(
            JSON, description="Entity lookup by name, optionally as of a date.",
            args={"name": _arg(required_string), "asOf": _arg(GraphQLString), "limit": _arg(GraphQLInt)},
            resolve=_resolver(CAP_ENTITY, {"asOf": "as_of"})),
        "precedents": GraphQLField(
            JSON, description="Explainable precedents for a policy version.",
            args={"policyVersionId": _arg(required_string), "limit": _arg(GraphQLInt)},
            resolve=_resolver(CAP_PRECEDENTS, {"policyVersionId": "policy_version_id"})),
        "evaluateKnowledge": GraphQLField(
            JSON, description="Evaluate the domain's active contextual-knowledge release.",
            args={
                "domain": _arg(required_string),
                "context": _arg(GraphQLNonNull(JSON)),
                "facts": _arg(JSON),
                "semanticCandidates": _arg(GraphQLList(GraphQLNonNull(GraphQLString))),
                "expectApplicableKnowledge": _arg(GraphQLBoolean),
            },
            resolve=_resolver(CAP_EVALUATE, {
                "semanticCandidates": "semantic_candidates",
                "expectApplicableKnowledge": "expect_applicable_knowledge"})),
        "capabilities": GraphQLField(
            JSON, description="Capabilities the caller is entitled to use.",
            resolve=_resolver(CAP_DISCOVER)),
    })
    return GraphQLSchema(query=query)


SCHEMA = _build_schema()


# ---------------------------------------------------------------------------
# guardrails


class _LimitsRule(ValidationRule):
    """Bounds selection depth and the number of top-level fields per operation.

    Fragments are followed (a cycle is cut off; the standard cycle rule reports
    it), so neither limit can be dodged by moving the selection into a fragment.
    """

    def __init__(self, context: ValidationContext) -> None:
        super().__init__(context)

    def enter_operation_definition(self, node: OperationDefinitionNode, *_args: Any) -> None:
        top_level = self._count_top_level(node.selection_set, frozenset())
        if top_level > MAX_TOP_LEVEL_FIELDS:
            self.report_error(GraphQLError(
                f"Query has {top_level} top-level fields; the limit is {MAX_TOP_LEVEL_FIELDS}.", node))
        depth = self._depth(node.selection_set, frozenset())
        if depth > MAX_DEPTH:
            self.report_error(GraphQLError(f"Query depth {depth} exceeds the limit of {MAX_DEPTH}.", node))

    def _fragment(self, name: str) -> FragmentDefinitionNode | None:
        return self.context.get_fragment(name)

    def _count_top_level(self, selection_set: SelectionSetNode, seen: frozenset[str]) -> int:
        total = 0
        for selection in selection_set.selections:
            if isinstance(selection, FieldNode):
                total += 1
            elif isinstance(selection, InlineFragmentNode):
                total += self._count_top_level(selection.selection_set, seen)
            elif isinstance(selection, FragmentSpreadNode) and selection.name.value not in seen:
                fragment = self._fragment(selection.name.value)
                if fragment is not None:
                    total += self._count_top_level(fragment.selection_set, seen | {selection.name.value})
        return total

    def _depth(self, selection_set: SelectionSetNode, seen: frozenset[str]) -> int:
        deepest = 0
        for selection in selection_set.selections:
            if isinstance(selection, FieldNode):
                child = self._depth(selection.selection_set, seen) if selection.selection_set else 0
                deepest = max(deepest, 1 + child)
            elif isinstance(selection, InlineFragmentNode):
                deepest = max(deepest, self._depth(selection.selection_set, seen))
            elif isinstance(selection, FragmentSpreadNode) and selection.name.value not in seen:
                fragment = self._fragment(selection.name.value)
                if fragment is not None:
                    deepest = max(deepest, self._depth(fragment.selection_set, seen | {selection.name.value}))
        return deepest


_RULES = [*specified_rules, _LimitsRule]


def _parse_and_validate(query: str) -> tuple[DocumentNode | None, list[GraphQLError]]:
    try:
        document = parse(query)
    except GraphQLSyntaxError as exc:
        return None, [exc]
    return document, validate(SCHEMA, document, _RULES)


def _error_payload(error: GraphQLError) -> dict[str, Any]:
    payload = error.formatted
    extensions = dict(payload.get("extensions") or {})
    extensions.setdefault("code", "graphql_validation_failed" if error.nodes else "bad_request")
    return {**payload, "extensions": extensions}


def _result_payload(result: ExecutionResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"data": result.data}
    if result.errors:
        payload["errors"] = [_error_payload(error) for error in result.errors]
    return payload


async def run_query(
    registry: CapabilityRegistry, identity: CallerIdentity, request: GraphQLRequest,
) -> dict[str, Any]:
    """Validate and execute one request; always returns a ``{data, errors}`` payload."""
    document, errors = _parse_and_validate(request.query)
    if errors or document is None:
        return {"data": None, "errors": [_error_payload(error) for error in errors]}
    outcome = execute(
        SCHEMA, document, context_value=RequestContext(registry, identity),
        variable_values=request.variables, operation_name=request.operationName)
    if not isinstance(outcome, ExecutionResult):
        outcome = await outcome
    return _result_payload(outcome)


@router.post("/graphql", dependencies=[Depends(require_scope("read"))])
async def graphql_endpoint(
    body: GraphQLRequest,
    user: dict = Depends(get_current_user),
    tenant: str = Depends(get_tenant),
) -> dict[str, Any]:
    """Run a read-only GraphQL query; every field is a governed capability call."""
    identity = CallerIdentity.from_claims(user, tenant=tenant)
    return await run_query(get_registry(), identity, body)

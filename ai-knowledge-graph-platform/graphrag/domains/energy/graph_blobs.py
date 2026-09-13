"""Content-addressed storage for published Energy RDF graphs.

A published version's bytes live in a file named by their own hash; the
governance database stores the hash, not the payload. That makes a publish
idempotent by construction -- two workers publishing byte-identical content
compute the same path and race harmlessly -- and keeps multi-megabyte graphs
out of the event-loop thread on every read.

Canonical form: sorted N-Triples
--------------------------------
The hash is taken over ``sorted(graph.serialize(format="nt").splitlines())``,
not over Turtle. Turtle's prefix selection, predicate grouping and blank-node
labelling are not stable across rdflib versions, so a Turtle-derived hash
would change on a dependency bump and silently defeat the idempotency it
exists to provide. N-Triples has one triple per line and no such freedom;
sorting removes the remaining serialisation-order variance.

Namespace prefixes are stored alongside, because N-Triples drops them and
``DatasetPublisher`` deliberately carries the candidate's prefixes onto the
published graph so ``export_turtle()`` stays readable.

Writes go through a temporary file in the destination directory followed by
``os.replace``, which is atomic on POSIX and on NTFS -- the same idiom
``graphrag/semantic_model/compiler.py`` already uses for generated artifacts.
A torn write can therefore never be observed as a valid version.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from rdflib import Graph

_ENCODING = "utf-8"


def canonical_bytes(graph: Graph) -> bytes:
    """The graph's canonical, order-independent N-Triples serialisation."""
    serialized = graph.serialize(format="nt")
    lines = sorted(line for line in serialized.splitlines() if line.strip())
    return ("\n".join(lines) + "\n").encode(_ENCODING)


def content_hash(graph: Graph) -> str:
    """sha256 of the canonical form -- the version's content identity."""
    return hashlib.sha256(canonical_bytes(graph)).hexdigest()


def prefixes_of(graph: Graph) -> dict[str, str]:
    return {prefix: str(namespace) for prefix, namespace in graph.namespaces()}


def blob_path(root: Path, digest: str) -> Path:
    """``<root>/<first two hex chars>/<digest>.nt`` -- sharded so a directory
    listing stays usable once there are many versions."""
    return Path(root) / digest[:2] / f"{digest}.nt"


def write_graph(graph: Graph, root: Path) -> tuple[str, Path, int]:
    """Store `graph`'s canonical bytes under `root`; return (hash, path, size).

    Writing content that is already stored is a no-op: the path is derived
    from the content, so an existing file with that name necessarily holds
    exactly these bytes.
    """
    payload = canonical_bytes(graph)
    digest = hashlib.sha256(payload).hexdigest()
    path = blob_path(root, digest)
    if path.exists():
        return digest, path, len(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{digest}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest, path, len(payload)


def read_graph(path: Path, prefixes: dict[str, str] | None = None) -> Graph:
    """Load a stored version back, rebinding its namespace prefixes."""
    graph = Graph()
    graph.parse(str(path), format="nt")
    for prefix, namespace in (prefixes or {}).items():
        graph.bind(prefix, namespace)
    return graph


def dumps_prefixes(prefixes: dict[str, str]) -> str:
    return json.dumps(prefixes, sort_keys=True)


def loads_prefixes(payload: str) -> dict[str, str]:
    return json.loads(payload) if payload else {}


__all__ = [
    "blob_path", "canonical_bytes", "content_hash", "dumps_prefixes",
    "loads_prefixes", "prefixes_of", "read_graph", "write_graph",
]

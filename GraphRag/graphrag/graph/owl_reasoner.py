"""OWL-RL reasoner wrapper for rdflib graphs.

Applies the OWL 2 RL entailment rules (a tractable, polynomial-time subset
of OWL DL) to an rdflib Graph, materialising implicit triples that follow from:

  - ``rdfs:subClassOf`` propagation (transitive closure)
  - ``owl:SymmetricProperty`` axioms  (A R B → B R A)
  - ``owl:InverseOf`` property pairs
  - Property domain / range constraints
  - Simple property chain axioms (OWL RL subset)

The closure is applied by the ``owlrl`` library (pure-Python, pip-installable).

Usage::

    from graphrag.graph.owl_reasoner import OWLRLReasoner

    reasoner = OWLRLReasoner.from_turtle("exports/graph_export.ttl")
    n_new    = reasoner.apply_closure()
    print(f"OWL-RL inferred {n_new} new triples")
    print(f"Graph consistent: {reasoner.is_consistent()}")
    reasoner.serialize("exports/graph_expanded.ttl")

Integration with export_rdf.py::

    python scripts/export_rdf.py --tenant default --infer
"""

from __future__ import annotations

from pathlib import Path

import structlog
from rdflib import Graph
from rdflib.namespace import OWL

log = structlog.get_logger(__name__)


class OWLRLReasoner:
    """Apply OWL 2 RL entailment rules to an rdflib Graph via owlrl.

    The graph is mutated **in-place** by ``apply_closure()``.  Clone the
    graph first if you need to preserve the original asserted triples:

    .. code-block:: python

        import copy
        original = copy.deepcopy(g)
        reasoner = OWLRLReasoner(g)
        reasoner.apply_closure()
        new_triples = reasoner.inferred_triples(original)

    Parameters
    ----------
    graph :
        rdflib Graph to reason over.
    """

    def __init__(self, graph: Graph) -> None:
        self._g               = graph
        self._closure_applied = False

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_turtle(cls, ttl_path: Path | str) -> "OWLRLReasoner":
        """Parse a Turtle file and return a reasoner instance.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        """
        path = Path(ttl_path)
        if not path.exists():
            raise FileNotFoundError(f"Turtle file not found: {path}")
        g = Graph()
        g.parse(str(path), format="turtle")
        log.info("owl_reasoner.loaded", path=str(path), triples=len(g))
        return cls(g)

    # ── Reasoning ──────────────────────────────────────────────────────────────

    def apply_closure(self) -> int:
        """Run OWL-RL entailment rules and return the count of new triples.

        The method is **idempotent** — calling it twice returns 0 on the
        second call without re-running the reasoner.

        Returns
        -------
        int
            Number of new triples materialised by the closure.

        Raises
        ------
        ImportError
            If the ``owlrl`` package is not installed.
        """
        try:
            import owlrl  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "owlrl is required for OWL-RL reasoning: "
                "pip install owlrl>=6.0.0"
            ) from exc

        if self._closure_applied:
            return 0

        before = len(self._g)
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(self._g)
        self._closure_applied = True
        added = len(self._g) - before
        log.info("owl_reasoner.closure_applied",
                 new_triples=added, total=len(self._g))
        return added

    def is_consistent(self) -> bool:
        """Return False if OWL-RL entailment derives owl:Nothing.

        Under OWL-RL semantics a graph is inconsistent when an individual
        is typed as ``owl:Nothing`` (the empty class) — i.e. there exists a
        triple ``(individual, rdf:type, owl:Nothing)``.  The ``owlrl``
        reasoner materialises such a triple when it detects a contradiction.

        Returns
        -------
        bool
            ``True`` if the graph is consistent, ``False`` otherwise.
        """
        from rdflib.namespace import RDF
        return not any(self._g.subjects(RDF.type, OWL.Nothing))

    def inferred_triples(self, original: Graph) -> list[tuple]:
        """Return only the triples added by the OWL-RL closure.

        Parameters
        ----------
        original :
            The graph state *before* ``apply_closure()`` was called.
            Typically a ``copy.deepcopy`` taken before constructing this
            reasoner instance.

        Returns
        -------
        list[tuple]
            ``[(subject, predicate, object)]`` for each inferred triple.
        """
        original_set = set(original)
        return [
            (s, p, o)
            for s, p, o in self._g
            if (s, p, o) not in original_set
        ]

    # ── Serialisation ──────────────────────────────────────────────────────────

    def serialize(self, path: Path | str, fmt: str = "turtle") -> None:
        """Write the expanded graph (asserted + inferred) to a file.

        Parameters
        ----------
        path :
            Destination file path.  Parent directories are created if absent.
        fmt :
            rdflib serialisation format (default ``"turtle"``).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._g.serialize(destination=str(out), format=fmt)
        log.info("owl_reasoner.serialized",
                 path=str(out), triples=len(self._g), format=fmt)

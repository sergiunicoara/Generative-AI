"""Smoke test for the GNN scorer — run with: python test_gnn.py"""
import numpy as np
from graphrag.graph.gnn_scorer import GNNScorer

def test_gat():
    scorer = GNNScorer(gnn_type="gat", num_layers=2, alpha=0.6, beta=0.4)
    dim = 16
    np.random.seed(42)
    q = np.random.randn(dim).tolist()

    chunk_entities = [
        {"chunk_id": "c1", "entity_name": "SpaceX",    "entity_type": "ORG",    "embedding": np.random.randn(dim).tolist()},
        {"chunk_id": "c1", "entity_name": "Elon Musk", "entity_type": "PERSON", "embedding": np.random.randn(dim).tolist()},
        {"chunk_id": "c2", "entity_name": "Tesla",     "entity_type": "ORG",    "embedding": np.random.randn(dim).tolist()},
    ]
    entity_edges = [
        {"src": "Elon Musk", "tgt": "SpaceX", "weight": 1.0},
        {"src": "Elon Musk", "tgt": "Tesla",  "weight": 1.0},
    ]
    chunks = [
        {"chunk_id": "c1", "text": "SpaceX launched Falcon 9", "rerank_score": 8.5},
        {"chunk_id": "c2", "text": "Tesla produced Model S",   "rerank_score": 3.1},
        {"chunk_id": "c3", "text": "No entity mention here",   "score": 0.2},
    ]

    result = scorer.score(q, chunks, chunk_entities, entity_edges)
    print("--- GAT scorer results ---")
    for c in result:
        print(f"  {c['chunk_id']:4s}  gnn={c['gnn_score']:.4f}  final={c['final_score']:.4f}")
    assert all("gnn_score" in c and "final_score" in c for c in result)
    print("GAT test PASSED\n")

def test_gcn():
    scorer = GNNScorer(gnn_type="gcn", num_layers=2, alpha=0.6, beta=0.4)
    dim = 16
    np.random.seed(7)
    q = np.random.randn(dim).tolist()

    chunk_entities = [
        {"chunk_id": "c1", "entity_name": "Apple",    "entity_type": "ORG",    "embedding": np.random.randn(dim).tolist()},
        {"chunk_id": "c2", "entity_name": "Tim Cook",  "entity_type": "PERSON", "embedding": np.random.randn(dim).tolist()},
    ]
    entity_edges = [
        {"src": "Apple", "tgt": "Tim Cook", "weight": 2.0},
    ]
    chunks = [
        {"chunk_id": "c1", "text": "Apple launched iPhone 16", "rerank_score": 5.0},
        {"chunk_id": "c2", "text": "Tim Cook is Apple CEO",    "rerank_score": 4.0},
    ]

    result = scorer.score(q, chunks, chunk_entities, entity_edges)
    print("--- GCN scorer results ---")
    for c in result:
        print(f"  {c['chunk_id']:4s}  gnn={c['gnn_score']:.4f}  final={c['final_score']:.4f}")
    assert all("gnn_score" in c and "final_score" in c for c in result)
    print("GCN test PASSED\n")

def test_empty_entities():
    scorer = GNNScorer()
    chunks = [{"chunk_id": "c1", "text": "some text", "score": 0.5}]
    result = scorer.score([0.1, 0.2], chunks, [], [])
    assert result[0]["gnn_score"] == 0.0
    print("Empty-entity fallback test PASSED\n")

if __name__ == "__main__":
    test_gat()
    test_gcn()
    test_empty_entities()
    print("All GNN tests passed.")

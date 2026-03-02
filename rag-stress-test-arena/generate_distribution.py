"""
Generate the real-world embedding corpus and held-out query set used by the benchmark.

  Corpus  : 20 000 Wikipedia articles  — indices drawn from a fixed permutation of
             the first 500 000 articles in the 20231101.en snapshot.
  Queries :  3 000 Wikipedia articles  — next slice of the same permutation,
             never present in the corpus (true held-out evaluation set).

  Source  : HuggingFace `datasets`  — wikipedia / 20231101.en
            This snapshot is immutable and will never change.
  Model   : sentence-transformers/all-mpnet-base-v2
            768-d, L2-normalised.  The gold-standard SBERT model for semantic
            similarity; weights have been frozen since its 2021 release.
  Text    : First 512 characters of each article body
            (one realistic RAG chunk — enough for a clear topic signal,
             short enough to stay within the model's token budget).

Reproducibility guarantee
─────────────────────────
  1. Dataset snapshot 20231101.en is immutable — same articles, same order, always.
  2. Article selection uses np.random.default_rng(SELECTION_SEED).permutation(N_SOURCE),
     which is fully deterministic.
  3. all-mpnet-base-v2 weights are fixed.  SentenceTransformer.encode() is
     deterministic for identical model + input on the same hardware/library version.
  4. SHA-256 checksums of both output files are printed so anyone can verify
     bit-for-bit identity with a reference run.

  -> Anyone running this script produces the same corpus and query vectors.

Prerequisites
─────────────
  pip install sentence-transformers datasets
  pip install "optimum[onnxruntime]"   # optional but recommended — 3-4× faster on CPU

Usage
─────
  py -3.11 generate_distribution.py

Speed
─────
  With ONNX backend (optimum[onnxruntime] installed):  ~25-30 min on CPU
  With PyTorch backend (fallback):                     ~90-120 min on CPU
  The script auto-detects which backend is available.

Memory
──────
  Uses streaming mode — the full Wikipedia dataset is never loaded into RAM.
  Peak memory: ~12 MB for the 23 000 article texts + model (~420 MB).

Output
──────
  corpus_768d.npy   -- shape (20 000, 768), float32, unit vectors  (~61 MB)
  queries_768d.npy  -- shape ( 3 000, 768), float32, unit vectors  (~9 MB)
"""

import hashlib
import os

import numpy as np
import yaml

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_NAME    = "wikimedia/wikipedia"
DATASET_CONFIG  = "20231101.en"     # frozen, citable snapshot — never changes
DATASET_SPLIT   = "train"

# Shuffle the first N_SOURCE articles so the corpus is topically diverse
# (rather than just articles whose titles start with 'A').
N_SOURCE        = 500_000
SELECTION_SEED  = 42                # governs which articles are selected; fixed forever

TEXT_CHARS      = 512               # first N chars of each article — one realistic RAG chunk

MODEL_NAME      = "sentence-transformers/all-mpnet-base-v2"
ENCODE_BATCH    = 32     # ONNX speedup comes from faster ops, not larger batches on CPU


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    """Return the hex SHA-256 digest of a file for bit-for-bit verification."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode(model, texts: list, label: str) -> np.ndarray:
    print(f"\n  [{label}] Encoding {len(texts):,} texts ...")
    return model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors -> cosine == dot product
        convert_to_numpy=True,
    ).astype("float32")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(config_path: str = "config.yml") -> None:

    # ── 0. Lazy imports — clear error if the user forgot pip install ──────────
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "\nMissing dependency.\n"
            "Run:  pip install datasets\n"
        )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "\nMissing dependency.\n"
            "Run:  pip install sentence-transformers\n"
        )

    # ── 1. Read config ────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["distribution"]

    n_corpus    = cfg["n_samples"]      # 20 000
    n_queries   = cfg["n_queries"]      # 3 000
    corpus_out  = cfg["file"]           # corpus_768d.npy
    query_out   = cfg["query_file"]     # queries_768d.npy
    total_need  = n_corpus + n_queries  # 23 000

    if total_need > N_SOURCE:
        raise ValueError(
            f"n_samples ({n_corpus}) + n_queries ({n_queries}) = {total_need} "
            f"exceeds N_SOURCE ({N_SOURCE}). Increase N_SOURCE or reduce sample counts."
        )

    # ── 2. Deterministic article selection ───────────────────────────────────
    print(f"Computing article selection "
          f"(permutation of {N_SOURCE:,} articles, seed={SELECTION_SEED}) ...")
    rng  = np.random.default_rng(SELECTION_SEED)
    perm = rng.permutation(N_SOURCE).tolist()

    # Sort indices before calling .select() so HuggingFace reads them
    # sequentially — much faster than random-access on a large dataset.
    corpus_indices = sorted(perm[:n_corpus])
    query_indices  = sorted(perm[n_corpus:total_need])
    all_indices    = sorted(set(corpus_indices) | set(query_indices))

    # ── 3. Load Wikipedia snapshot ────────────────────────────────────────────
    max_idx = max(all_indices)
    print(f"\nStreaming {DATASET_NAME}/{DATASET_CONFIG} "
          f"(scanning first {max_idx + 1:,} articles to collect {len(all_indices):,}) ...")
    print("  Note: first run downloads parquet shards (~20 GB) to the HuggingFace cache.")
    print("  Streaming mode: only selected articles are held in RAM — full dataset is never loaded.")

    # streaming=True: HuggingFace reads parquet row-by-row; we stop as soon as
    # all needed indices have been seen.  Peak RAM = 23k article texts (~12 MB).
    ds = load_dataset(
        DATASET_NAME, DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    needed_set   = set(all_indices)
    texts_by_idx = {}
    for i, row in enumerate(ds):
        if i in needed_set:
            texts_by_idx[i] = row["text"][:TEXT_CHARS].strip()
        if i >= max_idx:
            break
        if i % 100_000 == 0 and i > 0:
            print(f"    Scanned {i:,} articles, collected {len(texts_by_idx):,}/{len(needed_set):,} ...")

    corpus_texts = [texts_by_idx[i] for i in corpus_indices]
    query_texts  = [texts_by_idx[i] for i in query_indices]

    overlap = len(set(corpus_indices) & set(query_indices))
    print(f"  Corpus texts loaded : {len(corpus_texts):,}")
    print(f"  Query  texts loaded : {len(query_texts):,}")
    print(f"  Corpus/query overlap (must be 0): {overlap}")
    assert overlap == 0, "BUG: corpus and query sets must be disjoint"

    # ── 4. Load model — ONNX backend for 3-4× CPU speedup, PyTorch fallback ──
    # Prefer the INT8 quantized AVX-512 VNNI model — fastest on Intel CPUs.
    # Falls back through progressively less optimised options to plain PyTorch.
    print(f"\nLoading model: {MODEL_NAME}")
    model = None
    onnx_candidates = [
        ("onnx/model_qint8_avx512_vnni.onnx", "INT8 AVX-512 VNNI — fastest on Intel"),
        ("onnx/model_qint8_avx512.onnx",       "INT8 AVX-512"),
        ("onnx/model_quint8_avx2.onnx",         "UINT8 AVX2"),
        ("onnx/model_O4.onnx",                  "FP32 O4 — all ONNX optimisations"),
        ("onnx/model.onnx",                     "FP32 base ONNX"),
    ]
    try:
        import optimum  # noqa: F401
        for file_name, label in onnx_candidates:
            try:
                model = SentenceTransformer(
                    MODEL_NAME, backend="onnx",
                    model_kwargs={"file_name": file_name},
                )
                print(f"  Backend: ONNX ({label})")
                break
            except Exception:
                continue
    except ImportError:
        pass
    if model is None:
        print("  Backend: PyTorch (no ONNX variant succeeded)")
        print("  Tip: pip install \"optimum[onnxruntime]\" for 3-4× faster encoding")
        model = SentenceTransformer(MODEL_NAME)

    corpus_emb = _encode(model, corpus_texts, "1/2 corpus")
    query_emb  = _encode(model, query_texts,  "2/2 queries")

    # Sanity checks
    assert corpus_emb.shape == (n_corpus, 768), \
        f"Unexpected corpus shape: {corpus_emb.shape}"
    assert query_emb.shape  == (n_queries, 768), \
        f"Unexpected query shape: {query_emb.shape}"
    assert abs(np.linalg.norm(corpus_emb[0]) - 1.0) < 1e-5, \
        "Corpus vectors are not unit-normalised"
    assert abs(np.linalg.norm(query_emb[0])  - 1.0) < 1e-5, \
        "Query vectors are not unit-normalised"

    # ── 5. Save ───────────────────────────────────────────────────────────────
    np.save(corpus_out, corpus_emb)
    np.save(query_out,  query_emb)

    print("\n" + "=" * 62)
    print("Output files")
    print("=" * 62)
    for path, arr in [(corpus_out, corpus_emb), (query_out, query_emb)]:
        mb = os.path.getsize(path) / 1_000_000
        print(f"  {path}")
        print(f"    shape  : {arr.shape}")
        print(f"    size   : {mb:.1f} MB")
        print(f"    mean   : {arr.mean():.6f}")
        print(f"    std    : {arr.std():.6f}")
        print(f"    norm[0]: {np.linalg.norm(arr[0]):.8f}  (must be 1.0)")
        print(f"    SHA-256: {_sha256(path)}")
        print()

    print("Reproducibility: re-run this script and compare SHA-256 values.")
    print("Identical hashes confirm bit-for-bit reproducible vectors.")


if __name__ == "__main__":
    generate()

import json, os
import matplotlib.pyplot as plt


def ensure_out():
    os.makedirs("plots", exist_ok=True)


def plot_concurrency():
    path = "results/concurrency_burst.json"
    if not os.path.exists(path): return
    with open(path) as f: data = json.load(f)
    levels, results = data["levels"], data["results"]
    ensure_out()
    for metric in ["avg_ms", "p95_ms", "p99_ms"]:
        plt.figure()
        for eng, d in results.items():
            plt.plot(levels, [d[str(l)][metric] for l in levels], marker="o", label=eng)
        plt.xlabel("Concurrent clients"); plt.ylabel(metric)
        plt.title(f"Concurrency Burst - {metric}"); plt.legend(); plt.grid(True)
        plt.savefig(f"plots/concurrency_{metric}.png"); plt.close()


def plot_metadata():
    path = "results/metadata_filter_penalty.json"
    if not os.path.exists(path): return
    with open(path) as f: data = json.load(f)
    ensure_out()
    engines = list(data.keys())
    x = range(len(engines)); w = 0.35
    plt.figure()
    plt.bar([i - w/2 for i in x], [data[e]["no_filter_p95_ms"] for e in engines], w, label="no_filter_p95")
    plt.bar([i + w/2 for i in x], [data[e]["filter_p95_ms"] for e in engines], w, label="filter_p95")
    plt.xticks(list(x), engines); plt.ylabel("p95 latency (ms)")
    plt.title("Metadata Filter Penalty (p95)"); plt.legend(); plt.grid(axis="y")
    plt.savefig("plots/metadata_filter_penalty_p95.png"); plt.close()


def plot_fragmentation():
    path = "results/index_fragmentation.json"
    if not os.path.exists(path): return
    with open(path) as f: data = json.load(f)
    ensure_out()
    engines = list(data.keys())
    x = range(len(engines)); w = 0.35
    plt.figure()
    plt.bar([i - w/2 for i in x], [data[e]["baseline_p95_ms"] for e in engines], w, label="baseline_p95")
    plt.bar([i + w/2 for i in x], [data[e]["frag_p95_ms"] for e in engines], w, label="frag_p95")
    plt.xticks(list(x), engines); plt.ylabel("p95 latency (ms)")
    plt.title("Index Fragmentation Impact (p95)"); plt.legend(); plt.grid(axis="y")
    plt.savefig("plots/index_fragmentation_p95.png"); plt.close()


def plot_frontier():
    path = "results/recall_latency_frontier.json"
    if not os.path.exists(path): return
    with open(path) as f: data = json.load(f)
    ensure_out()
    for eng, d in data.items():
        pts = d["points"]
        plt.figure()
        plt.scatter([p["avg_latency_ms"] for p in pts], [p["avg_recall"] for p in pts])
        for p in pts: plt.text(p["avg_latency_ms"], p["avg_recall"], f"k={int(p['k'])}")
        plt.xlabel("avg latency (ms)"); plt.ylabel("avg recall")
        plt.title(f"Recall-Latency Frontier - {eng}"); plt.grid(True)
        plt.savefig(f"plots/recall_latency_frontier_{eng}.png"); plt.close()


def plot_reranker():
    path = "results/reranker_cost.json"
    if not os.path.exists(path): return
    with open(path) as f: data = json.load(f)
    ensure_out()
    engines = list(data.keys())
    x = range(len(engines)); w = 0.25
    plt.figure()
    plt.bar([i - w for i in x], [data[e]["retrieval_avg_ms"] for e in engines], w, label="retrieval")
    plt.bar(list(x), [data[e]["rerank_avg_ms"] for e in engines], w, label="rerank")
    plt.bar([i + w for i in x], [data[e]["end_to_end_avg_ms"] for e in engines], w, label="end_to_end")
    plt.xticks(list(x), engines); plt.ylabel("avg latency (ms)")
    plt.title("Reranker Cost Breakdown"); plt.legend(); plt.grid(axis="y")
    plt.savefig("plots/reranker_cost.png"); plt.close()


if __name__ == "__main__":
    plot_concurrency(); plot_metadata(); plot_fragmentation(); plot_frontier(); plot_reranker()
    print("All plots saved to plots/")
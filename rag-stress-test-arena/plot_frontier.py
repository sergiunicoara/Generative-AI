"""
Plot the recall-latency frontier for all engines.
Run after: py -3.11 run_all.py
Output:    results/summary/frontier.png
"""
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

RESULTS_PATH = "results/summary/frontier_results.json"
OUTPUT_PATH  = "results/summary/frontier.png"

ENGINE_COLORS = {
    "elasticsearch": "#E07B39",
    "qdrant":        "#6C63FF",
    "redis":         "#D62728",
    "pgvector":      "#2CA02C",
}


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))

    for eng_name, v in data.items():
        if eng_name == "metadata":
            continue
        pts    = v["points"]
        xs     = [p["avg_ms"]       for p in pts]
        ys     = [p["recall_at_10"] for p in pts]
        efs    = [p["ef_search"]    for p in pts]
        color  = ENGINE_COLORS.get(eng_name, "grey")

        ax.plot(xs, ys, marker="o", label=eng_name, color=color, linewidth=2, markersize=6)
        for x, y, ef in zip(xs, ys, efs):
            ax.annotate(
                f"ef={ef}",
                xy=(x, y),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

    meta = data.get("metadata", {})
    ax.set_xlabel("Avg latency per query (ms)", fontsize=11)
    ax.set_ylabel(f"Recall@{meta.get('recall_k', 10)}", fontsize=11)
    ax.set_title(
        f"Recall-Latency Frontier\n"
        f"(m=16, ef_construction=100, cosine distance, "
        f"{meta.get('ef_search_levels', [])} ef levels)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=1.05)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

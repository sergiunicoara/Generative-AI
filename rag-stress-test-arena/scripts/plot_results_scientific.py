import json
import matplotlib.pyplot as plt
import os

def plot_scientific():
    with open("results/summary/scientific_results.json", "r") as f:
        data = json.load(f)

    engines = list(data.keys())
    p95_means = [data[e]['p95_mean'] for e in engines]
    p95_cis = [data[e]['p95_ci'] for e in engines]
    recalls = [data[e]['recall_mean'] for e in engines]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(engines, p95_means, yerr=p95_cis, capsize=5, color='skyblue', label='P95 Latency (ms)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Scientific Benchmark: Latency vs Recall')

    ax2 = ax1.twinx()
    ax2.plot(engines, recalls, color='red', marker='o', label='Recall@10')
    ax2.set_ylabel('Recall')
    ax2.set_ylim(0, 1.1)

    fig.tight_layout()
    plt.savefig("plots/scientific_comparison.png")
    print("Plot saved to plots/scientific_comparison.png")

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_scientific()

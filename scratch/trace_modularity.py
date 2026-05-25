import numpy as np
import matplotlib.pyplot as plt
from algorithms.ho_rlsl import HORLSLRunner
from fcca import recursive_repartitioning
from analysis.paper_alignment import load_condition_stream, paper_horlsl_config
from core.config import config

def compute_modularity(adj, partition):
    m = np.sum(adj) / 2.0
    if m == 0: return 0.0
    n = adj.shape[0]
    k = np.sum(adj, axis=1)
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if partition[i] == partition[j]:
                Q += (adj[i, j] - (k[i] * k[j]) / (2.0 * m))
    return Q / (2.0 * m)

def communities_to_partition(communities, n_nodes):
    partition = np.zeros(n_nodes, dtype=int)
    for idx, comm in enumerate(communities):
        for node in comm:
            partition[node] = idx
    return partition

def trace_modularity():
    print("Tracing modularity frame-by-frame...")
    stream = load_condition_stream("incorrect")
    cfg = paper_horlsl_config()
    res = HORLSLRunner(cfg).run(stream)
    lowrank = res.lowrank_stream
    
    q_trace = []
    threshold = 0.15
    
    for t in range(lowrank.shape[0]):
        mean_adj = np.mean(lowrank[t], axis=2)
        mean_adj[mean_adj < threshold] = 0
        communities = recursive_repartitioning(mean_adj, nodes=list(range(mean_adj.shape[0])), min_size=4)
        partition = communities_to_partition(communities, mean_adj.shape[0])
        
        q_subs = []
        for s in range(lowrank.shape[3]):
            sub_adj = lowrank[t, :, :, s]
            sub_adj[sub_adj < threshold] = 0
            q_subs.append(compute_modularity(sub_adj, partition))
        q_trace.append(np.mean(q_subs))
        if t % 50 == 0: print(f" Frame {t}/{lowrank.shape[0]}...")

    peak_idx = np.argmax(q_trace)
    print(f"Peak Modularity: {q_trace[peak_idx]:.4f} at Frame {peak_idx} ({peak_idx-128}ms)")
    ern_q = np.mean(q_trace[114:146])
    print(f"Mean ERN Modularity (114-146): {ern_q:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(q_trace)) - 128, q_trace, label='Avg Modularity (Q)', color='purple')
    plt.axvline(0, color='red', linestyle='--')
    plt.savefig(config.paths.OUTPUTS_DIR / "modularity_trace.png")

if __name__ == "__main__":
    trace_modularity()

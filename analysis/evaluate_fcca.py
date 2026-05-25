import numpy as np
import pandas as pd
from core.config import config
from algorithms.ho_rlsl import HORLSLRunner
from fcca import fcca_on_lowrank_interval
from analysis.paper_alignment import derive_primary_ern_intervals
from analysis.paper_alignment import label_detected_intervals
from analysis.paper_alignment import load_condition_stream
from analysis.paper_alignment import paper_horlsl_config


def slugify_interval(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")

def compute_modularity(adj: np.ndarray, partition: np.ndarray) -> float:
    """
    Computes Q (Modularity) for a given partition of a network.
    Q = 1/(2m) * sum( (A_ij - k_i*k_j/2m) * delta(c_i, c_j) )
    """
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

def main():
    print("RUNNING FCCA DYNAMIC NETWORK EVALUATION...")

    stream = load_condition_stream("incorrect")
    cfg_rlsl = paper_horlsl_config()

    print(f" Extracting Denoised Low-Rank Tensors (sigma_min={cfg_rlsl.sigma_min})...")
    res_rlsl = HORLSLRunner(cfg_rlsl).run(stream)
    lowrank_stream = res_rlsl.lowrank_stream

    primary_intervals = derive_primary_ern_intervals(res_rlsl.change_points)
    detected_intervals = label_detected_intervals(res_rlsl.intervals)

    print("\nExecuting FCCA on primary ERN windows:")

    report = []
    threshold = 0.2
    for index, interval in enumerate(primary_intervals):
        print(f" -> Processing {interval.name} ({interval.start_idx}-{interval.end_idx})...")
        clusters, W = fcca_on_lowrank_interval(lowrank_stream, interval.start_idx, interval.end_idx, threshold=threshold)

        q_list = []
        for subject_index in range(stream.shape[3]):
            subject_adj = np.mean(lowrank_stream[interval.start_idx:interval.end_idx + 1, :, :, subject_index], axis=0)
            subject_adj[subject_adj < threshold] = 0
            q_list.append(compute_modularity(subject_adj, clusters))

        report.append({
            "Interval": interval.name,
            "Frames": f"{interval.start_idx}-{interval.end_idx}",
            "Avg Modularity (Q)": round(float(np.mean(q_list)), 4),
        })
        np.save(config.paths.OUTPUTS_DIR / f"fcca_W_{slugify_interval(interval.name)}_{index}.npy", W)

    print("\n========================================================")
    print("      FCCA NETWORK SEGREGATION (PRIMARY ERN WINDOWS)")
    print("========================================================")
    print(pd.DataFrame(report).to_string(index=False))
    print("========================================================")

    extra_post = [interval for interval in detected_intervals if interval.name.startswith("Post-ERN (")]
    if extra_post:
        print("\nSupplementary detected intervals:")
        for interval in extra_post:
            print(f"- {interval.name}: frames {interval.start_idx}-{interval.end_idx}")
        print("These late intervals are retained as supplementary detections rather than merged into the primary Table V-style summary.")

if __name__ == "__main__":
    main()

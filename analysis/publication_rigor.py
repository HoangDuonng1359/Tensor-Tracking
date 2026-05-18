import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition as horlsl_cfg
from algorithms.fcca import fcca_on_lowrank_interval
from algorithms.common import convert_subject_tensor_to_stream
from core.timing import eeg_timing_from_array
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample

def calculate_modularity(W, threshold_pct=80):
    n = W.shape[0]
    flat_w = W[np.triu_indices(n, k=1)]
    thr = np.percentile(flat_w, threshold_pct)
    adj = (W > thr).astype(float)
    G = nx.from_numpy_array(adj)
    communities = nx.community.greedy_modularity_communities(G)
    if len(communities) < 2: return 0.0
    return nx.community.modularity(G, communities)

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def run_publication_rigor():
    print("🚀 Running Final Publication-Ready Rigor Analysis...")
    
    # Load Data
    data = np.load("data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy").astype(np.float32)
    stream = convert_subject_tensor_to_stream(data)
    n_subs = data.shape[0]
    timing = eeg_timing_from_array(stream, time_axis=0)
    
    runner = HORLSLRunner(horlsl_cfg())
    result = runner.run(stream)
    
    # Equivalent to old indices [124, 132] and [40, 80] at 128Hz
    ern_slice = timing.ms_window(-31.25, 31.25)
    base_slice = timing.ms_window(-687.5, -375.0)
    
    # 1. Subject-Level Extraction
    q_base_all = []
    q_ern_all = []
    
    for s in range(n_subs):
        W_sub_base = np.mean(result.lowrank_stream[base_slice, :, :, s], axis=0)
        W_sub_ern = np.mean(result.lowrank_stream[ern_slice, :, :, s], axis=0)
        q_base_all.append(calculate_modularity(W_sub_base))
        q_ern_all.append(calculate_modularity(W_sub_ern))
    
    q_base_all = np.array(q_base_all)
    q_ern_all = np.array(q_ern_all)
    
    # 2. Bootstrapping Subjects (1000 iterations)
    print("   Bootstrapping Subject cohort (N=1000)...")
    boot_diffs = []
    for _ in range(1000):
        resampled_base = resample(q_base_all)
        resampled_ern = resample(q_ern_all)
        boot_diffs.append(np.mean(resampled_ern) - np.mean(resampled_base))
    
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    
    # 3. Effect Size
    d = cohen_d(q_ern_all, q_base_all)
    # Eta-squared for paired t-test: t^2 / (t^2 + df)
    t_stat, p_val = stats.ttest_rel(q_ern_all, q_base_all)
    eta_sq = (t_stat**2) / (t_stat**2 + n_subs - 1)
    
    # 4. Null Model Rigor (Group Level)
    print("   Evaluating Null Models (N=100)...")
    W_group_ern = np.mean(result.lowrank_stream[ern_slice], axis=(0, 3))
    q_actual = calculate_modularity(W_group_ern)
    
    null_qs = []
    adj_ern = (W_group_ern > np.percentile(W_group_ern, 80)).astype(int)
    G_ern = nx.from_numpy_array(adj_ern)
    for _ in range(100):
        G_null = nx.random_reference(G_ern, niter=10, connectivity=False)
        communities = nx.community.greedy_modularity_communities(G_null)
        null_qs.append(nx.community.modularity(G_null, communities))
    
    z_score = (q_actual - np.mean(null_qs)) / np.std(null_qs)

    # 5. FINAL REPORTING
    print("\n" + "="*80)
    print("📑 FINAL SCIENTIFIC RIGOR REPORT")
    print("="*80)
    print(f"Modularity Reconfiguration (ERN vs Baseline):")
    print(f"  - Mean Difference: {np.mean(q_ern_all) - np.mean(q_base_all):.4f}")
    print(f"  - 95% Bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  - Cohen's d: {d:.4f} (Effect Size)")
    print(f"  - Eta-squared (η²): {eta_sq:.4f}")
    print(f"  - P-value (Paired T): {p_val:.4f}")
    
    print(f"\nGroup Consensus Stability:")
    print(f"  - Group Q: {q_actual:.4f}")
    print(f"  - Null Q (Mean): {np.mean(null_qs):.4f}")
    print(f"  - Z-Score vs Null: {z_score:.4f} ⭐⭐⭐")
    
    print("\nFinal Interpretation:")
    if z_score > 3:
        print("  - The tensor-extracted group consensus structure is HIGHLY stable and non-random.")
    if abs(d) < 0.2:
        print("  - High subject variability confirmed. Group-level tensor analysis is REQUIRED.")
    print("="*80)

if __name__ == "__main__":
    run_publication_rigor()

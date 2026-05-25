import os
import sys
from typing import Dict, Any, Tuple
import numpy as np

from core.config import config
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition as horlsl_cfg
from algorithms.hosvd import HOSVDRunner, default_config_for_condition as hosvd_cfg
from algorithms.common import convert_subject_tensor_to_stream
from fcca import fcca_on_lowrank_interval

from analysis.metrics_utils import (
    extract_intervals,
    compute_network_metrics,
    calculate_distance,
    permutation_test_distance,
    compute_event_overlap,
    bootstrap_stability_evaluation
)

ZERO_INDEX = 128
N_BOOTS = 100
THRESHOLD = 0.1

def run_algorithm(algo_name: str, stream_incorr: np.ndarray, stream_corr: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
    if algo_name == "HO-RLSL":
        cfg = horlsl_cfg()
        cfg.lambda_sparse = 0.1
        runner = HORLSLRunner(cfg)
    else:
        cfg = hosvd_cfg()
        runner = HOSVDRunner(cfg)
        
    res_incorr = runner.run(stream_incorr)
    res_corr = runner.run(stream_corr)
    
    pre_i, ern_i, post_i = extract_intervals(res_incorr, ZERO_INDEX)
    pre_c, crn_c, post_c = extract_intervals(res_corr, ZERO_INDEX)
    
    part_ern_ern, W_ern_ern = fcca_on_lowrank_interval(res_incorr.lowrank_stream, ern_i[0], ern_i[1], threshold=THRESHOLD)
    _, W_ern_pre = fcca_on_lowrank_interval(res_incorr.lowrank_stream, pre_i[0], pre_i[1], threshold=THRESHOLD)
    _, W_ern_post = fcca_on_lowrank_interval(res_incorr.lowrank_stream, post_i[0], post_i[1], threshold=THRESHOLD)
    
    part_crn_crn, W_crn_crn = fcca_on_lowrank_interval(res_corr.lowrank_stream, crn_c[0], crn_c[1], threshold=THRESHOLD)
    _, W_crn_post = fcca_on_lowrank_interval(res_corr.lowrank_stream, post_c[0], post_c[1], threshold=THRESHOLD)
    
    metrics_ern = compute_network_metrics(W_ern_ern)
    metrics_crn = compute_network_metrics(W_crn_crn)
    dist_ern_crn = calculate_distance(W_ern_ern, W_crn_crn)
    
    return {
        "algo": algo_name,
        "n_cp_ern": len(res_incorr.change_points),
        "ern_interval": f"{times[ern_i[0]]:.1f}–{times[ern_i[1]]:.1f}",
        "mod_ern": metrics_ern[0],
        "wb_ern": metrics_ern[1],
        "dist_ern_crn": dist_ern_crn,
        "W_ern_ern": W_ern_ern,
        "W_crn_crn": W_crn_crn,
        "part_ern_ern": part_ern_ern,
        "res_incorr": res_incorr,
        "res_corr": res_corr,
        "intervals": {"pre_i": pre_i, "ern_i": ern_i, "post_i": post_i, "pre_c": pre_c, "crn_c": crn_c, "post_c": post_c}
    }

def print_detailed_modularity(name: str, met: Tuple[float, ...], det: Dict[str, Any]) -> None:
    print(f"{name} Details:")
    print(f"  Mod: {met[0]:.4f}, W/B: {met[1]:.4f}")
    print(f"  Cluster sizes: {det['cluster_sizes']}")
    print(f"  Within Edges: {det['count_within']} (mean W: {det['mean_within']:.4f})")
    print(f"  Between Edges: {det['count_between']} (mean B: {det['mean_between']:.4f})")


def run_permutation_test_comparison(algo_data: Dict[str, Any], n_perms: int = 100) -> None:
    res_inc = algo_data["res_incorr"]
    res_cor = algo_data["res_corr"]
    
    lowrank_inc = res_inc.lowrank_stream[algo_data["intervals"]["ern_i"][0]:algo_data["intervals"]["ern_i"][1] + 1]
    lowrank_cor = res_cor.lowrank_stream[algo_data["intervals"]["crn_c"][0]:algo_data["intervals"]["crn_c"][1] + 1]
    
    pval, mean_null = permutation_test_distance(lowrank_inc, lowrank_cor, algo_data['dist_ern_crn'], threshold=THRESHOLD, n_perms=n_perms)
    print(f"  => Observed Dist ({algo_data['algo']}): {algo_data['dist_ern_crn']:.4f}")
    print(f"  => Mean Null Dist ({algo_data['algo']}): {mean_null:.4f}")
    print(f"  => P-value ({algo_data['algo']}): {pval:.4f}")


def run_bootstrap_comparison(algo_name: str, tensor: np.ndarray, base_communities: Any, base_ern_interval: Tuple[int, int], times: np.ndarray) -> None:
    if algo_name == "HO-RLSL":
        def factory():
            cfg = horlsl_cfg()
            cfg.lambda_sparse = 0.1
            return HORLSLRunner(cfg)
    else:
        def factory():
            return HOSVDRunner(hosvd_cfg())
            
    cp_counts, ari_mean, ari_std = bootstrap_stability_evaluation(
        algorithm_runner_factory=factory,
        tensor=tensor,
        base_communities=base_communities,
        base_ern_interval=base_ern_interval,
        n_boots=N_BOOTS,
        algo_name=algo_name
    )
    
    print(f"\nBootstrap Change Point Frequencies ({algo_name}, {N_BOOTS} runs, around 0-200ms, tolerance = ±20ms):")
    total = N_BOOTS
    for cp, cnt in sorted(cp_counts.items()):
        if ZERO_INDEX - 10 <= cp <= ZERO_INDEX + 30:
            print(f"  CP {times[cp]:.1f} ms (Index {cp}): {cnt}/{total} times ({cnt/total*100:.1f}%)")
    print(f"  => Cluster Stability (ARI): {ari_mean:.4f} ± {ari_std:.4f}")

def main() -> None:
    print("Computing Extended Quantitative Metrics (Refactored)...")
    
    tensor_incorr = np.load(config.tensor_incorrect_file).astype(np.float32)
    stream_incorr = convert_subject_tensor_to_stream(tensor_incorr)
    
    tensor_corr = np.load(config.tensor_correct_file).astype(np.float32)
    stream_corr = convert_subject_tensor_to_stream(tensor_corr)
    
    times = np.linspace(-1000, 1000, 257)
    
    # 1. RUN HO-RLSL
    print("\n[1] Running HO-RLSL...")
    rlsl_data = run_algorithm("HO-RLSL", stream_incorr, stream_corr, times)
    
    _, W_crn_crn = fcca_on_lowrank_interval(rlsl_data["res_corr"].lowrank_stream, rlsl_data["intervals"]["crn_c"][0], rlsl_data["intervals"]["crn_c"][1], threshold=THRESHOLD)
    _, W_crn_pre = fcca_on_lowrank_interval(rlsl_data["res_corr"].lowrank_stream, rlsl_data["intervals"]["pre_c"][0], rlsl_data["intervals"]["pre_c"][1], threshold=THRESHOLD)
    
    met_crn_pre, det_crn_pre = compute_network_metrics(W_crn_pre, detailed=True)
    met_crn_crn, det_crn_crn = compute_network_metrics(W_crn_crn, detailed=True)
    
    print("\nInvestigating CRN Modularity Drop but W/B Increase:")
    print_detailed_modularity("PRE-CRN", met_crn_pre, det_crn_pre)
    print_detailed_modularity("CRN", met_crn_crn, det_crn_crn)
    
    # 2. Permutation Test
    print("\n[2] Computing Permutation Test for ERN vs CRN Distance (HO-RLSL)...")
    run_permutation_test_comparison(rlsl_data, n_perms=100)
    
    # 3. RUN HOSVD
    print("\n[3] Running HOSVD Baseline...")
    hosvd_data = run_algorithm("HOSVD", stream_incorr, stream_corr, times)
    
    overlap_rlsl = compute_event_overlap(times[rlsl_data["intervals"]["ern_i"][0]], times[rlsl_data["intervals"]["ern_i"][1]])
    overlap_hosvd = compute_event_overlap(times[hosvd_data["intervals"]["ern_i"][0]], times[hosvd_data["intervals"]["ern_i"][1]])
    
    print("\n=======================================================")
    print(" Bảng 5: HOSVD vs HO-RLSL Baseline Comparison")
    print("=======================================================")
    print(f"{'Method':<10} | {'Condition':<10} | {'#CP':<4} | {'Response interval':<18} | {'Overlap (0-150ms)':<17} | {'Modularity':<10} | {'W/B':<10} | {'ERN-CRN dist'}")
    print("-" * 115)
    
    for row, ov in zip([hosvd_data, rlsl_data], [overlap_hosvd, overlap_rlsl]):
        print(f"{row['algo']:<10} | ERN        | {row['n_cp_ern']:<4} | {row['ern_interval']:<18} | {ov*100:<16.1f}% | {row['mod_ern']:<10.4f} | {row['wb_ern']:<10.4f} | {row['dist_ern_crn']:.4f}")
    print("=======================================================\n")
    
    print("\n[3b] Computing Permutation Test for ERN vs CRN Distance (HOSVD)...")
    run_permutation_test_comparison(hosvd_data, n_perms=100)
    
    # 4. Bootstrap Stability
    print("\n[4] Bootstrap Subject Stability (HO-RLSL on ERN)...")
    run_bootstrap_comparison("HO-RLSL", tensor_incorr, rlsl_data["part_ern_ern"], rlsl_data["intervals"]["ern_i"], times)
            
    print("\n[5] Bootstrap Subject Stability (HOSVD on ERN)...")
    run_bootstrap_comparison("HOSVD", tensor_incorr, hosvd_data["part_ern_ern"], hosvd_data["intervals"]["ern_i"], times)
    
if __name__ == "__main__":
    main()

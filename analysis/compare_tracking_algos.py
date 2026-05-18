import time
import numpy as np
import pandas as pd
from algorithms.hosvd import HOSVDRunner, default_config_for_condition as hosvd_cfg
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition as horlsl_cfg
from algorithms.toucan import TOUCANRunner, default_config_for_toucan as toucan_cfg
from algorithms.common import convert_subject_tensor_to_stream, calculate_nmse
from core.timing import eeg_timing_from_array

def benchmark():
    print("🚀 Starting Multi-Algorithm Benchmark (Condition: Incorrect)...")
    
    # Load data
    data_path = "data/processed_v2/03_connectivity_tensors/tensor_incorrect_4d.npy"
    data = np.load(data_path).astype(np.float32)
    stream = convert_subject_tensor_to_stream(data)
    n_times = stream.shape[0]
    timing = eeg_timing_from_array(stream, time_axis=0)
    
    # Define Runners
    configs = {
        "HOSVD (Baseline)": hosvd_cfg(),
        "HO-RLSL (Tucker)": horlsl_cfg(),
        "TOUCAN (t-SVD)": toucan_cfg()
    }
    
    runners = {
        "HOSVD (Baseline)": HOSVDRunner(configs["HOSVD (Baseline)"]),
        "HO-RLSL (Tucker)": HORLSLRunner(configs["HO-RLSL (Tucker)"]),
        "TOUCAN (t-SVD)": TOUCANRunner(configs["TOUCAN (t-SVD)"])
    }
    
    results_summary = []
    
    for name, runner in runners.items():
        print(f"--- Running {name} ---")
        start_time = time.time()
        result = runner.run(stream)
        elapsed = time.time() - start_time
        
        # Metrics
        nmse = calculate_nmse(stream, result.lowrank_stream)
        n_cp = len(result.change_points)
        
        zero_idx = timing.zero_index
        # Check for ERN detection (near 0ms, window +/- ~150ms depending on frame count)
        # 20 samples is roughly 150ms at 128Hz
        ern_detected = any(zero_idx - 20 <= cp <= zero_idx + 20 for cp in result.change_points)
        
        # Closest CP to 0ms
        if n_cp > 0:
            closest_cp = result.change_points[np.argmin(np.abs(result.change_points - zero_idx))]
            offset_ms = timing.index_to_ms(closest_cp)
        else:
            offset_ms = np.nan

        results_summary.append({
            "Algorithm": name,
            "CP Count": n_cp,
            "NMSE": f"{nmse:.4f}",
            "Time (s)": f"{elapsed:.2f}",
            "ERN Detected": "Yes" if ern_detected else "No",
            "ERN Offset (ms)": f"{offset_ms:.1f}" if not np.isnan(offset_ms) else "N/A"
        })

    # Display results
    df = pd.DataFrame(results_summary)
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"Note: ERN Target is Frame {timing.zero_index} (0ms). Offset is relative to behavioral response.")

if __name__ == "__main__":
    benchmark()

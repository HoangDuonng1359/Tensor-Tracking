import numpy as np
from algorithms.hosvd import HOSVDRunner, default_config_for_condition as hosvd_cfg
from algorithms.ho_rlsl import HORLSLRunner, default_config_for_condition as horlsl_cfg
from algorithms.common import convert_subject_tensor_to_stream
from core.timing import eeg_timing_from_array

def get_phase_intervals(intervals, timing):
    # Find interval containing the 0ms index (e.g., 128)
    ern_interval = None
    pre_interval = None
    post_interval = None
    
    zero_idx = timing.zero_index
    
    for i, (start, end) in enumerate(intervals):
        if start <= zero_idx <= end:
            ern_interval = (start, end)
            if i > 0:
                pre_interval = (intervals[i-1][0], intervals[i-1][1])
            if i < len(intervals) - 1:
                post_interval = (intervals[i+1][0], intervals[i+1][1])
            break
    
    # If no interval perfectly wraps 0ms, take the closest ones
    if ern_interval is None:
        return "N/A", "N/A", "N/A"
        
    def fmt(start, end):
        return f"{int(timing.index_to_ms(start))} to {int(timing.index_to_ms(end))}"
        
    return fmt(*pre_interval) if pre_interval else "N/A", \
           fmt(*ern_interval), \
           fmt(*post_interval) if post_interval else "N/A"

def compare_results():
    print("📋 Comparing current results with Ozdemir (2017) baseline...")
    
    conditions = ["incorrect", "correct"]
    # Use more sensitive parameters for comparison
    h_cfg = horlsl_cfg()
    h_cfg.sigma_min = 0.11
    h_cfg.alpha = 8
    
    s_cfg = hosvd_cfg()
    s_cfg.sigma_min = 0.11
    s_cfg.alpha = 8

    algos = {
        "HOSVD": HOSVDRunner(s_cfg),
        "HO-RLSL": HORLSLRunner(h_cfg)
    }
    
    summary = {}
    
    for cond in conditions:
        data = np.load(f"data/processed_v2/03_connectivity_tensors/tensor_{cond}_4d.npy").astype(np.float32)
        stream = convert_subject_tensor_to_stream(data)
        timing = eeg_timing_from_array(stream, time_axis=0)
        
        summary[cond] = {}
        for name, runner in algos.items():
            result = runner.run(stream)
            summary[cond][name] = get_phase_intervals(result.intervals, timing)

    print("\n" + "="*80)
    print(f"{'Condition':<12} | {'Phase':<10} | {'HO-RLSL (Our)':<25} | {'HOSVD (Our)':<25}")
    print("-" * 80)
    
    for cond in conditions:
        phases = ["Pre", "During", "Post"]
        for i, phase in enumerate(phases):
            h_val = summary[cond]["HO-RLSL"][i]
            s_val = summary[cond]["HOSVD"][i]
            display_cond = cond.upper() if i == 0 else ""
            print(f"{display_cond:<12} | {phase:<10} | {h_val:<25} | {s_val:<25}")
        print("-" * 80)

if __name__ == "__main__":
    compare_results()

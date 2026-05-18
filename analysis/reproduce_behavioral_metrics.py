import pandas as pd
from algorithms.ho_rlsl import HORLSLRunner
from algorithms.hosvd import PaperComparisonConfig
from algorithms.hosvd import windowed_hosvd_tracking
from analysis.paper_alignment import PAPER_TABLE_V_ERN
from analysis.paper_alignment import change_points_to_ms
from analysis.paper_alignment import derive_primary_interval_strings
from analysis.paper_alignment import load_condition_stream
from analysis.paper_alignment import load_condition_tensor
from analysis.paper_alignment import paper_comparison_hosvd_config
from analysis.paper_alignment import paper_horlsl_config

def reproduce_paper_table_V():
    print("RUNNING REPRODUCTION OF TABLE V (Ozdemir 2017)...")
    
    stream = load_condition_stream("incorrect")
    data = load_condition_tensor("incorrect")
    cfg_rlsl = paper_horlsl_config()
    cfg_hosvd: PaperComparisonConfig = paper_comparison_hosvd_config()
    
    print(" Running HO-RLSL (paper-comparison config)...")
    res_rlsl = HORLSLRunner(cfg_rlsl).run(stream)
    cps_rlsl = res_rlsl.change_points
    
    print(" Running HoSVD (paper-comparison approximation)...")
    cps_hosvd = windowed_hosvd_tracking(data, cfg_hosvd)
    
    rlsl_intervals = derive_primary_interval_strings(cps_rlsl)
    hosvd_intervals = derive_primary_interval_strings(cps_hosvd)
    
    table_data = []
    for name in ["Pre-ERN", "ERN", "Post-ERN"]:
        table_data.append({
            "Interval": name,
            "HO-RLSL (ERP CORE)": dict(zip(["Pre-ERN", "ERN", "Post-ERN"], rlsl_intervals))[name],
            "HoSVD (ERP CORE)": dict(zip(["Pre-ERN", "ERN", "Post-ERN"], hosvd_intervals))[name],
            "Original Paper (HO-RLSL)": PAPER_TABLE_V_ERN["HO-RLSL"][name],
            "Original Paper (HoSVD)": PAPER_TABLE_V_ERN["HoSVD"][name],
        })
    df = pd.DataFrame(table_data)
    print("\n" + "="*70)
    print("  TABLE V-STYLE ERN INTERVAL COMPARISON")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    print("\nRaw Change Points (ms) for reference:")
    print(f"HO-RLSL: {change_points_to_ms(cps_rlsl).round(1)}")
    print(f"HoSVD:   {change_points_to_ms(cps_hosvd).round(1)}")
    print("\nNotes:")
    print("- HO-RLSL uses the canonical paper-comparison configuration shared across analysis scripts.")
    print("- HoSVD uses the explicit paper-comparison windowed approximation, not the canonical HOSVDRunner.")
    print("- Residual timing differences are expected on ERP CORE because the dataset has 40 subjects and 30 channels, not the paper's 91 subjects and 63 channels.")

if __name__ == "__main__":
    reproduce_paper_table_V()

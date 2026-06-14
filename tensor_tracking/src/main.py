import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure project dirs are in Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "src/preprocessing"))
sys.path.append(str(ROOT / "src/low_rank_extraction"))
sys.path.append(str(ROOT / "src/fcca"))

from preprocessing.config_1024 import config as prep_config
from preprocessing.pipeline_1024 import preprocess_individual_subject
from preprocessing.sampling_1024 import run_sampling_pipeline

from low_rank_extraction.hosvd_v2 import hosvd_low_rank
try:
    from low_rank_extraction.ho_rlsl_v2 import HoRLSL, HoRLSLConfig
    HAS_ho_rlsl_v2 = True
except ImportError:
    HAS_ho_rlsl_v2 = False

import utils

def main():
    parser = argparse.ArgumentParser(description="End-to-End Neuroscience Tensor Tracking Pipeline.")
    parser.add_argument("--num-subjects", type=int, default=None, 
                        help="Number of subjects to preprocess (default: all).")
    parser.add_argument("--low-rank-method", type=str, choices=["ho-rlsl-v2", "ho-rlsl-v1", "hosvd-v1", "hosvd", "raw"], default="ho-rlsl-v2",
                        help="Denoising method to use on connectivity tensors.")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip raw EEG preprocessing and start directly from pre-computed tensors.")
    args = parser.parse_args()

    print("=" * 80)
    print("STARTING END-TO-END NEUROSCIENCE TENSOR TRACKING PIPELINE")
    print("=" * 80)

    # Step 1: Preprocessing & Trial Matching
    if not args.skip_preprocessing:
        print("\n[STEP 1/5] Preprocessing EEG Data (CSD transformation and Epoching)...")
        sub_ids = sorted([d.name for d in prep_config.paths.DATA_RAW.glob("sub-*") if d.is_dir()])
        if args.num_subjects is not None:
            sub_ids = sub_ids[:args.num_subjects]
            print(f"Running for a subset of {args.num_subjects} subjects: {sub_ids}")
            
        for i, sub in enumerate(sub_ids):
            try:
                preprocess_individual_subject(sub)
                print(f"  [{i+1}/{len(sub_ids)}] {sub}: Preprocessing SUCCESS")
            except Exception as e:
                print(f"  [{i+1}/{len(sub_ids)}] {sub}: Preprocessing FAILED | {e}")
                
        print("\n[STEP 2/5] Running Temporal Matching and Trial Balancing...")
        run_sampling_pipeline()
        
        # Step 2: Compute Phase Locking Value (PLV) Tensors
        print("\n[STEP 3/5] Computing PLV Connectivity Tensors (using Hilbert Fallback)...")
        sub_files = sorted(list(prep_config.paths.REFINED_DIR.glob("*_theta_balanced-epo.fif")))
        if not sub_files:
            print("Error: No refined epoch files found. Cannot compute connectivity.")
            sys.exit(1)
            
        tensor_correct, tensor_incorrect = utils.compute_plv_tensors(sub_files, prep_config)
        
        # Save output tensors
        np.save(prep_config.tensor_correct_file, tensor_correct)
        np.save(prep_config.tensor_incorrect_file, tensor_incorrect)
        print(f"Saved computed tensors: {prep_config.tensor_correct_file.name}, {prep_config.tensor_incorrect_file.name}")
    else:
        print("\n[STEP 1-3] Skipping Preprocessing. Loading pre-computed tensors...")
        if not prep_config.tensor_incorrect_file.exists():
            print(f"Error: Pre-computed tensor not found at {prep_config.tensor_incorrect_file}")
            sys.exit(1)
        tensor_incorrect = np.load(prep_config.tensor_incorrect_file)
        print(f"Loaded pre-computed tensor with shape: {tensor_incorrect.shape}")

    # Step 3: Low-Rank Extraction
    print(f"\n[STEP 4/5] Extracting Low-Rank Connectivity Subspaces using {args.low_rank_method.upper()}...")
    cps_detected = None
    if args.low_rank_method == "ho-rlsl-v2":
        if not HAS_ho_rlsl_v2:
            print("Error: ho_rlsl_v2.py cannot be imported.")
            sys.exit(1)
        print("Running ho-rlsl-v2 recursive low-rank and sparse learning...")
        cfg = HoRLSLConfig(
            train_length=prep_config.algo.ho_rlsl_v2_TRAIN_LENGTH,
            alpha=prep_config.algo.ho_rlsl_v2_ALPHA,
            sigma_min=prep_config.algo.ho_rlsl_v2_SIGMA_MIN,
            sparse_solver=prep_config.algo.ho_rlsl_v2_SPARSE_SOLVER
        )
        tracker = HoRLSL(cfg)
        result = tracker.fit_transform_subject_first(tensor_incorrect)
        low_rank = result.low_rank
        cps_detected = np.asarray(result.filtered_change_points, dtype=int).tolist()
        print("ho-rlsl-v2 low-rank extraction completed. (Using native ho-rlsl-v2 change point detection.)")
        # Save results in the exact same file expected by streamlit
        ho_rlsl_v2_npz = ROOT / "outputs/ho_rlsl_v2_results.npz"
        ho_rlsl_v2_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ho_rlsl_v2_npz,
            low_rank=low_rank,
            sparse=result.sparse,
            change_points=np.asarray(result.change_points, dtype=int),
            filtered_change_points=np.asarray(result.filtered_change_points, dtype=int),
            change_point_ms=result.change_point_ms,
            update_times=np.asarray(result.update_times, dtype=int),
            change_scores=result.change_scores,
            change_score_times=result.change_score_times,
            sparse_solver=np.asarray(result.config.get("sparse_solver", "gtcs_s_omp")),
        )
        print(f"Saved ho-rlsl-v2 results to {ho_rlsl_v2_npz}")
    elif args.low_rank_method == "ho-rlsl-v1":
        from low_rank_extraction.ho_rlsl_v1 import HORLSLRunner
        from low_rank_extraction.common_v1 import convert_subject_tensor_to_stream, result_to_legacy_layout, DecompositionConfig
        
        print("Running ho-rlsl-v1 logic...")
        cfg = DecompositionConfig(
            train_steps=prep_config.algo.ho_rlsl_v2_TRAIN_LENGTH,
            alpha=prep_config.algo.ho_rlsl_v2_ALPHA,
            sigma_min=prep_config.algo.ho_rlsl_v2_SIGMA_MIN,
            lambda_sparse=0.05,
            max_rank=(10, 10, 10),
            symmetric_modes=True,
            name="ho-rlsl-v1_custom"
        )
        stream = convert_subject_tensor_to_stream(tensor_incorrect)
        tracker = HORLSLRunner(cfg)
        result = tracker.run(stream)
        low_rank, sparse = result_to_legacy_layout(result)
        cps_detected = np.asarray(result.change_points, dtype=int).tolist()
        print("ho-rlsl-v1 completed.")
        
        ho_rlsl_v1_npz = ROOT / "outputs/ho_rlsl_v1_results.npz"
        ho_rlsl_v1_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ho_rlsl_v1_npz,
            low_rank=low_rank,
            sparse=sparse,
            change_points=np.asarray(result.change_points, dtype=int),
        )
        print(f"Saved ho-rlsl-v1 results to {ho_rlsl_v1_npz}")
    elif args.low_rank_method == "hosvd-v1":
        from low_rank_extraction.hosvd_v1 import HOSVDRunner
        from low_rank_extraction.common_v1 import convert_subject_tensor_to_stream, result_to_legacy_layout, DecompositionConfig
        
        print("Running hosvd-v1 logic...")
        cfg = DecompositionConfig(
            train_steps=prep_config.algo.HO_RLSL_TRAIN_LENGTH,
            alpha=prep_config.algo.HO_RLSL_ALPHA,
            sigma_min=prep_config.algo.HO_RLSL_SIGMA_MIN,
            lambda_sparse=0.0,
            max_rank=4,
            symmetric_modes=True,
            name="hosvd-v1_custom"
        )
        stream = convert_subject_tensor_to_stream(tensor_incorrect)
        tracker = HOSVDRunner(cfg)
        result = tracker.run(stream)
        low_rank, sparse = result_to_legacy_layout(result)
        cps_detected = np.asarray(result.change_points, dtype=int).tolist()
        print("hosvd-v1 completed.")
        
        hosvd_v1_npz = ROOT / "outputs/hosvd_v1_results.npz"
        hosvd_v1_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            hosvd_v1_npz,
            low_rank=low_rank,
            sparse=sparse,
            change_points=np.asarray(result.change_points, dtype=int),
        )
        print(f"Saved hosvd-v1 results to {hosvd_v1_npz}")
    elif args.low_rank_method == "hosvd":
        print("Running static Tucker SVD decomposition...")
        low_rank = hosvd_low_rank(tensor_incorrect, ranks=prep_config.algo.HOSVD_RANKS)
        # Save change points results (from exact SSE)
        hosvd_npz = ROOT / "outputs/fcca_results/hosvd_change_points.npz"
        hosvd_npz.parent.mkdir(parents=True, exist_ok=True)
        # Compute exact change points on HOSVD
        features = utils.graph_feature_matrix(low_rank)
        cps_detected = utils.detect_change_points_exact(
            features, 
            n_bkps=prep_config.algo.N_CHANGE_POINTS, 
            min_size=prep_config.algo.MIN_INTERVAL_FRAMES
        )
        np.savez_compressed(
            hosvd_npz,
            change_points=np.asarray(cps_detected, dtype=int),
            method=np.asarray("HoSVD + exact SSE"),
        )
        print(f"Saved HoSVD change points to {hosvd_npz}")
    else:
        print("Using raw, un-denoised tensor...")
        low_rank = tensor_incorrect

    # Step 4 & 5: Change Point Detection & FCCA Clustering
    print("\n[STEP 5/5] Running Change Point Detection & FCCA Consensus Community Splits...")
    if cps_detected is None:
        features = utils.graph_feature_matrix(low_rank)
        cps_detected = utils.detect_change_points_exact(
            features, 
            n_bkps=prep_config.algo.N_CHANGE_POINTS, 
            min_size=prep_config.algo.MIN_INTERVAL_FRAMES
        )
        print(f"Exact SSE change point frames detected: {cps_detected}")

    n_times = low_rank.shape[-1]
    intervals = utils.make_intervals_from_change_points(n_times, cps_detected)
    
    # Run FCCA for each interval and compile outputs
    save_dict = {}
    if args.low_rank_method in ["ho-rlsl-v2", "ho-rlsl-v1", "hosvd-v1"]:
        save_dict["change_point_method"] = np.asarray("ho-rlsl-v2 update rule paper-like intervals")
    else:
        save_dict["change_point_method"] = np.asarray("Exact SSE change-point detection")

    for name, frames in intervals.items():
        print(f"  Clustering consensus networks for interval '{name}' (frames {frames[0]}-{frames[-1]})...")
        cooccur, labels, modularity, mean_adjacency = utils.run_fcca_for_interval(low_rank, frames)
        save_dict[f"{name}_consensus_matrix"] = cooccur
        save_dict[f"{name}_consensus_labels"] = labels
        save_dict[f"{name}_consensus_modularity"] = np.asarray(modularity)
        save_dict[f"{name}_mean_adjacency"] = mean_adjacency
        save_dict[f"{name}_community_method"] = np.asarray("Fiedler consensus clustering")

    # Save output NPZ
    fcca_results_file = prep_config.paths.FCCA_DIR / ("ho_rlsl_v2_fcca_results.npz" if args.low_rank_method == "ho-rlsl-v2" else "ho_rlsl_v1_fcca_results.npz" if args.low_rank_method == "ho-rlsl-v1" else "hosvd_v1_fcca_results.npz" if args.low_rank_method == "hosvd-v1" else "fcca_results.npz")
    fcca_results_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fcca_results_file, **save_dict)
    
    print(f"Saved FCCA consensus clustering results to {fcca_results_file}")

    if args.low_rank_method in ["ho-rlsl-v1", "hosvd-v1"]:
        print("\n[STEP 6/5] Running FCCA Paper Visualization using native change points...")
        try:
            import fcca.fcca_v1
            if args.low_rank_method == "hosvd-v1":
                fcca.fcca_v1.INPUT_DATA_FILE = "../../outputs/hosvd_v1_results.npz"
            else:
                fcca.fcca_v1.INPUT_DATA_FILE = "../../outputs/ho_rlsl_v1_results.npz"
            from fcca.fcca_v1 import cluster_ern_fcca_paper
            cluster_ern_fcca_paper()
        except Exception as e:
            print(f"Error running FCCA visualization: {e}")
    elif args.low_rank_method == "ho-rlsl-v2":
        print("\n[STEP 6/5] Running FCCA Paper Visualization using native change points...")
        try:
            from fcca.fcca_v2 import cluster_ern_fcca_paper_legacy_single_method
            cluster_ern_fcca_paper_legacy_single_method()
        except Exception as e:
            print(f"Error running FCCA visualization: {e}")

    print("\n" + "=" * 80)
    print("PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

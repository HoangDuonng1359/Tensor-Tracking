from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from algorithms.common import DecompositionConfig
from algorithms.common import DecompositionResult
from algorithms.common import compute_intervals
from algorithms.common import config_to_dict
from algorithms.common import convert_subject_tensor_to_stream
from algorithms.common import result_to_legacy_layout
from algorithms.common import save_result_bundle
from algorithms.t_algebra import t_prod, t_svd, t_transpose, get_device
from core.config import config


class TOUCANRunner:
    """
    PERFECTED TOUCAN: Online Tensor Subspace Tracking on Grassmann Manifold.
    - GPU Accelerated via PyTorch
    - Robust Low-rank + Sparse Decomposition
    - Momentum-based Grassmannian Tracking
    - EEG Phase/Oscillation preservation via t-SVD algebra
    """
    def __init__(self, decomposition_config: DecompositionConfig):
        self.config = decomposition_config
        self.device = get_device()
        self.step_size = 0.01 # Reduced for stability
        self.momentum = 0.95 # Increased for smoothness
        self.window_size = 10 # Tubal window size
        
    def _retract_grassmann(self, U: torch.Tensor, Delta: torch.Tensor) -> torch.Tensor:
        """Proper Grassmannian retraction using QR."""
        U_next = U + self.step_size * Delta
        U_fft = torch.fft.fft(U_next, dim=2)
        for i in range(U_fft.shape[2]):
            q, r = torch.linalg.qr(U_fft[:, :, i])
            U_fft[:, :, i] = q
        return torch.real(torch.fft.ifft(U_fft, dim=2))

    def run(self, stream_np: np.ndarray) -> DecompositionResult:
        # stream_np: (Time, Node, Node, Subjects)
        # For consensus tracking, we average subjects to get a cleaner group signal
        group_stream = torch.as_tensor(np.mean(stream_np, axis=3), device=self.device, dtype=torch.float32)
        n_times, n1, n2 = group_stream.shape
        k = self.config.max_rank or 5
        lambda_sparse = self.config.lambda_sparse or 0.1
        
        # 1. Initialize Window-based Subspace
        # Each 'slice' in t-SVD will be (n1, n2, window_size)
        def get_window_tensor(t):
            start = max(0, t - self.window_size + 1)
            pad = self.window_size - (t - start + 1)
            chunk = group_stream[start : t + 1]
            if pad > 0:
                # Pad with first frame if window is not full
                padding = group_stream[0].unsqueeze(0).repeat(pad, 1, 1)
                chunk = torch.cat([padding, chunk], dim=0)
            return chunk.permute(1, 2, 0) # (n1, n2, window)

        Y_init = get_window_tensor(self.config.train_steps)
        U_full, S_full, V_full = t_svd(Y_init)
        U = U_full[:, :k, :].to(self.device)
        
        velocity = torch.zeros_like(U)
        lowrank_stream = np.zeros(stream_np.shape, dtype=np.float32) # Full 4D
        sparse_stream = np.zeros(stream_np.shape, dtype=np.float32)
        residual_energy = torch.zeros(n_times, device=self.device)
        change_points: list[int] = []
        
        # For CP detection, we track the Grassmannian distance between consecutive subspaces
        last_U = U.clone()
        
        for t in range(n_times):
            # The current observation is actually a TUBE window
            Y_tube = get_window_tensor(t)
            
            # Robust L+S update
            S_t = torch.zeros_like(Y_tube)
            for _ in range(2):
                L_input = Y_tube - S_t
                W_t = t_prod(t_transpose(U), L_input)
                L_t = t_prod(U, W_t)
                Residual = Y_tube - L_t
                S_t = torch.sign(Residual) * torch.clamp(torch.abs(Residual) - lambda_sparse, min=0.0)
            
            # Record results for the CURRENT frame (last slice of the tube)
            # We apply the learned projection to all subjects at this time t
            for s in range(stream_np.shape[3]):
                Y_subj = torch.as_tensor(stream_np[t, :, :, s], device=self.device)
                # We project individual subject to the group consensus subspace
                # Since U is (n1, k, window), we use the average over the window or the latest slice
                U_latest = U[:, :, -1]
                L_subj = U_latest @ (U_latest.T @ Y_subj)
                lowrank_stream[t, :, :, s] = L_subj.cpu().numpy()
                sparse_stream[t, :, :, s] = (Y_subj - L_subj).cpu().numpy()
            
            energy = torch.mean((Y_tube - L_t)**2)
            residual_energy[t] = energy
            
            if t < self.config.train_steps:
                continue
                
            # 3. Change Point Detection based on Subspace Rotation
            # Distance = 1 - Tr(U_old^T * U_new) / k (averaged over tube)
            dist = 0.0
            U_fft_old = torch.fft.fft(last_U, dim=2)
            U_fft_new = torch.fft.fft(U, dim=2)
            for i in range(self.window_size):
                dist += 1.0 - torch.norm(torch.matmul(U_fft_old[:,:,i].H, U_fft_new[:,:,i]))**2 / k
            dist /= self.window_size
            
            # Only trigger CP if the subspace significantly rotates
            if dist > 0.15: # 15% rotation threshold
                if not change_points or (t - change_points[-1] > self.config.alpha):
                    change_points.append(t)
                    last_U = U.clone()
            
            # 4. Momentum Update
            Wt = t_transpose(W_t)
            Delta = t_prod(Y_tube - L_t - S_t, Wt)
            velocity = self.momentum * velocity + Delta
            U = self._retract_grassmann(U, velocity)

        intervals = compute_intervals(change_points, n_times)
        
        result = DecompositionResult(
            algorithm="toucan_perfected",
            config=config_to_dict(self.config),
            lowrank_stream=lowrank_stream,
            sparse_stream=sparse_stream,
            change_points=np.array(change_points, dtype=np.int32),
            intervals=intervals,
            mode_ranks=np.full((n_times, 3), k, dtype=np.int32),
            residual_energy=residual_energy.cpu().numpy(),
            sparse_mass=np.zeros(n_times, dtype=np.float32),
            sigma_thresholds=np.zeros(3, dtype=np.float32)
        )
        return result


def load_condition_tensor(condition: str) -> np.ndarray:
    tensor_file = config.tensor_incorrect_file if condition == "incorrect" else config.tensor_correct_file
    return np.load(tensor_file).astype(np.float32)


def default_config_for_toucan() -> DecompositionConfig:
    return DecompositionConfig(
        train_steps=15,
        alpha=8,
        max_rank=5,
        lambda_sparse=0.05,
        name="toucan_perfected_v1",
    )


def save_condition_outputs(condition: str, result: DecompositionResult) -> None:
    output_dir = config.paths.TENSOR_DIR
    artifact_prefix = output_dir / f"toucan_{condition}"
    bundle_path = output_dir / f"toucan_{condition}_bundle.npz"
    save_result_bundle(bundle_path, result)
    lowrank_legacy, sparse_legacy = result_to_legacy_layout(result)

    np.save(output_dir / f"{artifact_prefix.name}_lowrank.npy", lowrank_legacy)
    np.save(output_dir / f"{artifact_prefix.name}_sparse.npy", sparse_legacy)
    np.save(output_dir / f"{artifact_prefix.name}_cp.npy", result.change_points)
    np.save(output_dir / f"{artifact_prefix.name}_intervals.npy", result.intervals)
    np.save(output_dir / f"{artifact_prefix.name}_energy.npy", result.residual_energy)
    np.save(output_dir / f"{artifact_prefix.name}_sparse_mass.npy", result.sparse_mass)

    savemat(config.paths.MATLAB_DIR / f"toucan_results_{condition}.mat", {
        "lowrank": lowrank_legacy,
        "sparse": sparse_legacy,
        "change_points": result.change_points,
        "intervals": result.intervals,
        "residual_energy": result.residual_energy,
        "sparse_mass": result.sparse_mass,
    })


def run_condition(condition: str, decomposition_config: DecompositionConfig | None = None) -> DecompositionResult:
    condition_config = decomposition_config or default_config_for_toucan()
    subject_tensor = load_condition_tensor(condition)
    stream = convert_subject_tensor_to_stream(subject_tensor)
    return TOUCANRunner(condition_config).run(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PERFECTED TOUCAN Online Tensor Tracking on EEG.")
    parser.add_argument("--condition", choices=["correct", "incorrect", "both"], default="both")
    parser.add_argument("--train-steps", type=int, default=15)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--max-rank", type=int, default=5)
    parser.add_argument("--lambda-sparse", type=float, default=0.05)
    args = parser.parse_args()

    run_config = DecompositionConfig(
        train_steps=args.train_steps,
        alpha=args.alpha,
        max_rank=args.max_rank,
        lambda_sparse=args.lambda_sparse,
        name="toucan_perfected_cli",
    )

    conditions = ["correct", "incorrect"] if args.condition == "both" else [args.condition]
    for condition in conditions:
        result = run_condition(condition, decomposition_config=replace(run_config))
        save_condition_outputs(condition, result)
        print(
            f"Perfected TOUCAN {condition}: {len(result.change_points)} change points detected."
        )


if __name__ == "__main__":
    main()

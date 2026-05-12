import numpy as np

from algorithms.benchmark import generate_synthetic_tensor
from algorithms.common import DecompositionConfig
from algorithms.common import convert_stream_to_subject_tensor
from algorithms.common import convert_subject_tensor_to_stream
from algorithms.ho_rlsl import HORLSLRunner
from algorithms.hosvd import HOSVDRunner


def test_tensor_layout_roundtrip():
  tensor = np.random.default_rng(0).random((3, 4, 4, 5), dtype=np.float32)
  stream = convert_subject_tensor_to_stream(tensor)
  restored = convert_stream_to_subject_tensor(stream)
  assert restored.shape == tensor.shape
  assert np.allclose(restored, tensor)


def test_algorithm_output_contract_matches():
  observed, _, _ = generate_synthetic_tensor(
    network_type="modular",
    n_nodes=12,
    n_subjects=6,
    n_times=15,
    noise_sparsity=0.1,
    seed=3,
  )
  run_config = DecompositionConfig(
    train_steps=4,
    alpha=3,
    sigma_min=None,
    sigma_scale=0.1,
    lambda_sparse=0.05,
    symmetric_modes=True,
  )
  ho_rlsl_result = HORLSLRunner(run_config).run(observed)
  hosvd_result = HOSVDRunner(run_config).run(observed)

  assert ho_rlsl_result.lowrank_stream.shape == observed.shape
  assert hosvd_result.lowrank_stream.shape == observed.shape
  assert ho_rlsl_result.intervals.shape[1] == 2
  assert hosvd_result.intervals.shape[1] == 2
  assert ho_rlsl_result.mode_ranks.shape == hosvd_result.mode_ranks.shape


def test_symmetry_is_preserved():
  observed, _, _ = generate_synthetic_tensor(
    network_type="overlapping",
    n_nodes=10,
    n_subjects=4,
    n_times=12,
    noise_sparsity=0.2,
    seed=7,
  )
  result = HORLSLRunner(
    DecompositionConfig(
      train_steps=4,
      alpha=2,
      sigma_min=None,
      sigma_scale=0.1,
      lambda_sparse=0.05,
      symmetric_modes=True,
    )
  ).run(observed)

  assert np.allclose(result.lowrank_stream, np.swapaxes(result.lowrank_stream, 1, 2), atol=1e-5)


def test_ho_rlsl_beats_hosvd_on_sparse_synthetic_data():
  observed, truth, truth_intervals = generate_synthetic_tensor(
    network_type="modular",
    n_nodes=16,
    n_subjects=10,
    n_times=20,
    noise_sparsity=0.35,
    seed=11,
  )
  config = DecompositionConfig(
    train_steps=5,
    alpha=5,
    sigma_min=None,
    sigma_scale=0.1,
    lambda_sparse=0.08,
    symmetric_modes=True,
  )

  ho_rlsl = HORLSLRunner(config).run(observed)
  hosvd = HOSVDRunner(config).run(observed)
  ho_rlsl_mse = np.mean((ho_rlsl.lowrank_stream - truth) ** 2)
  hosvd_mse = np.mean((hosvd.lowrank_stream - truth) ** 2)
  assert ho_rlsl_mse <= hosvd_mse
  assert truth_intervals.shape == (3, 2)


def test_change_points_are_not_degenerate():
  observed, _, _ = generate_synthetic_tensor(
    network_type="hierarchical",
    n_nodes=16,
    n_subjects=8,
    n_times=20,
    noise_sparsity=0.15,
    seed=2,
  )
  config = DecompositionConfig(
    train_steps=5,
    alpha=5,
    sigma_min=None,
    sigma_scale=0.1,
    lambda_sparse=0.05,
    symmetric_modes=True,
  )
  result = HORLSLRunner(config).run(observed)
  assert result.intervals.shape[0] >= 1
  assert result.change_points.ndim == 1

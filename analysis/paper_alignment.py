from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.common import DecompositionConfig
from algorithms.common import convert_subject_tensor_to_stream
from algorithms.hosvd import PaperComparisonConfig as PaperComparisonHOSVDConfig
from core.config import config
from core.timing import EEGTiming
from core.timing import eeg_timing

PAPER_TABLE_V_ERN = {
  "HO-RLSL": {
    "Pre-ERN": "-484 to -109 ms",
    "ERN": "-109 to +141 ms",
    "Post-ERN": "+141 to +766 ms",
  },
  "HoSVD": {
    "Pre-ERN": "-484 to -47 ms",
    "ERN": "-47 to +141 ms",
    "Post-ERN": "+141 to +766 ms",
  },
}


@dataclass(frozen=True)
class NamedInterval:
  name: str
  start_idx: int
  end_idx: int

  def ms_label(self, timing: EEGTiming) -> str:
    start_ms = timing.index_to_ms(self.start_idx)
    end_ms = timing.index_to_ms(self.end_idx)
    return f"{start_ms:+.0f} to {end_ms:+.0f} ms"


def paper_horlsl_config() -> DecompositionConfig:
  return DecompositionConfig(
    train_steps=10,
    alpha=8,
    sigma_min=0.005,
    max_rank=15,
    lambda_sparse=0.05,
    symmetric_modes=True,
    name="paper_compare",
  )


def canonical_hosvd_config() -> DecompositionConfig:
  return DecompositionConfig(
    train_steps=10,
    alpha=8,
    sigma_min=0.11,
    lambda_sparse=0.0,
    max_rank=4,
    symmetric_modes=True,
    name="paper_aligned_hosvd",
  )


def paper_comparison_hosvd_config() -> PaperComparisonHOSVDConfig:
  return PaperComparisonHOSVDConfig(
    train_steps=10,
    alpha=8,
    sigma_min=0.11,
    max_rank=3,
  )


def load_condition_tensor(condition: str = "incorrect") -> np.ndarray:
  path = config.tensor_incorrect_file if condition == "incorrect" else config.tensor_correct_file
  return np.load(path).astype(np.float32)


def load_condition_stream(condition: str = "incorrect") -> np.ndarray:
  return convert_subject_tensor_to_stream(load_condition_tensor(condition))


def default_timing() -> EEGTiming:
  n_times = int(round((config.eeg.TMAX - config.eeg.TMIN) * config.eeg.SFREQ)) + 1
  return eeg_timing(n_times)


def change_points_to_ms(change_points: np.ndarray | list[int], timing: EEGTiming | None = None) -> np.ndarray:
  timing = timing or default_timing()
  cps = np.asarray(change_points, dtype=int)
  return np.asarray([timing.index_to_ms(cp) for cp in cps], dtype=np.float32)


def format_change_points_ms(change_points: np.ndarray | list[int], timing: EEGTiming | None = None) -> np.ndarray:
  return np.round(change_points_to_ms(change_points, timing), 1)


def _pick_change_point(
  change_points: np.ndarray,
  timing: EEGTiming,
  lower_ms: float,
  upper_ms: float,
  *,
  fallback_ms: float,
  pick: str,
) -> int:
  cps = np.asarray(change_points, dtype=int)
  if cps.size == 0:
    return timing.ms_to_index(fallback_ms)

  cps_ms = change_points_to_ms(cps, timing)
  mask = (cps_ms >= lower_ms) & (cps_ms <= upper_ms)
  candidates = cps[mask]
  if candidates.size == 0:
    return timing.ms_to_index(fallback_ms)
  if pick == "last":
    return int(candidates[-1])
  return int(candidates[0])


def derive_primary_ern_intervals(
  change_points: np.ndarray | list[int],
  timing: EEGTiming | None = None,
) -> list[NamedInterval]:
  timing = timing or default_timing()
  cps = np.asarray(change_points, dtype=int)
  onset_idx = _pick_change_point(cps, timing, -200.0, 0.0, fallback_ms=-109.0, pick="last")
  offset_idx = _pick_change_point(cps, timing, 80.0, 200.0, fallback_ms=141.0, pick="first")

  start_idx = timing.ms_to_index(-484.0)
  end_idx = timing.ms_to_index(766.0)
  onset_idx = max(start_idx, min(onset_idx, end_idx))
  offset_idx = max(onset_idx, min(offset_idx, end_idx))

  return [
    NamedInterval("Pre-ERN", start_idx, onset_idx),
    NamedInterval("ERN", onset_idx, offset_idx),
    NamedInterval("Post-ERN", offset_idx, end_idx),
  ]


def derive_primary_interval_strings(
  change_points: np.ndarray | list[int],
  timing: EEGTiming | None = None,
) -> list[str]:
  timing = timing or default_timing()
  return [interval.ms_label(timing) for interval in derive_primary_ern_intervals(change_points, timing)]


def label_detected_intervals(
  intervals: np.ndarray,
  timing: EEGTiming | None = None,
) -> list[NamedInterval]:
  timing = timing or default_timing()
  rows = [NamedInterval(f"Interval {idx + 1}", int(start), int(end)) for idx, (start, end) in enumerate(intervals.tolist())]
  labeled: list[NamedInterval] = []
  post_count = 0
  for row in rows:
    start_ms = timing.index_to_ms(row.start_idx)
    end_ms = timing.index_to_ms(row.end_idx)
    if end_ms < 0:
      name = "Pre-ERN"
    elif start_ms <= 0 <= end_ms:
      name = "ERN"
    else:
      post_count += 1
      name = "Post-ERN" if post_count == 1 else f"Post-ERN ({post_count})"
    labeled.append(NamedInterval(name, row.start_idx, row.end_idx))
  return labeled

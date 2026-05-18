from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import config


@dataclass(frozen=True)
class EEGTiming:
  n_times: int
  tmin_s: float
  tmax_s: float
  zero_reference: str = "response"

  @property
  def time_s(self) -> np.ndarray:
    return np.linspace(self.tmin_s, self.tmax_s, self.n_times, dtype=np.float32)

  @property
  def time_ms(self) -> np.ndarray:
    return self.time_s * 1000.0

  @property
  def zero_index(self) -> int:
    return int(np.argmin(np.abs(self.time_s)))

  def index_to_ms(self, index: int) -> float:
    return float(self.time_ms[int(index)])

  def index_to_s(self, index: int) -> float:
    return float(self.time_s[int(index)])

  def ms_to_index(self, time_ms: float) -> int:
    return int(np.argmin(np.abs(self.time_ms - float(time_ms))))

  def ms_window(self, start_ms: float, end_ms: float) -> slice:
    start_idx = self.ms_to_index(start_ms)
    end_idx = self.ms_to_index(end_ms)
    lower = min(start_idx, end_idx)
    upper = max(start_idx, end_idx)
    return slice(lower, upper + 1)

  def interval_rows(self, intervals: np.ndarray) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for interval_index, (start_idx, end_idx) in enumerate(intervals.tolist(), start=1):
      rows.append({
        "interval_index": interval_index,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_s": self.index_to_s(int(start_idx)),
        "end_s": self.index_to_s(int(end_idx)),
        "start_ms": self.index_to_ms(int(start_idx)),
        "end_ms": self.index_to_ms(int(end_idx)),
        "zero_reference": self.zero_reference,
      })
    return rows

  def to_metadata(self) -> dict[str, Any]:
    return {
      "n_times": self.n_times,
      "tmin_s": self.tmin_s,
      "tmax_s": self.tmax_s,
      "zero_index": self.zero_index,
      "zero_reference": self.zero_reference,
    }


def eeg_timing(
  n_times: int,
  *,
  tmin_s: float | None = None,
  tmax_s: float | None = None,
  zero_reference: str = "response",
) -> EEGTiming:
  return EEGTiming(
    n_times=int(n_times),
    tmin_s=float(config.eeg.TMIN if tmin_s is None else tmin_s),
    tmax_s=float(config.eeg.TMAX if tmax_s is None else tmax_s),
    zero_reference=zero_reference,
  )


def eeg_timing_from_array(
  array: np.ndarray,
  *,
  time_axis: int = -1,
  tmin_s: float | None = None,
  tmax_s: float | None = None,
  zero_reference: str = "response",
) -> EEGTiming:
  n_times = int(array.shape[time_axis])
  return eeg_timing(n_times, tmin_s=tmin_s, tmax_s=tmax_s, zero_reference=zero_reference)


def bundle_metadata_path(bundle_path: Path) -> Path:
  return bundle_path.with_suffix(".json")


def load_bundle_metadata(bundle_path: Path) -> dict[str, Any]:
  metadata_path = bundle_metadata_path(bundle_path)
  if not metadata_path.exists():
    return {}
  return json.loads(metadata_path.read_text(encoding="utf-8"))


def eeg_timing_from_bundle(bundle_path: Path, *, fallback_n_times: int | None = None) -> EEGTiming:
  metadata = load_bundle_metadata(bundle_path)
  timing = metadata.get("timing")
  if timing:
    return eeg_timing(
      int(timing["n_times"]),
      tmin_s=float(timing["tmin_s"]),
      tmax_s=float(timing["tmax_s"]),
      zero_reference=str(timing.get("zero_reference", "response")),
    )
  if fallback_n_times is None:
    with np.load(bundle_path) as bundle:
      fallback_n_times = int(bundle["lowrank_stream"].shape[0])
  return eeg_timing(fallback_n_times)

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np


DEFAULT_SOURCES = (
    Path("fcca_results/ho_rlsl_results.npz"),
    Path("fcca_results/hosvd_change_points.npz"),
    Path("fcca_results/ho_rlsl_fcca_results.npz"),
    Path("fcca_results/fcca_results.npz"),
)
DEFAULT_TENSOR = Path("tensor_4d/tensor_incorrect_4d.npy")
DEFAULT_HOSVD_OUTPUT = Path("fcca_results/hosvd_change_points.npz")
DEFAULT_PREPROCESSING_CONFIG = Path("preprocessing/core/config.py")
DEFAULT_PREPROCESSING_PIPELINE = Path("preprocessing/preprocessing/pipeline.py")


def literal_assignment_from_class(source: str, class_name: str, attr_name: str) -> object | None:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            target = None
            value = None
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                target = item.target.id
                value = item.value
            elif isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                target = item.targets[0].id
                value = item.value
            if target == attr_name and value is not None:
                return ast.literal_eval(value)
    return None


def preprocessing_event_lock(pipeline_path: Path = DEFAULT_PREPROCESSING_PIPELINE) -> str:
    if not pipeline_path.exists():
        return "unknown"
    text = pipeline_path.read_text(encoding="utf-8")
    if "resp_id" in text and "len(l) == 3" in text and "mne.Epochs" in text:
        return "response"
    return "unknown"


def load_preprocessing_time_axis(config_path: Path = DEFAULT_PREPROCESSING_CONFIG) -> dict[str, object] | None:
    if not config_path.exists():
        return None

    source = config_path.read_text(encoding="utf-8")
    sfreq = literal_assignment_from_class(source, "EEGConfig", "SFREQ")
    tmin = literal_assignment_from_class(source, "EEGConfig", "TMIN")
    tmax = literal_assignment_from_class(source, "EEGConfig", "TMAX")
    baseline = literal_assignment_from_class(source, "EEGConfig", "BASELINE")
    if sfreq is None or tmin is None or tmax is None:
        return None

    sfreq = float(sfreq)
    tmin = float(tmin)
    tmax = float(tmax)
    expected_exclusive = int(round((tmax - tmin) * sfreq))
    expected_inclusive = expected_exclusive + 1
    return {
        "start_ms": 1000.0 * tmin,
        "end_ms": 1000.0 * tmax,
        "sfreq": sfreq,
        "tmin": tmin,
        "tmax": tmax,
        "baseline": baseline,
        "event_lock": preprocessing_event_lock(),
        "expected_n_times": (expected_exclusive, expected_inclusive),
        "source": str(config_path),
    }


def resolve_time_axis(args: argparse.Namespace) -> dict[str, object]:
    preprocessing_axis = None if args.no_preprocessing_config else load_preprocessing_time_axis(args.preprocessing_config)

    if preprocessing_axis is not None:
        start_ms = float(args.start_ms) if args.start_ms is not None else float(preprocessing_axis["start_ms"])
        end_ms = float(args.end_ms) if args.end_ms is not None else float(preprocessing_axis["end_ms"])
        source = str(preprocessing_axis["source"])
        event_lock = str(preprocessing_axis["event_lock"])
        expected_n_times = tuple(int(value) for value in preprocessing_axis["expected_n_times"])
        sfreq = float(preprocessing_axis["sfreq"])
    else:
        start_ms = float(args.start_ms) if args.start_ms is not None else -1000.0
        end_ms = float(args.end_ms) if args.end_ms is not None else 1000.0
        source = "CLI/default fallback"
        event_lock = "unknown"
        expected_n_times = ()
        sfreq = None

    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source": source,
        "event_lock": event_lock,
        "expected_n_times": expected_n_times,
        "sfreq": sfreq,
    }


def infer_sample_rate_from_n_times(n_times: int, start_ms: float, end_ms: float) -> float | None:
    duration_seconds = (float(end_ms) - float(start_ms)) / 1000.0
    if duration_seconds <= 0 or n_times <= 1:
        return None
    return float((int(n_times) - 1) / duration_seconds)


def expected_counts_for_sample_rate(sfreq: float, start_ms: float, end_ms: float) -> tuple[int, int]:
    duration_seconds = (float(end_ms) - float(start_ms)) / 1000.0
    exclusive = int(round(duration_seconds * float(sfreq)))
    return exclusive, exclusive + 1


def parse_rank_tuple(value: str) -> tuple[int | None, int | None, int | None]:
    entries: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"none", "null", "-"}:
            entries.append(None)
        else:
            entries.append(int(item))
    if len(entries) != 3:
        raise ValueError("Expected three HoSVD ranks, e.g. '10,10,10'")
    return entries[0], entries[1], entries[2]


def method_label(source: Path, npz_data: np.lib.npyio.NpzFile | None = None) -> str:
    solver = None
    if npz_data is not None and "sparse_solver" in npz_data.files:
        try:
            solver = str(np.asarray(npz_data["sparse_solver"]).item())
        except Exception:
            solver = None
    normalized = source.as_posix().lower()
    if "ho_rlsl_results" in normalized:
        return "Ho-RLSL update rule" + (f" ({solver})" if solver else "")
    if "hosvd" in normalized:
        return "HoSVD"
    if "ho_rlsl_fcca" in normalized:
        return "Ho-RLSL FCCA interval detector"
    if "fcca_results" in normalized:
        return "Tucker/FCCA interval detector"
    if npz_data is not None and "method" in npz_data:
        try:
            return str(np.asarray(npz_data["method"]).item())
        except Exception:
            pass
    return source.stem


def frames_to_ms(
    n_times: int,
    start_ms: float = -1000.0,
    end_ms: float = 1000.0,
    sfreq: float | None = None,
) -> np.ndarray:
    if n_times <= 1:
        raise ValueError("n_times must be greater than 1")
    if sfreq is not None and sfreq > 0:
        return float(start_ms) + 1000.0 * np.arange(int(n_times), dtype=float) / float(sfreq)
    return np.linspace(float(start_ms), float(end_ms), int(n_times))


def infer_n_times(npz_data: np.lib.npyio.NpzFile, tensor_path: Path | None) -> int | None:
    for key in ("low_rank", "sparse"):
        if key in npz_data:
            return int(npz_data[key].shape[-1])

    for key in ("original_shape", "internal_shape"):
        if key in npz_data:
            shape = np.asarray(npz_data[key], dtype=int).ravel()
            if shape.size > 0:
                return int(shape[-1])

    if tensor_path is not None and tensor_path.exists():
        return int(np.load(tensor_path, mmap_mode="r").shape[-1])

    return None


def evaluate_change_points(
    change_points: np.ndarray,
    n_times: int,
    start_ms: float,
    end_ms: float,
    target_start_ms: float,
    target_end_ms: float,
    time_axis_source: str = "unknown",
    event_lock: str = "unknown",
    expected_n_times: tuple[int, ...] = (),
    sfreq: float | None = None,
) -> dict[str, object]:
    change_points = np.asarray(change_points, dtype=int).ravel()
    valid_points = change_points[(change_points >= 0) & (change_points < n_times)]
    times_ms = frames_to_ms(n_times, start_ms=start_ms, end_ms=end_ms, sfreq=sfreq)
    point_times = times_ms[valid_points] if valid_points.size else np.asarray([], dtype=float)

    in_window_mask = (point_times >= target_start_ms) & (point_times <= target_end_ms)
    hits = valid_points[in_window_mask]
    hit_times = point_times[in_window_mask]
    false_points = valid_points[~in_window_mask]
    false_times = point_times[~in_window_mask]

    target_center = 0.5 * (target_start_ms + target_end_ms)
    if point_times.size:
        closest_idx = int(np.argmin(np.abs(point_times - target_center)))
        closest_point = int(valid_points[closest_idx])
        closest_ms = float(point_times[closest_idx])
        closest_error_ms = float(closest_ms - target_center)
    else:
        closest_point = None
        closest_ms = None
        closest_error_ms = None

    return {
        "n_times": int(n_times),
        "change_points": valid_points.astype(int),
        "change_point_times_ms": point_times.astype(float),
        "target_window_ms": (float(target_start_ms), float(target_end_ms)),
        "target_center_ms": float(target_center),
        "time_axis_ms": (float(start_ms), float(end_ms)),
        "time_axis_source": str(time_axis_source),
        "event_lock": str(event_lock),
        "expected_n_times": tuple(int(value) for value in expected_n_times),
        "n_times_matches_preprocessing": bool(not expected_n_times or int(n_times) in set(expected_n_times)),
        "sfreq": None if sfreq is None else float(sfreq),
        "actual_end_ms": float(times_ms[-1]),
        "hits": hits.astype(int),
        "hit_times_ms": hit_times.astype(float),
        "false_positive_points": false_points.astype(int),
        "false_positive_times_ms": false_times.astype(float),
        "hit_count": int(hits.size),
        "total_count": int(valid_points.size),
        "false_positive_count": int(false_points.size),
        "hit_rate": float(hits.size / valid_points.size) if valid_points.size else 0.0,
        "precision_like": float(hits.size / valid_points.size) if valid_points.size else 0.0,
        "false_positive_rate": float(false_points.size / valid_points.size) if valid_points.size else 0.0,
        "has_hit": bool(hits.size > 0),
        "closest_point": closest_point,
        "closest_ms": closest_ms,
        "closest_error_ms": closest_error_ms,
    }


def change_history_mode_counts(npz_data: np.lib.npyio.NpzFile) -> dict[int, int]:
    if "change_point_modes" in npz_data.files:
        modes = np.asarray(npz_data["change_point_modes"], dtype=int).ravel()
        counts: dict[int, int] = {}
        for mode in modes:
            counts[int(mode)] = counts.get(int(mode), 0) + 1
        return counts
    if "change_history" not in npz_data.files:
        return {}
    counts: dict[int, int] = {}
    for event in npz_data["change_history"]:
        event = dict(event)
        if int(event.get("changed", 0)):
            mode = int(event.get("mode", -1))
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def format_ms(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return "[]"
    return "[" + ", ".join(f"{value:.1f}" for value in values) + "]"


def print_report(source: Path, label: str, result: dict[str, object]) -> None:
    target_start, target_end = result["target_window_ms"]
    axis_start, axis_end = result["time_axis_ms"]
    print(f"\nSource: {source}")
    print(f"method: {label}")
    print(f"n_times: {result['n_times']}")
    print(f"time axis: {axis_start:.1f} to {axis_end:.1f} ms ({result['time_axis_source']})")
    if result["sfreq"] is not None:
        print(f"sample axis: {float(result['sfreq']):.1f} Hz, actual last frame {float(result['actual_end_ms']):.1f} ms")
    print(f"event lock: {result['event_lock']}")
    if result["expected_n_times"] and not result["n_times_matches_preprocessing"]:
        print(
            "warning: n_times does not match preprocessing SFREQ/TMIN/TMAX expected values "
            f"{list(result['expected_n_times'])}; evaluating by interpolated time axis anyway"
        )
    target_anchor = "response" if result["event_lock"] == "response" else "tensor zero point"
    print(f"target window: {target_start:.1f} to {target_end:.1f} ms after {target_anchor}")
    print(f"change point frames: {np.asarray(result['change_points'], dtype=int).tolist()}")
    print(f"change point ms: {format_ms(np.asarray(result['change_point_times_ms'], dtype=float))}")
    print(
        "hits in window: "
        f"{np.asarray(result['hits'], dtype=int).tolist()} "
        f"at {format_ms(np.asarray(result['hit_times_ms'], dtype=float))}"
    )
    print(f"hit count: {result['hit_count']}/{result['total_count']}")
    print(f"hit rate: {100.0 * float(result['hit_rate']):.1f}%")
    print(f"precision-like score: {100.0 * float(result['precision_like']):.1f}%")
    print(f"false positives outside window: {result['false_positive_count']}")
    print(f"false positive rate: {100.0 * float(result['false_positive_rate']):.1f}%")
    if result.get("mode_counts"):
        print(f"changed modes: {result['mode_counts']}")

    target_center = float(result["target_center_ms"])
    if result["closest_point"] is None:
        print(f"closest to {target_center:.1f} ms: n/a")
    else:
        print(
            f"closest to {target_center:.1f} ms: "
            f"frame {result['closest_point']} at {float(result['closest_ms']):.1f} ms "
            f"(error {float(result['closest_error_ms']):+.1f} ms)"
        )

    verdict = "PASS" if result["has_hit"] else "FAIL"
    print(f"verdict: {verdict}")


def print_comparison(rows: list[tuple[str, Path, dict[str, object]]]) -> None:
    if len(rows) < 2:
        return

    print("\nComparison")
    print("method                         verdict  hits    FP      precision  closest_ms  error_to_center_ms")
    print("-----------------------------  -------  ------  ------  ---------  ----------  ------------------")
    for label, _, result in rows:
        verdict = "PASS" if result["has_hit"] else "FAIL"
        closest_ms = "n/a" if result["closest_ms"] is None else f"{float(result['closest_ms']):.1f}"
        error_ms = "n/a" if result["closest_error_ms"] is None else f"{float(result['closest_error_ms']):+.1f}"
        hits = f"{result['hit_count']}/{result['total_count']}"
        fp_count = str(result.get("false_positive_count", "n/a"))
        precision = f"{100.0 * float(result.get('precision_like', 0.0)):.1f}%"
        print(f"{label[:29]:29}  {verdict:7}  {hits:6}  {fp_count:6}  {precision:9}  {closest_ms:10}  {error_ms:18}")

    best = min(
        rows,
        key=lambda row: (
            0 if row[2]["has_hit"] else 1,
            -float(row[2].get("precision_like", 0.0)),
            abs(float(row[2]["closest_error_ms"])) if row[2]["closest_error_ms"] is not None else np.inf,
            int(row[2]["total_count"]),
        ),
    )
    print(f"best by target-window hit then closest-to-center error: {best[0]}")


def source_paths_from_args(values: list[Path] | None) -> list[Path]:
    if values:
        return values
    return [path for path in DEFAULT_SOURCES if path.exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether detected change points fall in the expected ERN "
            "latency window after the response."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        help="Result .npz containing a change_points array. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tensor",
        type=Path,
        default=DEFAULT_TENSOR,
        help="Optional tensor .npy used only to infer n_times when the source does not store shape.",
    )
    parser.add_argument("--n-times", type=int, default=None, help="Override the number of time frames.")
    parser.add_argument(
        "--start-ms",
        type=float,
        default=None,
        help="Override time of frame 0 in ms. Defaults to preprocessing EEGConfig.TMIN.",
    )
    parser.add_argument(
        "--end-ms",
        type=float,
        default=None,
        help="Override time of the last frame in ms. Defaults to preprocessing EEGConfig.TMAX.",
    )
    parser.add_argument("--target-start-ms", type=float, default=25.0)
    parser.add_argument("--target-end-ms", type=float, default=75.0)
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=DEFAULT_PREPROCESSING_CONFIG,
        help="preprocessing/core/config.py used to read SFREQ, TMIN and TMAX.",
    )
    parser.add_argument(
        "--no-preprocessing-config",
        action="store_true",
        help="Ignore preprocessing config and use explicit/default --start-ms/--end-ms.",
    )
    parser.add_argument(
        "--allow-rescaled-time-axis",
        action="store_true",
        help=(
            "If n_times does not match preprocessing SFREQ/TMIN/TMAX, still map frames "
            "linearly from start-ms to end-ms. Without this, mismatched sources are skipped."
        ),
    )
    parser.add_argument(
        "--tensor-sfreq",
        type=float,
        default=None,
        help=(
            "Actual sampling rate of the saved tensor. Use 1024 for 2049-frame "
            "response-locked tensors from -1s to +1s. If omitted, it is inferred from n_times."
        ),
    )
    parser.add_argument(
        "--compute-hosvd",
        action="store_true",
        help="Compute HoSVD change points from --tensor before evaluation.",
    )
    parser.add_argument("--hosvd-output", type=Path, default=DEFAULT_HOSVD_OUTPUT)
    parser.add_argument("--hosvd-ranks", default="10,10,10")
    parser.add_argument("--hosvd-n-bkps", type=int, default=2)
    parser.add_argument("--hosvd-min-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_axis = resolve_time_axis(args)
    if args.compute_hosvd:
        from HoSVD import find_change_points_from_file, save_result

        if args.tensor is None or not args.tensor.exists():
            raise FileNotFoundError(f"HoSVD tensor input not found: {args.tensor}")
        hosvd_result = find_change_points_from_file(
            args.tensor,
            ranks=parse_rank_tuple(args.hosvd_ranks),
            n_bkps=args.hosvd_n_bkps,
            min_size=args.hosvd_min_size,
            use_low_rank=True,
        )
        save_result(hosvd_result, args.hosvd_output)
        print(f"Computed HoSVD change points and saved: {args.hosvd_output}")

    sources = source_paths_from_args(args.source)
    if args.compute_hosvd and args.hosvd_output not in sources:
        sources.append(args.hosvd_output)
    if not sources:
        raise FileNotFoundError("No source .npz files found. Pass --source path/to/results.npz.")

    comparison_rows: list[tuple[str, Path, dict[str, object]]] = []
    for source in sources:
        if not source.exists():
            print(f"\nSource: {source}")
            print("SKIP: file not found")
            continue

        with np.load(source, allow_pickle=True) as data:
            label = method_label(source, data)
            if "change_points" not in data:
                print(f"\nSource: {source}")
                print("SKIP: no change_points array")
                continue

            n_times = args.n_times or infer_n_times(data, args.tensor)
            if n_times is None:
                print(f"\nSource: {source}")
                print("SKIP: could not infer n_times. Pass --n-times.")
                continue

            expected_n_times = tuple(int(value) for value in time_axis["expected_n_times"])
            matches_preprocessing = not expected_n_times or int(n_times) in set(expected_n_times)
            tensor_sfreq = args.tensor_sfreq
            inferred_sfreq = infer_sample_rate_from_n_times(
                n_times,
                float(time_axis["start_ms"]),
                float(time_axis["end_ms"]),
            )
            if tensor_sfreq is None:
                tensor_sfreq = inferred_sfreq

            tensor_expected_n_times = ()
            matches_tensor_sfreq = False
            if tensor_sfreq is not None:
                tensor_expected_n_times = expected_counts_for_sample_rate(
                    tensor_sfreq,
                    float(time_axis["start_ms"]),
                    float(time_axis["end_ms"]),
                )
                matches_tensor_sfreq = int(n_times) in set(tensor_expected_n_times)

            if (
                expected_n_times
                and not matches_preprocessing
                and not matches_tensor_sfreq
                and not args.allow_rescaled_time_axis
            ):
                print(f"\nSource: {source}")
                print(f"method: {label}")
                print(f"n_times: {n_times}")
                print(
                    "SKIP: n_times does not match preprocessing config. "
                    f"Expected one of {list(expected_n_times)} from "
                    f"SFREQ={time_axis['sfreq']} Hz, "
                    f"TMIN={float(time_axis['start_ms']) / 1000.0:.3f}s, "
                    f"TMAX={float(time_axis['end_ms']) / 1000.0:.3f}s. "
                    "Also could not validate the tensor sampling rate. "
                    "Pass --tensor-sfreq 1024 for 2049-frame tensors or "
                    "--allow-rescaled-time-axis only for legacy tensors whose axis is known "
                    "to span the same response-locked interval."
                )
                continue

            eval_sfreq = None
            source_label = str(time_axis["source"])
            eval_expected_n_times = expected_n_times
            if matches_preprocessing and time_axis["sfreq"]:
                eval_sfreq = float(time_axis["sfreq"])
            elif matches_tensor_sfreq and tensor_sfreq is not None:
                eval_sfreq = float(tensor_sfreq)
                eval_expected_n_times = tuple(int(value) for value in tensor_expected_n_times)
                source_label = (
                    f"{source_label} + tensor_sfreq={float(tensor_sfreq):.1f} Hz "
                    f"(inferred from n_times)" if args.tensor_sfreq is None
                    else f"{source_label} + tensor_sfreq={float(tensor_sfreq):.1f} Hz"
                )
            elif expected_n_times and not matches_preprocessing and args.allow_rescaled_time_axis:
                source_label = f"{source_label} + rescaled linear axis"

            result = evaluate_change_points(
                data["change_points"],
                n_times=n_times,
                start_ms=float(time_axis["start_ms"]),
                end_ms=float(time_axis["end_ms"]),
                target_start_ms=args.target_start_ms,
                target_end_ms=args.target_end_ms,
                time_axis_source=source_label,
                event_lock=str(time_axis["event_lock"]),
                expected_n_times=eval_expected_n_times,
                sfreq=eval_sfreq,
            )
            mode_counts = change_history_mode_counts(data)
            if mode_counts:
                result["mode_counts"] = {int(mode): int(count) for mode, count in mode_counts.items()}
            print_report(source, label, result)
            comparison_rows.append((label, source, result))

    print_comparison(comparison_rows)


if __name__ == "__main__":
    main()

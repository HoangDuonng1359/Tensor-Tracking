# How Ho-RLSL Finds Change Points

## Raw change points

Ho-RLSL processes connectivity as:

```text
M_t = L_t + S_t
```

After every update window of length `alpha`, it updates the low-rank subspaces. A raw change point is created when a subspace direction is added or deleted.

Raw points are useful for debugging, but they are usually noisy at 1024 Hz.

```text
raw_change_points = all add/delete update events
```

## Score components

Each update window receives a composite score:

```text
reconstruction_error_score
```

How much sparse recovery struggles to explain the compressed residual.

```text
support_change_score
```

How much the sparse support changes inside the update window.

```text
subspace_angle_score
```

Projection distance between old and new subspaces.

```text
mode_energy_score
```

Residual energy outside the tracked mode subspaces.

```text
direction_update_score
```

Number of added/deleted directions, weighted by mode.

The final score is:

```text
0.30 reconstruction
0.20 support
0.30 subspace angle
0.15 mode energy
0.05 direction update
```

## Mode weighting

Internal modes:

```text
0: connectivity node mode
1: connectivity node mode
2: subject mode
```

Default weights:

```text
1.0, 1.0, 0.4
```

This means subject-only changes still exist in the report, but they are less likely to become final change points.

## Filtering

Final change points are produced from raw change points by:

1. Smoothing the final score.
2. Computing an adaptive MAD threshold.
3. Keeping local peaks above threshold.
4. Enforcing `min_cp_distance_ms`.

The saved arrays mean:

```text
raw_change_points       before filtering
filtered_change_points  after filtering
change_points           same as filtered_change_points
```

## Evaluation

For the current response-locked ERN tensor, evaluation uses:

```text
target window: 25-75 ms after response
target center: 50 ms
```

Good behavior:

- at least one filtered change point in 25-75 ms
- fewer false positives than raw change points
- small closest-to-50ms error
- mode contribution not only subject mode

Run:

```powershell
python evaluate_change_points.py --source fcca_results/ho_rlsl_results.npz --source fcca_results/hosvd_change_points.npz
```


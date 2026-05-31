# Ho-RLSL Algorithm

## 1. Data layout

Repo tensor:

```text
subjects x nodes x nodes x time
```

Internal paper-like tensor:

```text
nodes x nodes x subjects x time
```

Internal modes:

```text
mode 0: node/connectivity
mode 1: node/connectivity
mode 2: subject
mode 3: time
```

Mode 0 and mode 1 are tied by default because connectivity matrices are symmetric.

## 2. Initialization

The first `train_length` frames initialize the Tucker-mode subspaces by HOSVD.

For each non-time mode:

```text
unfold training tensor by mode
run SVD
keep singular vectors above sigma_min threshold
cap rank by max_ranks
```

The result is a basis list:

```text
P_0, P_1, P_2
```

## 3. Low-rank and sparse separation

For each time frame `t`, Ho-RLSL builds orthogonal projectors:

```text
phi_i = I - P_i P_i^T
```

Then it computes the compressed measurement:

```text
Y_t = M_t x_1 phi_1 x_2 phi_2 x_3 phi_3
```

Sparse recovery can use:

```text
gtcs_s_omp: GTCS-S-inspired OMP tensor pursuit
fista_l1:   L1/FISTA baseline
proxy:      fast threshold proxy
```

After sparse recovery:

```text
L_hat_t = M_t - S_hat_t
```

`L_hat_t` is sanitized by symmetry and zero diagonal before it is stored.

## 4. GTCS-S-inspired OMP

`gtcs_s_omp` solves sparse recovery by greedy pursuit:

1. Start with residual `R = Y_t`.
2. Backproject residual with `A*`.
3. Select the canonical tensor atom with maximum absolute correlation.
4. Build the compressed dictionary columns for selected atoms.
5. Refit coefficients by least squares.
6. Stop at `sparsity`, `max_atoms`, or `residual_tol`.

This is not the official GTCS-S implementation, but it is closer to the paper's sparse pursuit idea than using FISTA only.

## 5. Recursive subspace update

After `train_length`, frames are collected into update buffers of length `alpha`.

For each update buffer:

```text
unfold low-rank tensors by mode
delete low-energy directions
add high-residual directions
tie symmetric modes again
```

Every add/delete event is stored in `change_history`.

## 6. Change point scoring

Raw updates are not accepted directly as final change points. Each update window gets a score:

```text
final_score =
  0.30 * reconstruction_error_score
+ 0.20 * support_change_score
+ 0.30 * subspace_angle_score
+ 0.15 * mode_energy_score
+ 0.05 * direction_update_score
```

The default mode weights are:

```text
mode 0: 1.0
mode 1: 1.0
mode 2: 0.4
```

This reduces subject-mode-dominated false positives.

## 7. Final change point filter

The detector stores:

```text
raw_change_points
filtered_change_points
change_points
```

Filtering steps:

1. Smooth final scores.
2. Compute adaptive threshold with MAD.
3. Keep local peaks above threshold.
4. Merge peaks closer than `min_cp_distance_ms`.
5. Use the remaining points as `change_points`.

## 8. Paper-like ERN intervals

The pipeline uses filtered Ho-RLSL change points to create:

```text
pre_ern
ern
post_ern
```

The ERN anchor is the best change point in 25-75 ms after response. If none exists, the closest point to 50 ms is marked as fallback.


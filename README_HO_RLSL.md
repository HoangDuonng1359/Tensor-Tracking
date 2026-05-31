# Ho-RLSL

## Muc tieu

`Ho_RLSL.py` chay Higher-Order Recursive Low-Rank + Sparse Learning tren tensor connectivity:

```text
subjects x nodes x nodes x time
```

No tu chuyen ve layout paper-like:

```text
nodes x nodes x subjects x time
```

Moi frame thoi gian duoc tach thanh:

```text
M_t = L_t + S_t
```

- `L_t`: thanh phan low-rank, dung cho FCCA va community detection.
- `S_t`: thanh phan sparse/outlier.

## Solver sparse

Ban hien tai co 3 solver:

```text
gtcs_s_omp  default, GTCS-S-inspired greedy sparse tensor pursuit
fista_l1    L1/FISTA baseline
proxy       thresholding nhanh, chi dung de thu nghiem
```

`gtcs_s_omp` khong phai GTCS-S MATLAB chinh thuc cua bai bao, nhung gan hon FISTA vi no giai sparse pursuit tren compressed tensor measurement:

```text
Y_t = M_t x_1 phi_1 x_2 phi_2 x_3 phi_3
```

Moi vong lap chon canonical tensor atom co correlation lon nhat voi residual, fit lai coefficients bang least squares, roi dung khi dat `sparsity` hoac `residual_tol`.

## Change points

File `.npz` luu hai loai change point:

```text
raw_change_points       tat ca update windows co add/delete subspace direction
filtered_change_points  raw points sau khi score, smooth, threshold va min-distance
change_points           alias cua filtered_change_points
```

`change_points` la danh sach nen dung cho evaluate, Streamlit va interval paper-like.

Score gom:

```text
reconstruction_error_score
support_change_score
subspace_angle_score
mode_energy_score
direction_update_score
final_score
```

Detector loc nhieu bang:

```text
MAD adaptive threshold
score smoothing
minimum change-point distance, mac dinh 50 ms
mode_weights = 1.0,1.0,0.4
```

Mode 0/1 la connectivity; mode 2 la subjects. Subject-mode-only thay doi bi giam trong so de bot nhieu.

## Lenh chay khuyen nghi

```powershell
conda activate tensor
python tensor_de_v2.py --low-rank-method ho-rlsl --skip-timecourse --save-ho-rlsl-tensors --ho-train-length 80 --ho-alpha 64 --ho-sparse-solver gtcs_s_omp --ho-sparsity 8 --ho-min-cp-distance-ms 50 --ho-score-smoothing-ms 25 --ho-threshold-k 3.0
```

Danh gia:

```powershell
python evaluate_change_points.py --source fcca_results/ho_rlsl_results.npz --source fcca_results/hosvd_change_points.npz
```

Mo Streamlit:

```powershell
streamlit run streamlit_app.py
```

Trong Streamlit chon:

```text
Analysis tensor -> Ho-RLSL saved low-rank
```

## Output chinh

```text
fcca_results/ho_rlsl_results.npz
fcca_results/ho_rlsl_fcca_results.npz
fcca_results/paper_like_ho_rlsl_report.json
```

`paper_like_ho_rlsl_report.json` luu config, solver params, change points, mode contribution va interval pre-ERN/ERN/post-ERN.

## Cach doc ket qua

Ket qua tot hon khi:

- Co it nhat mot `change_points` trong 25-75 ms sau response.
- So false positives ngoai 25-75 ms giam so voi raw points.
- Change point ERN khong chi bi chi phoi boi subject mode.
- FCCA interval ERN co community structure ro hon pre/post.


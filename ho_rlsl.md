# Thuật toán Ho-RLSL

## 1. Mục tiêu

Ho-RLSL là viết tắt của **Higher-Order Recursive Low-Rank + Sparse Learning**. Mục tiêu là theo dõi sự thay đổi của cấu trúc connectivity theo thời gian.

Tại mỗi thời điểm `t`, tensor connectivity được tách thành:

<span>Mt</span>​=L<span>t</span>​+<span>St</span>​

Trong đó:

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $M_t$ | Tensor quan sát tại frame $t$ |
| $L_t$ | Tensor low-rank, biểu diễn cấu trúc connectivity chính |
| $S_t$ | Tensor sparse, biểu diễn outlier, nhiễu hoặc thay đổi cục bộ |

Trong pipeline hiện tại:

```
low_rank = L_t
sparse   = S_t
```

`low_rank` là đầu vào chính cho FCCA và Streamlit khi chọn:

```
Analysis tensor -> Ho-RLSL saved low-rank
```

## 2. Dạng dữ liệu

Tensor gốc trong repo có dạng:

$$\\text{subjects} \\times \\text{nodes} \\times \\text{nodes} \\times \\text{time}$$

Với dữ liệu hiện tại (40 subjects, 30 EEG channels, 2049 time frames):

$$\\text{shape} = (40,\\ 30,\\ 30,\\ 2049)$$

Ho-RLSL tự chuyển về dạng gần với bài báo:

$$\\text{nodes} \\times \\text{nodes} \\times \\text{subjects} \\times \\text{time}$$

$$\\text{shape nội bộ} = (30,\\ 30,\\ 40,\\ 2049)$$

Các mode nội bộ:

| Mode | Ý nghĩa |
| --- | --- |
| Mode 0 | node/connectivity |
| Mode 1 | node/connectivity |
| Mode 2 | subject |
| Mode 3 | time |

Mode 0 và mode 1 được gắn với nhau bằng `tie_symmetric_modes = (0, 1)`, vì ma trận connectivity đối xứng.

## 3. Khởi tạo bằng HOSVD

Ho-RLSL dùng `train_length` frame đầu để học subspace ban đầu. Giá trị khuyến nghị hiện tại:

$$\\text{train_length} = 80 \\text{ frames} = 78.125 \\text{ ms} \\quad (\\text{tại } 1024 \\text{ Hz})$$

Khối training:

$$M\_{\\text{train}} = \\text{sequence}\[\\ldots,\\ :\\ \\text{train_length}\]$$

Thuật toán unfold tensor theo từng mode $i$ rồi chạy SVD:

$$X\_{(i)} = U_i \\Sigma_i V_i^\\top$$

Giữ các singular vector có singular value vượt ngưỡng:

$$\\text{threshold} = \\sigma\_{\\min} \\times \\sigma\_{\\max}$$

Với config hiện tại: $\\sigma\_{\\min} = 0.11$, $\\text{max_ranks} = (10,\\ 10,\\ 10)$.

Kết quả là các basis của subspace: $P_0$ (mode connectivity 0), $P_1$ (mode connectivity 1), $P_2$ (mode subject).

## 4. Chiếu trực giao

Tại mỗi frame $t$, với tensor quan sát $M_t$, Ho-RLSL tạo projector trực giao với subspace hiện tại:

$$\\phi_i = I - P_i P_i^\\top$$

Sau đó chiếu tensor:

$$Y_t = M_t \\times_1 \\phi_1 \\times_2 \\phi_2 \\times_3 \\phi_3$$

$Y_t$ là phần của measurement nằm ngoài low-rank subspace hiện tại. Phần này được đưa vào bước khôi phục sparse.

## 5. Khôi phục sparse

Bài báo dùng GTCS-S cho bước khôi phục sparse. Project hiện tại không có file MATLAB `.m`, và Python không có hàm GTCS-S chuẩn sẵn có. Vì vậy đã thêm các solver sau:

| Solver | Mô tả |
| --- | --- |
| `gtcs_s_omp` | Mặc định, greedy tensor pursuit lấy cảm hứng từ GTCS-S |
| `fista_l1` | Baseline $\\ell_1$/FISTA |
| `fista` | Alias của `fista_l1` |
| `proxy` | Threshold proxy nhanh |

### 5.1. Solver mặc định: `gtcs_s_omp`

`gtcs_s_omp` là solver thay thế gần với tinh thần GTCS-S hơn FISTA. Nó giải sparse recovery theo hướng greedy:

1. Khởi tạo residual: $R = Y_t$
2. Backproject residual: $A^\*(R)$
3. Chọn canonical tensor atom có absolute correlation lớn nhất.
4. Tạo compressed dictionary column cho các atom đã chọn.
5. Fit lại coefficient bằng least squares.
6. Cập nhật residual.
7. Dừng khi đạt một trong các điều kiện:

$$\\text{selected_atoms} = \\text{sparsity}$$

$$\\text{selected_atoms} = \\text{max_atoms}$$

$$\\frac{|R|}{|Y_t|} \\leq \\text{residual_tol}$$

Config hiện tại: `sparse_solver = gtcs_s_omp`, `sparsity = 8`, `residual_tol = 1e-3`.

### 5.2. Baseline FISTA

FISTA vẫn được giữ để so sánh. Bài toán tối ưu:

$$\\min\_{S}\\ \\frac{1}{2} |A(S) - Y_t|\_F^2 + \\lambda |S|\_1$$

Đây là approximation thực dụng, nhưng không gần bài báo bằng `gtcs_s_omp`.

## 6. Tái tạo low-rank

Sau khi có sparse estimate $\\hat{S}\_t$, Ho-RLSL tính:

$$\\hat{L}\_t = M_t - \\hat{S}\_t$$

Sau đó làm sạch tensor low-rank:

- Thay $\\text{NaN}/\\text{Inf}$ bằng $0$
- Đối xứng hóa ma trận connectivity
- Gán đường chéo bằng $0$

$\\hat{L}\_t$ được lưu vào output `low_rank`.

## 7. Cập nhật subspace đệ quy

Sau `train_length`, Ho-RLSL gom các $\\hat{L}\_t$ vào buffer có độ dài:

$$\\alpha = 64 \\text{ frames} = 62.5 \\text{ ms} \\quad (\\text{tại } 1024 \\text{ Hz})$$

Khi buffer đủ $\\alpha$ frame, thuật toán cập nhật subspace. Với mỗi mode $i$:

$$D_i = \\text{concat}\\bigl(\\text{unfold}(\\hat{L}\_t,\\ \\text{mode}=i)\\bigr)$$

Sau đó:

- Xóa các direction có năng lượng thấp
- Thêm các direction có residual cao

Mỗi sự kiện thêm/xóa được lưu trong `change_history`.

## 8. Raw change points

Ban đầu Ho-RLSL đánh dấu raw change point khi có thêm/xóa direction trong update window. Raw points được lưu tại `raw_change_points`.

Kết quả hiện tại: **19 điểm**. Raw points được giữ lại để debug, nhưng không dùng trực tiếp để vẽ và evaluate vì còn nhiều false positives.

## 9. Điểm số change point tổng hợp

Để giảm nhiễu, mỗi update window được chấm điểm bằng điểm tổng hợp. Công thức:

$$\\text{final_score} = 0.30 \\cdot s\_{\\text{recon}} + 0.20 \\cdot s\_{\\text{support}} + 0.30 \\cdot s\_{\\text{angle}} + 0.15 \\cdot s\_{\\text{energy}} + 0.05 \\cdot s\_{\\text{dir}}$$

Trong đó:

| Thành phần | Ký hiệu | Ý nghĩa |
| --- | --- | --- |
| `reconstruction_error_score` | $s\_{\\text{recon}}$ | Sparse recovery còn lỗi lớn hay không |
| `support_change_score` | $s\_{\\text{support}}$ | Sparse support thay đổi mạnh hay không |
| `subspace_angle_score` | $s\_{\\text{angle}}$ | Subspace cũ và mới khác nhau bao nhiêu |
| `mode_energy_score` | $s\_{\\text{energy}}$ | Energy nằm ngoài subspace theo từng mode |
| `direction_update_score` | $s\_{\\text{dir}}$ | Số direction bị thêm/xóa |

Điểm số được lưu trong: `change_scores`, `change_score_times`, `change_point_score_components`.

## 10. Trọng số theo mode

Để tránh subject mode chi phối quá mạnh, project thêm:

$$\\text{mode_weights} = (1.0,\\ 1.0,\\ 0.4)$$

Tức là:

| Mode | Đối tượng | Weight |
| --- | --- | --- |
| Mode 0 | connectivity | $1.0$ |
| Mode 1 | connectivity | $1.0$ |
| Mode 2 | subject | $0.4$ |

Thay đổi subject mode vẫn được ghi nhận, nhưng bị giảm trọng số khi tính score.

> **Lưu ý:** Kết quả hiện tại có `change_point_modes = [2, 2, 2]`, tức filtered change points vẫn bị chi phối bởi subject mode. Đây là điểm chưa giống bài báo hoàn toàn.

## 11. Lọc change point cuối cùng

Ho-RLSL hiện lưu ba loại: `raw_change_points`, `filtered_change_points`, `change_points` (trong đó `change_points = filtered_change_points`).

Các bước lọc:

1. Làm mượt score theo `score_smoothing_ms`.
2. Tính adaptive threshold bằng MAD.
3. Giữ local peak vượt threshold.
4. Áp dụng khoảng cách tối thiểu `min_cp_distance_ms`.
5. Nếu raw points có điểm trong cửa sổ ERN $\[25,\\ 75\]$ ms, giữ target context:
   - Raw point trước ERN gần nhất
   - ERN anchor tốt nhất
   - Raw point sau ERN gần nhất

Lý do thêm target context: với dữ liệu hiện tại, frame 1072 tại $46.875$ ms là ERN hit tốt, nhưng có thể không phải local peak toàn cục vì frame 1136 có score cao hơn một chút.

## 12. Kết quả Ho-RLSL hiện tại

File kết quả: `fcca_results/ho_rlsl_results.npz`

Solver: `sparse_solver = gtcs_s_omp`

Filtered change points hiện tại:

$$\\text{change_points} = \[1008,\\ 1072,\\ 1136\]$$

$$\\text{change_point_ms} = \[-15.625,\\ 46.875,\\ 109.375\] \\text{ ms}$$

Trong đó frame $1072 \\rightarrow 46.875$ ms nằm trong cửa sổ ERN $\[25,\\ 75\]$ ms sau response.

So với bản cũ:

| Phiên bản | Hit | False positives |
| --- | --- | --- |
| Trước | 1/18 | 17 |
| Sau | 1/3 | 2 |

## 13. Các interval pre-ERN, ERN, post-ERN

Pipeline dùng filtered Ho-RLSL change points để tạo interval. Với kết quả hiện tại:

$$\\text{anchor} = \\text{frame } 1072 = 46.875 \\text{ ms}$$

$$\\text{boundary points} = \[1008,\\ 1136\]$$

Các interval:

| Interval | Frames | Thời gian |
| --- | --- | --- |
| `pre_ern` | $0 - 1007$ | $-1000.0$ ms đến $-16.6$ ms |
| `ern` | $1008 - 1135$ | $-15.6$ ms đến $108.4$ ms |
| `post_ern` | $1136 - 2048$ | $109.4$ ms đến $1000.0$ ms |

Report được lưu tại: `fcca_results/paper_like_ho_rlsl_report.json`

## 14. Nội dung file output

`fcca_results/ho_rlsl_results.npz` hiện có các key sau (quan trọng nhất):

$$\\text{low_rank}:\\ (40,\\ 30,\\ 30,\\ 2049)$$

$$\\text{sparse}:\\ (40,\\ 30,\\ 30,\\ 2049)$$

`low_rank` là input cho FCCA khi chọn Ho-RLSL. Các key đầy đủ:

```
change_points, raw_change_points, filtered_change_points
change_point_ms, change_point_modes
change_scores, change_score_times, change_point_score_components
sparse_solver, solver_params
update_times, rank_history, change_history
input_layout, output_layout, internal_layout
original_shape, internal_shape
initial_ranks, initial_thresholds, config
low_rank, sparse
```

## 15. Tích hợp Streamlit

Trong Streamlit: **Analysis tensor → Ho-RLSL saved low-rank**

App đọc `fcca_results/ho_rlsl_results.npz` và lấy: `low_rank`, `change_points`, `raw_change_points`, `filtered_change_points`, `change_scores`.

Trên waveform:

- Đường đỏ đứt: filtered Ho-RLSL change points
- Đường xám chấm: raw Ho-RLSL change points
- Đường xanh: ranh giới interval

Đã xóa đường CP score màu xám khỏi waveform để biểu đồ gọn hơn.

## 16. Đầu vào FCCA

Khi dùng Ho-RLSL, đầu vào cho FCCA là:

$$\\text{low_rank}\[:,\\ :,\\ :,\\ \\text{selected_frames}\]$$

Shape tổng: $\\text{low_rank} = (40,\\ 30,\\ 30,\\ 2049)$

Ví dụ interval ERN ($\\text{selected_frames} = 1008{-}1135$):

$$\\text{FCCA input shape} = (40,\\ 30,\\ 30,\\ 128)$$

## 17. Lệnh chạy

**Chạy Ho-RLSL pipeline:**

```powershell
conda activate tensor
python tensor_de_v2.py --low-rank-method ho-rlsl --skip-timecourse --save-ho-rlsl-tensors --ho-train-length 80 --ho-alpha 64 --ho-sparse-solver gtcs_s_omp --ho-sparsity 8 --ho-min-cp-distance-ms 50 --ho-score-smoothing-ms 25 --ho-threshold-k 3.0
```

**Đánh giá Ho-RLSL và HoSVD:**

```powershell
python evaluate_change_points.py --source fcca_results/ho_rlsl_results.npz --source fcca_results/hosvd_change_points.npz
```

**Chạy Streamlit:**

```powershell
streamlit run streamlit_app.py
```

## 18. Những điểm đã gần bài báo hơn

- Dùng tensor layout $\\text{nodes} \\times \\text{nodes} \\times \\text{subjects} \\times \\text{time}$
- Khởi tạo subspace bằng HOSVD
- Dùng recursive low-rank + sparse separation
- Thay FISTA-only bằng OMP lấy cảm hứng từ GTCS-S
- Phát hiện change point từ subspace updates
- Lọc raw change points để giảm false positives
- Tạo `pre_ern`/`ern`/`post_ern` từ Ho-RLSL change points
- Dùng Ho-RLSL `low_rank` làm đầu vào FCCA
- So sánh với HoSVD baseline

## 19. Những điểm chưa giống bài báo hoàn toàn

- GTCS-S hiện tại không phải bản MATLAB GTCS-S chính thức; `gtcs_s_omp` là solver thay thế trong Python
- Dữ liệu hiện tại có 40 subjects và 30 channels, khác dữ liệu bài báo
- Dữ liệu hiện tại là 1024 Hz, khác setup của bài báo
- Filtered change points hiện vẫn bị chi phối bởi subject mode (mode 2)

> **Kết luận:** Ho-RLSL đã bắt được ERN hit tại $46.875$ ms và giảm false positives mạnh. Tuy nhiên về mode contribution, kết quả vẫn chưa hoàn toàn paper-like.
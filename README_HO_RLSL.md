# Ho-RLSL

## Ý tưởng chính

Tại mỗi thời điểm, dữ liệu connectivity được xem là một tensor 3D:

```text
nodes x nodes x subjects
```

Trong repo này tensor gốc đang lưu theo dạng:

```text
subjects x nodes x nodes x time
```

`Ho_RLSL.py` tự chuyển về layout paper-like:

```text
nodes x nodes x subjects x time
```

Thuật toán giả định:

```text
M_t = L_t + S_t
```

Trong đó:

- `M_t`: tensor quan sát tại thời điểm `t`.
- `L_t`: phần low-rank, biểu diễn cấu trúc mạng não ổn định/chậm thay đổi.
- `S_t`: phần sparse, biểu diễn nhiễu/outlier/các cạnh bất thường.

## Pipeline

1. Khởi tạo subspace bằng HOSVD trên các time window đầu.
2. Tạo projector trực giao:

```text
phi_i = I - P_i P_i^T
```

3. Project tensor:

```text
Y_t = M_t x_1 phi_1 x_2 phi_2 x_3 phi_3
```

4. Recover sparse bằng FISTA:

```text
min_S 0.5 * ||A(S) - Y_t||_F^2 + lambda * ||S||_1
```

5. Tính:

```text
L_hat_t = M_t - S_hat_t
```

6. Cập nhật subspace theo cửa sổ `alpha`.
7. Ghi change point nếu có add/delete direction trong subspace.
8. Dùng low-rank tensor để chạy FCCA và vẽ các cụm pre-ERN, ERN, post-ERN.

## Cách chạy

Kích hoạt môi trường:

```powershell
conda activate tensor
```

Chạy Ho-RLSL trong pipeline chính:

```powershell
python tensor_de_v2.py --low-rank-method ho-rlsl --skip-timecourse
```

Nếu muốn Streamlit đọc lại low-rank tensor Ho-RLSL:

```powershell
python tensor_de_v2.py --low-rank-method ho-rlsl --skip-timecourse --save-ho-rlsl-tensors
```

Kết quả được lưu vào:

```text
fcca_results/ho_rlsl_results.npz
fcca_results/ho_rlsl_fcca_results.npz
fcca_results/figures/
```

Mở giao diện:

```powershell
streamlit run streamlit_app.py
```

Trong Streamlit, chọn:

```text
Analysis tensor -> Ho-RLSL saved low-rank
Graph layout -> Scalp electrode map
```

## Tham số quan trọng

### `--ho-train-length`

Số time window đầu dùng để khởi tạo subspace.

```powershell
--ho-train-length 10
```

### `--ho-alpha`

Số frame trong một cửa sổ cập nhật subspace.

```powershell
--ho-alpha 8
```

Nhỏ quá dễ bắt nhiễu thành change point giả. Lớn quá dễ bỏ lỡ thay đổi nhanh.

### `--ho-sigma-min`

Ngưỡng giữ/thêm/xóa basis direction.

```powershell
--ho-sigma-min 0.11
```

Tham số này rất nhạy. Nếu quá thấp, subspace giữ cả nhiễu. Nếu quá cao, thuật toán bỏ mất thay đổi thật.

### `--ho-max-ranks`

Giới hạn rank cho 3 mode:

```powershell
--ho-max-ranks 10,10,10
```

Với tensor hiện tại `30 nodes x 30 nodes x 40 subjects`, giá trị thực dụng thường là:

```text
8,8,8
10,10,10
12,12,10
```

### `--ho-sparse-solver`

Chọn solver sparse:

```powershell
--ho-sparse-solver fista
```

Hoặc chạy nhanh hơn nhưng thô hơn:

```powershell
--ho-sparse-solver proxy
```

### `--ho-fista-max-iter`

Số vòng lặp FISTA:

```powershell
--ho-fista-max-iter 20
```

Giảm xuống nếu chạy chậm:

```powershell
--ho-fista-max-iter 5
```

## Clustering pre-ERN, ERN, post-ERN

Project hiện dùng clustering theo profile interval:

```text
pre_ern  -> compact: ít cụm hơn
ern      -> detailed: nhiều cụm hơn
post_ern -> compact: ít cụm hơn
```

Số cụm không bị ép cố định. Thuật toán tự chọn dựa trên Louvain/Greedy và adaptive recursive Fiedler.

Với kết quả hiện tại, mục tiêu là:

```text
pre_ern:  ít cụm
ern:      nhiều cụm hơn
post_ern: ít cụm
```

Node trong graph được tô theo community. Cạnh nối hai node cùng community dùng màu của community đó; cạnh nối khác community là xám nhạt.

## Đánh giá Ho-RLSL tốt hay không

Chạy synthetic benchmark:

```powershell
python test_ho_rlsl_synthetic.py
```

Các chỉ số cần xem:

- `Ho-RLSL MSE intervals`
- `HoSVD MSE intervals`
- `cp_true`
- `cp_detected`
- `cp_error`

Ho-RLSL tốt nếu:

```text
Ho-RLSL MSE < HoSVD MSE
cp_error nhỏ
change points ổn định
ERN có modularity hoặc segregation rõ hơn pre/post
```

## Khác biệt với bài báo gốc

Module hiện tại là bản **paper-like**, chưa phải bản gốc 100%.

Khác biệt chính:

- Bài báo dùng GTCS-S cho sparse recovery; code hiện dùng FISTA approximation.
- Dữ liệu project hiện tại có `40 subjects x 30 nodes x 2049 time windows`, khác bài báo gốc `91 subjects x 63 channels x 256 time points`.
- Clustering trong project đã được điều chỉnh để phục vụ visualization pre-ERN/ERN/post-ERN.
- Tham số cần tune theo dữ liệu thật.

## Lệnh gợi ý

Chạy nhanh:

```powershell
python tensor_de_v2.py --low-rank-method ho-rlsl --skip-timecourse --ho-sparse-solver proxy --ho-max-ranks 8,8,8
```

Chạy FISTA nhẹ:

```powershell
python tensor_de_v2.py --low-rank-method ho-rlsl --skip-timecourse --ho-fista-max-iter 5 --ho-max-ranks 8,8,8
```

Chạy đầy đủ hơn:

```powershell
python tensor_de_v2.py --low-rank-method ho-rlsl --skip-timecourse --save-ho-rlsl-tensors --ho-fista-max-iter 20 --ho-max-ranks 10,10,10
```

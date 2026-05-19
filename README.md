# Tensor Tracking ERN/CRN

Repo này dùng để phân tích kết nối não theo thời gian từ dữ liệu EEG trong thí nghiệm ERN/CRN.

Nói ngắn gọn: code đọc tensor kết nối não 4 chiều, tìm các giai đoạn thời gian quan trọng quanh lỗi phản ứng, gom cụm các kênh não theo mức kết nối, rồi lưu hình và file kết quả để xem lại.

## Cấu trúc chính

```text
tensor_de_v2.py                 Script phân tích chính
streamlit_app.py                Giao diện web để xem tensor, ERP và kết quả FCCA
tensor_4d/                      Tensor kết nối não đã tạo sẵn
fcca_results/                   Kết quả phân tích và hình xuất ra
ERN_Raw_Data_BIDS-Compatible/   Dữ liệu EEG gốc dạng BIDS/EEGLAB
```

## Dữ liệu đầu vào

Script chính dùng hai file tensor:

```text
tensor_4d/tensor_incorrect_4d.npy
tensor_4d/tensor_correct_4d.npy
```

Mỗi tensor có dạng:

```text
(subjects, nodes, nodes, time_windows)
```

Ý nghĩa:

- `subjects`: số người tham gia.
- `nodes`: số kênh hoặc vùng não.
- `nodes x nodes`: ma trận kết nối giữa các kênh não.
- `time_windows`: các cửa sổ thời gian quanh sự kiện ERN/CRN.

## File `tensor_de_v2.py` làm gì?

Luồng xử lý chính nằm trong hàm `main()`:

1. Vẽ sóng ERP/CRN và các điểm đổi trạng thái kết nối.
2. Đọc tensor incorrect mặc định từ `tensor_4d/tensor_incorrect_4d.npy`.
3. Giảm nhiễu tensor bằng Tucker low-rank decomposition.
4. Tìm 2 điểm thay đổi trên trục thời gian.
5. Chia thời gian thành 3 giai đoạn:
   - `pre_ern`: trước ERN.
   - `ern`: giai đoạn ERN chính.
   - `post_ern`: sau ERN.
6. Với từng giai đoạn, chạy FCCA đơn giản:
   - gom cụm graph của từng subject và từng thời điểm;
   - tạo consensus matrix;
   - gom cụm lại trên consensus matrix;
   - tính modularity để đánh giá chất lượng cộng đồng.
7. Lưu kết quả `.npz` và các hình `.png`.

## Các hàm quan trọng

### `tucker_low_rank_decomposition`

Giảm nhiễu tensor bằng phân rã Tucker, sử dụng thư viện tensorly.

Kết quả gồm:

- `L`: phần low-rank, được xem là tín hiệu sạch hơn.
- `S`: phần dư/nhiễu.
- `core`, `factors`: thành phần phân rã Tucker.

### `detect_change_points`

Tìm các điểm mà cấu trúc kết nối não thay đổi theo thời gian.

Nếu có thư viện `ruptures`, code dùng `ruptures.Binseg`.
Nếu không có, code tự dùng dynamic programming theo lỗi SSE.

### `make_change_point_intervals`

Từ các điểm thay đổi, chia thời gian thành 3 đoạn:

```text
pre_ern -> ern -> post_ern
```

### `run_fcca_interval`

Chạy phân tích FCCA cho một đoạn thời gian.

Ý tưởng đơn giản:

- Mỗi ma trận kết nối là một graph.
- Mỗi graph được chia thành các cộng đồng bằng modularity.
- Các lần chia cụm được gom lại thành một consensus matrix.
- Consensus matrix cho biết hai node thường nằm cùng cộng đồng đến mức nào.

### `visualize_fcca_results`

Vẽ 3 hình cho mỗi giai đoạn:

- consensus matrix;
- graph trung bình;
- graph vòng tròn sau khi threshold cạnh mạnh.

### `visualize_condition_timecourses`

Vẽ ERP/CRN trung bình cho điều kiện incorrect và correct.

Nếu đọc được EEG gốc hoặc epoch FIF bằng `mne`, hình sẽ dùng sóng EEG thật.
Nếu không, code dùng trung bình kết nối từ tensor làm đường thay thế.

## Cách chạy script phân tích

Cài các thư viện cần thiết:

```bash
pip install numpy scipy matplotlib tensorly networkx mne streamlit pandas ruptures dyconnmap
```

Chạy phân tích:

```bash
python tensor_de_v2.py
```

Kết quả được lưu vào:

```text
fcca_results/fcca_results.npz
fcca_results/figures/
```

Các hình chính:

```text
fcca_results/figures/incorrect_correct_tensor_timecourses.png
fcca_results/figures/pre_ern_fcca.png
fcca_results/figures/ern_fcca.png
fcca_results/figures/post_ern_fcca.png
```

## Cách chạy giao diện Streamlit

```bash
streamlit run streamlit_app.py
```

Giao diện này dùng để xem dữ liệu và kết quả trực quan hơn, không cần đọc trực tiếp file `.npz`.

## Ý nghĩa kết quả

`fcca_results.npz` chứa các mảng kết quả cho từng giai đoạn:

- `frames`: các frame thời gian thuộc giai đoạn đó.
- `graph_labels`: nhãn cộng đồng của từng graph.
- `graph_modularity_scores`: điểm modularity của từng graph.
- `consensus_matrix`: ma trận đồng cụm giữa các node.
- `consensus_labels`: nhãn cộng đồng cuối cùng.
- `consensus_modularity`: điểm modularity của consensus graph.
- `mean_adjacency`: ma trận kết nối trung bình.
- `change_points`: các frame được phát hiện là điểm đổi trạng thái.
- `change_point_method`: phương pháp tìm change point.

## Tóm tắt dễ hiểu

Code này trả lời câu hỏi:

> Kết nối giữa các kênh não thay đổi như thế nào trước, trong và sau ERN/CRN?

Nó làm việc này bằng cách:

1. Dùng tensor 4D biểu diễn kết nối não theo subject và thời gian.
2. Làm sạch tensor bằng Tucker decomposition.
3. Tự tìm ranh giới thời gian giữa các giai đoạn.
4. Gom cụm các kênh não thành cộng đồng kết nối.
5. Xuất hình và file kết quả để phân tích tiếp.

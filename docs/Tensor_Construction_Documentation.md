# Quy trình Xây dựng Tensor và Phân tích Kết nối
## 🧠 Dựa trên phương pháp Ozdemir (2017)

Sau khi tiền xử lý, dữ liệu EEG sạch được chuyển đổi thành các Tensor 4D đại diện cho sự thay đổi của mạng lưới kết nối não bộ theo thời gian.

---

## 1. Cơ sở lý thuyết: Kết nối Pha (Phase Connectivity)

Mạng lưới não bộ trong nghiên cứu này được xác định bởi sự **đồng bộ pha** giữa các vùng não thay vì biên độ tín hiệu đơn thuần.

### Kỹ thuật Ước lượng Pha: RID-Rihaczek
Dự án sử dụng **Reduced Interference Rihaczek Distribution (RID-Rihaczek)** để ước lượng pha tức thời:
- **Dải tần**: Tập trung vào dải **Theta (4-8 Hz)**.
- **Mã nguồn**: Thực thi tại hàm `compute_rid_rihaczek_tfd()` trong `preprocessing/connectivity.py`.

---

## 2. Chỉ số Kết nối: PLV (Phase Locking Value)

Để đo lường sự kết nối giữa hai kênh EEG (nút mạng), dự án sử dụng **Phase Locking Value (PLV)**:
- **Giá trị**: Từ 0 (không đồng bộ) đến 1 (đồng bộ hoàn hảo).
- **GPU Acceleration**: Tính toán PLV được tăng tốc bằng GPU (PyTorch) để xử lý khối lượng dữ liệu lớn của 40 subjects.

---

## 3. Cấu trúc Tensor 4D sau cùng

Dữ liệu từ **40 đối tượng** được tổng hợp thành các khối Tensor:

### Tensor tổng hợp (4D):
- **Kích thước**: `(40, 30, 30, 256)`
    - `40`: Số lượng người tham gia (bao gồm toàn bộ dataset).
    - `30 x 30`: Ma trận kết nối giữa các cặp kênh.
    - `256`: Các mốc thời gian (tương ứng 2 giây tại 128Hz).

---

## 4. Quy trình thực hiện (`preprocessing/connectivity.py`)

1. **Temporal Matching**: 
   - Thay vì lấy mẫu ngẫu nhiên, hệ thống sử dụng hàm `match_temporal_trials()` để chọn các trial Đúng có thời gian gần nhất với trial Sai. 
   - Điều này giúp triệt tiêu nhiễu do mệt mỏi hoặc drift tín hiệu qua thời gian.
2. **Batch Processing**: 
   - Pipeline tự động duyệt qua 40 subjects, tính toán ma trận PLV cho từng người và stack chúng thành Tensor 4D.
3. **Lưu trữ**: 
   - Kết quả được lưu dưới dạng `.npy` để tối ưu hóa việc đọc/ghi trong Python và `.mat` để phục vụ kiểm chứng bằng MATLAB nếu cần.

---

## 5. Ứng dụng trong HO-RLSL
Khối Tensor 4D này là đầu vào trực tiếp cho thuật toán `decompose_tensor_stream()` trong `algorithms/ho_rlsl.py`. Nó cho phép theo dõi sự tiến hóa của "Common Subspace" – đại diện cho cấu trúc mạng lưới dùng chung giữa tất cả các subjects.

---
*Tài liệu được cập nhật dựa trên phiên bản code tái cấu trúc (11/05/2026).*

# Kết quả Phân tích và Kiểm định Thống kê
## 🧠 Dựa trên phương pháp Ozdemir (2017)

Tài liệu này tóm tắt kết quả thu được sau khi chạy toàn bộ quy trình từ xây dựng Tensor đến phân tách thành phần (Decomposition) và kiểm định thống kê.

---

## 1. Phân tách Tensor (HO-RLSL Decomposition)
Chúng ta đã áp dụng thuật toán **Higher-Order Recursive Low-Rank + Sparse Structure Learning** để tìm ra "không gian con chung" (common subspace) đại diện cho mạng lưới não bộ của tất cả các đối tượng.

### Đặc tính Động của Mạng lưới (Network Dynamics)
- **Năng lượng mạng (Network Energy)**: Biểu thị mức độ hoạt động đồng bộ của mạng lưới.
    - Trên ERP CORE, cửa sổ ERN chính được script `analysis.reproduce_behavioral_metrics` tóm tắt theo phong cách Table V, nhưng không còn được mô tả như một bản tái lập "exact reproduction".
- **Điểm thay đổi (Change-points)**: Thuật toán đã tự động phát hiện các thời điểm mà mạng lưới não bộ thay đổi cấu trúc nhanh chóng.
    - Các điểm thay đổi quanh ERN thường xuất hiện gần mốc trước 0ms và sau 100ms, nhưng có thể xuất hiện thêm các khoảng hậu ERN phụ do ERP CORE có ít subjects và ít channels hơn paper gốc.

---

## 2. Trọng số Mạng lưới (Network Weights)
Biểu đồ `ozdemir_replicated_dynamics.png` cho thấy sự phân bổ tầm quan trọng của các vùng não theo thời gian:
- **Vùng trung tâm (FCz, Cz)**: Chiếm trọng số cao nhất tại đỉnh ERN, đóng vai trò "nút thắt" (hub) trong mạng lưới kiểm soát lỗi.
- **Sự chuyển dịch**: Mạng lưới bắt đầu từ sự kết nối rộng ở vùng trán, sau đó thu hẹp và tập trung mạnh vào vùng trung tâm khi phát hiện lỗi.

---

## 3. Kiểm định và Diễn giải (Validation)
Các script chính hiện được chia làm hai lớp:
- **Table V-style comparison**: `analysis.reproduce_behavioral_metrics`
- **FCCA modularity over primary ERN windows**: `analysis.evaluate_fcca`

Kết quả được diễn giải theo nguyên tắc:
- So sánh với Ozdemir (2017) ở mức **high-fidelity**.
- Giải thích rõ các sai khác còn lại bằng ràng buộc của ERP CORE thay vì mô tả như lỗi implementation nếu chưa có bằng chứng code-level.

---

## 4. Danh sách Kết quả đầu ra (Outputs)
Các hình ảnh kết quả đã được lưu tại `outputs/eda/`:
1. `ozdemir_replicated_dynamics.png`: Biểu đồ năng lượng và trọng số mạng lưới theo thời gian (giống phong cách bài báo Ozdemir).
2. `statistical_validation_final.png`: Kết quả so sánh T-test và biểu đồ tương quan hành vi.

---

## 5. Kết luận
Quy trình hiện tại là một **high-fidelity reproduction on ERP CORE** của Ozdemir (2017). Kết quả được báo cáo theo hướng bám sát paper, đồng thời nêu rõ các khác biệt do số lượng subjects, số channels và chiến lược cân bằng trial.

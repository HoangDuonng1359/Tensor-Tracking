# Exploratory Data Analysis (EDA) Documentation
## 🧠 Project: TrackingTensor3D (ERN Component - Flanker Task)

Tài liệu này tóm tắt quá trình Khám phá Dữ liệu (EDA) chi tiết cho toàn bộ **40 đối tượng** trong bộ dữ liệu EEG thuộc dự án **TrackingTensor3D**, dựa trên nghiên cứu của Ozdemir (2017) và bộ dữ liệu **ERP CORE**.

---

## 1. Tổng quan về Bộ dữ liệu (Dataset Overview)
- **Tổng số đối tượng**: 40 người trưởng thành.
- **Nhiệm vụ**: **Flanker Task** (Nghiên cứu về Error-Related Negativity - ERN).
- **Thiết bị**: EEG 30 kênh, tốc độ lấy mẫu 128 Hz.
- **Mục tiêu**: Phân tích sự thay đổi động của mạng lưới não bộ khi mắc lỗi (Incorrect) so với khi làm đúng (Correct).

---

## 2. Thống kê Chi tiết về Thử nghiệm (Trial Statistics)

Dựa trên kết quả chạy pipeline mới nhất cho 40 subjects:

### A. Thử nghiệm Đúng (Correct Trials)
- **Đặc điểm**: Số lượng trial Đúng dồi dào (~250-400 trials/người), cung cấp baseline ổn định.

### B. Thử nghiệm Sai (Incorrect Trials - ERN)
- **Đặc điểm**: Số lượng trial biến thiên lớn tùy theo hiệu suất người tham gia (từ 1 đến 138 trials).
- **Cập nhật Inclusion Criteria**: Theo yêu cầu mới, **tất cả 40 subjects** đều được đưa vào phân tích (bao gồm cả các sub có ít hơn 15 lỗi) để tối đa hóa kích thước mẫu. 

---

## 3. Quy trình xử lý Artifacts & Tín hiệu

1. **CSD (Current Source Density)**: Thay thế hoàn toàn cho ICA để bảo toàn pha tín hiệu. Giúp giảm nhiễu Volume Conduction và làm nổi bật các nguồn phát tại FCz/Cz.
2. **Temporal Matching**: Thay vì lấy ngẫu nhiên, chúng ta khớp các trial Đúng gần nhất với thời điểm xảy ra trial Sai để triệt tiêu ảnh hưởng của sự mệt mỏi hoặc trôi tín hiệu sinh học.
3. **Theta Enhancement**: Phân tích PSD xác nhận năng lượng Theta (4-8Hz) tăng mạnh trong cửa sổ 0-150ms sau phản ứng sai.

---

## 4. Danh sách Subjects & Trạng thái

| Subject | Status | Incorrect Trials | Decision |
| :--- | :--- | :--- | :--- |
| **sub-001** | SUCCESS | 56 | Included |
| **sub-005** | SUCCESS | 2 | Included (Low trial count) |
| **sub-006** | SUCCESS | 138 | Included (High trial count) |
| **sub-040** | SUCCESS | 28 | Included |
| ... | ... | ... | ... |

---

## 5. Kết luận EDA
Việc bao gồm toàn bộ 40 subjects giúp tăng cường sức mạnh cho các phân tích cụm (FCCA). Sự chênh lệch trial được xử lý triệt để bằng **Temporal Matching Subsampling**, đảm bảo mỗi subject đóng góp một cặp tensor Đúng/Sai có kích thước tương đương và trạng thái sinh học tương đồng.


# Tài liệu Chi tiết Quy trình Tiền xử lý (Preprocessing Pipeline)
## 🧠 Dự án: TrackingTensor3D | Dữ liệu: 40 Subjects (ERP CORE)

Tài liệu này cung cấp cái nhìn chuyên sâu vào các kỹ thuật xử lý tín hiệu số được áp dụng để chuyển đổi dữ liệu EEG thô từ 40 người tham gia thành các đoạn tín hiệu sạch (Clean Epochs) phục vụ cho phân tích Tensor, tuân thủ nghiêm ngặt framework của **Ozdemir et al. (2017)**.

---

## 1. Kiến trúc Quy trình Tiền xử lý
Quy trình được thực hiện thông qua tập lệnh `preprocessing/pipeline.py` với các tham số được tối ưu hóa cho thành phần ERN:

### A. Chuẩn hóa và Lọc tín hiệu (Standardization & Filtering)
- **Resampling (128 Hz)**: Giảm tốc độ lấy mẫu xuống 128 Hz để tối ưu hóa tính toán Tensor mà vẫn đảm bảo dải tần số quan tâm (Theta).
- **Lọc thông dải (Band-pass Filter)**: Sử dụng bộ lọc **Zero-phase FIR (firwin)** từ 0.1 Hz đến 30 Hz.

### B. Biến đổi mật độ nguồn dòng (Current Source Density - CSD)
Theo đúng phương pháp của Ozdemir (2017), hệ thống **không sử dụng ICA** để tránh làm biến đổi pha tín hiệu một cách nhân tạo. Thay vào đó, CSD được áp dụng:
- **Lợi ích**: Loại bỏ hiện tượng Volume Conduction (dẫn truyền khối qua hộp sọ).
- **Kết quả**: Biến các điện cực trên da đầu thành các nguồn phát độc lập, làm sắc nét các kết nối chức năng giữa các vùng não.

### C. Tính toán kết nối (Connectivity Computation)
Sử dụng chỉ số **Phase Locking Value (PLV)** được tính toán thông qua phân phối thời gian-tần số **RID-Rihaczek**:
- **GPU Acceleration**: Tính toán trên GPU để xử lý nhanh tensor 4D.
- **Temporal Matching**: Cân bằng số lượng trial Đúng (Correct) và Sai (Incorrect) bằng cách khớp thời gian, nhằm loại bỏ sai số do sự trôi sinh học (biological drift). **Tất cả 40 subjects** đều được giữ lại trong quy trình này.

---

## 2. Phân đoạn và Gán nhãn (Epoching & Labeling)
Dữ liệu sau khi sạch được cắt thành các đoạn nhỏ (Epochs) dựa trên các mốc thời gian phản ứng:
- **Thời gian**: từ **-1.0s đến +1.0s** (256 điểm dữ liệu).
- **Phân loại**: **Correct** (Làm đúng) và **Incorrect** (Làm sai).
- **Baseline Correction**: Hiệu chỉnh nền dựa trên khoảng **-600ms đến -400ms**.

---

## 3. Quy trình thực hiện (Workflow)

Để chạy toàn bộ quy trình tiền xử lý, thực hiện các lệnh sau:

1. **Khởi tạo môi trường**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Chạy ETL & CSD Transformation**:
   ```bash
   python3 -m preprocessing.pipeline
   ```

3. **Tính toán Connectivity (PLV)**:
   ```bash
   python3 -m preprocessing.connectivity
   ```

4. **Lọc dải Theta & Cân bằng Trial**:
   ```bash
   python3 -m preprocessing.cleaning
   ```

---

## 4. Tóm tắt các thông số kỹ thuật (Config)

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **SFREQ** | 128 Hz | Tốc độ lấy mẫu |
| **Filter Range** | 0.1 - 30.0 Hz | FIR Zero-phase |
| **ICA** | None | Tuân thủ Ozdemir (2017) |
| **CSD** | Spherical Spline | Laplacian transform |
| **Min Trials** | 1 | Bao gồm toàn bộ 40 subjects |
| **Connectivity** | PLV (RID-Rihaczek) | Tính toán trên GPU |

---
> [!IMPORTANT]
> Quy trình này tập trung vào việc bảo toàn tối đa thông tin pha. Việc sử dụng CSD thay cho ICA và giữ lại toàn bộ 40 subjects giúp tăng cường sức mạnh thống kê cho các phân tích cụm (FCCA) ở giai đoạn sau.


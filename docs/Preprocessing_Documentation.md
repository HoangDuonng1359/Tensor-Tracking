# Tài liệu Chi tiết Quy trình Tiền xử lý (Preprocessing Pipeline)
## 🧠 Dự án: TrackingTensor3D | Dữ liệu: 40 Subjects (ERP CORE)

Tài liệu này cung cấp cái nhìn chuyên sâu vào các kỹ thuật xử lý tín hiệu số được áp dụng để chuyển đổi dữ liệu EEG thô từ 40 người tham gia thành các đoạn tín hiệu sạch (Clean Epochs) phục vụ cho phân tích Tensor, tuân thủ nghiêm ngặt framework của **Ozdemir et al. (2017)**.

---

## 1. Kiến trúc Quy trình Tiền xử lý
Quy trình được thực hiện thông qua tập lệnh `preprocessing/pipeline.py` với các tham số được tối ưu hóa cho thành phần ERN:

### A. Chuẩn hóa tín hiệu (Standardization)
- **Resampling (128 Hz)**: Giảm tốc độ lấy mẫu xuống 128 Hz để phù hợp với baseline của Ozdemir.
- **Broadband raw path**: `preprocessing/pipeline.py` hiện không áp dụng bộ lọc 0.1-30Hz trên raw trước khi epoching. Theta filtering được thực hiện ở bước sampling để giữ đúng pipeline PLV theo dải tần mục tiêu.

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
- **Thời gian**: từ **-1.0s đến +1.0s** (257 điểm dữ liệu).
- **Phân loại**: **Correct** (Làm đúng) và **Incorrect** (Làm sai).
- **Baseline Correction**: Không áp dụng trong `preprocessing/pipeline.py`; epochs được giữ ở trạng thái broadband + CSD trước bước theta balancing.

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

3. **Lọc dải Theta & Cân bằng Trial**:
   ```bash
   python3 -m preprocessing.sampling
   ```

4. **Tính toán Connectivity (PLV)**:
   ```bash
   python3 -m preprocessing.connectivity
   ```

---

## 4. Tóm tắt các thông số kỹ thuật (Config)

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **SFREQ** | 128 Hz | Tốc độ lấy mẫu |
| **Filter Range** | Theta 4.0 - 8.0 Hz | Áp dụng trong bước sampling |
| **ICA** | None | Tuân thủ Ozdemir (2017) |
| **CSD** | Spherical Spline | Laplacian transform |
| **Min Trials** | 1 | Bao gồm toàn bộ 40 subjects |
| **Connectivity** | PLV (RID-Rihaczek) | Tính toán trên GPU |

---
> [!IMPORTANT]
> Quy trình hiện tại được tổ chức theo hướng broadband + CSD trước, sau đó mới lọc theta để xây dựng PLV. Đây là pipeline paper-aligned đang được dùng bởi các script phân tích chính trong repo.

---

## 5. Kết quả Đầu ra (Tensor 4D)

Quá trình tiền xử lý kết thúc bằng việc tạo ra các Tensor 4D, phục vụ trực tiếp cho quá trình trích xuất đặc trưng bằng HO-RLSL và HOSVD.
- **Kích thước Tensor**: `[Times, Channels, Channels, Subjects]`
  - Cụ thể: `[257, 30, 30, 40]` (257 thời điểm, ma trận liên kết 30x30 kênh, 40 subjects).
- **Lưu trữ**: Tensor được lưu dưới dạng `.npy` (Python) và `.mat` (MATLAB) để thuận tiện đối chiếu.

---

## 6. Kết quả Thực nghiệm & Đánh giá (Empirical Benchmarks)

Sau khi đưa Tensor 4D vào các thuật toán tracking, chúng tôi thu được những kết quả xác thực hiệu năng quan trọng của HO-RLSL so với HOSVD (Baseline):

### A. Change Point (CP) Extraction & Network Topology
HO-RLSL nhận diện rõ ràng khoảng Response-related:

| Điều kiện | Số lượng CPs | Response Interval (ms) | Modularity | W/B Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **ERN (Incorrect)** | 6 | **15.6 – 132.8** | 0.5593 | 3.9804 |
| **CRN (Correct)** | 6 | **15.6 – 132.8** | 0.5427 | 6.4429 |

*Ghi chú về CRN Functional Integration*: Tại cửa sổ CRN, tỷ lệ Within/Between (W/B) tăng vọt lên **6.44**, tuy nhiên Modularity lại giảm nhẹ. Điều này cho thấy sự tái cấu trúc mạnh mẽ của mạng lưới: thay vì duy trì các cộng đồng rời rạc, bộ não tích hợp thành một cộng đồng lớn (giant component) để xử lý tín hiệu theo dõi phản hồi đúng (correct-response monitoring).

### B. So sánh HO-RLSL và HOSVD

| Thuật toán | Response Interval | Overlap (0-150ms) | Modularity | W/B Ratio | Dist ERN-CRN | Permutation p-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HOSVD** | 0.0–117.2 ms | 78.1% | 0.0000 | N/A (1 cluster) | **7.7460** | < 0.0001 |
| **HO-RLSL** | **15.6–132.8 ms**| **78.1%** | 0.5593 | **3.9804** | 1.3340 | < 0.0001 |

### C. Khả năng ổn định (Subject Bootstrapping Stability)
Kiểm định lặp (100 runs, ±20ms tolerance, khoảng 0-200ms):

| Thuật toán | Change Point (ms) | Độ ổn định (Stability) | Cluster Stability (ARI) |
| :--- | :--- | :--- | :--- |
| **HO-RLSL** | -54.7, 7.8, 70.3, 132.8, 195.3 | ~31 - 37% | **0.6187 ± 0.1283** |
| **HOSVD** | Không phát hiện | 0% | 0.0038 ± 0.0647 |

**Kết luận thực nghiệm:** 
Permutation test chứng minh cả hai thuật toán đều phân loại ERN và CRN với ý nghĩa thống kê cao ($p < 0.0001$). Tuy nhiên, HO-RLSL vượt trội hơn hẳn trong việc tracking động học cấu trúc: nó duy trì cấu trúc cộng đồng rõ rệt (W/B ratio cao) và tạo ra các mốc change points ổn định quanh khoảng thời gian phản ứng dưới điều kiện bootstrapping, trong khi HOSVD bị suy thoái mạng lưới (network degeneracy với Modularity = 0) và hoàn toàn mất ổn định trên tập mẫu phụ (sub-samples).

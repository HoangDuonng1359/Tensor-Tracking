# 📄 Tài Liệu Mô Tả Chi Tiết Bộ Dữ Liệu Điện Não Đồ Thô (Raw BIDS EEG Dataset)
## Dự án Phân tích Động lực học Mạng lưới Não bộ (Tensor Subspace Tracking Pipeline)

Tài liệu này mô tả chi tiết cấu trúc, thông số kỹ thuật, thiết kế thí nghiệm và mã hóa sự kiện của **bộ dữ liệu điện não đồ (EEG) thô** hiện có trong thư mục `data/raw/ERN Raw Data BIDS-Compatible/` của dự án.

---

## 1. THIẾT KẾ THÍ NGHIỆM & NHIỆM VỤ (TASK DESCRIPTION)

Dữ liệu thô được thu thập từ tiểu nhiệm vụ **Error-Related Negativity (ERN)** nằm trong dự án ERP CORE quy mô lớn. 

*   **Tác vụ thí nghiệm (Task):** Biến thể của **Eriksen Flanker Task** (Eriksen và Eriksen, 1974).
*   **Mục tiêu tác vụ:** Kích thích đối tượng thực hiện phản ứng nhanh và chính xác dưới áp lực thời gian, từ đó kích hoạt cơ chế nhận diện lỗi sai tự động của não bộ (sinh ra sóng âm ERN khi phạm lỗi, và sóng CRN khi trả lời đúng).
*   **Cơ chế kích thích:** 
    *   Đối tượng được quan sát một nhóm gồm **5 mũi tên/ký tự** trên màn hình.
    *   Nhiệm vụ của đối tượng là nhấn nút chỉ định hướng của **mũi tên nằm ở chính giữa** (trái hoặc phải).
    *   **Thử nghiệm Đồng hướng (Congruent):** Các mũi tên hai bên chỉ cùng hướng với mũi tên trung tâm (ví dụ: `<<<<<` hoặc `>>>>>`).
    *   **Thử nghiệm Ngược hướng (Incongruent):** Các mũi tên hai bên chỉ ngược hướng với mũi tên trung tâm (ví dụ: `<<><<` hoặc `>>><>>`). Điều này tạo ra sự xung đột nhận thức mạnh mẽ, làm tăng đáng kể tỉ lệ mắc lỗi sai (Incorrect responses).
*   **Hướng dẫn đối tượng (Instructions):** Nhấn nút `[LEFT_BUTTON]` khi mũi tên trung tâm chỉ sang trái, nhấn `[RIGHT_BUTTON]` khi chỉ sang phải. Phản hồi yêu cầu tốc độ tối đa nhưng vẫn phải đảm bảo độ chính xác.

---

## 2. CẤU TRÚC THƯ MỤC CHUẨN BIDS-COMPATIBLE

Dữ liệu thô được tổ chức nghiêm ngặt theo chuẩn quốc tế **BIDS (Brain Imaging Data Structure)**:

```
data/raw/ERN Raw Data BIDS-Compatible/
├── sub-001/
├── sub-002/
...
└── sub-040/
    └── eeg/
        ├── sub-040_task-ERN_eeg.set          # Tệp tiêu đề EEGLAB/MNE chứa metadata bản ghi
        ├── sub-040_task-ERN_eeg.fdt          # Tệp nhị phân chứa chuỗi thời gian điện thế thô
        ├── sub-040_task-ERN_eeg.json         # Siêu dữ liệu BIDS của thiết bị và bản ghi
        ├── sub-040_task-ERN_channels.tsv     # Danh sách chi tiết các kênh đo, đơn vị và loại kênh
        ├── sub-040_task-ERN_electrodes.tsv   # Tọa độ 3D không gian của các điện cực trên da đầu
        ├── sub-040_task-ERN_coordsystem.json # Định nghĩa hệ tọa độ không gian điện cực
        └── sub-040_task-ERN_events.tsv       # Nhật ký sự kiện thời gian thực (Triggers)
```

---

## 3. THÔNG SỐ KỸ THUẬT PHẦN CỨNG & BẢN GHI (TECHNICAL SPECS)

Thông tin được trích xuất trực tiếp từ cấu trúc tệp `.json` thực tế của đối tượng:

*   **Nhà sản xuất thiết bị (Manufacturer):** **BioSemi** (Hà Lan) - Hãng công nghệ EEG nghiên cứu hàng đầu thế giới.
*   **Dòng máy (Model):** **ActiveTwo** (Hệ thống điện cực hoạt động tích cực - Active electrode system).
*   **Tần số lấy mẫu gốc (Original Sampling Frequency):** **1024 Hz** (Độ phân giải thời gian cực cao, khoảng cách giữa các mẫu điện thế là $0.976\text{ ms}$).
*   **Điện cực tham chiếu (EEG Reference):** **CMS (Common Mode Sense)** kết hợp với điện cực **DRL (Driven Right Leg)** tạo thành một vòng phản hồi chủ động để giảm thiểu nhiễu chế độ chung và nhiễu điện từ mà không cần điện cực tham chiếu vật lý thụ động truyền thống.
*   **Tần số điện lưới (Power Line Frequency):** **60 Hz** (Thiết kế thu thập tại Bắc Mỹ).
*   **Bộ lọc phần mềm thô (Software Filters):** **n/a (Unfiltered)**. Dữ liệu thô hoàn toàn nguyên bản từ bộ ADC khuếch đại, không áp dụng bất kỳ bộ lọc tần số hay bộ lọc notch nào trước đó.
*   **Thời lượng ghi (Recording Duration):** **~697 giây (~11.6 phút)** liên tục cho mỗi đối tượng.

---

## 4. PHÂN LOẠI KÊNH ĐO (EEG/EOG CHANNEL TYPES)

Bản ghi thô chứa tổng cộng **33 kênh đo** vật lý, đều sử dụng đơn vị điện thế là **microVolt ($\mu\text{V}$)**:

### A. 30 Kênh Điện Não Đồ (EEG Kênh):
Được bố trí theo sơ đồ chuẩn quốc tế **10-20 System** để phủ đều các bán cầu não:
*   **Vùng trán (Frontal):** `Fp1`, `Fp2`, `Fz`, `F3`, `F4`, `F7`, `F8`
*   **Vùng trước trung tâm (Frontocentral):** `FCz`, `FC3`, `FC4`
*   **Vùng trung tâm (Central):** `Cz`, `C3`, `C4`, `C5`, `C6`
*   **Vùng đỉnh (Parietal):** `Pz`, `P3`, `P4`, `P7`, `P8`, `P9`, `P10`
*   **Vùng trước chẩm (Parieto-occipital):** `CPz`, `POz`, `PO3`, `PO4`, `PO7`, `PO8`
*   **Vùng chẩm (Occipital):** `Oz`, `O1`, `O2`

### B. 3 Kênh Điện Nhãn Đồ (EOG Kênh):
Dùng để đo lường các chuyển động mắt phục vụ khâu khử nhiễu nháy mắt (Blink artifacts):
*   `HEOG_left`, `HEOG_right`: Đo chuyển động mắt theo phương ngang (Horizontal EOG).
*   `VEOG_lower`: Đo chuyển động mắt theo phương dọc (Vertical EOG), được đặt dưới mắt đối tượng.

---

## 5. CẤU TRÚC MÃ HÓA SỰ KIỆN THỜI GIAN THỰC (`events.tsv`)

Tệp `events.tsv` đóng vai trò là cột xương sống liên kết hành vi của đối tượng với sóng điện não. Mỗi hàng biểu diễn một sự kiện được ghi nhận thời gian thực với các cột:
*   `onset`: Thời điểm xảy ra sự kiện tính bằng giây (độ chính xác đến số thập phân thứ 4).
*   `duration`: Khoảng thời gian kích thích xuất hiện (thường là `0.2s` đối với stimulus và `0s` đối với response).
*   `sample`: Chỉ số mẫu tương ứng trong tín hiệu thô gốc (bằng `onset` * 1024).
*   `trial_type`: Phân loại sự kiện (`stimulus` hoặc `response`).
*   `value`: **Mã Trigger số** mang tính định danh cao:

### Phân tích ý nghĩa mã Trigger (`value`):

| Giá trị Trigger | Phân loại | Ý nghĩa sinh lý học hành vi |
| :--- | :--- | :--- |
| **11** | Stimulus | Congruent Stimulus, Mũi tên trung tâm chỉ sang TRÁI (`<<<<<`) |
| **22** | Stimulus | Congruent Stimulus, Mũi tên trung tâm chỉ sang PHẢI (`>>>>>`) |
| **12** | Stimulus | Incongruent Stimulus, Mũi tên trung tâm chỉ sang TRÁI (`>><>>`) |
| **21** | Stimulus | Incongruent Stimulus, Mũi tên trung tâm chỉ sang PHẢI (`<<><<`) |
| **111** | Response | Phản hồi TRÁI chính xác (Correct Left hand buttonpress) |
| **222** | Response | Phản hồi PHẢI chính xác (Correct Right hand buttonpress) |
| **121** | Response | Phản hồi TRÁI nhưng là phản ứng SAI (Incorrect Left hand) |
| **212** | Response | Phản hồi PHẢI nhưng là phản ứng SAI (Incorrect Right hand) |
| **112**, **122**, **211**, **221** | Response | Các phản hồi hỗn hợp khác (bao gồm lỗi bấm nhầm tay, phản hồi quá sớm hoặc các trường hợp đặc biệt) |

---

## 6. KHẢ NĂNG TIỀN XỬ LÝ TỪ DỮ LIỆU THÔ SANG TENSOR

Dữ liệu thô này chính là điểm khởi đầu cho toàn bộ pipeline xử lý chất lượng cao của dự án:
1.  Đọc chuỗi thời gian điện thế từ các tệp `.set`/`.fdt` thô.
2.  Áp dụng bộ lọc FIR cắt dải tần số cao không mong muốn và hạ mẫu tín hiệu từ **1024 Hz** xuống **128 Hz** dựa trên mốc sự kiện phản hồi (`trial_type == 'response'`).
3.  Áp dụng giải thuật Spherical Spline CSD để chuyển đổi $\mu\text{V}$ thành mật độ dòng nguồn $\mu\text{V}/\text{m}^2$, cô lập nguồn phát ERN vùng Frontocentral.
4.  Lọc dải tần Theta ($4\text{--}8\text{ Hz}$) và phân mảnh tín hiệu thành các epoch từ $-1.0\text{ s}$ đến $1.0\text{ s}$ bao quanh thời điểm bấm nút phản hồi (mốc $0\text{ ms}$).

*Tài liệu kỹ thuật được xây dựng tự động dựa trên phân tích cấu trúc siêu dữ liệu thực tế của bộ dữ liệu ERP CORE ERN.*

# 📄 Phương Pháp Cân Bằng Thử Nghiệm Theo Thời Gian (Temporal Matching Subsampling)
## Dự án Phân tích Động lực học Mạng lưới Não bộ (Tensor Subspace Tracking Pipeline)

Tài liệu này giải thích chi tiết về mặt toán học, vật lý sinh học và cấu trúc lập trình của khâu **Cân bằng thử nghiệm theo thời gian (Temporal Matching)** nằm trong tệp `preprocessing/sampling.py`. Đây là một bước cực kỳ then chốt nhằm bảo đảm tính chính xác không sai lệch của các phép tính liên kết pha Tensor sau đó.

---

## 1. THÁCH THỨC TOÁN HỌC: LỖI CHỆCH KÍCH THƯỚC MẪU (SAMPLE-SIZE BIAS IN PLV)

Trong các thí nghiệm kích thích ERN (như Eriksen Flanker Task), đối tượng thực hiện đúng phần lớn các thử nghiệm (thường đạt tỉ lệ chính xác $80\text{--}90\%$). Điều này dẫn đến sự mất cân bằng số lượng thử nghiệm nghiêm trọng giữa hai điều kiện hành vi:
*   Số lượng thử nghiệm Đúng (Correct trials - CRN): $N_{\text{correct}} \approx 200 \text{ đến } 300$ thử nghiệm.
*   Số lượng thử nghiệm Sai (Incorrect trials - ERN): $N_{\text{incorrect}} \approx 20 \text{ đến } 50$ thử nghiệm.

### Tại sao không thể tính PLV trực tiếp trên số lượng mẫu mất cân bằng này?
Chỉ số Khóa pha (Phase Locking Value - PLV) giữa hai kênh đo $x$ và $y$ được tính toán qua $N$ thử nghiệm tại một điểm thời gian:

$$\text{PLV} = \left| \frac{1}{N} \sum_{k=1}^{N} e^{j(\theta_k^x - \theta_k^y)} \right|$$

Về mặt thống kê toán học, chỉ số PLV có một **độ chệch hệ thống phụ thuộc vào kích thước mẫu (sample-size bias)**. Khi số lượng mẫu $N$ càng nhỏ, nhiễu nền ngẫu nhiên sẽ không thể triệt tiêu hết, làm cho giá trị PLV ước lượng bị **khuếch đại giả tạo (artificially inflated)** theo tỷ lệ:

$$\text{PLV}_{\text{noise}} \propto \frac{1}{\sqrt{N}}$$

*   Nếu tính PLV cho điều kiện **Incorrect** ($N = 30$), giá trị PLV nền do nhiễu sẽ dao động quanh mức $1 / \sqrt{30} \approx 0.18$.
*   Nếu tính PLV cho điều kiện **Correct** ($N = 240$), giá trị PLV nền do nhiễu chỉ dao động quanh mức $1 / \sqrt{240} \approx 0.06$.

**Hậu quả:** Nếu so sánh trực tiếp, điều kiện Incorrect (Sai) sẽ luôn có vẻ như có độ đồng bộ pha cao hơn hẳn so với điều kiện Correct (Đúng), nhưng thực chất đó chỉ là **sai số toán học do cỡ mẫu Incorrect quá ít**. 

Để loại bỏ hoàn toàn độ chệch này, bắt buộc phải cân bằng số lượng thử nghiệm chính xác **1:1** ($N_{\text{correct}} = N_{\text{incorrect}}$) cho từng đối tượng trước khi tính toán Tensor PLV.

---

## 2. BẢN CHẤT SINH HỌC: TẠI SAO LÀ TEMPORAL MATCHING?

Nếu chỉ cần cân bằng tỷ lệ 1:1, giải pháp đơn giản nhất là chọn ngẫu nhiên (Random Subsampling) $N$ thử nghiệm Đúng để so khớp với $N$ thử nghiệm Sai. Tuy nhiên, cách tiếp cận này vi phạm nghiêm trọng bản chất sinh học của não bộ:

1.  **Sự mệt mỏi sinh học (Biological Fatigue):** Trong suốt 12 phút thực hiện tác vụ căng thẳng, trạng thái nền của não bộ đối tượng thay đổi liên tục. Ở những phút cuối, sự mệt mỏi sinh học tăng cao, làm suy giảm năng lượng sóng Theta nền.
2.  **Hiệu ứng học tập (Learning & Adaptation):** Đối tượng sẽ dần thích nghi với quy luật Flanker, làm thay đổi cấu trúc kết nối chức năng của thùy trán.
3.  **Trôi dạt trở kháng vật lý (Impedance Drift):** Lớp gel dẫn điện trên da đầu bị khô dần theo thời gian, làm tăng trở kháng tiếp xúc của cảm biến.

```
Đầu buổi thí nghiệm (Gel tốt, tỉnh táo)  ────────▶ Cuối buổi thí nghiệm (Gel khô, mệt mỏi)
[Correct Trial 1, 2, 3...]                         [Incorrect Trial 45 (Phạm lỗi do mệt mỏi)]
       │                                                      ▲
       └─────────── Chênh lệch trạng thái sinh học nền ───────┘
```

Nếu chọn ngẫu nhiên, ta có thể ghép cặp một thử nghiệm Sai ở cuối buổi thí nghiệm với một thử nghiệm Đúng ở đầu buổi thí nghiệm. Sự so sánh lúc này không còn thuần khiết là ERN vs CRN nữa, mà đã bị lẫn lộn (confounded) bởi sự mệt mỏi sinh học và sự thay đổi trở kháng.

**Giải pháp Temporal Matching:** Đối với mỗi thử nghiệm Sai, thuật toán sẽ tìm kiếm thử nghiệm Đúng **xảy ra gần nó nhất về mặt thời gian thực** để ghép cặp. Điều này giữ cho trạng thái sinh học nền và trở kháng điện cực giữa hai thử nghiệm được **cố định và giống nhau tối đa**, triệt tiêu triệt để các biến nhiễu ngoại lai.

---

## 3. GIẢI THUẬT TOÁN HỌC TRONG `preprocessing/sampling.py`

Giải thuật ghép cặp thời gian lân cận được lập trình như sau:

```
[Bắt đầu] ──▶ Xác định danh sách thời gian của Correct và Incorrect
               │
               ▼
   Lặp qua từng thử nghiệm Incorrect (thời điểm t_inc)
               │
               ▼
   Tính khoảng cách thời gian d_j = |t_correct_j - t_inc| cho tất cả Correct chưa dùng
               │
               ▼
   Tìm thử nghiệm Correct gần nhất: j* = argmin(d_j)
               │
               ▼
   Đánh dấu Correct[j*] đã dùng (loại khỏi pool để đảm bảo độc bản 1:1)
               │
               ▼
[Kết thúc] ──▶ Gộp chung Incorrect + Correct đã ghép cặp, sắp xếp theo trình tự thời gian
```

---

## 4. PHÂN TÍCH CHI TIẾT MÃ NGUỒN PYTHON

Hàm xử lý cốt lõi `match_temporal_trials` hoạt động chi tiết như sau:

```python
def match_temporal_trials(epochs: mne.Epochs) -> Optional[mne.Epochs]:
  events = epochs.events
  event_id = epochs.event_id
  
  # 1. Trích xuất chỉ số hàng của các thử nghiệm Correct và Incorrect
  correct_idx = np.where(np.isin(events[:, 2], [event_id[k] for k in event_id if k.startswith('Correct')]))[0]
  incorrect_idx = np.where(np.isin(events[:, 2], [event_id[k] for k in event_id if k.startswith('Incorrect')]))[0]
  
  # Nếu đối tượng không mắc bất kỳ lỗi sai nào (không có ERN), bỏ qua đối tượng này
  if len(incorrect_idx) == 0:
    return None

  matched_correct_idx = []
  
  # 2. Duyệt qua từng thử nghiệm phạm lỗi (Incorrect)
  for inc_idx in incorrect_idx:
    inc_time = events[inc_idx, 0] # Thời điểm (tính bằng mili-giây mẫu) xảy ra lỗi
    
    # 3. Tính toán khoảng cách tuyệt đối đến toàn bộ các thử nghiệm Đúng còn lại
    distances = np.abs(events[correct_idx, 0] - inc_time)
    
    # 4. Tìm vị trí thử nghiệm Đúng có khoảng cách nhỏ nhất
    nearest_cor_idx = correct_idx[np.argmin(distances)]
    matched_correct_idx.append(nearest_cor_idx)
    
    # 5. Loại bỏ thử nghiệm Đúng đã chọn khỏi danh sách lựa chọn tiếp theo (Unique 1:1)
    correct_idx = correct_idx[correct_idx != nearest_cor_idx]
    if len(correct_idx) == 0:
      break
      
  # 6. Gộp danh sách Incorrect và các Correct được ghép cặp, sắp xếp theo thứ tự thời gian gốc
  final_indices = np.concatenate([incorrect_idx[:len(matched_correct_idx)], matched_correct_idx])
  final_indices.sort()
  
  # Trả về đối tượng Epochs đã được lọc sạch và cân bằng
  return epochs[final_indices]
```

### Quá trình thực thi Pipeline (`run_sampling_pipeline`):
1.  Đọc tệp epoch thô `sub-XXX_master-epo.fif` chứa đầy đủ thử nghiệm từ Bước 1.
2.  Thực hiện trích xuất và áp dụng hàm cân bằng `match_temporal_trials`.
3.  Lưu tệp Epoch tinh chế đạt chuẩn 1:1 thành `sub-XXX_theta_balanced-epo.fif` để phục vụ khâu tính toán Tensor PLV tiếp theo.

---

## 5. KIỂM ĐỊNH CHẤT LƯỢNG (QA/QC VERIFICATION)

Hiệu quả hoạt động của bước này được ghi nhận trực tiếp trong biểu đồ chẩn đoán kỹ thuật:
👉 **[outputs/eda/preprocessing_diagnostics/diag_04_temporal_matching.png](file:///home/chinh303/code/tensortracking/TrackingTensor3D/outputs/eda/preprocessing_diagnostics/diag_04_temporal_matching.png)**

Biểu đồ chẩn đoán này chỉ ra rõ ràng:
*   Trước khi lọc, các thử nghiệm Đúng phân bổ dày đặc xuyên suốt bản ghi, trong khi thử nghiệm Sai xuất hiện thưa thớt.
*   Sau khi chạy thuật toán, các thử nghiệm Đúng được lựa chọn có vị trí đứng **cận kề hoặc xen kẽ sát nút** với các thử nghiệm Sai tương ứng, đảm bảo trạng thái sinh lý thần kinh của hai trạng thái so sánh là tương đương tuyệt đối!

*Tài liệu phân tích thuật toán tiền xử lý cao cấp dự án EEG Tensor Tracking.*

# 📄 Xây Dựng Kết Nối Chức Năng Sóng Theta (Connectivity Construction Pipeline)
## Dự án Phân tích Động lực học Mạng lưới Não bộ (Tensor Subspace Tracking Pipeline)

Tài liệu này giải thích chi tiết về mặt toán học, vật lý sinh học và cấu trúc lập trình của khâu **Xây dựng kết nối chức năng dải Theta (Connectivity Construction)** nằm trong tệp `preprocessing/connectivity.py`. Đây là khâu chịu trách nhiệm chuyển đổi các dữ liệu chuỗi thời gian điện não (Epochs) thành ma trận kết nối pha 4D khổng lồ (Tensor 4D) để cung cấp cho giải thuật Subspace Tracking thời gian thực.

---

## 📄 CẤU TRÚC 5 PHẦN CỦA QUY TRÌNH XÂY DỰNG KẾT NỐI

```
                  [Tệp Epochs Đã Cân Bằng (Balanced Epochs)]
                                      │
                                      ▼
             [PHẦN 1: Biến đổi Thời gian - Tần số RID-Rihaczek]
                                      │
                                      ▼
             [PHẦN 2: Trích xuất Góc Pha Phức dải Theta (PyTorch)]
                                      │
                                      ▼
         [PHẦN 3: Tính toán Ma Trận Khóa Pha TFD-PLV (Cross-Channel)]
                                      │
                                      ▼
             [PHẦN 4: Ghép nối & Đồng bộ hóa Tensor Nhóm 4D]
                                      │
                                      ▼
            [PHẦN 5: Lưu trữ Cấu trúc Dữ liệu MATLAB (.mat)]
```

---

## PHẦN 1: BẢN CHẤT TOÁN HỌC - RID-RIHACZEK TFD & TFD-PLV

Để đo đạc sự đồng bộ pha thời gian thực giữa hai vùng não một cách chính xác trong môi trường không dừng (non-stationary) của điện não, quy trình sử dụng bộ lọc kết hợp giữa **Phân bố Thời gian - Tần số Giảm nhiễu (Reduced Interference Distribution - RID)** của Rihaczek và **Chỉ số khóa pha (Phase Locking Value - PLV)**.

### 1.1 Phân bố Rihaczek TFD
Phân bố thời gian - tần số Rihaczek nguyên bản của một tín hiệu $x(t)$ được định nghĩa là:

$$\text{RD}_x(t, f) = x(t) X^*(f) e^{-j 2 \pi f t}$$

Trong đó $X(f)$ là biến đổi Fourier của $x(t)$, và $*$ ký hiệu số phức liên hợp. Nhược điểm lớn của lớp biểu diễn song tuyến tính Wigner-Ville hay Rihaczek nguyên bản là xuất hiện **Nhiễu xuyên biên (Cross-terms)** khi tín hiệu có nhiều thành phần tần số.

Để triệt tiêu nhiễu xuyên biên mà không làm nhòe độ phân giải thời gian-tần số, thuật toán áp dụng **Hàm nhân giảm nhiễu (Reduced Interference Kernel)** trong miền Ambiguity:

$$\text{TFD}_x(t, f) = \iint \phi(\eta, \tau) A_x(\eta, \tau) e^{j2\pi(\eta t - \tau f)} d\eta d\tau$$

Hàm nhân $\phi(\eta, \tau)$ đóng vai trò như một bộ lọc thông thấp hai chiều trong miền Ambiguity, chỉ cho phép các thành phần tự phát (Auto-terms) đi qua và lọc bỏ hoàn toàn các nhiễu xuyên biên tần số cao (Cross-terms).

### 1.2 Chỉ số khóa pha TFD-PLV
Sau khi tính được phổ thời gian - tần số phức $\text{TFD}_x(t, f)$ và $\text{TFD}_y(t, f)$ cho hai kênh $x$ và $y$, ta trích xuất góc pha phức chuẩn hóa (chỉ giữ lại pha, triệt tiêu biên độ):

$$\psi_x(k, t, f) = \frac{\text{TFD}_x(k, t, f)}{|\text{TFD}_x(k, t, f)|} = e^{j \phi_{x, k}(t, f)}$$

Trong đó $k$ là chỉ số thử nghiệm (trial index), $t$ là thời điểm, và $f$ là tần số. 
Chỉ số Khóa pha thời gian - tần số (TFD-PLV) giữa kênh $x$ và $y$ qua $N$ thử nghiệm được tính bằng:

$$\text{PLV}(t, f) = \left| \frac{1}{N} \sum_{k=1}^{N} \psi_x(k, t, f) \psi_y^*(k, t, f) \right|$$

Giá trị $\text{PLV}(t, f)$ nằm trong khoảng $[0, 1]$:
*   $\text{PLV} = 1$: Hai kênh có sự khóa pha đồng bộ hoàn hảo tại thời điểm $t$ và tần số $f$.
*   $\text{PLV} = 0$: Pha của hai kênh phân bổ ngẫu nhiên, không có sự liên kết chức năng.

---

## PHẦN 2: TỐI ƯU HÓA GPU (PYTORCH ACCELERATION)

Tính toán TFD-PLV cho **30 kênh**, trên **257 điểm thời gian**, qua hàng chục thử nghiệm, cho **40 đối tượng** đòi hỏi hàng tỷ phép tính số phức. Nếu chạy trên CPU truyền thống sẽ mất khoảng **4-5 tiếng** cho một đối tượng.

Quy trình trong `connectivity.py` tận dụng tối đa sức mạnh tính toán song song của **GPU NVIDIA (thông qua PyTorch)** để tăng tốc độ xử lý lên gấp **100 lần**:
1.  **Chuyển mảng lên GPU:** Toàn bộ dữ liệu Epochs của đối tượng được đưa trực tiếp vào VRAM của GPU:
    ```python
    data = torch.tensor(epochs[condition].get_data(), dtype=torch.float32, device=device)
    ```
2.  **Tính toán song song các phép nhân phức:** Phép nhân số phức liên hợp và tính trung bình thử nghiệm được vector hóa toàn bộ trên nhân CUDA của GPU, loại bỏ hoàn toàn các vòng lặp CPU chậm chạp:
    ```python
    sync = torch.abs(torch.mean(theta_phases[:, ch1, :, :] * torch.conj(theta_phases[:, ch2, :, :]), dim=0))
    ```

---

## PHẦN 3: GIẢI THÍCH CHI TIẾT TỪNG DÒNG MÃ NGUỒN

Quy trình xây dựng kết nối gồm các bước lập trình chính sau:

### 3.1 Khởi tạo bộ nhớ & Xác định dải tần Theta
```python
sub_files = sorted(list(config.paths.REFINED_DIR.glob("*_theta_balanced-epo.fif")))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
*   Quét toàn bộ các tệp Epochs đã cân bằng ở Bước 2.
*   Cấu hình thiết bị tính toán: Ưu tiên `cuda` (GPU), tự động rơi về `cpu` nếu máy không có card đồ họa chuyên dụng.

### 3.2 Thiết lập Bộ lọc Mặt nạ dải Theta (Theta Filter Mask)
```python
# Tính trục tần số thực tế dựa trên độ phân giải thời gian và tần số lấy mẫu (128Hz)
freq_axis = torch.fft.fftfreq(n_points, d=1/config.eeg.SFREQ, device=device)
# Thiết lập bộ lọc logic để cô lập dải tần 4 - 8 Hz
theta_mask = (freq_axis >= config.proc.THETA_BAND[0]) & (freq_axis <= config.proc.THETA_BAND[1])
theta_freq_indices = torch.where(theta_mask)[0]
```
*   Tạo ra một bộ lọc mặt nạ nhị phân trên GPU để cô lập chính xác các bin tần số thuộc dải sóng Theta ($4\text{--}8\text{ Hz}$) trong không gian Fourier.

### 3.3 Trích xuất góc pha phức dải Theta
```python
theta_phases = torch.zeros((n_trials, n_channels, n_points, len(theta_freq_indices)), 
                           dtype=torch.complex64, device=device)

for ch in range(n_channels):
    # Tính phổ thời gian - tần số RID-Rihaczek cho toàn bộ thử nghiệm của kênh ch
    tfr = tfd_engine.compute_tfd(data[:, ch, :]) # Hình dạng đầu ra: (trials, time, freq)
    
    # Chỉ giữ lại các tần số thuộc dải Theta
    theta_samples = tfr[:, :, theta_mask] # Hình dạng: (trials, time, n_theta)
    
    # Chuẩn hóa số phức để triệt tiêu biên độ, chỉ giữ lại góc pha e^(j*phi)
    theta_phases[:, ch, :, :] = theta_samples / (torch.abs(theta_samples) + 1e-12)
```
*   `tfd_engine.compute_tfd` tính toán song song ma trận phân bố TFR thời gian-tần số phức.
*   Phép toán chia cho trị tuyệt đối số phức `theta_samples / (torch.abs(theta_samples) + 1e-12)` là bước chuẩn hóa pha cốt lõi, loại bỏ hoàn toàn sự thống trị của biên độ sóng để đo đạc thuần túy sự liên kết đồng bộ pha.

### 3.4 Tính toán mạng lưới liên kết pha (Cross-Channel Coherence)
```python
for ch1 in range(n_channels):
    for ch2 in range(ch1 + 1, n_channels):
        # Nhân pha phức của kênh ch1 với số phức liên hợp của kênh ch2 qua các thử nghiệm
        sync = torch.abs(torch.mean(theta_phases[:, ch1, :, :] * torch.conj(theta_phases[:, ch2, :, :]), dim=0))
        
        # Tính trung bình trên các bin tần số thuộc dải Theta
        plv_time_series = torch.mean(sync, dim=1)
        
        # Lưu đối xứng vào ma trận kết nối
        storage[ch1, ch2, :] = plv_time_series.cpu().numpy()
        storage[ch2, ch1, :] = plv_time_series.cpu().numpy()
```
*   Vòng lặp chạy qua toàn bộ $\frac{30 \times 29}{2} = 435$ cặp liên kết kênh không lặp lại.
*   Tích phức chéo `theta_phases[:, ch1, :, :] * torch.conj(theta_phases[:, ch2, :, :])` tính toán hiệu pha phức thời gian thực.
*   Hàm `torch.abs(torch.mean(..., dim=0))` tính độ dài vectơ trung bình để cho ra chỉ số PLV chạy trong khoảng $[0, 1]$.
*   Sau khi xử lý xong trên GPU, kết quả được chuyển ngược về RAM CPU bằng lệnh `.cpu().numpy()` để giải phóng VRAM.

---

## PHẦN 4: CẤU TRÚC TENSOR NHÓM 4D ĐẦU RA

Sau khi quét qua tất cả các đối tượng (ví dụ $40$ đối tượng), toàn bộ ma trận kết nối 3D của từng cá thể được xếp chồng (stack) lại để tạo ra **Tensor Nhóm 4 chiều khổng lồ**:

```python
tensor_correct = np.stack(all_corr, axis=0)   # Kích thước: [40, 30, 30, 257]
tensor_incorrect = np.stack(all_inc, axis=0) # Kích thước: [40, 30, 30, 257]
```

### Giải nghĩa các chiều của Tensor 4D `[Subjects x Channels x Channels x Times]`:

1.  **Chiều thứ nhất (`Subjects` - 40):** Danh sách các đối tượng nghiên cứu cá thể. Chiều này cho phép phân tích sự khác biệt nhóm hoặc theo dõi sự biến đổi không gian con của từng người.
2.  **Chiều thứ hai (`Channels` - 30):** Các cảm biến điện cực xuất phát.
3.  **Chiều thứ ba (`Channels` - 30):** Các cảm biến điện cực đích.
    *   *Lưu ý:* Mặt cắt `[30 x 30]` tại mỗi điểm thời gian là một **Ma trận liên kết chức năng (Functional Connectivity Matrix)** đối xứng qua đường chéo, mô tả cấu trúc mạng lưới não bộ toàn cục.
4.  **Chiều thứ tư (`Times` - 257):** Dòng thời gian chạy từ $-1.0\text{ s}$ đến $1.0\text{ s}$ bao quanh mốc phản hồi (với tần số lấy mẫu $128\text{ Hz}$). Đây là chiều động lực học giúp giải thuật Subspace Tracking theo dõi sự trượt chuyển của mạng lưới thời gian thực.

---

## PHẦN 5: LƯU TRỮ CẤU TRÚC DỮ LIỆU MATLAB (`.mat`)

Để đảm bảo tính liên thông và tương thích ngược hoàn hảo với các mã nguồn thuật toán tối ưu hóa viết bằng MATLAB, quy trình tự động xuất dữ liệu ra tệp tin chuẩn MATLAB:

```python
savemat(config.paths.MATLAB_DIR / "connectivity_balanced_4d.mat", {
  'tensor_correct': tensor_correct,
  'tensor_incorrect': tensor_incorrect,
  'subjects': processed_subs,
  'channels': sample_epochs.ch_names
})
```

Tệp tin này chứa đầy đủ thông tin giải phẫu kênh và định danh đối tượng, cho phép các phần mềm Matlab khác đọc trực tiếp thông qua hàm `load('connectivity_balanced_4d.mat')` để thực hiện phép tách rã tensor HOSVD hoặc HO-RLSL.

---
*Tài liệu phân tích cấu trúc toán học xây dựng Tensor PLV dải Theta - EEG Tensor Tracking.*

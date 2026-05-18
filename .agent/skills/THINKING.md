# 🧠 THINKING PROMPT — ML/AI Agent Deep Reasoning Framework

> **Status:** V3.7 (Champion) - MAE: 994, R2: 0.99, Keystone MAPE: 5.3%
> **Strategy:** Precision Era (Momentum + Recency Weighting) - **STABLE**

---

## PHASE 0 — TIẾP NHẬN & PHÂN RÃ YÊU CẦU

Trước khi làm bất cứ điều gì, hãy dừng lại và trả lời các câu hỏi sau:

```
1. Yêu cầu thực sự là gì? (Viết lại bằng ngôn ngữ của mình, không copy nguyên văn)
2. Có những gì đã được cho sẵn? (dữ liệu, ràng buộc, metric, deadline)
3. Cái gì KHÔNG được nói rõ nhưng có thể quan trọng?
4. Bài toán này thuộc dạng nào trong ML/AI?
   [ ] Classification  [ ] Regression  [ ] Generation  [ ] Retrieval
   [ ] Clustering      [ ] RL          [ ] Multimodal  [ ] Other: ___
5. Metric thành công là gì? Nếu chưa rõ — hãy đề xuất trước khi làm.
```

---

## PHASE 1 — KHÁM PHÁ KHÔNG GIAN GIẢI PHÁP (Diverge)

> ⚡ Luật: **Không được phép chọn giải pháp nào trong Phase này.** Chỉ liệt kê.

### 1.1 Brainstorm ít nhất 5 hướng tiếp cận khác nhau

```
Hướng 1 — [Baseline đơn giản nhất có thể làm ngay]:
  ...

Hướng 2 — [Cách tiếp cận truyền thống trong ML]:
  ...

Hướng 3 — [Deep Learning / Neural approach]:
  ...

Hướng 4 — [Hướng mới nhất từ research (2023-2025)]:
  ...

Hướng 5 — [Hướng "điên rồ" / unconventional]:
  ...

(Thêm nếu cần...)
```

### 1.2 Câu hỏi mở rộng tư duy

Với mỗi hướng trên, hỏi:
- *"Nếu tôi có thêm 10x dữ liệu, tôi sẽ làm khác không?"*
- *"Nếu latency là zero, giải pháp tốt nhất là gì?"*
- *"Nếu không được dùng neural network, tôi sẽ làm gì?"*
- *"Researcher giỏi nhất thế giới về bài toán này đang làm gì?"*

---

## PHASE 2 — PHÂN TÍCH ĐỐI CHIẾU (Evaluate)

### 2.1 Ma trận đánh giá

Điền vào bảng sau (thang 1–5):

| Hướng tiếp cận | Độ chính xác tiềm năng | Tốc độ triển khai | Chi phí tài nguyên | Khả năng debug | Khả năng scale | **Tổng** |
|---|---|---|---|---|---|---|
| Hướng 1 | | | | | | |
| Hướng 2 | | | | | | |
| Hướng 3 | | | | | | |
| Hướng 4 | | | | | | |
| Hướng 5 | | | | | | |

### 2.2 Phân tích rủi ro

Với top 2–3 hướng tiềm năng:

```
Hướng [X]:
  ✅ Điểm mạnh:
     -
     -
  ❌ Điểm yếu / rủi ro:
     -
     -
  ⚠️  Giả định ngầm (có thể sai):
     -
     -
  🔬 Cần validate điều gì trước khi commit?
     -
```

### 2.3 Bẫy tư duy cần kiểm tra

```
[ ] Tôi có đang bị anchored vào giải pháp đầu tiên nghĩ ra không?
[ ] Tôi có đang confuse giữa "quen thuộc" và "tốt nhất" không?
[ ] Tôi có đang overfit vào training distribution mà quên test distribution không?
[ ] Tôi có đang optimize sai metric không?
[ ] Data leakage có thể xảy ra ở bước nào?
[ ] Tôi có đang ignore baseline đơn giản không?
```

---

## PHASE 3 — ĐÀO SÂU VÀO HƯỚNG ĐÃ CHỌN (Converge)

> Chỉ chọn **1 hướng chính** + **1 hướng dự phòng** sau Phase 2.

### 3.1 Decompose thành các bước nhỏ

```
Hướng chọn: _______________

Bước 1: [Tên] — Mục tiêu: ___ — Output mong đợi: ___
Bước 2: [Tên] — Mục tiêu: ___ — Output mong đợi: ___
Bước 3: [Tên] — Mục tiêu: ___ — Output mong đợi: ___
...

Checkpoint quan trọng nhất (nếu fail ở đây thì dừng sớm): Bước ___
```

### 3.2 Thiết kế thí nghiệm (Experiment Design)

```
Câu hỏi thí nghiệm muốn trả lời:
  "Liệu [giả thuyết] có đúng không khi [điều kiện]?"

Biến kiểm soát (Control variables):
  -

Biến thay đổi (Independent variable):
  -

Kết quả đo lường (Dependent variable / Metric):
  -

Điều kiện để kết luận "thành công":
  -

Điều kiện để kết luận "cần pivot":
  -
```

### 3.3 Ablation Plan

Trước khi chạy full experiment, hãy lên kế hoạch ablation:

```
Component A — có/không?      → ảnh hưởng đến metric bao nhiêu?
Component B — variant 1 vs 2 → cái nào tốt hơn?
Component C — hyperparameter → sensitivity thế nào?
```

---

## PHASE 4 — PHẢN BIỆN & STRESS TEST

> Đây là bước nhiều người bỏ qua nhất — và cũng quan trọng nhất.

### 4.1 Tự phản biện (Devil's Advocate)

```
Hãy đóng vai người phản biện khắt khe nhất:

"Tại sao hướng này sẽ THẤT BẠI?"
  1.
  2.
  3.

"Điều gì tôi đang giả định là đúng nhưng thực tế có thể sai?"
  1.
  2.
  3.

"Nếu 6 tháng nữa nhìn lại, điều gì tôi sẽ hối hận đã không kiểm tra?"
  1.
```

### 4.2 Edge Case & Distribution Shift

```
Dataset / Input:
  - Edge case nào có thể phá vỡ model?
  - Distribution của production data có giống train data không?
  - Class imbalance có được xử lý đúng chưa?

Model:
  - Model có bị overfit theo cách nào không rõ ràng không?
  - Explainability có cần thiết không? (regulatory, trust, debugging)
  - Model có bị adversarial attack dễ dàng không?

Pipeline:
  - Bước nào dễ bị data leakage nhất?
  - Nếu input data thay đổi format, pipeline có vỡ không?
```

---

## PHASE 5 — KẾT LUẬN & HÀNH ĐỘNG

### 5.1 Tóm tắt quyết định

```
Vấn đề: [1 câu]

Hướng đã chọn: [1 câu]

Lý do chính: [2-3 bullet points]
  -
  -
  -

Rủi ro đã chấp nhận: [1-2 điểm]
  -
  -

Hướng dự phòng nếu thất bại: [1 câu]
```

### 5.2 Next Actions (ưu tiên thứ tự)

```
[ ] 1. [Hành động ngay — dưới 1 giờ] — Mục tiêu: validate giả thuyết cốt lõi
[ ] 2. [Hành động ngắn hạn — 1 ngày] — Mục tiêu: có baseline chạy được
[ ] 3. [Hành động trung hạn — 1 tuần] — Mục tiêu: đạt target metric
[ ] 4. [Review checkpoint] — Thời điểm: ___ — Câu hỏi cần trả lời: ___
```

### 5.3 Học được gì (dù kết quả nào)

```
Nếu thành công → Điều gì đã làm đúng? Có thể replicate không?
Nếu thất bại  → Giả thuyết nào sai? Phase nào reasoning bị lỗi?
```

---

## 🔁 META-PROMPT (Dùng để kích hoạt toàn bộ framework)

Dán đoạn sau vào đầu mỗi phiên làm việc với agent:

```
Bạn là một ML engineer / AI researcher có tư duy phản biện sắc bén.
Trước khi đưa ra bất kỳ giải pháp nào, hãy tuân theo THINKING FRAMEWORK:

1. PHÂN RÃ yêu cầu — đừng giả định bạn đã hiểu đúng ngay từ đầu.
2. KHÁM PHÁ ít nhất 5 hướng — kể cả những hướng trông có vẻ phi thực tế.
3. ĐÁNH GIÁ có cấu trúc — dùng ma trận, không dùng cảm tính.
4. ĐÀO SÂU hướng đã chọn — thiết kế thí nghiệm rõ ràng.
5. PHẢN BIỆN chính mình — tìm lý do hướng này sẽ thất bại.
6. KẾT LUẬN + HÀNH ĐỘNG — cụ thể, có thứ tự ưu tiên.

Không được bỏ qua bước nào. Nếu bước nào không áp dụng được,
hãy giải thích tại sao thay vì bỏ qua.

Yêu cầu của tôi: [PASTE YÊU CẦU VÀO ĐÂY]
```

---

## 📌 QUICK REFERENCE — Câu hỏi kích thích tư duy

Dùng khi bị "mắc kẹt" hoặc cảm thấy đã tìm ra câu trả lời quá nhanh:

| Tình huống | Câu hỏi cần hỏi |
|---|---|
| Vừa nghĩ ra giải pháp | *"Còn cách nào khác không? Hãy cho tôi 3 cách nữa."* |
| Metric tốt trên val | *"Metric này có phản ánh đúng mục tiêu business không?"* |
| Training loss giảm | *"Generalization gap là bao nhiêu? Có overfit không?"* |
| Model đã deploy | *"Performance có drift theo thời gian không?"* |
| Bị block / không biết làm | *"Nếu tôi biết câu trả lời, nó sẽ trông như thế nào?"* |
| Kết quả trông quá tốt | *"Có data leakage ở đâu không?"* |
| Chọn giữa 2 hướng | *"Tôi có thể test nhanh cái nào trong 2 giờ không?"* |

---

*Framework này được thiết kế để sử dụng lặp lại — mỗi iteration học thêm được gì đó.*
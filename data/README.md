# Data Directory

Thư mục này được sử dụng để chứa dữ liệu thô (raw) và dữ liệu đã qua tiền xử lý (processed). Do kích thước dữ liệu lớn, toàn bộ các file `.npy`, `.mat`, `.set`, `.fdt` bên trong thư mục này (ngoại trừ file `.gitkeep` và `README.md`) sẽ bị bỏ qua (ignored) bởi Git.

## Cấu trúc thư mục yêu cầu

Để chạy được mã nguồn, bạn cần cấu trúc thư mục `data/` như sau:

```text
data/
├── raw/
│   └── ERN Raw Data BIDS-Compatible/
│       ├── sub-001/
│       ├── sub-002/
│       └── ...
├── processed_v2/
│   ├── 01_epochs/                 # Dữ liệu sau khi CSD và cắt Epochs
│   ├── 02_sampling/               # Dữ liệu sau khi lọc Theta và cân bằng trial
│   └── 03_connectivity_tensors/   # Output cuối cùng: Tensors 4D (.npy và .mat)
```

## Hướng dẫn
1. Tải bộ dữ liệu **ERP CORE** (ERN BIDS-Compatible) và đặt vào `data/raw/ERN Raw Data BIDS-Compatible/`.
2. Chạy lần lượt các script trong thư mục `preprocessing/` (như `pipeline.py`, `sampling.py`, `connectivity.py`) để tạo ra Tensor 4D lưu tại `data/processed_v2/03_connectivity_tensors/`.

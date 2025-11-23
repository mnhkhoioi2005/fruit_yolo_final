# Fruit Detection with YOLO

Hệ thống nhận diện trái cây sử dụng YOLO với khả năng xử lý video và camera real-time.

## Tính năng

-  **Upload Video**: Upload video để nhận diện và đếm trái cây
-  **Camera Real-time**: Nhận diện trái cây trực tiếp từ webcam
-  **Khung nhận diện**: Vùng focus để nhận diện chính xác hơn
-  **Xuất video**: Tải video đã được xử lý với bounding boxes

## Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8+
- Webcam (cho tính năng camera)

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## Cấu trúc project

```
Fruit/
├── main.py                     # Chương trình chính 
├── fruit_camera_with_frame.py  # Chạy riêng nhận diện bằng camera
├── train_yolo_complete.py      # Script training YOLO model
├── requirements.txt            # Dependencies
├── README.md                   # Tài liệu này
├── runs/
│   └── detect/
│       └── fruit_yolo_final/
│           └── weights/
│               └── best.pt     # YOLO model đã train
├── uploads/                    # Video upload
├── outputs/                    # Video đã xử lý
├── venv/                       # Virtual environment
└── test/                       # Hình ảnh và video để test
```

## Sử dụng

### Chạy ứng dụng
```bash
python main.py   #chương trình chính
```

```bash
python fruit_camera_with_frame.py   #chạy riêng nhận diện bằng camera
```


Truy cập: **http://localhost:5001**

### Giao diện web

#### Tab 1: Upload Video
1. Click vào khung upload để chọn video
2. Nhấn "Xử Lý Video"
3. Đợi xử lý hoàn tất
4. Xem kết quả video đã xử lý ở phần outputs
#### Tab 2: Camera Real-time
1. Nhấn "Bắt Đầu Camera"
2. Đặt trái cây vào khung nhận diện
3. Hệ thống sẽ tự động nhận diện
4. Nhấn "Tắt Camera" để dừng


# fruit_yolo_final

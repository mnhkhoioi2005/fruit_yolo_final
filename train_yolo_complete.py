import os
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil  

try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

class YOLOFruitTrainer:
    def __init__(self, data_yaml_path, model_size='s'):
        self.data_yaml = Path(data_yaml_path)
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {data_yaml_path}")
        
        self.model_size = model_size.lower()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        print(f"Device: {self.device.upper()}")

    def load_model(self):
        model_name = f"yolov8{self.model_size}.pt"
        print(f"\n{'='*60}")
        print(f"ĐANG TẢI MÔ HÌNH: {model_name}")
        print(f"{'='*60}")
        self.model = YOLO(model_name)
        print(f"Đã tải xong!")

    def train(self, epochs=100, imgsz=640, batch=16, patience=50):
        print(f"\n{'='*60}")
        print("BẮT ĐẦU TRAINING")
        print(f"{'='*60}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU VRAM: {vram_gb:.1f}GB")
            if vram_gb <= 6:
                batch = 8
            elif vram_gb < 10:
                batch = 16
            else:
                batch = 32
        
        print(f"Epochs: {epochs} | Image size: {imgsz} | Batch: {batch}")
        
        self.model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=self.device,
            project='runs/detect',
            name='fruit_yolo_final',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            seed=42,
            deterministic=True,
            amp=True,
            workers=4,
            cache=False,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            plots=True,
            save=True,
        )
        
        print(f"\nTRAINING HOÀN TẤT!")
        print(f"Model lưu tại: runs/detect/fruit_yolo_final/weights/best.pt")

    def validate(self):
        print(f"\n{'='*60}")
        print("VALIDATION SET - ĐÁNH GIÁ")
        print(f"{'='*60}")
        
        metrics = self.model.val(
            data=str(self.data_yaml),
            split='val',
            imgsz=640,
            batch=16,
            conf=0.25,
            iou=0.6,
            device=self.device
        )
        
        print(f"\nKết quả Validation:")
        print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"  Precision    : {metrics.box.mp:.4f}")
        print(f"  Recall       : {metrics.box.mr:.4f}")
        return metrics

    def test(self):
        print(f"\n{'='*60}")
        print("TEST SET - ĐÁNH GIÁ CUỐI CÙNG")
        print(f"{'='*60}")
        
        with open(self.data_yaml) as f:
            data = yaml.safe_load(f)
        
        if 'test' not in data or not data['test']:
            print("Không có test set, bỏ qua...")
            return None
        
        metrics = self.model.val(
            data=str(self.data_yaml),
            split='test',
            imgsz=640,
            batch=16,
            conf=0.25,
            iou=0.6,
            device=self.device
        )
        
        print(f"\nKết quả Test:")
        print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"  Precision    : {metrics.box.mp:.4f}")
        print(f"  Recall       : {metrics.box.mr:.4f}")
        return metrics

    def predict_samples(self, num_samples=10):
        print(f"\n{'='*60}")
        print(f"PREDICT TRÊN {num_samples} ẢNH MẪU TỪ TEST SET")
        print(f"{'='*60}")
        
        with open(self.data_yaml) as f:
            data = yaml.safe_load(f)
        
        test_path = data.get('test', data.get('val', ''))
        if Path(test_path).is_absolute():
            test_dir = Path(test_path).parent / 'images'
        else:
            test_dir = self.data_yaml.parent / Path(test_path).parent / 'images'
        
        import glob
        test_imgs = glob.glob(str(test_dir / '*.jpg'))[:num_samples]
        if not test_imgs:
            test_imgs = glob.glob(str(test_dir / '*.png'))[:num_samples]
        
        if not test_imgs:
            print("Không tìm thấy ảnh test")
            return
        
        print(f"Đang predict trên {len(test_imgs)} ảnh...")
        self.model.predict(
            source=test_imgs,
            conf=0.25,
            iou=0.45,
            save=True,
            project='runs/detect',
            name='test_predictions',
            exist_ok=True,
            imgsz=640
        )
        print(f"Kết quả lưu tại: runs/detect/test_predictions/")

    def export_model(self):
        print(f"\n{'='*60}")
        print("EXPORT MODEL")
        print(f"{'='*60}")
        
        try:
            onnx_path = self.model.export(format='onnx', imgsz=640, simplify=True)
            print(f"ONNX: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
        
        try:
            tflite_path = self.model.export(format='tflite', imgsz=640)
            print(f"TFLite: {tflite_path}")
        except Exception as e:
            print(f"TFLite export failed: {e}")

    def save_to_drive(self, drive_path='/content/drive/MyDrive/yolo_results/'):
        if not IS_COLAB:
            print("Không phải môi trường Colab, bỏ qua lưu vào Drive.")
            return
        
        source_dir = 'runs'  
        if not os.path.exists(source_dir):
            print("Không tìm thấy thư mục runs để lưu.")
            return
        
        try:
           
            shutil.copytree(source_dir, drive_path, dirs_exist_ok=True) 
            print(f"Đã lưu kết quả vào Drive: {drive_path}")
        except Exception as e:
            print(f"Lỗi lưu vào Drive: {e}")

def main():
   
   
    DATA_YAML = "/content/drive/MyDrive/datasetyolo/data.yaml"  
    MODEL_SIZE = 's'  
    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    
    print(f"\n{'='*60}")
    print("YOLO FRUIT DETECTION - TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"{'='*60}\n")
   
    trainer = YOLOFruitTrainer(DATA_YAML, MODEL_SIZE)
    trainer.load_model()
    trainer.train(epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, patience=50)
    trainer.save_to_drive()  
    val_metrics = trainer.validate()
    test_metrics = trainer.test()
    trainer.predict_samples(num_samples=10)
    trainer.export_model()
    
    print(f"\n{'='*60}")
    print("HOÀN TẤT!")
    print(f"{'='*60}")
    print(f"Model: runs/detect/fruit_yolo_final/weights/best.pt")
    print(f"Validation mAP@0.5: {val_metrics.box.map50:.4f}")
    if test_metrics:
        print(f"Test mAP@0.5: {test_metrics.box.map50:.4f}")
    print(f"Predictions: runs/detect/test_predictions/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('runs/detect/fruit_yolo_final/weights/best.pt')

def draw_detection_frame(frame, x, y, w, h):
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    corner_length = 30
    corner_thickness = 5
    
    cv2.line(frame, (x, y), (x + corner_length, y), (0, 255, 255), corner_thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), (0, 255, 255), corner_thickness)
    
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), (0, 255, 255), corner_thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), (0, 255, 255), corner_thickness)
    
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), (0, 255, 255), corner_thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), (0, 255, 255), corner_thickness)
    
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), (0, 255, 255), corner_thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), (0, 255, 255), corner_thickness)
    
    cv2.putText(frame, "Dat trai cay vao khung", (x, y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def is_in_detection_frame(box, frame_x, frame_y, frame_w, frame_h):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return (frame_x <= center_x <= frame_x + frame_w and 
            frame_y <= center_y <= frame_y + frame_h)

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_width = 300
    frame_height = 300
    frame_x = (640 - frame_width) // 2
    frame_y = (480 - frame_height) // 2
    
    print("YOLO Fruit Detection với khung nhận diện")
    print("Đưa trái cây vào khung xanh để nhận diện tốt hơn")
    print("Controls: 'q'=thoát, 's'=chụp ảnh, '+'=tăng khung, '-'=giảm khung")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        draw_detection_frame(frame, frame_x, frame_y, frame_width, frame_height)
        
        results = model(frame, conf=0.5, verbose=False)
        
        in_frame_detections = []
        out_frame_detections = []
        
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                if is_in_detection_frame([x1, y1, x2, y2], frame_x, frame_y, frame_width, frame_height):
                    in_frame_detections.append((class_name, confidence))
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    out_frame_detections.append((class_name, confidence))
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                               (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(frame, f"Trong khung: {len(in_frame_detections)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Ngoai khung: {len(out_frame_detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if in_frame_detections:
            best_detection = max(in_frame_detections, key=lambda x: x[1])
            cv2.putText(frame, f"PHAT HIEN: {best_detection[0]}", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        cv2.imshow("Fruit Detection với Khung", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"fruit_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Đã lưu: {filename}")
        elif key == ord('+') or key == ord('='):
            frame_width = min(frame_width + 20, 500)
            frame_height = min(frame_height + 20, 400)
            frame_x = (640 - frame_width) // 2
            frame_y = (480 - frame_height) // 2
        elif key == ord('-'):
            frame_width = max(frame_width - 20, 100)
            frame_height = max(frame_height - 20, 100)
            frame_x = (640 - frame_width) // 2
            frame_y = (480 - frame_height) // 2
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Chương trình đã kết thúc!")

if __name__ == "__main__":
    main()

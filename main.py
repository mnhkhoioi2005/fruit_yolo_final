from flask import Flask, request, jsonify, render_template_string, send_file, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
from collections import Counter
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

model = YOLO('runs/detect/fruit_yolo_final/weights/best.pt')

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_width = 300
    frame_height = 300
    frame_x = (640 - frame_width) // 2
    frame_y = (480 - frame_height) // 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        cv2.rectangle(frame, (frame_x, frame_y), (frame_x + frame_width, frame_y + frame_height), (0, 255, 0), 3)
        
        corner_length = 30
        corner_thickness = 5
        
        cv2.line(frame, (frame_x, frame_y), (frame_x + corner_length, frame_y), (0, 255, 255), corner_thickness)
        cv2.line(frame, (frame_x, frame_y), (frame_x, frame_y + corner_length), (0, 255, 255), corner_thickness)
        
        cv2.line(frame, (frame_x + frame_width, frame_y), (frame_x + frame_width - corner_length, frame_y), (0, 255, 255), corner_thickness)
        cv2.line(frame, (frame_x + frame_width, frame_y), (frame_x + frame_width, frame_y + corner_length), (0, 255, 255), corner_thickness)
        
        cv2.line(frame, (frame_x, frame_y + frame_height), (frame_x + corner_length, frame_y + frame_height), (0, 255, 255), corner_thickness)
        cv2.line(frame, (frame_x, frame_y + frame_height), (frame_x, frame_y + frame_height - corner_length), (0, 255, 255), corner_thickness)
        
        cv2.line(frame, (frame_x + frame_width, frame_y + frame_height), (frame_x + frame_width - corner_length, frame_y + frame_height), (0, 255, 255), corner_thickness)
        cv2.line(frame, (frame_x + frame_width, frame_y + frame_height), (frame_x + frame_width, frame_y + frame_height - corner_length), (0, 255, 255), corner_thickness)
        
        cv2.putText(frame, "Dat trai cay vao khung", (frame_x, frame_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        results = model(frame, conf=0.5, verbose=False)
        
        in_frame_detections = []
        out_frame_detections = []
        
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if (frame_x <= center_x <= frame_x + frame_width and 
                    frame_y <= center_y <= frame_y + frame_height):
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
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

HTML = '''
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎥 Nhận Diện Trái Cây</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { text-align: center; color: #333; margin-bottom: 20px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 40px; }
        .tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .tab-button {
            background: #f0f0f0;
            border: none;
            padding: 15px 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            border-radius: 25px 25px 0 0;
        }
        .tab-button.active {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover { border-color: #667eea; background: #f9f9f9; }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .btn:hover { transform: scale(1.05); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .progress-container {
            display: none;
            margin: 20px 0;
            background: #f0f0f0;
            border-radius: 10px;
            padding: 20px;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .result {
            display: none;
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        .download-btn {
            display: inline-block;
            margin-top: 20px;
            padding: 12px 30px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
        }
        .download-btn:hover { background: #218838; }
        video { width: 100%; border-radius: 10px; margin-top: 15px; }
        #realtimeVideo {
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Nhận Diện Trái Cây</h1>
        <p class="subtitle">Upload video hoặc sử dụng camera thời gian thực để nhận diện trái cây</p>
        
        <div class="tabs">
            <button class="tab-button active" onclick="openTab('upload')">Upload Video</button>
            <button class="tab-button" onclick="openTab('realtime')">Camera Thời Gian Thực</button>
        </div>
        
        <div id="upload" class="tab-content active">
            <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                <div style="font-size: 60px;">📹</div>
                <p style="margin-top: 10px; font-size: 18px;">Click để chọn video</p>
                <p style="color: #999; margin-top: 5px;">Hỗ trợ: MP4, AVI, MOV (tối đa 500MB)</p>
            </div>
            
            <input type="file" id="videoInput" accept="video/*">
            <div style="text-align: center;">
                <button class="btn" id="processBtn" disabled>Xử Lý Video</button>
            </div>
            
            <div class="progress-container" id="progressContainer">
                <p id="statusText">Đang xử lý video...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill">0%</div>
                </div>
                <p style="color: #666; margin-top: 10px;">Thời gian ước tính: <span id="timeEstimate">Đang tính...</span></p>
            </div>
            
            <div class="result" id="result">
                <h2 style="text-align:center; color:green; font-weight:bold;">
                    ✔ Video đã xử lý! Hãy xem file tại thư mục OUTPUT.
                </h2>
            </div>
        </div>
        
        <div id="realtime" class="tab-content">
            <div style="text-align: center; margin-bottom: 20px;">
                <button class="btn" id="toggleRealtime">Bắt Đầu Camera</button>
            </div>
            <img id="realtimeVideo" src="" alt="Realtime Video Feed" style="display: none;">
        </div>
    </div>

    <script>
        function openTab(tabName) {
            var tabs = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            document.getElementById(tabName).classList.add('active');
            
            var buttons = document.getElementsByClassName('tab-button');
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove('active');
            }
            event.currentTarget.classList.add('active');
        }

        // Upload Video Script
        const videoInput = document.getElementById('videoInput');
        const processBtn = document.getElementById('processBtn');
        const progressContainer = document.getElementById('progressContainer');
        const result = document.getElementById('result');
        let selectedFile = null;

        videoInput.addEventListener('change', (e) => {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                processBtn.disabled = false;
                processBtn.textContent = 'Xử Lý: ' + selectedFile.name;
            }
        });

        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            processBtn.disabled = true;
            progressContainer.style.display = 'block';
            result.style.display = 'none';
            
            const formData = new FormData();
            formData.append('video', selectedFile);
            
            try {
                const response = await fetch('/process_video', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Lỗi: ' + data.error);
                    return;
                }
                
                
                
                progressContainer.style.display = 'none';
                result.style.display = 'block';
                
            } catch (error) {
                alert('Lỗi kết nối: ' + error);
            } finally {
                processBtn.disabled = false;
                processBtn.textContent = 'Xử Lý Video';
            }
        });

        // Realtime Camera Script
        const toggleRealtime = document.getElementById('toggleRealtime');
        const realtimeVideo = document.getElementById('realtimeVideo');
        let isRunning = false;

        toggleRealtime.addEventListener('click', () => {
            if (!isRunning) {
                realtimeVideo.src = '/video_feed';
                realtimeVideo.style.display = 'block';
                toggleRealtime.textContent = 'Tắt Camera';
                isRunning = true;
            } else {
                realtimeVideo.src = '';
                realtimeVideo.style.display = 'none';
                toggleRealtime.textContent = 'Bắt Đầu Camera';
                isRunning = false;
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Không có video'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Chưa chọn video'}), 400
    
    filename = secure_filename(video_file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(input_path)
    
    output_filename = f"detected_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.3, verbose=False)
        annotated_frame = results[0].plot()
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            all_detections.append(class_name)
        
        out.write(annotated_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    fruit_counts = dict(Counter(all_detections))
    
    return jsonify({
        'success': True,
        'output_filename': output_filename,
        'total_frames': total_frames,
        'total_detections': len(all_detections),
        'unique_fruits': len(fruit_counts),
        'fruit_counts': fruit_counts
    })

@app.route('/download/<filename>')
def download(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=False
    )

if __name__ == '__main__':
    print("Server chạy tại: http://localhost:5001")
    print("Upload video hoặc sử dụng camera để nhận diện trái cây!")
    app.run(host='0.0.0.0', port=5001, debug=True)
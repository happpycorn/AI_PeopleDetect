from flask import Flask, Response, jsonify
from cv2 import imencode
import threading
import time
from camera import Camera
from people_detect import PeopleCounter

app = Flask(__name__)

latest_frame = None
latest_count = 0
lock = threading.Lock()  # 避免同時讀寫衝突

AWAIT_TIME = 1

cam = Camera()
counter = PeopleCounter("yolov8n.pt")

def update_loop():
    global latest_frame, latest_count
    while True:
        frame = cam.capture_frame()
        if frame is None: continue
        
        count, frame = counter.count(frame)
        
        with lock:
            latest_frame = frame
            latest_count = count
        
        time.sleep(AWAIT_TIME)

def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            _, buffer = imencode('.jpg', latest_frame)
            frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.1)

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count')
def people_count():
    with lock:
        count = latest_count
    return jsonify({"people_count": count})

if __name__ == '__main__':
    t = threading.Thread(target=update_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000)

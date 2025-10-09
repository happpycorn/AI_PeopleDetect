from people_detect import count_people
import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

interval = 0.05  # 每秒抓一張影像
last_time = 0

while True:
    current_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if current_time - last_time < interval: continue

    ret, frame = cap.read()
    if not ret: 
        print("Can't receive frame (stream end?). Exiting ...")
        break

    people_count = count_people(frame)
    print(f"Detected people: {people_count}")

    cv2.imshow('Webcam People Counter', frame)

    last_time = current_time

    # cv2.imshow('Dummy', cv2.UMat(1,1,cv2.CV_8UC3))  # 幾乎看不到的窗口

    
cap.release()
cv2.destroyAllWindows()

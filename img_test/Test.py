from ultralytics import YOLO
import cv2

# 載入模型 (n = nano 輕量，適合測試)
model = YOLO("yolov8n.pt")

# 讀取靜態照片
image = cv2.imread("test.jpg")

# 推論
results = model(image)[0]

# 過濾只保留人 (class 0 = person)
person_boxes = [box.xyxy[0] for box, cls in zip(results.boxes, results.boxes.cls) if int(cls) == 0]

# # 畫框
# for box in person_boxes:
#     x1, y1, x2, y2 = map(int, box)
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 顯示人數
print("Detected people:", len(person_boxes))
# cv2.putText(image, f"People: {len(person_boxes)}", (20, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# cv2.imshow("People Counter", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

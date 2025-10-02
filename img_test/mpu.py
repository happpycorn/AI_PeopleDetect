from ultralytics import YOLO
import torch
import cv2

# 選擇裝置
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO("yolov8n.pt")
model.to(device)

image = cv2.imread("test.jpg")
results = model(image, device=device)[0]

print("Detected people:", sum(int(cls==0) for cls in results.boxes.cls))

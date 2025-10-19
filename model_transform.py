from ultralytics import YOLO
import onnx

model = YOLO("yolov8n.pt")

onnx_model_path = "yolov8n.onnx"
model.export(
    format="onnx",
    opset=16,
    dynamic=False,
    imgsz=(640, 640)
)
print(f"ONNX model exported to {onnx_model_path}")
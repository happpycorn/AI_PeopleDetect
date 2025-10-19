from ultralytics import YOLO
import onnx
from onnxsim import simplify

model = YOLO("yolov8n.pt")

onnx_model_path = "yolov8n.onnx"
model.export(
    format="onnx",
    opset=16,
    dynamic=False,
    imgsz=(640, 640)
)
print(f"ONNX model exported to {onnx_model_path}")

onnx_model = onnx.load(onnx_model_path)

simplified_model, check = simplify(onnx_model)

if not check: raise RuntimeError("ONNX simplifier failed!")

simplified_model_path = "yolov8n_simplified.onnx"
onnx.save(simplified_model, simplified_model_path)
print(f"Simplified ONNX model saved to {simplified_model_path}")
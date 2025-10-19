from abc import ABC, abstractmethod
from ultralytics import YOLO
import numpy as np
import cv2

class PeopleCounter:
    def __init__(self, using_pt=True, model_path="yolov8n.onnx"):
        if using_pt:
            self.module = PtModule(model_path.replace(".onnx", ".pt"))
        else:
            self.module = OnnxModule(model_path)
        print(f"[INFO] PeopleCounter initialized (using_pt={using_pt})")

    def count(self, image):
        if isinstance(self.module, PtModule):
            results = self.module.infer(image)
            people_mask = (results.boxes.cls == 0)
            count = int(people_mask.sum().item())
            annotated = self.module.draw(image, results)
            return count, annotated

        elif isinstance(self.module, OnnxModule):
            results = self.module.infer(image)
            annotated, count = self.module.draw(image.copy(), results)
            return count, annotated

class BaseYoloModule(ABC):
    @abstractmethod
    def infer(self, image: np.ndarray):
        pass

    @abstractmethod
    def draw(self, image: np.ndarray, results):
        pass

class PtModule(BaseYoloModule):
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        print(f"[INFO] Loaded PyTorch model: {model_path}")

    def infer(self, image: np.ndarray):
        return self.model(image, verbose=False)[0]

    def draw(self, image: np.ndarray, results):
        annotated = image.copy()
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, "person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated
    
class OnnxModule:
    def __init__(self, model_path="yolov8n.onnx", conf_thres=0.5, nms_thres=0.4):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        print(f"[INFO] ONNX model loaded: {model_path}")

    def infer(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        outputs = np.squeeze(outputs)

        boxes, confidences, class_ids = [], [], []
        for detection in outputs.T:  # YOLOv8: (84, N)
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.conf_thres and class_id == 0:  # 只偵測人
                cx, cy, bw, bh = detection[:4]
                x = int((cx - bw / 2) * w / 640)
                y = int((cy - bh / 2) * h / 640)
                width = int(bw * w / 640)
                height = int(bh * h / 640)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.nms_thres)
        results = (boxes, confidences, class_ids, indices)
        return results

    def draw(self, image, results):
        boxes, confidences, class_ids, indices = results
        person_count = 0

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(image, "person", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2 )
                person_count += 1

        return image, person_count

if __name__ == "__main__":
    counter = PeopleCounter(using_pt=False, model_path="yolov8n.onnx")
    # counter = PeopleCounter(using_pt=True, model_path="yolov8n.pt")

    img = cv2.imread("test.jpg")
    count, annotated = counter.count(img)
    print("Detected people:", count)

    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

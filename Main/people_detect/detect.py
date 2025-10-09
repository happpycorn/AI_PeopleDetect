from ultralytics import YOLO
import numpy as np
import cv2

class PeopleCounter:
    def __init__(self, model_path: str = "yolov8n.pt", using_NPU: bool = False):

        self.model = YOLO(model_path)
        self.using_NPU = using_NPU
        print(f"[INFO] YOLO model loaded from '{model_path}'")

    def count(self, image: np.ndarray) -> tuple[int, np.ndarray]:

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid OpenCV image (numpy.ndarray).")

        results = self.model(image, verbose=False)[0]
        
        people_mask = (results.boxes.cls == 0)
        people_count = int(people_mask.sum().item())
        
        annotated_image = self.drawRect(image, results)

        return people_count, annotated_image

    def drawRect(self, image, results) -> np.ndarray:
        annotated_image = image.copy()
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, "person", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated_image

if __name__ == "__main__":
    counter = PeopleCounter("yolov8n.pt")
    image = cv2.imread("test.jpg")
    
    count, annotated = counter.count(image)
    print("Detected people:", count)
    
    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

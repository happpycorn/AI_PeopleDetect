from ultralytics import YOLO
import numpy as np

# 初始化模型 (放在外面，只需載入一次)
model = YOLO("yolov8n.pt")

def count_people(image: np.ndarray) -> int:
    """
    偵測影像中的人數
    :param image: OpenCV 讀取的影像 (np.ndarray)
    :return: 偵測到的人數
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid OpenCV image (numpy.ndarray).")
    
    results = model(image, verbose=False)[0]
    people_count = (results.boxes.cls == 0).sum().item()
    return people_count

if __name__ == "__main__":
    import cv2
    image = cv2.imread("test.jpg")
    print("Detected people:", count_people(image))

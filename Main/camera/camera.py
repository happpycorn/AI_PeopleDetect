import cv2

class Camera:
    def __init__(self, device_index=1):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Can't receive frame (stream end?).")
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = Camera()
    try:
        frame = cam.capture_frame()
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
    finally:
        cam.release()

from people_detect import count_people

import cv2
image = cv2.imread("test.jpg")
print("Detected people:", count_people(image))
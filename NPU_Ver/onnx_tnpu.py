import sys
sys.path.append('./')
import cv2
import numpy as np
import ktc
import torch

class NPUOnnxModule:
    def __init__(self, model_path="yolov8n.onnx", conf_thres=0.5, nms_thres=0.4, platform=730):
        self.onnx_path = model_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.platform = platform
        print(f"[INFO] NPU ONNX module initialized: {model_path} on platform {platform}")

    def letterbox(self, img, new_shape=(640, 640), color=(114,114,114)):
        h, w = img.shape[:2]
        r = min(new_shape[0]/h, new_shape[1]/w)
        new_unpad_w, new_unpad_h = int(round(w*r)), int(round(h*r))  # 使用 round
        dw, dh = new_shape[1]-new_unpad_w, new_shape[0]-new_unpad_h
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img_padded, r, (dw, dh)

    def preprocess(self, img, model_input_w=640, model_input_h=640):
        img_resized, r, (dw, dh) = self.letterbox(img, (model_input_w, model_input_h))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = img_resized.astype(np.float32) / 255.0
        img_chw = img_resized.transpose(2,0,1)[None,...]
        return img_chw, r, dw, dh

    def post_process(self, model_pred, r, dw, dh, conf_thres=0.1, iou_thres=0.45, target_class_id=0):

        boxes, confidences, class_ids = [], [], []
        for detection in model_pred[0][0].T:  # YOLOv8: (84, N)
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > self.conf_thres and class_id == 0:  # 只偵測人
                cx, cy, bw, bh = detection[:4]
                x = int((cx - bw/2 - (dw/2)) / r)
                y = int((cy - bh/2 - (dh/2)) / r)
                width = int(bw / r)
                height = int(bh / r)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.nms_thres)
        results = (boxes, confidences, class_ids, indices)
        return results
    
    def infer(self, image):
        img, r, dw, dh = self.preprocess(image)
        model_pred = ktc.kneron_inference(
            [img], input_names=['images'],
            onnx_file=self.onnx_path, platform=self.platform
        )
        det = self.post_process(model_pred, r, dw, dh)
        return det

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

# ===== 使用範例 =====
if __name__ == "__main__":
    from constant import ONNX_PATH, ONNX_NAME
    from os import path

    image_path = "test.jpg"
    im0 = cv2.imread(image_path)

    model = NPUOnnxModule(
        model_path=path.join(ONNX_PATH, f"Output_{ONNX_NAME}"), 
        conf_thres=0.5, nms_thres=0.4, platform=730
    )
    det = model.infer(im0)
    out_img, count = model.draw(im0, det)

    save_path = "data/test_result.jpg"
    cv2.imwrite(save_path, out_img)
    print(f"偵測完成，總人數: {count}, 圖片已儲存至: {save_path}")

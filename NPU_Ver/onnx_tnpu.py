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

    def torch_nms(self, boxes, scores, iou_threshold=0.45):
        """
        Pure PyTorch NMS
        boxes: [N, 4] tensor, xyxy
        scores: [N] tensor
        """
        keep = []
        if boxes.numel() == 0:
            return torch.tensor(keep, dtype=torch.long)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, idxs = scores.sort(descending=True)

        while idxs.numel() > 0:
            i = idxs[0]
            keep.append(i.item())

            if idxs.numel() == 1:
                break

            xx1 = torch.maximum(x1[i], x1[idxs[1:]])
            yy1 = torch.maximum(y1[i], y1[idxs[1:]])
            xx2 = torch.minimum(x2[i], x2[idxs[1:]])
            yy2 = torch.minimum(y2[i], y2[idxs[1:]])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[idxs[1:]] - inter)
            idxs = idxs[1:][iou <= iou_threshold]

        return torch.tensor(keep, dtype=torch.long)

    def preprocess(self, img, model_input_w=640, model_input_h=640):
        img_resized = cv2.resize(img, (model_input_w, model_input_h)).astype(np.float32) / 255.0
        img_chw = img_resized.transpose(2, 0, 1)[None, ...]
        return img_chw

    def post_process(self, model_pred, conf_thres=0.25, iou_thres=0.45):
        batch_det = []
        for pred in model_pred:
            if len(pred) == 0:
                batch_det.append([])
                continue

            # confidence 篩選
            mask = pred[:, 4] > conf_thres
            pred = pred[mask]
            if len(pred) == 0:
                batch_det.append([])
                continue

            # NMS
            boxes = torch.tensor(pred[:, :4], dtype=torch.float32)
            scores = torch.tensor(pred[:, 4], dtype=torch.float32)
            keep = self.torch_nms(boxes, scores, iou_thres)
            pred = pred[keep]

            batch_det.append(pred)
        return batch_det

    def infer(self, image):
        img = self.preprocess(image, model_input_w=640, model_input_h=640)

        model_pred = ktc.kneron_inference(
            [img], input_names=['images'],
            onnx_file=self.onnx_path, platform=self.platform
        )

        batch_det = self.post_process(model_pred, conf_thres=self.conf_thres, iou_thres=self.nms_thres)
        return batch_det[0]

    def draw(self, image, det):
        person_count = 0
        if len(det):
            for d in det:
                # 安全 unpack
                if len(d) >= 6:
                    x1, y1, x2, y2, conf, cls = d
                else:
                    x1, y1, x2, y2, conf = d
                    cls = 0  # 如果沒有 cls, 假設人

                if int(cls) != 0: continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image, f"person {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                person_count += 1
        return image, person_count

# ===== 使用範例 =====
if __name__ == "__main__":
    from constant import ONNX_PATH, ONNX_NAME
    from os import path

    image_path = "test.jpg"
    im0 = cv2.imread(image_path)

    model = NPUOnnxModule(
        model_path=path.join(ONNX_PATH, ONNX_NAME), 
        conf_thres=0.5, nms_thres=0.4, platform=730
    )
    det = model.infer(im0)
    out_img, count = model.draw(im0, det)

    save_path = "test_result.jpg"
    cv2.imwrite(save_path, out_img)
    print(f"偵測完成，總人數: {count}, 圖片已儲存至: {save_path}")

import os 
os.chdir('./yolov7')
import sys
sys.path.append('./')
from utils.general import  non_max_suppression, scale_coords
from utils.plots import plot_one_box

def preprocess(img, model_input_w, model_input_h):

    img = cv2.resize(img, (model_input_w, model_input_h)).astype(np.float32)
    img = np.transpose(img, (2,0,1))
    img /= 255.0
    img = np.expand_dims(img, axis=[0])

    return img

def post_process(model_inf_data):
    pred = []
    for o_n in model_inf_data:
        pred.append(torch.from_numpy(o_n))

    anchors = []
    anchors.append([12,16, 19,36, 40,28])  
    anchors.append([36,75, 76,55, 72,146])  
    anchors.append([142,110, 192,243, 459,401])  

    nl = len(anchors)  # number of detection layers
    grid = [torch.zeros(1)] * nl  # init grid

    a = torch.tensor(anchors).float().view(nl, -1, 2)
    anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2)
    stride = torch.tensor([8. , 16. , 32.])  

    for idx, every_stride in enumerate(np.array(stride.view(-1, 1, 1)).squeeze()):
        anchors[idx] /= every_stride

    scaled_res = []

    for i in range(len(anchors)):
        if grid[i].shape[2:4] != pred[i].shape[2:4]:
            bs, _, ny, nx, _ = pred[i].shape
            grid[i] = _make_grid(nx, ny).to('cpu')

        y = sigmoid(pred[i])

        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        scaled_res.append(y.reshape([bs,-1,85]))

    concat_pred = torch.concat(scaled_res,dim=1)

    # Apply NMS
    batch_det = non_max_suppression(concat_pred, conf_thres=0.25, iou_thres=0.45)
    return batch_det
## onnx model check using onnxruntime
im0 = cv2.imread('/data1/yolov7/inference/images/bus.jpg')

# resize and normalize input data
img = preprocess(im0, model_input_w = 640, model_input_h = 640)

# onnx inference using onnxruntime
OPTIMIZED_ONNX_PATH = "/data1/kneopi_yolov7-tiny_opt.onnx" 
ort_session = ort.InferenceSession(OPTIMIZED_ONNX_PATH)
model_pred = ort_session.run(None, {'images': img})

# onnx output data processing
batch_det = post_process(model_pred)
det = batch_det[0] #only one image

# visualize segmentation result to img
if len(det):
    cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in cls_names]

    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    # Write results
    for *xyxy, conf, cls in reversed(det):
        label = f'{cls_names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    # Save results (image with detections)
    save_path = os.path.abspath("/data1/kneopi_onnxruntime_inf.jpg")
    cv2.imwrite(save_path, im0)

    print("save result to " + save_path)
## onnx model check using ktc.inference
im0 = cv2.imread('/data1/yolov7/inference/images/bus.jpg')

# resize and normalize input data
img = preprocess(im0, model_input_w = 640, model_input_h = 640)

# onnx inference using ktc.inference
OPTIMIZED_ONNX_PATH = "/data1/kneopi_yolov7-tiny_opt.onnx" 
model_pred = ktc.kneron_inference([img], input_names=['images'], onnx_file=OPTIMIZED_ONNX_PATH, platform=730)

# onnx output data processing
batch_det = post_process(model_pred)
det = batch_det[0] #only one image

# visualize detection result to img
if len(det):
    cls_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in cls_names]

    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    # Write results
    for *xyxy, conf, cls in reversed(det):
        label = f'{cls_names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    # Save results (image with detections)
    save_path = os.path.abspath("/data1/kneopi_tc_onnx_inf.jpg")
    cv2.imwrite(save_path, im0)

    print("save result to " + save_path)
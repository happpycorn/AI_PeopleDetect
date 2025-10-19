from constant import ONNX_PATH, ONNX_NAME

import onnx
from os import path
import ktc 

m = onnx.load(path.join(ONNX_PATH, ONNX_NAME))
md_opt = ktc.onnx_optimizer.onnx2onnx_flow(m)


km = ktc.ModelConfig(11111, "0001", "730", onnx_model=md_opt)

onnx.save(md_opt, path.join(ONNX_PATH, f"Output_{ONNX_NAME}"))
from constant import ONNX_PATH, ONNX_NAME

import onnx
from os import path
import ktc 
from onnxsim import simplify

m = onnx.load(path.join(ONNX_PATH, ONNX_NAME))

# m, check = simplify(m)

# if not check: raise RuntimeError("ONNX simplifier failed!")

m = ktc.onnx_optimizer.onnx2onnx_flow(m, duplicate_shared_weights=False)

km = ktc.ModelConfig(11111, "0001", "730", onnx_model=m)

onnx.save(m, path.join(ONNX_PATH, f"Output_{ONNX_NAME}"))
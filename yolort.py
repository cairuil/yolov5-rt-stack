import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import numpy as np
import cv2

import torch
from torch import nn
import torchvision
from yolort.models import yolov5s
# from yolort.relay import get_trace_module
from yolort.utils import get_image_from_url
from tvm.contrib import utils,pipeline_executor
from tvm.contrib import graph_executor


in_size = 640
input_shape = (in_size, in_size)

export_script_name = 'yolort.torchscript.pt'
# model_func=yolov5s(pretrained=True,size=(in_size,in_size))
# script_module=get_trace_module(model_func,input_shape=input_shape)

# script_module.save(export_script_name)
# script_module=export_script_name

script_module=torch.jit.load(export_script_name)
print(script_module.graph)


img= cv2.imread("bus.jpg")

img1 = cv2.resize(img, (in_size, in_size))
img = img1.astype("float32")
# img1 = cv2.resize(img, (in_size, in_size))

img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)


input_name = "input0"
shape_list = [(input_name, (1, 3, *input_shape))]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)

target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    lib= relay.vm.compile(mod, target=target, params=params)


ctx = tvm.cpu()

vm = VirtualMachine(lib, ctx)
vm.set_input("main", **{input_name: img})
tvm_res = vm.run()

# print(tvm_res[].asnumpy().tolist())

print(tvm_res().asnumpy().tolist())

print(tvm_res[1].asnumpy().tolist())
print(tvm_res[2].asnumpy().tolist())


score_threshold = 0.6
boxes = tvm_res[0].asnumpy().tolist()
valid_boxes = []
for i, score in enumerate(tvm_res[1].asnumpy().tolist()):
    if score > score_threshold:
        valid_boxes.append(boxes[i])
    else:
        break

print(f"Get {len(valid_boxes)} valid boxes")


for i in range(0,len(valid_boxes)):
    img_src=cv2.rectangle(img1,(int(valid_boxes[i][0]),int(valid_boxes[i][1])),(int(valid_boxes[i][2]),int(valid_boxes[i][3])),color=(0,255,255),thickness=2,lineType=cv2.LINE_AA)
cv2.namedWindow("demo")
cv2.imshow("demo",img_src)
cv2.waitKey(0)
cv2.destroyAllWindows()

# temp = utils.tempdir()
# path_lib = temp.relpath("yolort.tar")
# lib.export_library(path_lib)
# print(temp.listdir())

# for i in range(3):
#     torch.testing.assert_allclose(torch_res[i], tvm_res[i].asnumpy(), rtol=1e-4, atol=1e-4)
#
# print("looks good!")
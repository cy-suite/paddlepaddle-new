# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorrt as trt

from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.grid_sample", trt_version="8.x")
def grid_sample_converter(network, paddle_op, inputs):
    input_tensor, grid_tensor = inputs
    padding = paddle_op.attrs().get("paddings", [0, 0])

    mode = paddle_op.attrs().get("mode", "bilinear")
    padding_mode = paddle_op.attrs().get("padding_mode", "zeros")
    align_corners = paddle_op.attrs().get("align_corners", True)

    if padding_mode == "zeros":
        sample_mode = trt.SampleMode.FILL
    elif padding_mode == "border":
        sample_mode = trt.SampleMode.CLAMP
    elif padding_mode == "reflection":
        sample_mode = trt.SampleMode.REFLECT

    if mode == "nearest":
        interpolation_mode = trt.InterpolationMode.NEAREST
    elif mode == "bilinear":
        interpolation_mode = trt.InterpolationMode.LINEAR

    grid_sample_layer = network.add_grid_sample(input_tensor, grid_tensor)

    grid_sample_layer.interpolation_mode = interpolation_mode
    grid_sample_layer.align_corners = align_corners
    grid_sample_layer.sample_mode = sample_mode
    return grid_sample_layer.get_output(0)

@converter_registry.register("pd_op.yolo_box_head", trt_version="8.x")
def yolo_box_head_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    anchors = paddle_op.attrs().get("anchors", [])
    class_num = paddle_op.attrs().get("class_num", 0)

    yolo_box_plugin = YoloBoxHeadPlugin(anchors, class_num)

    yolo_box_inputs = [input_tensor]
    yolo_box_head_layer = network.add_plugin_v2(yolo_box_inputs, yolo_box_plugin)

    output_name = paddle_op.output("Out")[0]
    return yolo_box_head_layer.get_output(0)
class YoloBoxHeadPlugin(trt.IPluginV2):
    def __init__(self, anchors, class_num):
        self.anchors = anchors
        self.class_num = class_num

    def get_plugin_type(self):
        return "yolo_box_head_plugin"

    def get_plugin_version(self):
        return "1"

    def initialize(self):
        return 0

    def get_output_dimensions(self, index, inputs, nb_input_dims):
        assert index == 0
        assert nb_input_dims == 1
        return inputs[0]

    def enqueue(self, batch_size, inputs, outputs, workspace, stream):
        pass

    def serialize(self):
        return bytearray(self.anchors) + bytearray([self.class_num])

    def deserialize(self, data):
        self.anchors = list(data[:len(data) - 4])
        self.class_num = int(data[-4:])
        return self

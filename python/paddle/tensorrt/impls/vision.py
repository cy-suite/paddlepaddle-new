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


@converter_registry.register("pd_op.roi_align", trt_version="8.x")
def roi_align_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    rois_tensor = inputs[1]

    pooled_height = paddle_op.attrs().get("pooled_height", 0)
    pooled_width = paddle_op.attrs().get("pooled_width", 0)
    spatial_scale = paddle_op.attrs().get("spatial_scale", 0.0)
    sampling_ratio = paddle_op.attrs().get("sampling_ratio", -1)
    aligned = paddle_op.attrs().get("aligned", True)

    data_type = trt.DataType.FLOAT16

    roi_align_plugin = trt.plugins.RoiAlignPluginDynamic(
        data_type,
        pooled_height,
        pooled_width,
        spatial_scale,
        sampling_ratio,
        aligned,
    )

    roi_align_layer = network.add_plugin_v2(
        [input_tensor, rois_tensor], roi_align_plugin
    )
    return roi_align_layer.get_output(0)

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


@converter_registry.register("pd_op.anchor_generator", trt_version="8.x")
def anchor_generator_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_dims = input_tensor.shape
    print(input_dims)
    anchor_sizes = paddle_op.attrs()["anchor_sizes"]
    aspect_ratios = paddle_op.attrs()["aspect_ratios"]
    stride = paddle_op.attrs()["stride"]
    variances = paddle_op.attrs()["variances"]
    offset = paddle_op.attrs()["offset"]

    num_anchors = len(aspect_ratios) * len(anchor_sizes)
    is_dynamic = True
    height = input_dims[1]
    width = input_dims[2]
    box_num = width * height * num_anchors
    data_type = trt.DataType.FLOAT

    if is_dynamic:
        plugin_name = "AnchorGeneratorPluginDynamic"
        fields = [
            trt.PluginField(
                "data_type",
                np.array(data_type, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "anchor_sizes",
                np.array(anchor_sizes, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "aspect_ratios",
                np.array(aspect_ratios, dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "stride",
                np.array(stride, dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "variances",
                np.array(variances, dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "offset",
                np.array([offset], dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "num_anchors",
                np.array(num_anchors, dtype=np.int32),
                trt.PluginFieldType.INT32,
            )
        ]
    else:
        plugin_name = "AnchorGeneratorPlugin"
        fields = [
            trt.PluginField(
                "data_type",
                np.array(data_type, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "anchor_sizes",
                np.array(anchor_sizes, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "aspect_ratios",
                np.array(aspect_ratios, dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "stride",
                np.array(stride, dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "variances",
                np.array(variances, dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "offset",
                np.array([offset], dtype=np.float32),
                trt.PluginFieldType.FLOAT32,
            ),
            trt.PluginField(
                "num_anchors",
                np.array(num_anchors, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "height",
                np.array(height, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "width",
                np.array(width, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "box_num",
                np.array(box_num, dtype=np.int32),
                trt.PluginFieldType.INT32,
            )
        ]
    field_collection = trt.PluginFieldCollection(fields)
    plugin_version = "1"
    anchor_generator_plugin = get_trt_plugin(
        plugin_name, field_collection, plugin_version
    )
    anchor_generator_layer = network.add_plugin_v2(
        [input_tensor], anchor_generator_plugin
    )
    return anchor_generator_layer.get_output(
        0
    ), anchor_generator_layer.get_output(1)

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

from paddle.tensorrt.converter_utils import (
    add_elementwise_layer,
)
from paddle.tensorrt.register import converter_registry

logic_type_map = {
    "pd_op.greater_than": trt.ElementWiseOperation.GREATER,
    "pd_op.less_than": trt.ElementWiseOperation.LESS,
}


@converter_registry.register("pd_op.greater_than", trt_version="8.x")
@converter_registry.register("pd_op.less_than", trt_version="8.x")
def logic_converter(network, paddle_op, inputs):
    layer_output = add_elementwise_layer(
        network, paddle_op, inputs, logic_type_map[paddle_op.name()]
    )
    return layer_output


@converter_registry.register("pd_op.equal", trt_version="8.x")
@converter_registry.register("pd_op.equal_", trt_version="8.x")
def equal_converter(network, paddle_op, inputs):
    X = inputs[0]
    Y = inputs[1]
    dims_x = X.shape
    dims_y = Y.shape
    axis = abs(len(dims_x) - len(dims_y))

    if len(dims_x) > len(dims_y):
        expand_shape = [1] * len(dims_x)
        expand_shape[axis:axis+len(dims_y)] = dims_y
        expand_layer = network.add_shuffle(Y)
        expand_layer.reshape_dims = tuple(expand_shape)
        Y = expand_layer.get_output(0)
    elif len(dims_x) < len(dims_y):
        expand_shape = [1] * len(dims_y)
        expand_shape[axis:axis+len(dims_x)] = dims_x
        expand_layer = network.add_shuffle(X)
        expand_layer.reshape_dims = tuple(expand_shape)
        X = expand_layer.get_output(0)

    equal_layer = network.add_elementwise(X, Y, trt.ElementWiseOperation.EQUAL)
    return equal_layer.get_output(0)


@converter_registry.register("pd_op.not_equal", trt_version="8.x")
@converter_registry.register("pd_op.not_equal_", trt_version="8.x")
def not_equal_converter(network, paddle_op, inputs):
    X = inputs[0]
    Y = inputs[1]
    dims_x = X.shape
    dims_y = Y.shape
    axis = abs(len(dims_x) - len(dims_y))

    if len(dims_x) > len(dims_y):
        expand_shape = [1] * len(dims_x)
        expand_shape[axis:axis+len(dims_y)] = dims_y
        expand_layer = network.add_shuffle(Y)
        expand_layer.reshape_dims = tuple(expand_shape)
        Y = expand_layer.get_output(0)
    elif len(dims_x) < len(dims_y):
        expand_shape = [1] * len(dims_y)
        expand_shape[axis:axis+len(dims_x)] = dims_x
        expand_layer = network.add_shuffle(X)
        expand_layer.reshape_dims = tuple(expand_shape)
        X = expand_layer.get_output(0)

    equal_layer = network.add_elementwise(X, Y, trt.ElementWiseOperation.EQUAL)

    not_layer = network.add_unary(equal_layer.get_output(0), trt.UnaryOperation.NOT)
    layer_output = not_layer.get_output(0)
    return layer_output

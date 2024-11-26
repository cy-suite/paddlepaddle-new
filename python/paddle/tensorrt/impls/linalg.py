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
    broadcast,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.matmul", trt_version="8.x")
def matmul_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    transpose_x = paddle_op.attrs()["transpose_x"]
    transpose_y = paddle_op.attrs()["transpose_y"]
    self_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_x
        else trt.MatrixOperation.NONE
    )
    other_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_y
        else trt.MatrixOperation.NONE
    )

    weight_tensor = inputs[1]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)

    if len(weight_shape) == 1:
        layer = network.add_shuffle(weight_tensor)
        layer.reshape_dims = (*tuple(weight_shape), 1)
        weight_tensor = layer.get_output(0)

    lhs_val, rhs_val = broadcast(
        network, inputs[0], weight_tensor, inputs[0].name, weight_tensor.name
    )
    out = network.add_matrix_multiply(
        lhs_val, self_matrix_op, rhs_val, other_matrix_op
    )
    return out.get_output(0)


@converter_registry.register("pd_op.transpose", trt_version="8.x")
def transpose_converter(network, paddle_op, inputs):
    perm = paddle_op.attrs()["perm"]
    transposed_tensor = network.add_shuffle(inputs[0])
    transposed_tensor.second_transpose = perm
    return transposed_tensor.get_output(0)


@converter_registry.register("pd_op.bmm", trt_version="8.x")
def bmm_converter(network, paddle_op, inputs):
    out = network.add_matrix_multiply(
        inputs[0], trt.MatrixOperation.NONE, inputs[1], trt.MatrixOperation.NONE
    )
    return out.get_output(0)


@converter_registry.register("pd_op.flip", trt_version="8.x")
def flip_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape_layer = network.add_shape(input_tensor)
    input_shape_layer.name = f"{input_tensor.name}_shape"

    input_shape = input_shape_layer.get_output(0)
    rank = len(input_tensor.shape)

    axis = paddle_op.attrs()["axis"]
    if isinstance(axis, int):
        axis = [axis]
    axis = [get_positive_dim(a, rank) for a in axis]

    start_tensors = []
    stride_tensors = []
    size_tensors = []

    for i in range(rank):
        dim_tensor = get_shape_tensor_element(network, input_shape, i)
        if i in axis:
            # start = dim - 1, stride = -1, size = dim
            start_tensors.append(trt_sub(network, dim_tensor, add_1D_constant_layer(network, [1])))
            stride_tensors.append(add_1D_constant_layer(network, [-1]))
            size_tensors.append(dim_tensor)
        else:
            # start = 0, stride = 1, size = dim
            start_tensors.append(add_1D_constant_layer(network, [0]))
            stride_tensors.append(add_1D_constant_layer(network, [1]))
            size_tensors.append(dim_tensor)

    start_tensor = trt_concat(network, start_tensors)
    stride_tensor = trt_concat(network, stride_tensors)
    size_tensor = trt_concat(network, size_tensors)

    slice_layer = network.add_slice(
        input_tensor, start=(0,) * rank, shape=(1,) * rank, stride=(1,) * rank
    )
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, size_tensor)
    slice_layer.set_input(3, stride_tensor)
    slice_layer.name = f"{input_tensor.name}_flip"
    return slice_layer.get_output(0)

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

import numpy as np
import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    add_cast_reduce_layer,
    add_elementwise_layer,
    add_reduce_layer,
    broadcast,
    get_axes_for_reduce_op,
    trt_cast,
    trt_div,
    trt_floor_div,
    trt_mul,
    trt_sub,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.add", trt_version="8.x")
@converter_registry.register("pd_op.add_", trt_version="8.x")
def add_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.SUM
    )


@converter_registry.register("pd_op.scale", trt_version="8.x")
def scale_converter(network, paddle_op, inputs):
    scale = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    bias = paddle_op.attrs().get("bias", 0.0)
    power = paddle_op.attrs().get("power", 1.0)

    # Convert scale, bias, and power to TensorRT weights
    scale_weight = trt.Weights(np.array([scale], dtype=np.float32))
    bias_weight = trt.Weights(np.array([bias], dtype=np.float32))
    power_weight = trt.Weights(np.array([power], dtype=np.float32))

    scale_layer = network.add_scale(
        inputs[0],
        mode=trt.ScaleMode.UNIFORM,
        shift=bias_weight,
        scale=scale_weight,
        power=power_weight,
    )
    return scale_layer.get_output(0)


@converter_registry.register("pd_op.max", trt_version="8.x")
def max_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    axis = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    input_shape = paddle_op.operands()[0].source().shape
    keepdim = paddle_op.attrs()["keepdim"]
    if network.has_implicit_batch_dimension:
        assert (
            axis != 0
        ), "can't reduce on axis == 0 when network has implicit batch dimension"
    output_shape = []
    if len(axis) == 0:
        axis = list(range(len(input_shape)))
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] = len(input_shape) + axis[i]
    layer = network.add_reduce(
        input_tensor,
        trt.ReduceOperation.MAX,
        axes=get_axes_for_reduce_op(axis),
        keep_dims=keepdim,
    )
    return layer.get_output(0)


@converter_registry.register("pd_op.divide", trt_version="8.x")
def divide_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.DIV
    )


@converter_registry.register("pd_op.subtract", trt_version="8.x")
def substract_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.SUB
    )


@converter_registry.register("pd_op.multiply", trt_version="8.x")
def multiply_converter(network, paddle_op, inputs):
    return add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.PROD
    )


@converter_registry.register("pd_op.remainder", trt_version="8.x")
@converter_registry.register("pd_op.remainder_", trt_version="8.x")
def remainder_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    input_shape = paddle_op.operands()[0].source().shape

    weight_tensor = inputs[1]
    input_tensor = inputs[0]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)
    if type(inputs[0]) == trt.Weights:
        input_tensor = network.add_constant(input_shape, inputs[0]).get_output(
            0
        )

    lhs_val, rhs_val = broadcast(
        network,
        input_tensor,
        weight_tensor,
        input_tensor.name,
        weight_tensor.name,
    )

    # Check if floor division is needed
    is_floor_div = input_tensor.dtype != trt.DataType.INT32

    # Floor division
    quotient = (
        trt_floor_div(network, lhs_val, rhs_val)
        if is_floor_div
        else trt_div(network, lhs_val, rhs_val)
    )

    # Multiply rhs by the quotient
    product = trt_mul(network, rhs_val, quotient)

    # Subtract the product from lhs to get the remainder
    remainder = trt_sub(network, lhs_val, product)

    return remainder


@converter_registry.register("pd_op.min", trt_version="8.x")
def min_converter(network, paddle_op, inputs):
    return add_reduce_layer(network, paddle_op, inputs, trt.ReduceOperation.MIN)


@converter_registry.register("pd_op.sum", trt_version="8.x")
def sum_converter(network, paddle_op, inputs):
    return add_reduce_layer(network, paddle_op, inputs, trt.ReduceOperation.SUM)


@converter_registry.register("pd_op.any", trt_version="8.x")
def any_converter(network, paddle_op, inputs):
    return add_cast_reduce_layer(
        network, paddle_op, inputs, trt.ReduceOperation.MAX
    )


@converter_registry.register("pd_op.all", trt_version="8.x")
def all_converter(network, paddle_op, inputs):
    return add_cast_reduce_layer(
        network, paddle_op, inputs, trt.ReduceOperation.MIN
    )


@converter_registry.register("pd_op.cumsum", trt_version="8.x")
def cumsum_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    dtype = input_tensor.dtype
    axis = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    input_shape = input_tensor.shape
    rank = len(input_shape)

    if axis < 0:
        axis += rank
    axis = int(axis)

    # Obtain the number of cycles
    if input_shape[axis] > 0:
        axis_tensor = np.array(input_shape[axis], dtype=np.int32)
        trip_limit = network.add_constant((), axis_tensor)
    else:
        dynamic_shape = network.add_shape(input_tensor).get_output(0)
        axis_tensor = np.array(axis, dtype=np.int32)
        index = network.add_constant((), axis_tensor).get_output(0)
        trip_limit = network.add_gather(dynamic_shape, index, 0)

    # Obtain the slice shape
    shape_list = []
    for i in range(rank):
        if i == axis:
            shape_list.append(add_1D_constant_layer(network, [1]))
        elif input_shape[i] < 0:
            dynamic_shape = network.add_shape(input_tensor).get_output(0)
            index = network.add_constant(
                (), np.array(i, dtype=np.int32)
            ).get_output(0)
            shape_index = network.add_gather(dynamic_shape, index, 0)
            shuffle_layer = network.add_shuffle(shape_index.get_output(0))
            shuffle_layer.reshape_dims = (1,)
            shape_list.append(shuffle_layer.get_output(0))
        else:
            shape_list.append(add_1D_constant_layer(network, input_shape[i]))
    slice_shape = network.add_concatenation(shape_list).get_output(0)

    start = [0] * rank
    size = [1] * rank
    stride = [1] * rank
    input_sliced = network.add_slice(input_tensor, start, size, stride)
    input_sliced.set_input(2, slice_shape)

    # squeeze axis
    shape_list.pop(axis)
    new_shape = network.add_concatenation(shape_list).get_output(0)
    squeeze_layer = network.add_shuffle(input_sliced.get_output(0))
    squeeze_layer.set_input(1, new_shape)

    loop = network.add_loop()
    loop.add_trip_limit(trip_limit.get_output(0), trt.TripLimit.COUNT)

    iterator = loop.add_iterator(input_tensor, axis)
    data = iterator.get_output(0)

    # create zero tensor
    zero_vec = np.array([0.0], dtype=np.float32)
    zero = network.add_constant((1,), zero_vec).get_output(0)
    lhs_val, rhs_val = broadcast(
        network,
        squeeze_layer.get_output(0),
        zero,
        squeeze_layer.get_output(0).name,
        zero.name,
    )
    cast_tensor = trt_cast(network, rhs_val, dtype)
    zero_tensor = network.add_elementwise(
        lhs_val, cast_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)

    # Cycle and add according to the axis
    running_sum = loop.add_recurrence(zero_tensor)
    running_sum_tensor = running_sum.get_output(0)

    cur_sum = network.add_elementwise(
        data, running_sum_tensor, trt.ElementWiseOperation.SUM
    ).get_output(0)

    running_sum.set_input(1, cur_sum)

    reverse_flag = trt.LoopOutput.CONCATENATE
    loop_out = loop.add_loop_output(cur_sum, reverse_flag, axis)
    loop_out.set_input(1, trip_limit.get_output(0))

    return loop_out.get_output(0)

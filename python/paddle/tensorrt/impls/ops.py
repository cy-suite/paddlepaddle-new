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

ops_type_map = {
    "pd_op.sqrt": trt.UnaryOperation.SQRT,
    "pd_op.sqrt_": trt.UnaryOperation.SQRT,
    "pd_op.floor": trt.UnaryOperation.FLOOR,
    "pd_op.exp": trt.UnaryOperation.EXP,
    "pd_op.abs": trt.UnaryOperation.ABS,
    "pd_op.abs_": trt.UnaryOperation.ABS,
    "pd_op.sin": trt.UnaryOperation.SIN,
    "pd_op.cos": trt.UnaryOperation.COS,
    "pd_op.sinh": trt.UnaryOperation.SINH,
    "pd_op.cosh": trt.UnaryOperation.COSH,
    "pd_op.asinh": trt.UnaryOperation.ASINH,
    "pd_op.acosh": trt.UnaryOperation.ACOSH,
    "pd_op.atanh": trt.UnaryOperation.ATANH,
    "pd_op.ceil": trt.UnaryOperation.CEIL,
    "pd_op.reciprocal": trt.UnaryOperation.RECIP,
    "pd_op.erf": trt.UnaryOperation.ERF,
    "pd_op.sign": trt.UnaryOperation.SIGN,
    "pd_op.round": trt.UnaryOperation.ROUND,
}


@converter_registry.register("pd_op.sqrt", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sqrt_", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.floor", trt_version="8.x")
@converter_registry.register("pd_op.exp", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.abs", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.abs_", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sin", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.cos", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sinh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.cosh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.asinh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.acosh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.atanh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.ceil", trt_version="trt_version_ge=8.0")
@converter_registry.register(
    "pd_op.reciprocal", trt_version="trt_version_ge=8.0"
)
@converter_registry.register("pd_op.erf", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sign", trt_version="trt_version_ge=8.2")
@converter_registry.register("pd_op.round", trt_version="trt_version_ge=8.2")
def Unary_Op_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    layer = network.add_unary(input_tensor, ops_type_map[paddle_op.name()])
    return layer.get_output(0)


@converter_registry.register("pd_op.rsqrt", trt_version="trt_version_ge=8.0")
def Rsqrt_Op_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    sqrt_layer = network.add_unary(input_tensor, trt.UnaryOperation.SQRT)
    rsqrt_layer = network.add_unary(
        sqrt_layer.get_output(0), trt.UnaryOperation.RECIP
    )
    return rsqrt_layer.get_output(0)

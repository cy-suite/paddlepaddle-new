# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle import pir
from paddle.base import core
from paddle.pir.core import create_parameter


class TestRmsNormFusePattern(PassTest):
    r"""
     x                   x       w
     |                   |       |
    pow                  |       |
     |                   |       |
    mean     epilson     |       |
       \     /           |       |
        rsqrt            |       |
          |              |       |
            \          /         |
              multiply           |
                 |               |
                    \          /
                      multiply
    """

    def fused_rms_norm_pattern(self):
        python_ctx = pir.DrrPatternContext()
        python_pat = python_ctx.SourcePattern()

        def constraint_function(match_ctx):
            axis = match_ctx.VectorInt64Attr("axis")
            if len(axis) > 1:
                return False
            return True

        # Source Pattern
        pow = python_pat.Op("pd_op.pow")

        mean = python_pat.Op("pd_op.mean", {"axis": python_pat.Attr("axis")})

        full = python_pat.Op("pd_op.full")

        scale = python_pat.Op("pd_op.scale", {"bias": python_pat.Attr("bias")})

        rsqrt = python_pat.Op("pd_op.rsqrt")
        multiply1 = python_pat.Op("pd_op.multiply")
        multiply2 = python_pat.Op("pd_op.multiply")

        # Operation connections
        pow([python_pat.Tensor("x")], [python_pat.Tensor("pow_out")])

        mean([python_pat.Tensor("pow_out")], [python_pat.Tensor("mean_out")])

        full([], [python_pat.Tensor("full_out")])

        scale(
            [python_pat.Tensor("mean_out"), python_pat.Tensor("full_out")],
            [python_pat.Tensor("scale_out")],
        )

        rsqrt(
            [python_pat.Tensor("scale_out")], [python_pat.Tensor("rsqrt_out")]
        )

        multiply1(
            [python_pat.Tensor("rsqrt_out"), python_pat.Tensor("x")],
            [python_pat.Tensor("multiply_out1")],
        )

        multiply2(
            [python_pat.Tensor("multiply_out1"), python_pat.Tensor("w")],
            [python_pat.Tensor("multiply_out2")],
        )

        python_pat.AddConstraint(constraint_function)

        # Result Pattern
        python_res = python_pat.ResultPattern()

        def compute_begin_norm_axis(match_ctx):
            axis = match_ctx.VectorInt64Attr("axis")
            pow_out_shape = match_ctx.Tensor("pow_out").shape
            return (
                len(pow_out_shape) - 1 if axis[0] == -1 else axis[0],
                "int32",
            )

        begin_norm_axis = python_res.ComputeAttr(compute_begin_norm_axis)

        rms_norm = python_res.Op(
            "pd_op.rms_norm",
            {
                "epsilon": python_pat.Attr("bias"),
                "begin_norm_axis": begin_norm_axis,
                "quant_scale": python_res.Float32Attr(-1.0),
                "quant_round_type": python_res.Int32Attr(0),
                "quant_max_bound": python_res.Float32Attr(0.0),
                "quant_min_bound": python_res.Float32Attr(0.0),
            },
        )

        rms_norm(
            [
                python_res.Tensor("x"),
                python_res.InputNoneTensor(),
                python_res.InputNoneTensor(),
                python_res.Tensor("w"),
                python_res.InputNoneTensor(),
            ],
            [
                python_res.Tensor("multiply_out2"),
                python_res.Tensor("residual_out"),
                python_res.Tensor("inv_var"),
            ],
        )

        return python_ctx

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        fused_rms_norm_ctx = self.fused_rms_norm_pattern()
        for x_shape in [[1, 1, 4096]]:
            for w_shape in [[4096]]:
                for w_type in ['float32']:
                    for epilson in [1e-6]:
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float32'
                                )
                                w = create_parameter(
                                    name="w",
                                    shape=w_shape,
                                    dtype=w_type,
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.random(w_shape).astype(w_type)
                                    ),
                                )
                                variance = x.pow(2).mean(-1, keepdim=True)
                                x = paddle.rsqrt(variance + 1e-6) * x
                                out = x * w
                                out = paddle.assign(out)
                                self.pass_attr_list = [
                                    {
                                        'py_add_norm_fuse_pass': [
                                            fused_rms_norm_ctx
                                        ]
                                    }
                                ]
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float32"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.pow": 0,
                                    "pd_op.mean": 0,
                                    "pd_op.full": 0,
                                    "pd_op.scale": 0,
                                    "pd_op.rsqrt": 0,
                                    "pd_op.multiply": 0,
                                    "pd_op.rms_norm": 1,
                                }

                                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()

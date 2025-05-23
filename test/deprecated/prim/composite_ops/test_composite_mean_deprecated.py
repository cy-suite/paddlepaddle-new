# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from prim.composite_ops.utils import TOLERANCE

import paddle
from paddle import tensor
from paddle.base import core
from paddle.incubate.autograd import primapi


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class Attr:
    def __init__(self) -> None:
        self.dtype = "float32"
        self.keepdim = False
        self.axis = None
        self.shape = None

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype

    def set_keepdim(self, keepdim) -> None:
        self.keepdim = keepdim

    def set_axis(self, axis) -> None:
        self.axis = axis

    def set_shape(self, shape) -> None:
        self.shape = shape

    def get_rtol(self, flag):
        rtol = TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(x):
    return tensor.mean(x, axis=attrs.axis, keepdim=attrs.keepdim)


def expect_forward(inputs):
    return fn(inputs)


class TestCompositeMean(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float16", "float32", "float64"]
        self.keepdim = [False, True]
        self.shapes = [[16, 16, 64, 64], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite(self, inputs):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y = fn(x)
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that reduce_mean in original block
            self.assertTrue('reduce_mean' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that reduce_mean is split into small ops
            self.assertTrue('reduce_mean' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_forward(self):
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)

        expect = expect_forward(tensor_data).numpy()
        actual = self.cal_composite(np_data)[0]

        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    def test_forward(self):
        for i in self.axes:
            for j in self.dtypes:
                for t in self.shapes:
                    for k in self.keepdim:
                        # mean-kernel on cpu not support float16
                        if (
                            paddle.device.get_device() == "cpu"
                            and j == "float16"
                        ):
                            print("need pass this case")
                            continue
                        attrs.set_axis(i)
                        attrs.set_dtype(j)
                        attrs.set_shape(t)
                        attrs.set_keepdim(k)
                        self.compare_forward()


if __name__ == '__main__':
    unittest.main()

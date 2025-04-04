# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import utils

import paddle
from paddle.static import InputSpec


class TestTranspose(unittest.TestCase):
    def eval(self, dy_compute, inputs, input_spec=None):
        dy_out = dy_compute(*inputs)

        static_compute = utils.apply_to_static(
            dy_compute, use_cinn=True, input_spec=input_spec
        )
        st_out = static_compute(*inputs)

        for a, b in zip(
            paddle.utils.flatten(dy_out), paddle.utils.flatten(st_out)
        ):
            numpy.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)

    def test_small_021(self):
        def func(x):
            x = x.transpose([0, 2, 1])
            return x + 1

        x = paddle.uniform([13, 47, 29])

        self.eval(func, [x])

    def test_large_0231(self):
        def func(x):
            x = x.transpose([0, 2, 3, 1])
            return x * (x + 1)

        x = paddle.uniform([32, 128, 14, 14])

        self.eval(func, [x])

    def test_all_permuted_3210(self):
        def func(x):
            x = x.transpose([3, 2, 1, 0])
            return x * (x + 1)

        x = paddle.uniform([83, 67, 5, 109])

        self.eval(func, [x])

    def test_multi_upstream_2310(self):
        def func(x, y, z):
            u = x * x + y
            return u.transpose([2, 3, 1, 0]) + z

        x = paddle.uniform([100, 32, 7, 7])
        y = paddle.uniform([100, 32, 7, 7])
        z = paddle.uniform([32, 100])

        self.eval(func, [x, y, z])

    def test_multi_downstream_0231(self):
        def func(x, y):
            x = x.transpose([0, 2, 3, 1])
            z = x * 3.14
            return x * y + z, z

        x = paddle.uniform([32, 128, 14, 14])
        y = paddle.uniform([32, 14, 14, 128])

        self.eval(func, [x, y])

    def test_self_cross(self):
        def func(x):
            x_t = x.transpose([0, 2, 1])
            return x + x_t

        x = paddle.uniform([100, 200, 200])

        self.eval(func, [x])

    def test_unit_loop_0231(self):
        def func(x):
            return x.transpose([0, 2, 3, 1]) + 1

        x = paddle.uniform([128, 256, 1, 1])

        self.eval(func, [x])

    def test_unit_loop_0312(self):
        def func(x):
            return x.transpose([0, 3, 1, 2]) + 1

        x = paddle.uniform([128, 1, 1, 256])

        self.eval(func, [x])

    def test_unit_loop_210(self):
        def func(x):
            return x.transpose([2, 1, 0]) + 1

        x = paddle.uniform([1, 1024, 1])

        self.eval(func, [x])

    def test_large_0231_dynshape(self):
        def func(x):
            return x.transpose([0, 2, 3, 1]) + 1

        x = paddle.uniform([32, 128, 14, 14])
        x_spec = InputSpec([None, 128, None, None])

        self.eval(func, [x], [x_spec])

    def test_all_permuted_3210_dynshape(self):
        def func(x):
            x = x.transpose([3, 2, 1, 0])
            return x * (x + 1)

        x = paddle.uniform([32, 14, 14, 128])
        x_spec = InputSpec([None, None, None, None])

        self.eval(func, [x], [x_spec])

    def test_unit_loop_021_dynshape(self):
        def func(x):
            return x.transpose([0, 2, 1]) + 1

        x = paddle.uniform([1024, 1, 1])
        x_spec = InputSpec([None, None, 1])

        self.eval(func, [x], [x_spec])


if __name__ == "__main__":
    unittest.main()

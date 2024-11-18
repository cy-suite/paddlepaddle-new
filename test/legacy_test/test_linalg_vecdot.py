#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import sys
import unittest

import numpy as np

import paddle

if sys.platform == 'win32':
    RTOL = {'float32': 1e-02, 'float64': 1e-04}
    ATOL = {'float32': 1e-02, 'float64': 1e-04}
else:
    RTOL = {'float32': 1e-06, 'float64': 1e-15}
    ATOL = {'float32': 1e-06, 'float64': 1e-15}


class VecDotTestCase(unittest.TestCase):
    def setUp(self):
        self.init_config()
        self.generate_input()
        self.generate_expected_output()
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def generate_input(self):
        np.random.seed(123)
        self.x = np.random.random(self.input_shape).astype(self.dtype)
        self.y = np.random.random(self.input_shape).astype(self.dtype)

    def generate_expected_output(self):
        self.expected_output = np.sum(self.x * self.y, axis=self.axis)

    def init_config(self):
        self.dtype = 'float64'
        self.input_shape = (3, 4)
        self.axis = -1

    def test_dygraph(self):
        for place in self.places:
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x, place=place)
            y_tensor = paddle.to_tensor(self.y, place=place)
            result = paddle.vecdot(x_tensor, y_tensor, axis=self.axis)

            np.testing.assert_allclose(
                result.numpy(),
                self.expected_output,
                rtol=RTOL[self.dtype],
                atol=ATOL[self.dtype],
            )

    def test_static(self):
        paddle.enable_static()
        for place in self.places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="x", shape=self.input_shape, dtype=self.dtype
                )
                y = paddle.static.data(
                    name="y", shape=self.input_shape, dtype=self.dtype
                )

                result = paddle.vecdot(x, y, axis=self.axis)
                exe = paddle.static.Executor(place)
                output = exe.run(
                    feed={"x": self.x, "y": self.y},
                    fetch_list=[result],
                )[0]

            np.testing.assert_allclose(
                output,
                self.expected_output,
                rtol=RTOL[self.dtype],
                atol=ATOL[self.dtype],
            )


class VecDotTestCaseFloat32(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.input_shape = (3, 4)
        self.axis = -1


class VecDotTestCaseHigherDim(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.input_shape = (2, 3, 4)
        self.axis = -1


class VecDotTestCaseAxis(VecDotTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.input_shape = (3, 4, 5)
        self.axis = 1


class VecDotTestCaseError(unittest.TestCase):
    def test_axis_mismatch(self):
        with self.assertRaises(ValueError):
            x = paddle.rand([3, 4], dtype="float32")
            y = paddle.rand([3, 5], dtype="float32")
            paddle.vecdot(x, y, axis=-1)

    def test_dtype_mismatch(self):
        with self.assertRaises(TypeError):
            x = paddle.rand([3, 4], dtype="float32")
            y = paddle.rand([3, 4], dtype="int32")
            paddle.vecdot(x, y, axis=-1)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import unittest

import numpy as np
from utils import dygraph_guard, static_guard

import paddle
from paddle import base, static


class TestSubtractAPI(unittest.TestCase):
    def setUp(self):
        self.places = []
        self.places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

        self.dtypes = [
            'int16',
            'int32',
            'int64',
            'float32',
            'float64',
            'complex64',
            'complex128',
        ]
        self.shapes = [
            ((), ()),
            ((), (3,)),
            ((0,), (0,)),
            ((3,), (3,)),
            ((1,), (3,)),
            ((5,), (1,)),
            ((0,), ()),
            ((0, 3), (3,)),
            ((3, 1), (3, 4)),
            ((10, 15), (10, 15)),
            ((2, 0), (1, 0)),
            ((3, 4, 5), ()),
            ((1, 1, 1), (3, 4, 5)),
            ((1, 3, 5), (4, 1, 5)),
            ((0, 3, 1), (1, 3, 5)),
            ((1, 0, 2), (3, 0, 1)),
        ]

    def generate_random_data(self, shape, dtype):
        if np.issubdtype(np.dtype(dtype), np.integer):
            return np.random.randint(0, 100, size=shape).astype(dtype)
        elif np.issubdtype(np.dtype(dtype), np.floating):
            return np.random.uniform(-100, 100, size=shape).astype(dtype)
        elif np.issubdtype(np.dtype(dtype), np.complexfloating):
            real_part = np.random.uniform(-100, 100, size=shape)
            imag_part = np.random.uniform(-100, 100, size=shape)
            return (real_part + 1j * imag_part).astype(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def test_dygraph(self):
        with dygraph_guard():
            for shape, dtype in itertools.product(self.shapes, self.dtypes):
                x_np = self.generate_random_data(shape[0], dtype)
                y_np = self.generate_random_data(shape[1], dtype)
                expected = np.subtract(x_np, y_np)
                for place in self.places:
                    x = paddle.to_tensor(x_np, dtype=dtype, place=place)
                    y = paddle.to_tensor(y_np, dtype=dtype, place=place)
                    result = paddle.subtract(x, y)
                    np.testing.assert_allclose(
                        result, expected, rtol=1e-05, atol=1e-05
                    )

    def test_static(self):
        with static_guard():
            for shape, dtype in itertools.product(self.shapes, self.dtypes):
                x_np = self.generate_random_data(shape[0], dtype)
                y_np = self.generate_random_data(shape[1], dtype)
                expected = np.subtract(x_np, y_np)
                for place in self.places:
                    with static.program_guard(
                        static.Program(), static.Program()
                    ):
                        x = paddle.static.data(
                            name="x", shape=shape[0], dtype=dtype
                        )
                        y = paddle.static.data(
                            name="y", shape=shape[1], dtype=dtype
                        )
                        result = paddle.subtract(x, y)
                        exe = base.Executor(place=place)
                        fetches = exe.run(
                            feed={"x": x_np, "y": y_np},
                            fetch_list=[result],
                        )
                        np.testing.assert_allclose(
                            fetches[0], expected, rtol=1e-05, atol=1e-05
                        )


if __name__ == '__main__':
    unittest.main()

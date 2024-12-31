#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestZeroSizeParameter(unittest.TestCase):
    def setUp(self):
        self.parameter_dtypes = [
            'float16',
            'float32',
            'float64',
        ]
        self.zero_size_shapes = [
            [0, 4],
            [0, 0],
            [4, 0],
            [0, 5, 6],
            [6, 5, 0, 0],
            [0, 0, 0, 12],
        ]

    def test_create_parameter(self):
        for parameter_dtype in self.parameter_dtypes:
            for zero_size_shape in self.zero_size_shapes:

                class Model(paddle.nn.Layer):
                    def __init__(self) -> None:
                        super().__init__()
                        self.dummy_linear = paddle.nn.Linear(3, 4)
                        self.w = self.create_parameter(
                            shape=zero_size_shape, dtype=parameter_dtype
                        )

                model = Model()
                self.assertEqual(
                    model.w.shape,
                    zero_size_shape,
                    msg=f"error occurs at: {parameter_dtype}, {zero_size_shape}",
                )
                self.assertEqual(
                    model.w.data_ptr(),
                    0,
                    msg=f"error occurs at: {parameter_dtype}, {zero_size_shape}",
                )
                self.assertEqual(
                    str(model.w.place),
                    str(model.dummy_linear.weight.place),
                    msg=f"error occurs at: {parameter_dtype}, {zero_size_shape}",
                )
                self.assertEqual(
                    model.w.strides,
                    zero_size_shape,
                    msg=f"error occurs at: {parameter_dtype}, {zero_size_shape}",
                )
                self.assertEqual(
                    model.w.is_contiguous(),
                    True,
                    msg=f"error occurs at: {parameter_dtype}, {zero_size_shape}",
                )


class TestZeroSizeForward(unittest.TestCase):
    def test_forward_eager(self):
        pass


class TestZeroSizeBackward(unittest.TestCase): ...


if __name__ == '__main__':
    unittest.main()

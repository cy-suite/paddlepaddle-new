#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestBaddBmmUnderlineAPI(unittest.TestCase):
    def test_api_error(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)

        paddle.disable_static()

        def test_error1():
            data_x_wrong = np.ones((2, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error1)

        def test_error2():
            data_x_wrong = np.ones((2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error2)

        def test_error3():
            data_input_wrong = np.ones((2, 2, 2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error3)

        def test_error4():
            data_input_wrong = np.ones((2, 5)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error4)

        def test_error5():
            data_input_wrong = np.ones((3, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error5)

        def test_error6():
            data_input_wrong = np.ones((3, 2, 1)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error6)

        def test_error7():
            data_input_wrong = np.ones((1, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error7)

        def test_error_y1():
            data_y_wrong = np.ones((2, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error_y1)

        def test_error_y2():
            data_y_wrong = np.ones((2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error_y2)

        def test_error_y3():
            data_y_wrong = np.ones((1, 2, 3)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y_wrong)
            input = paddle.to_tensor(data_input)
            out = paddle.baddbmm_(input=input, x=x, y=y, beta=0.5, alpha=5.0)

        self.assertRaises(ValueError, test_error_y3)

        paddle.enable_static()

    def test_api_normal_1(self):
        data_x = np.ones((2, 2, 2)).astype(np.float32)
        data_y = np.ones((2, 2, 2)).astype(np.float32)
        data_input = np.ones((2, 2, 2)).astype(np.float32)
        data_alpha = 0.1
        data_beta = 1.0

        paddle.disable_static()

        x = paddle.to_tensor(data_x)
        y = paddle.to_tensor(data_y)
        input = paddle.to_tensor(data_input)

        numpy_output = data_beta * data_input + data_alpha * np.matmul(
            data_x, data_y
        )

        paddle_output = paddle.baddbmm_(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )

        np.testing.assert_allclose(
            numpy_output, paddle_output.numpy(), rtol=1e-05
        )

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

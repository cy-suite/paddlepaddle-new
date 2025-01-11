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

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


class TestBaddBmmOp(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.baddbmm
        self.public_python_api = paddle.baddbmm
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((3, 20, 15)).astype(self.dtype),
            'X': np.random.random((3, 20, 10)).astype(self.dtype),
            'Y': np.random.random((3, 10, 15)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input', 'X', 'Y'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_x(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_y(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )


class TestBaddBmmFP16Op(TestBaddBmmOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestBaddBmmBF16Op(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.python_api = paddle.baddbmm
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((3, 20, 15)).astype(self.dtype),
            'X': np.random.random((3, 20, 10)).astype(self.dtype),
            'Y': np.random.random((3, 10, 15)).astype(self.dtype),
        }
        self.outputs = {
            'Out': self.inputs['Input']
            + np.matmul(self.inputs['X'], self.inputs['Y'])
        }

        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def init_dtype_type(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad_with_place(self.place, ['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', no_grad_set=None
        )


class TestBaddBmmOp2(TestBaddBmmOp):
    # test alpha and beta
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.baddbmm
        self.public_python_api = paddle.baddbmm
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((3, 20, 15)).astype(self.dtype),
            'X': np.random.random((3, 20, 10)).astype(self.dtype),
            'Y': np.random.random((3, 10, 15)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.1,
            'Beta': 1.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }


class TestBaddBmmOp3(OpTest):
    def setUp(self):
        self.op_type = "baddbmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.baddbmm
        self.public_python_api = paddle.baddbmm
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((3, 20, 15)).astype(self.dtype),
            'X': np.random.random((3, 20, 10)).astype(self.dtype),
            'Y': np.random.random((3, 10, 15)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }
        self.outputs = {
            'Out': self.attrs['Beta'] * self.inputs['Input']
            + self.attrs['Alpha']
            * np.matmul(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['Input', 'X', 'Y'], 'Out', check_pir=True, check_prim_pir=True
        )

    def test_check_grad_x(self):
        self.check_grad(
            ['X'], 'Out', no_grad_set=None, check_pir=True, check_prim_pir=True
        )

    def test_check_grad_y(self):
        self.check_grad(
            ['Y'], 'Out', no_grad_set=None, check_pir=True, check_prim_pir=True
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['Input'],
            'Out',
            no_grad_set=None,
            check_pir=True,
            check_prim_pir=True,
        )


class TestBaddBmmAPI(unittest.TestCase):
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
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error1)

        def test_error2():
            data_x_wrong = np.ones((2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x_wrong)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error2)

        def test_error3():
            data_input_wrong = np.ones((2, 2, 2, 2)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error3)

        def test_error4():
            data_input_wrong = np.ones((2, 5)).astype(np.float32)
            x = paddle.to_tensor(data_x)
            y = paddle.to_tensor(data_y)
            input = paddle.to_tensor(data_input_wrong)
            out = paddle.tensor.baddbmm(
                input=input, x=x, y=y, beta=0.5, alpha=5.0
            )

        self.assertRaises(ValueError, test_error4)

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
        paddle_output = paddle.tensor.baddbmm(
            input=input, x=x, y=y, beta=data_beta, alpha=data_alpha
        )
        numpy_output = data_beta * data_input + data_alpha * np.matmul(
            data_x, data_y
        )

        np.testing.assert_allclose(
            numpy_output, paddle_output.numpy(), rtol=1e-05
        )

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

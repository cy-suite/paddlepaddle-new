#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np
from op_test import OpTest

import paddle

class TestAddOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.init_inputs()
    
    def init_inputs(self):
        x = np.random.random([10, 10]).astype("float64")
        y = np.random.random([10, 10]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.attrs = {'axis': -1}
        self.outputs = {'Out': x + y}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_fp64(self):
        self.check_grad(['X', 'Y'], 'Out')

class TestElementwiseAddWithAxis0(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.init_config()

    def init_config(self):
        self.inputs = {
            'X': np.random.rand(10, 10, 10).astype("float64"), 
            'Y': np.random.rand(10, 10, 1).astype("float64")  
        }
        self.attrs = {'axis': 0}
        self.outputs = {'Out': self.inputs['X'] + self.inputs['Y']}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_fp64(self):
        self.check_grad(['X', 'Y'], 'Out')

class TestElementwiseAddBroadcast(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.init_config()

    def init_config(self):
        self.inputs = {
            'X': np.random.rand(10, 10, 10).astype("float64"),  
            'Y': np.random.rand(10, 10).astype("float64")     
        }
        self.attrs = {}
        self.outputs = {'Out': self.inputs['X'] + self.inputs['Y']}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_fp64(self):
        self.check_grad(['X', 'Y'], 'Out')

class TestElementwiseAdd0D(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.init_config()

    def init_config(self):
        self.inputs = {
            'X': np.array(0.5).astype("float64"),  
            'Y': np.array(1.5).astype("float64")
        }
        self.attrs = {}
        self.outputs = {'Out': self.inputs['X'] + self.inputs['Y']}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

class TestAddnOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.l = 32
        self.x_np = np.random.random([self.l, 16, 256])

    def check_main(self, x_np, dtype, axis=None, mixed_dtype=False):
        paddle.disable_static()
        x = []
        for i in range(x_np.shape[0]):
            if mixed_dtype and i == 0:
                val = paddle.to_tensor(x_np[i].astype('float32'))
            else:
                val = paddle.to_tensor(x_np[i].astype(dtype))
            val.stop_gradient = False
            x.append(val)

        y = paddle.add_n(x)
        x_g = paddle.grad(y, x)
        y_np = y.numpy().astype(dtype)
        x_g_np = []
        for val in x_g:
            x_g_np.append(val.numpy().astype(dtype))
        paddle.enable_static()
        return y_np, x_g_np

    def test_add_n_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        y_np_16, x_g_np_16 = self.check_main(self.x_np, 'float16')
        y_np_32, x_g_np_32 = self.check_main(self.x_np, 'float32')

        np.testing.assert_allclose(y_np_16, y_np_32, rtol=1e-03)
        for i in range(len(x_g_np_32)):
            np.testing.assert_allclose(x_g_np_16[i], x_g_np_32[i], rtol=1e-03)

    def test_add_n_fp16_mixed_dtype(self):
        if not paddle.is_compiled_with_cuda():
            return
        y_np_16, x_g_np_16 = self.check_main(
            self.x_np, 'float16', mixed_dtype=True
        )
        y_np_32, x_g_np_32 = self.check_main(self.x_np, 'float32')

        np.testing.assert_allclose(y_np_16, y_np_32, rtol=1e-03)
        for i in range(len(x_g_np_32)):
            np.testing.assert_allclose(x_g_np_16[i], x_g_np_32[i], rtol=1e-03)

    def test_add_n_api(self):
        if not paddle.is_compiled_with_cuda():
            return
        dtypes = ['float32', 'complex64', 'complex128']
        for dtype in dtypes:
            if dtype == 'complex64' or dtype == 'complex128':
                self.x_np = (
                    np.random.random([self.l, 16, 256])
                    + 1j * np.random.random([self.l, 16, 256])
                ).astype(dtype)

            y_np_32, x_g_np_32 = self.check_main(self.x_np, dtype)
            y_np_gt = np.sum(self.x_np, axis=0).astype(dtype)
            np.testing.assert_allclose(y_np_32, y_np_gt, rtol=1e-06)


if __name__ == "__main__":
    unittest.main()

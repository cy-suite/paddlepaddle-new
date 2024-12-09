# # Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import unittest

# import numpy as np
# from op_test import OpTest

# import paddle
# from paddle import base
# from paddle.base import core


# class TestClipTensorOp(OpTest):
#     def setUp(self):
#         self.max_relative_error = 0.006
#         self.op_type = "clip_tensor"
#         self.python_api = paddle.tensor.math.clip_tensor
#         self.initTestCase()
#         self.x = np.random.random(size=self.shape).astype(self.dtype)
#         self.min = np.full(self.shape, self.min_value).astype(self.dtype)
#         self.max = np.full(self.shape, self.max_value).astype(self.dtype)
#         self.x[np.abs(self.x - self.min) < self.max_relative_error] = 0.5
#         self.x[np.abs(self.x - self.max) < self.max_relative_error] = 0.5

#         self.inputs = {'x': self.x, 'min': self.min, 'max': self.max}
#         out = np.clip(self.x, self.min, self.max)
#         self.outputs = {'out': out}

#     def initTestCase(self):
#         self.dtype = np.float64
#         self.shape = (10, 10)
#         self.min_value = 0.3
#         self.max_value = 0.8

#     def test_check_output(self):
#         self.check_output(check_pir=True)

#     def test_check_grad(self):
#         self.check_grad(['x'], 'out', check_pir=True)


# class TestCase1(TestClipTensorOp):
#     def initTestCase(self):
#         self.dtype = np.float64
#         self.shape = (8, 16, 8)
#         self.min_value = 0.3
#         self.max_value = 0.8


# class TestCase2(TestClipTensorOp):
#     def initTestCase(self):
#         self.dtype = np.float64
#         self.shape = (8, 16, 8)
#         self.min_value = 0.3
#         self.max_value = 0.8


# class TestCase3(TestClipTensorOp):
#     def initTestCase(self):
#         self.dtype = np.float64
#         self.shape = (8, 16, 8)
#         self.min_value = 0.0
#         self.max_value = 1.0


# @unittest.skipIf(
#     not core.is_compiled_with_cuda()
#     or not core.is_bfloat16_supported(core.CUDAPlace(0)),
#     "core is not compiled with CUDA.",
# )
# class TestGPUClipTensorOp(OpTest):
#     def setUp(self):
#         self.max_relative_error = 0.006
#         self.op_type = "clip_tensor"
#         self.python_api = paddle.tensor.math.clip_tensor
#         self.initTestCase()
#         self.x = np.random.random(size=self.shape).astype(self.dtype)
#         self.min = np.full(self.shape, self.min_value).astype(self.dtype)
#         self.max = np.full(self.shape, self.max_value).astype(self.dtype)
#         self.x[np.abs(self.x - self.min) < self.max_relative_error] = 0.5
#         self.x[np.abs(self.x - self.max) < self.max_relative_error] = 0.5

#         self.inputs = {'x': self.x, 'min': self.min, 'max': self.max}
#         out = np.clip(self.x, self.min, self.max)
#         self.outputs = {'out': out}

#     def initTestCase(self):
#         self.dtype = np.float64
#         self.shape = (10, 10)
#         self.min_value = 0.3
#         self.max_value = 0.8

#     def test_check_output(self):
#         self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

#     def test_check_grad(self):
#         self.check_grad_with_place(core.CUDAPlace(0), ['x'], 'out', check_pir=True)

# if __name__ == '__main__':
#     unittest.main()
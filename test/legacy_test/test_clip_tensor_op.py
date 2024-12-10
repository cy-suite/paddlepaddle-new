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
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


class TestClipTensorOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.op_type = "clip_tensor"
        self.python_api = paddle.tensor.math.clip_tensor

        self.initTestCase()

        self.x = np.random.random(size=self.shape).astype(self.dtype)
        self.min = np.full(self.shape, 0.3).astype(self.dtype)
        self.max = np.full(self.shape, 0.8).astype(self.dtype)
        self.x[np.abs(self.x - self.min) < self.max_relative_error] = 0.5
        self.x[np.abs(self.x - self.max) < self.max_relative_error] = 0.5

        self.inputs = {'x': self.x, 'min': self.min, 'max': self.max}
        out = np.clip(self.x, self.min, self.max)
        self.outputs = {'out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False, check_cinn=True)

    def test_check_grad(self):
        self.check_grad(['x'], 'out', check_pir=True, check_cinn=True)

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (10, 10)


class TestCase1(TestClipTensorOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (10, 4, 5)

if __name__ == '__main__':
    unittest.main()

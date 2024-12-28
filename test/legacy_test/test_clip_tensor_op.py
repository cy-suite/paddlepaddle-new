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


def clip_tensor_wrapper(x, min, max):
    return paddle._C_ops.clip_tensor(x, min, max)

class TestClipTensorOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.python_api = clip_tensor_wrapper

        self.dtype = np.float64
        self.initTestCase()

        self.op_type = "clip"
        self.inputs['x'] = np.random.random(self.shape).astype('float64')
        self.inputs['min'] = np.full(self.min_shape, self.min_value).astype('float64')
        self.inputs['max'] = np.full(self.max_shape, self.max_value).astype('float64')
        min_v = self.inputs['min']
        max_v = self.inputs['max']

        self.inputs['x'][np.abs(self.inputs['x'] - min_v) < self.max_relative_error] = 0.5
        self.inputs['x'][np.abs(self.inputs['x'] - max_v) < self.max_relative_error] = 0.5

        self.outputs = {'out': np.clip(self.inputs['x'], min_v, max_v)}

    def test_check_output(self):
        self.check_output_()

    def test_check_grad_normal(self):
        self.check_grad(place, ['x'], 'out')

    def initTestCase(self):
        self.x_shape = (10, 1, 10)
        self.max_shape = (1, 10)
        self.min_shape = (10)
        self.min_value = 0.8
        self.max_value = 0.3


class TestCase1(TestClipOp):
    def initTestCase(self):
        self.x_shape = (8, 16, 8)
        self.max_shape = (1, 8)
        self.min_shape = (16, 8)
        self.max_value = 0.7
        self.min_value = 0.0


class TestCase2(TestClipOp):
    def initTestCase(self):
        self.x_shape = (8, 16)
        self.max_shape = (8, 8, 16)
        self.min_shape = (16)
        self.max_value = 1.0
        self.min_value = 0.0


class TestCase3(TestClipOp):
    def initTestCase(self):
        self.x_shape = (8, 16)
        self.max_shape = (16)
        self.min_shape = (4, 8, 16)
        self.max_value = 0.7
        self.min_value = 0.2


if __name__ == '__main__':
    unittest.main()
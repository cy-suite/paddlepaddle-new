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

import paddle


class TestClipTensorAPI(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.init_input()
    
    def init_input(self):
        self.x = np.random.random([5, 9, 4]).astype('float32')
        self.min1 = np.random.random([5, 9, 4]).astype('float32')
        self.max1 = np.random.random([5, 9, 4]).astype('float32')
    
    def test_static_api(self):
        x_shape = self.x.shape
        min_shape = self.min1.shape
        max_shape = self.max1.shape

        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                'x', x_shape, dtype=self.x.dtype
            )
            min = paddle.static.data(
                'min', min_shape, dtype=self.min1.dtype
            )
            max = paddle.static.data(
                'max', max_shape, dtype=self.max1.dtype
            )
            out = paddle.clip(x, min, max)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.x, 'min': self.min1, 'max': self.max1},
                fetch_list=[out],
            )
            np_out = np.clip(self.x, self.min1, self.max1)
            np.allclose(res[0], np_out)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        min = paddle.to_tensor(self.min1)
        max = paddle.to_tensor(self.max1)
        out = paddle.clip(x, min, max)
        np_out = np.clip(self.x, self.min1, self.max1)
        np.allclose(np_out, out.numpy())
        paddle.enable_static()
    
    def test_dygraph_api_inplace(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        min = paddle.to_tensor(self.min1)
        max = paddle.to_tensor(self.max1)
        x.clip_(min, max)
        np_out = np.clip(self.x, self.min1, self.max1)
        np.allclose(np_out, x.numpy())
        paddle.enable_static()


class TestClipTensorAPICase1(TestClipTensorAPI):
    def init_input(self):
        self.x = np.random.random([5, 9, 4]).astype('float32')
        self.min1 = np.random.random([9, 4]).astype('float32')
        self.max1 = np.random.random([4]).astype('float32')


class TestClipTensorAPICase2(TestClipTensorAPI):
    def init_input(self):
        self.x = np.random.random([5, 9, 4]).astype('float64')
        self.min1 = np.random.random([1]).astype('float64')
        self.max1 = np.random.random([4]).astype('float64')


class TestClipTensorAPICase3(TestClipTensorAPI):
    def init_input(self):
        self.x = np.random.random([5, 9, 4]).astype('int32')
        self.min1 = np.random.random([1]).astype('int32')
        self.max1 = np.random.random([4]).astype('int32')


class TestClipTensorAPICase4(TestClipTensorAPI):
    def init_input(self):
        self.x = np.random.random([5, 9, 4]).astype('float32')
        self.min1 = np.random.random([4]).astype('float32')
        self.max1 = np.random.random([9, 4]).astype('float32')


if __name__ == '__main__':
    unittest.main()

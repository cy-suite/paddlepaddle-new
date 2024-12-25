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
from paddle import base
from paddle.base import core


def np_pd_equal(x_shape, min_shape=None, max_shape=None, dtype='float32'):
    paddle.disable_static()
    x = np.random.randn(*x_shape).astype(dtype)
    max = np.random.randn(*max_shape).astype(dtype)
    min = np.random.randn(*min_shape).astype(dtype)
    np_out = np.clip(x, min, max)
    x_pd = paddle.to_tensor(x, dtype=dtype)
    min_pd = paddle.to_tensor(min, dtype=dtype)
    max_pd = paddle.to_tensor(max, dtype=dtype)
    pd_out = paddle.clip(x_pd, min_pd, max_pd)
    np.allclose(pd_out.numpy(), np_out)


def np_pd_static_equal(
    x_shape, min_shape=None, max_shape=None, dtype='float32'
):
    paddle.enable_static()
    x = np.random.randn(*x_shape).astype(dtype)
    max = np.random.randn(*max_shape).astype(dtype)
    min = np.random.randn(*min_shape).astype(dtype)
    np_out = np.clip(x, min, max)

    place = base.CPUPlace()
    if core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)

    with paddle.static.program_guard(
        paddle.static.Program(), paddle.static.Program()
    ):
        x_pd = paddle.static.data("x", shape=x_shape, dtype=dtype)
        min_pd = paddle.static.data("min", shape=min_shape, dtype=dtype)
        max_pd = paddle.static.data("max", shape=max_shape, dtype=dtype)
        pd_out = paddle.clip(x_pd, min_pd, max_pd)
        exe = base.Executor(place)
        (res,) = exe.run(
            feed={"x": x, "min": min, "max": max}, fetch_list=[pd_out]
        )
        np.allclose(res, np_out)

    paddle.disable_static()


class TestClipTensorAPI(unittest.TestCase):

    def test_check_output_int32(self):
        np_pd_equal([4, 5], [5], [1], 'int32')

    def test_check_output_float32(self):
        np_pd_equal([4], [5, 4], [4], 'float32')

    def test_check_output_int64(self):
        np_pd_equal([4, 5], [5], [4, 5], 'int64')

    def test_check_output_Nonemin(self):
        paddle.disable_static()
        x = np.random.randn(4, 5).astype('float32')
        max = np.random.randn(4, 4, 5).astype('float32')
        min = float(np.finfo(np.float32).min)
        np_out = np.clip(x, min, max)
        x_pd = paddle.to_tensor(x, dtype='float32')
        max_pd = paddle.to_tensor(max, dtype='float32')
        pd_out = paddle.clip(x_pd, None, max_pd)
        np.allclose(pd_out.numpy(), np_out)

    def test_check_static_output_int32(self):
        np_pd_static_equal([4], [5, 4], [6, 5, 4], 'int32')

    def test_check_static_output_int64(self):
        np_pd_static_equal([4, 5], [5], [4, 5], 'int64')

    def test_check_static_output_float32(self):
        np_pd_static_equal([4], [5, 4], [4], 'float32')

    def test_check_static_output_Nonemin(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            x = np.random.randn(4, 5).astype('float32')
            max = np.random.randn(4, 4, 5).astype('float32')
            min = float(np.finfo(np.float32).min)
            np_out = np.clip(x, min, max)

            place = paddle.CPUPlace()
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            x_pd = paddle.static.data("x", shape=[4, 5], dtype='float32')
            max_pd = paddle.static.data("max", shape=[4, 4, 5], dtype='float32')
            pd_out = paddle.clip(x_pd, None, max_pd)
            exe = base.Executor(place)
            res = exe.run(feed={'x': x, 'max': max}, fetch_list=[pd_out])
            np.allclose(res[0], np_out)
        paddle.disable_static()

    def test_fp16(self):
        if base.core.is_compiled_with_cuda():
            paddle.enable_static()
            data_shape = [1, 9, 9, 4]
            data = np.random.random(data_shape).astype('float16')
            min1 = np.random.random(data_shape).astype('float16')
            max2 = np.random.random(data_shape).astype('float16')

            with paddle.static.program_guard(paddle.static.Program()):
                images = paddle.static.data(
                    name='x', shape=data_shape, dtype='float16'
                )
                min = paddle.static.data(
                    name='min', shape=data_shape, dtype='float16'
                )
                max = paddle.static.data(
                    name='max', shape=data_shape, dtype='float16'
                )
                out = paddle.tensor.math.clip_tensor(images, min, max)
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                res1 = exe.run(
                    feed={
                        "x": data,
                        "min": min1,
                        "max": max2,
                    },
                    fetch_list=[out],
                )
            paddle.disable_static()


class TestClipTensor_API(unittest.TestCase):
    def setUp(self):
        self.x_shape = [4, 5, 5]
        self.min_shape = [5, 5]
        self.max_shape = [4, 5, 5]

    def test_check_output(self):
        paddle.disable_static()
        x = np.random.randn(*self.x_shape).astype('float32')
        max_np = np.random.randn(*self.max_shape).astype('float32')
        min_np = np.random.randn(*self.min_shape).astype('float32')
        np_out = np.clip(x, min_np, max_np)
        x_pd = paddle.to_tensor(x, dtype='float32')
        min_pd = paddle.to_tensor(min_np, dtype='float32')
        max_pd = paddle.to_tensor(max_np, dtype='float32')
        paddle.clip_(x_pd, min_pd, max_pd)
        np.allclose(x_pd.numpy(), np_out)
        paddle.enable_static()

    def test_check_error_shape(self):
        paddle.disable_stataic()
        with self.assertRaises(ValueError):
            x_pd = paddle.randn([4], dtype='float32')
            min_pd = paddle.randn([4, 4, 5], dtype='float32')
            max_pd = paddle.randn([4, 4, 5], dtype='float32')
            paddle.clip_(x_pd, min_pd, max_pd)
        paddle.enable_static()

    def test_check_None(self):
        paddle.disable_static()
        x = np.random.randn(4, 5, 5).astype('float32')
        max_np = np.random.randn(5, 5).astype('float32')
        min_np = float(np.finfo(np.float32).min)
        np_out = np.clip(x, min_np, max_np)
        x_pd = paddle.to_tensor(x, dtype='float32')
        max_pd = paddle.to_tensor(max_np, dtype='float32')
        min_pd = paddle.to_tensor(min_np, dtype='float32')
        paddle.clip_(x_pd, min_pd, max_pd)
        np.allclose(x_pd.numpy(), np_out)

        x = np.random.randn(4, 5, 5).astype('float32')
        max_np = float(np.finfo(np.float32).max)
        min_np = np.random.randn(5, 5).astype('float32')
        np_out = np.clip(x, min_np, max_np)
        x_pd = paddle.to_tensor(x, dtype='float32')
        max_pd = paddle.to_tensor(max_np, dtype='float32')
        min_pd = paddle.to_tensor(min_np, dtype='float32')
        paddle.clip_(x_pd, min_pd, max_pd)
        np.allclose(x_pd.numpy(), np_out)
        paddle.enable_static()


class TestClipTensor_API1(TestClipTensor_API):
    def setUp(self):
        self.x_shape = [4, 5, 5]
        self.min_shape = [5]
        self.max_shape = [5, 5]


class TestClipTensor_API2(TestClipTensor_API):
    def setUp(self):
        self.x_shape = [9, 5, 5]
        self.min_shape = [5, 5]
        self.max_shape = [9, 5, 5]


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

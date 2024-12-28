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

import math
import unittest

import numpy as np

import paddle
from paddle import base


class TestClipTensorAPI(unittest.TestCase):
    def set_up(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10]
        self.max_shape = [10]
        self.data_type = 'float32'

    def check_dygraph_api(self):
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        paddle.disable_static(place)
        x = np.random.random(self.x_shape).astype(self.data_type)
        min = (
            np.random.random(self.min_shape).astype(self.data_type)
            if self.min_shape is not None
            else self.min_shape
        )
        max = (
            np.random.random(self.max_shape).astype(self.data_type)
            if self.max_shape is not None
            else self.max_shape
        )
        out_np = x.clip(min, max)
        x_pd = paddle.to_tensor(x, dtype=self.data_type)
        min_pd = (
            paddle.to_tensor(min, dtype=self.data_type)
            if min is not None
            else min
        )
        max_pd = (
            paddle.to_tensor(max, dtype=self.data_type)
            if max is not None
            else max
        )
        out_pd = paddle.clip(x_pd, min_pd, max_pd)
        np.testing.assert_allclose(out_pd.numpy(), out_np, rtol=1e-05)

        min_new_shape = self.min_shape if self.min_shape is not None else [1]
        max_new_shape = self.max_shape if self.max_shape is not None else [1]
        if math.prod(self.x_shape) >= math.prod(min_new_shape) and math.prod(
            self.x_shape
        ) >= math.prod(max_new_shape):
            x_pd.clip_(min_pd, max_pd)
            np.testing.assert_allclose(x_pd.numpy(), out_np, rtol=1e-05)

    def check_static_api(self):
        paddle.enable_static()
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        main = paddle.static.Program()
        startup = paddle.static.Program()
        x = np.random.random(self.x_shape).astype(self.data_type)
        min = (
            np.random.random(self.min_shape).astype(self.data_type)
            if self.min_shape is not None
            else self.min_shape
        )
        max = (
            np.random.random(self.max_shape).astype(self.data_type)
            if self.max_shape is not None
            else self.max_shape
        )
        out_np = x.clip(min, max)
        with paddle.static.program_guard(main, startup):
            x_pd = paddle.static.data(
                name='x', shape=self.x_shape, dtype=self.data_type
            )
            min_pd = (
                paddle.static.data(
                    name='min', shape=self.min_shape, dtype=self.data_type
                )
                if self.min_shape is not None
                else self.min_shape
            )
            max_pd = (
                paddle.static.data(
                    name='max', shape=self.max_shape, dtype=self.data_type
                )
                if self.max_shape is not None
                else self.max_shape
            )
            out_pd = paddle.clip(x_pd, min_pd, max_pd)
            (res) = exe.run(
                main, feed={'x': x, 'min': min, 'max': max}, fetch_list=[out_pd]
            )
            np.testing.assert_allclose(res, out_np, rtol=1e-05)

    def test_fp16(self):
        if base.core.is_compiled_with_cuda():
            paddle.enable_static()
            x = np.random.random(self.x_shape).astype('float16')
            min = (
                np.random.random(self.min_shape).astype('float16')
                if self.min_shape is not None
                else self.min_shape
            )
            max = (
                np.random.random(self.max_shape).astype('float16')
                if self.max_shape is not None
                else self.max_shape
            )
            out_np = x.clip(min, max)

            with paddle.static.program_guard(paddle.static.Program()):
                x_pd = paddle.static.data(
                    name='x', shape=self.x_shape, dtype='float16'
                )
                min_pd = (
                    paddle.static.data(
                        name='min', shape=self.min_shape, dtype='float16'
                    )
                    if self.min_shape is not None
                    else self.min_shape
                )
                max_pd = (
                    paddle.static.data(
                        name='max', shape=self.max_shape, dtype='float16'
                    )
                    if self.max_shape is not None
                    else self.max_shape
                )
                out_pd = paddle.clip(x_pd, min_pd, max_pd)
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                (res) = exe.run(
                    feed={
                        "x": x,
                        "min": min,
                        "max": max,
                    },
                    fetch_list=[out_pd],
                )
                np.testing.assert_allclose(res, out_np, rtol=1e-05)

    def test_shape_error(self):
        paddle.disable_static()
        with self.assertRaises(ValueError):
            x = np.random.random([10])
            min = np.random.random([10, 1, 10])
            max = np.random.random([10, 1, 10])
            x = paddle.to_tensor(x)
            min = paddle.to_tensor(min)
            max = paddle.to_tensor(max)
            x.clip_(min, max)


class TestClipTensorAPI1(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10, 1, 10]
        self.max_shape = [10]
        self.data_type = 'float32'


class TestClipTensorAPI2(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10]
        self.min_shape = [10, 1, 10]
        self.max_shape = [10, 1, 10]
        self.data_type = 'float32'


class TestClipTensorAPI3(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10]
        self.min_shape = [1, 10]
        self.max_shape = [10, 1, 10]
        self.data_type = 'float32'


class TestClipTensorAPI4(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = [10, 1, 10]
        self.max_shape = [10]
        self.data_type = 'int32'


class TestClipTensorAPI5(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10]
        self.min_shape = [10, 1, 10]
        self.max_shape = [10, 1, 10]
        self.data_type = 'int32'


class TestClipTensorAPI6(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10]
        self.min_shape = [1, 10]
        self.max_shape = [10, 1, 10]
        self.data_type = 'int32'


class TestClipTensorAPI7(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10, 1, 10]
        self.min_shape = None
        self.max_shape = [10]
        self.data_type = 'float32'


class TestClipTensorAPI8(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10]
        self.min_shape = None
        self.max_shape = [10, 1, 10]
        self.data_type = 'int32'


class TestClipTensorAPI9(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10]
        self.min_shape = [10, 1, 10]
        self.max_shape = None
        self.data_type = 'int64'


class TestClipTensorAPI10(TestClipTensorAPI):
    def set_up(self):
        self.x_shape = [10]
        self.min_shape = [1, 10]
        self.max_shape = [10, 1, 10]
        self.data_type = 'int64'


if __name__ == '__main__':
    unittest.main()
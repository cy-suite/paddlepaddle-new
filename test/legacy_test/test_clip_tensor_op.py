#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TestClipTensor(unittest.TestCase):
    def test_static_clip(self):
        data_shape = [1, 2, 3, 4]
        self.place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        max_data = np.random.uniform(1.5, 2.5, data_shape[-3:]).astype(
            'float32'
        )
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=data_shape, dtype='float32')
            min = paddle.static.data(
                name='min', shape=data_shape[-2:], dtype='float32'
            )
            max = paddle.static.data(
                name='max', shape=data_shape[-3:], dtype='float32'
            )
            out = paddle.clip(x, min, max)
            exe = base.Executor(self.place)
            res = exe.run(
                feed={
                    "x": data,
                    'min': min_data,
                    'max': max_data,
                },
                fetch_list=[out],
            )
            res_np = np.clip(data, min_data, max_data)
            np.testing.assert_allclose(res_np, res[0], rtol=1e-05)
        paddle.disable_static()

        data_shape = [1, 2, 3, 4]
        self.place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )

        data = np.random.uniform(-2.0, 7.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=data_shape, dtype='float32')
            min = paddle.static.data(
                name='min', shape=data_shape[-2:], dtype='float32'
            )
            max = 5.0
            out = paddle.clip(x, min, max)
            exe = base.Executor(self.place)
            res = exe.run(
                feed={
                    "x": data,
                    'min': min_data,
                    'max': 5.0,
                },
                fetch_list=[out],
            )
            res_np = np.clip(data, min_data, 5.0)
            np.testing.assert_allclose(res_np, res[0], rtol=1e-05)
        paddle.disable_static()

        data_shape = [1, 2, 3, 4]
        self.place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        max_data = float(np.finfo(np.float32).max)
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=data_shape, dtype='float32')
            min = paddle.static.data(
                name='min', shape=data_shape[-2:], dtype='float32'
            )
            out = paddle.clip(x, min)
            exe = base.Executor(self.place)
            res = exe.run(
                feed={"x": data, 'min': min_data},
                fetch_list=[out],
            )
            res_np = np.clip(data, min_data, max_data)
            np.testing.assert_allclose(res_np, res[0], rtol=1e-05)
        paddle.disable_static()

        data_shape = [1, 2, 3, 4]
        self.place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, []).astype('float32')
        max_data = np.random.uniform(1.5, 2.5, []).astype('float32')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=data_shape, dtype='float32')
            min = paddle.static.data(name='min', shape=[], dtype='float32')
            max = paddle.static.data(name='max', shape=[], dtype='float32')
            out = paddle.clip(x, min, max)
            exe = base.Executor(self.place)
            res = exe.run(
                feed={"x": data, 'min': min_data, 'max': max_data},
                fetch_list=[out],
            )
            res_np = np.clip(data, min_data, max_data)
            np.testing.assert_allclose(res_np, res[0], rtol=1e-05)
        paddle.disable_static()

        data_shape = [1, 2, 3, 4]
        self.place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, [1]).astype('float32')
        max_data = np.random.uniform(1.5, 2.5, [1]).astype('float32')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=data_shape, dtype='float32')
            min = paddle.static.data(name='min', shape=[1], dtype='float32')
            max = paddle.static.data(name='max', shape=[1], dtype='float32')
            out = paddle.clip(x, min, max)
            exe = base.Executor(self.place)
            res = exe.run(
                feed={"x": data, 'min': min_data, 'max': max_data},
                fetch_list=[out],
            )
            res_np = np.clip(data, min_data, max_data)
            np.testing.assert_allclose(res_np, res[0], rtol=1e-05)
        paddle.disable_static()

        if base.core.is_compiled_with_cuda():
            paddle.enable_static()
            data_shape = [1, 9, 9, 4]
            data = np.random.random(data_shape).astype('float16')

            with paddle.static.program_guard(paddle.static.Program()):
                images = paddle.static.data(
                    name='image1', shape=data_shape, dtype='float16'
                )
                min = paddle.static.data(
                    name='min1', shape=[1], dtype='float16'
                )
                max = paddle.static.data(
                    name='max1', shape=[1], dtype='float16'
                )
                out = paddle.clip(images, min, max)
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                res1 = exe.run(
                    feed={
                        "image1": data,
                        "min1": np.array([0.2]).astype('float16'),
                        "max1": np.array([0.8]).astype('float16'),
                    },
                    fetch_list=[out],
                )
            paddle.disable_static()

    def test_dygraph_clip(self):
        self.place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        paddle.disable_static(self.place)
        data_shape = [1, 2, 3, 4]
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        max_data = np.random.uniform(1.5, 2.5, data_shape[-3:]).astype(
            'float32'
        )
        out_np = np.clip(data, min_data, max_data)
        data = paddle.to_tensor(data)
        min_data = paddle.to_tensor(min_data)
        max_data = paddle.to_tensor(max_data)
        out = paddle.clip(data, min_data, max_data)
        np.testing.assert_allclose(out.numpy(), out_np, rtol=1e-05)

        data_shape = [1, 2, 3, 4]
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        max_data = np.random.uniform(1.5, 2.5, data_shape[-1:]).astype(
            'float32'
        )
        out_np = np.clip(data, min_data, max_data)
        data = paddle.to_tensor(data)
        min_data = paddle.to_tensor(min_data)
        max_data = paddle.to_tensor(max_data)
        out = paddle.clip(data, min_data, max_data)
        np.testing.assert_allclose(out.numpy(), out_np, rtol=1e-05)

        data_shape = [1, 2, 3, 4]
        data = np.random.uniform(-2.0, 7.0, data_shape).astype('int32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'int32'
        )
        max_data = 5
        out_np = np.clip(data, min_data, max_data)
        data = paddle.to_tensor(data, dtype='int32')
        min_data = paddle.to_tensor(min_data, dtype='int32')
        max_data = paddle.to_tensor([max_data], dtype='int32')
        out = paddle.clip(data, min_data, max_data)
        np.testing.assert_allclose(out.numpy(), out_np, rtol=1e-05)

        data_shape = [1, 2, 3, 4]
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        max_data = float(np.finfo(np.float32).max)
        out_np = np.clip(data, min_data, max_data)
        data = paddle.to_tensor(data)
        min_data = paddle.to_tensor(min_data)
        out = paddle.clip(data, min_data)
        np.testing.assert_allclose(out.numpy(), out_np, rtol=1e-05)

        data_shape = [1, 2, 3, 4]
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        max_data = 5
        out_np = np.clip(data, min_data, max_data)
        data = paddle.to_tensor(data)
        min_data = paddle.to_tensor(min_data)
        out = paddle.clip(data, min_data, max_data)
        np.testing.assert_allclose(out.numpy(), out_np, rtol=1e-05)

        if base.core.is_compiled_with_cuda():
            data_shape = [1, 2, 3, 4]
            data = np.random.uniform(-2.0, 2.0, data_shape).astype('float16')
            min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
                'float16'
            )
            max_data = np.random.uniform(1.5, 2.5, data_shape[-1:]).astype(
                'float16'
            )
            out_np = np.clip(data, min_data, max_data)
            data = paddle.to_tensor(data)
            min_data = paddle.to_tensor(min_data)
            max_data = paddle.to_tensor(max_data)
            out = paddle.clip(data, min_data, max_data)
            np.testing.assert_allclose(out.numpy(), out_np, rtol=1e-05)

            data_shape = [1, 2, 3, 4]
            data = np.random.random(data_shape).astype('float16')
            min_data = -10.0
            max_data = 10.0
            out_np = np.clip(data, min_data, max_data)
            data = paddle.to_tensor(data)
            min_data = paddle.to_tensor(min_data)
            max_data = paddle.to_tensor(max_data)
            out = paddle.clip(data, min_data, max_data)
            np.testing.assert_allclose(out.numpy(), out_np, rtol=1e-05)

        paddle.enable_static()

    def test_shapeerror_clip(self):
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        data = paddle.to_tensor(data)
        with self.assertRaises(ValueError):
            paddle.clip(data, paddle.rand([2]))

        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        data = paddle.to_tensor(data)
        with self.assertRaises(ValueError):
            paddle.clip(
                data, min=paddle.to_tensor([1, 2, 3, 4]), max=paddle.rand([0])
            )

    def test_tensor_clip_(self):
        data_shape = [1, 9, 9, 4]
        data = np.random.uniform(-2.0, 2.0, data_shape).astype('float32')
        data = paddle.to_tensor(data)
        min_data = np.random.uniform(-2.5, -1.5, data_shape[-2:]).astype(
            'float32'
        )
        min = paddle.to_tensor(min_data)
        max_data = np.random.uniform(1.5, 2.5, data_shape[-3:]).astype(
            'float32'
        )
        max = paddle.to_tensor(max_data)
        data.clip_(min, max)


if __name__ == '__main__':
    unittest.main()

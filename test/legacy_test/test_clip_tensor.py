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
    if max_shape is None:
        if dtype == 'int32':
            max = np.iinfo(np.int32).max - 2**7
        elif dtype == 'int64':
            max = np.iinfo(np.int64).max - 2**39
        elif dtype == 'float16':
            max = float(np.finfo(np.float16).max)
        else:
            max = float(np.finfo(np.float32).max)
    else:
        max = np.random.randn(*max_shape).astype(dtype)
    if min_shape is None:
        if dtype == 'int32':
            min = np.iinfo(np.int32).min
        elif dtype == 'int64':
            min = np.iinfo(np.int64).min
        elif dtype == 'float16':
            min = float(np.finfo(np.float16).min)
        else:
            min = float(np.finfo(np.float32).min)
    else:
        min = np.random.randn(*min_shape).astype(dtype)
    np_out = np.clip(x, min, max)
    x_pd = paddle.to_tensor(x, dtype=dtype)
    min_pd = paddle.to_tensor(min, dtype=dtype)
    max_pd = paddle.to_tensor(max, dtype=dtype)
    pd_out = paddle.clip(x_pd, min_pd, max_pd)
    np.allclose(pd_out.numpy(), np_out)

    x_pd.clip_(min_pd, max_pd)
    np.allclose(x_pd.numpy(), np_out)
    paddle.enable_static()


def np_pd_static_equal(
    x_shape, min_shape=None, max_shape=None, dtype='float32'
):
    paddle.enable_static()
    x = np.random.randn(*x_shape).astype(dtype)
    if max_shape is None:
        if dtype == 'int32':
            max = np.iinfo(np.int32).max - 2**7
        elif dtype == 'int64':
            max = np.iinfo(np.int64).max - 2**39
        elif dtype == 'float16':
            max = float(np.finfo(np.float16).max)
        else:
            max = float(np.finfo(np.float32).max)
    else:
        max = np.random.randn(*max_shape).astype(dtype)
    if min_shape is None:
        if dtype == 'int32':
            min = np.iinfo(np.int32).min
        elif dtype == 'int64':
            min = np.iinfo(np.int64).min
        elif dtype == 'float16':
            min = float(np.finfo(np.float16).min)
        else:
            min = float(np.finfo(np.float32).min)
    else:
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

    def test_check_output(self):
        paddle.disable_static()
        np_pd_equal([5], [5], [1])
        np_pd_equal([4, 5], [5], [1], 'int32')
        np_pd_equal([4, 5], [5], [4, 5], 'int64')
        paddle.enable_static()

    def test_check_static_output(self):
        paddle.enable_static()
        np_pd_static_equal([5], [5], [1])
        np_pd_static_equal([4, 5], [5], [1], 'int32')
        np_pd_static_equal([4, 5], [5], [4, 5], 'int64')
        paddle.disable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

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
from tensorrt_test_base import TensorRTBaseTest

import paddle


def Avgpool3d_wrapper(x):
    pool = paddle.nn.AvgPool3D(kernel_size=2, stride=2, padding=0)
    return pool(x)


def Avgpool3d_python_api(x, padding="SAME", stride=(1, 1, 1)):
    pool = paddle.nn.AvgPool3D(kernel_size=2, stride=stride, padding=padding)
    return pool(x)


def Maxpool3d_wrapper(x):
    pool = paddle.nn.MaxPool3D(kernel_size=2, stride=2, padding=0)
    return pool(x)


def Maxpool3d_python_api(x, padding="SAME", stride=(1, 1, 1)):
    pool = paddle.nn.MaxPool3D(kernel_size=2, stride=stride, padding=padding)
    return pool(x)


class TestAvgPool3dTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = Avgpool3d_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMaxPool3dTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = Maxpool3d_wrapper
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAvgPool3dPaddingAlgorithmTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = Avgpool3d_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "padding": "VALID",
            "stride": (1, 2, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMaxPool3dPaddingAlgorithmTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = Maxpool3d_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "padding": "VALID",
            "stride": (1, 2, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAvgPool3dStrideTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = Avgpool3d_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "padding": "SAME",
            "stride": (2, 2, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMaxPool3dStrideTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = Maxpool3d_python_api
        self.api_args = {
            "x": np.random.random([2, 3, 8, 8, 8]).astype("float32"),
            "padding": "SAME",
            "stride": (2, 2, 2),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3, 8, 8, 8]}
        self.max_shape = {"x": [10, 3, 8, 8, 8]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()

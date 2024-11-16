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


def embedding_warpper_func(x, dtype):
    pre_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    layer = paddle.nn.Embedding(4, 3)
    paddle.set_default_dtype(pre_dtype)
    return layer(x)


class TestEmbeddingFloat1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.embedding
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "weight": np.random.randn(8, 10).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "weight"]}
        self.min_shape = {"x": [1, 3], "weight": [6, 10]}
        self.max_shape = {"x": [5, 3], "weight": [10, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEmbeddingFloat2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = embedding_warpper_func
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "dtype": "float32",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEmbeddingInt1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.embedding
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "weight": np.random.randn(8, 10).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "weight"]}
        self.min_shape = {"x": [1, 3], "weight": [6, 10]}
        self.max_shape = {"x": [5, 3], "weight": [10, 10]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEmbeddingInt2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = embedding_warpper_func
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "dtype": "int64",
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == "__main__":
    unittest.main()

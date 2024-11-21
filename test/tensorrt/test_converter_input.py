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


def embedding_warpper_func(x):
    layer = paddle.nn.Embedding(64, 4)
    return layer(x)


class TestEmbeddingCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.embedding
        self.api_args = {
            "x": np.array([[3, 16, 24], [6, 4, 47]]).astype("int64"),
            "weight": np.random.uniform(-1, 1, [64, 4]).astype('float32'),
        }
        self.program_config = {"feed_list": ["x", "weight"]}
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(1, 64, size=shape).astype(
                np.int64
            ),
        }
        self.min_shape = {"x": [1, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEmbeddingCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = embedding_warpper_func
        self.api_args = {
            "x": np.array([[3, 16, 24], [6, 4, 47]]).astype("int64"),
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(1, 64, size=shape).astype(
                np.int64
            ),
        }
        self.program_config = {"feed_list": ["x"]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


if __name__ == "__main__":
    unittest.main()

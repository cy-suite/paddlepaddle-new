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

import paddle.nn.functional as F


class TestGridSampleTRTPatternBase(TensorRTBaseTest):
    def setUp(self):
        self.python_api = F.grid_sample
        self.api_args = {
            "x": np.array(
                [[[[-0.6, 0.8, -0.5], [-0.5, 0.2, 1.2], [1.4, 0.3, -0.2]]]]
            ).astype("float32"),
            "grid": np.array(
                [
                    [
                        [[0.2, 0.3], [-0.4, -0.3], [-0.9, 0.3], [-0.9, -0.6]],
                        [[0.4, 0.1], [0.9, -0.8], [0.4, 0.5], [0.5, -0.2]],
                        [[0.1, -0.8], [-0.3, -1.0], [0.7, 0.4], [0.2, 0.8]],
                    ]
                ],
                dtype='float32',
            ),
        }
        self.program_config = {"feed_list": ["x", "grid"]}
        self.min_shape = {"x": [1, 1, 3, 3], "grid": [1, 3, 4, 2]}
        self.max_shape = {"x": [5, 1, 3, 3], "grid": [5, 3, 4, 2]}


class TestGridSampleTRTPatternCase1(TestGridSampleTRTPatternBase):
    """default:mode='bilinear', padding_mode='zeros', align_corners=True"""

    def test_trt_result(self):
        self.check_trt_result()


class TestGridSampleTRTPatternCase2(TestGridSampleTRTPatternBase):
    """default:mode='nearest', padding_mode='reflection', align_corners=False"""

    def setUp(self):
        super().setUp()
        self.api_args.update(
            {
                "mode": "nearest",
                "padding_mode": "reflection",
                "align_corner": False,
            }
        )

    def test_trt_result(self):
        self.check_trt_result()


class TestGridSampleTRTPatternCase3(TestGridSampleTRTPatternBase):
    """default:mode='nearest', padding_mode='border', align_corners=True"""

    def setUp(self):
        super().setUp()
        self.api_args.update({"mode": "nearest", "padding_mode": "border"})

    def test_trt_result(self):
        self.check_trt_result()


class TestGridSampleTRTPatternCase4(TestGridSampleTRTPatternBase):
    """default:mode='bilinear', padding_mode='border', align_corners=False"""

    def setUp(self):
        super().setUp()
        self.api_args.update(
            {
                "mode": "bilinear",
                "padding_mode": "border",
                "align_corner": False,
            },
        )

    def test_trt_result(self):
        self.check_trt_result()

class TestYoloBoxHeadTRTPatternBase(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.fluid.layers.yolo_box_head
        self.api_args = {
            "x": np.random.rand(1, 3, 416, 416).astype("float32"),  
            "anchors": np.array([10.0, 13.0, 16.0, 30.0, 33.0, 23.0], dtype="float32"), 
            "class_num": 80,  
        }
        self.program_config = {
            "feed_list": ["x", "anchors", "class_num"],
        }
        self.min_shape = {
            "x": [1, 3, 416, 416],
            "anchors": [6], 
            "class_num": [1],  
        }
        self.max_shape = {
            "x": [5, 3, 416, 416],
            "anchors": [6],
            "class_num": [1],
        }

class TestYoloBoxHeadTRTPatternCase1(TestYoloBoxHeadTRTPatternBase):
    def test_trt_result(self):
        self.check_trt_result() 

class TestYoloBoxHeadTRTPatternCase2(TestYoloBoxHeadTRTPatternBase):
    def setUp(self):
        super().setUp()
        self.api_args.update(
            {
                "anchors": np.array([12.0, 16.0, 30.0, 45.0, 50.0, 60.0], dtype="float32"),  
                "class_num": 90,  
            }
        )

    def test_trt_result(self):
        self.check_trt_result()

class TestYoloBoxHeadTRTPatternCase3(TestYoloBoxHeadTRTPatternBase):
    def setUp(self):
        super().setUp()
        self.api_args.update(
            {
                "anchors": np.array([12.0, 15.0, 20.0, 35.0, 45.0, 60.0], dtype="float32"),
                "class_num": 80,  
            }
        )
    def test_trt_result(self):
        self.check_trt_result()

class TestYoloBoxHeadTRTPatternCase4(TestYoloBoxHeadTRTPatternBase):
    def setUp(self):
        super().setUp()
        self.api_args.update(
            {
                "anchors": np.array([8.0, 14.0, 18.0, 25.0, 33.0, 38.0], dtype="float32"),
                "class_num": 100,  
            }
        )
    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()

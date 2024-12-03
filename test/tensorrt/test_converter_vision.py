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
from collections.abc import Sequence

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

import paddle.nn.functional as F
from paddle import Tensor, to_tensor


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


def anchor_generator(
    x: Tensor,
    anchor_sizes: Sequence[float],
    aspect_ratios: Sequence[float],
    stride: Sequence[float],
    variances: Sequence[float],
    offset: float = 0.5,
) -> Tensor:
    num_anchors = len(aspect_ratios) * len(anchor_sizes)
    layer_h = x.shape[2]
    layer_w = x.shape[3]
    out_dim = (layer_h, layer_w, num_anchors, 4)
    out_anchors = np.zeros(out_dim).astype('float32')

    for h_idx in range(layer_h):
        for w_idx in range(layer_w):
            x_ctr = (w_idx * stride[0]) + offset * (stride[0] - 1)
            y_ctr = (h_idx * stride[1]) + offset * (stride[1] - 1)
            idx = 0
            for r in range(len(aspect_ratios)):
                ar = aspect_ratios[r]
                for s in range(len(anchor_sizes)):
                    anchor_size = anchor_sizes[s]
                    area = stride[0] * stride[1]
                    area_ratios = area / ar
                    base_w = np.round(np.sqrt(area_ratios))
                    base_h = np.round(base_w * ar)
                    scale_w = anchor_size / stride[0]
                    scale_h = anchor_size / stride[1]
                    w = scale_w * base_w
                    h = scale_h * base_h
                    out_anchors[h_idx, w_idx, idx, :] = [
                        (x_ctr - 0.5 * (w - 1)),
                        (y_ctr - 0.5 * (h - 1)),
                        (x_ctr + 0.5 * (w - 1)),
                        (y_ctr + 0.5 * (h - 1)),
                    ]
                    idx += 1

    # Set the variance
    out_var = np.tile(variances, (layer_h, layer_w, num_anchors, 1))
    out_anchors = out_anchors.astype('float32')
    out_var = out_var.astype('float32')
    out_anchors = paddle.to_tensor(out_anchors)
    out_var = paddle.to_tensor(out_var)
    return out_anchors, out_var


class TestAnchorGeneratorTRTPatternBase(TensorRTBaseTest):
    def setUp(self):
        self.python_api = anchor_generator
        self.api_args = {
            "x": np.random.rand(1, 32, 16, 16).astype("float32"),
            "anchor_sizes": [64, 128, 256, 512],
            "aspect_ratios": [0.5, 1.0, 2.0],
            "stride": [16.0, 16.0],
            "variances": [0.1, 0.1, 0.2, 0.2],
            "offset": 0.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 32, 16, 16]}
        self.max_shape = {"x": [4, 48, 16, 16]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()

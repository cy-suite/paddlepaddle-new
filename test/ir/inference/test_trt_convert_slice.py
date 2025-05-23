# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest
from functools import partial
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertSliceTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        out_shape = list(inputs['input_data'].shape)
        for x in range(len(attrs[0]["axes"])):
            start = 0
            end = 0
            if attrs[0]["starts"][x] < 0:
                start = (
                    attrs[0]["starts"][x]
                    + inputs['input_data'].shape[attrs[0]["axes"][x]]
                )
            else:
                start = attrs[0]["starts"][x]
            if attrs[0]["ends"][x] < 0:
                end = (
                    attrs[0]["ends"][x]
                    + inputs['input_data'].shape[attrs[0]["axes"][x]]
                )
            else:
                end = attrs[0]["ends"][x]
            start = max(0, start)
            end = max(0, end)
            out_shape[attrs[0]["axes"][x]] = end - start
            if start >= end:
                return False
        for x in attrs[0]["decrease_axis"]:
            if x < 0:
                return False
            if out_shape[x] != 1:
                return False
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: list[dict[str, Any]]):
            return np.random.random([6, 6, 64, 64]).astype(np.float32)

        for axes in [[0, 1], [1, 3], [2, 3]]:
            for starts in [[0, 1]]:
                for ends in [[2, 2], [5, 5], [1, -1]]:
                    for decrease_axis in [[], [1], [2], [-1], [-100]]:
                        for infer_flags in [[-1]]:
                            dics = [
                                {
                                    "axes": axes,
                                    "starts": starts,
                                    "ends": ends,
                                    "decrease_axis": decrease_axis,
                                    "infer_flags": infer_flags,
                                }
                            ]

                            ops_config = [
                                {
                                    "op_type": "slice",
                                    "op_inputs": {"Input": ["input_data"]},
                                    "op_outputs": {
                                        "Out": ["slice_output_data"]
                                    },
                                    "op_attrs": dics[0],
                                }
                            ]
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={},
                                inputs={
                                    "input_data": TensorConfig(
                                        data_gen=partial(generate_input1, dics)
                                    )
                                },
                                outputs=["slice_output_data"],
                            )

                            yield program_config

    def generate_dynamic_shape(self):
        self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
        self.dynamic_shape.max_input_shape = {"input_data": [8, 8, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"input_data": [6, 6, 64, 64]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape:
                for x in attrs[0]["axes"]:
                    if x == 0:
                        return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        self.trt_param.max_batch_size = 9
        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-3

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def test_old_ir(self):
        # TODO(inference): fix.
        # trt6 and trt7.1 has bug.
        # trt7.2 deserialize has bug.
        self.run_test()

    def test_pir(self):
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()

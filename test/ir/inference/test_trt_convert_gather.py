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

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertGatherTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if len(inputs['input_data'].shape) <= attrs[0]['axis']:
            return False

        return True

    def sample_program_configs(self):
        def generate_input1(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_input2(index):
            return np.array(index).astype(np.int32)

        def generate_input4(index):
            return np.array(index).astype(np.int64)

        def generate_input3(axis):
            return np.array([axis]).astype(np.int32)

        for shape in [[32], [16, 64], [32, 16, 16], [32, 64, 16, 32]]:
            for index in [[1, 4], [4, 8]]:
                for axis in [0, 1, 2, 3]:
                    for overwrite in [True, False]:
                        for input in [
                            {"X": ["input_data"], "Index": ["index_data"]},
                        ]:
                            for index_type_int32 in [True, False]:
                                self.shape = shape
                                self.axis = axis
                                self.input_num = len(input)
                                self.index_type_int32 = index_type_int32
                                dics = [{"overwrite": overwrite, "axis": axis}]
                                ops_config = [
                                    {
                                        "op_type": "gather",
                                        "op_inputs": input,
                                        "op_outputs": {"Out": ["output_data"]},
                                        "op_attrs": dics[0],
                                    }
                                ]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={},
                                    inputs=(
                                        {
                                            "index_data": TensorConfig(
                                                data_gen=partial(
                                                    (
                                                        generate_input2
                                                        if index_type_int32
                                                        else generate_input4
                                                    ),
                                                    index,
                                                )
                                            ),
                                            "input_data": TensorConfig(
                                                data_gen=partial(
                                                    generate_input1, shape
                                                )
                                            ),
                                        }
                                    ),
                                    outputs=["output_data"],
                                )

                                yield program_config

    def generate_dynamic_shape(self):
        if len(self.shape) == 1:
            self.dynamic_shape.min_input_shape = {
                "input_data": [1],
                "index_data": [2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [128],
                "index_data": [2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [16],
                "index_data": [2],
            }
        elif len(self.shape) == 2:
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 4],
                "index_data": [2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [256, 256],
                "index_data": [2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [64, 32],
                "index_data": [2],
            }
        elif len(self.shape) == 3:
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 4, 4],
                "index_data": [2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [128, 256, 256],
                "index_data": [2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [16, 64, 32],
                "index_data": [2],
            }
        elif len(self.shape) == 4:
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 4, 4, 4],
                "index_data": [2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [128, 256, 64, 128],
                "index_data": [2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [16, 64, 16, 32],
                "index_data": [2],
            }
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(dynamic_shape):
            if self.input_num == 3:
                return 0, 5
            else:
                if dynamic_shape:
                    return 1, 3
                else:
                    return 0, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if not run_pir:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield self.create_inference_config(), generate_trt_nodes_num(
                False
            ), 1e-5
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield self.create_inference_config(), generate_trt_nodes_num(
                False
            ), 1e-3

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-3

    def add_skip_trt_case(self):
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 7000:

            def teller1(program_config, predictor_config):
                if len(self.dynamic_shape.min_input_shape) != 0:
                    inputs = program_config.inputs
                    if (
                        len(inputs['input_data'].shape) == 1
                        or len(inputs['index_data'].shape) == 1
                    ):
                        return True
                return False

            self.add_skip_case(
                teller1,
                SkipReasons.TRT_NOT_SUPPORT,
                "Need to repair the case: trt reshape out failed for dynamic shape mode when inputs' dims==1. under trt7.0 ",
            )

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()

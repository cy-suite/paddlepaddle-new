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


class TrtConvertReduceTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # dim should be in (-rank, rank), and not NONE
        rank = len(inputs['input_data'].shape)
        for x in attrs[0]["dim"]:
            if x >= rank or x <= -rank:
                return False
        if len(attrs[0]["dim"]) == 0:
            return False

        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 7000:
            if attrs[0]['out_dtype'] == 2:
                return False

        return True

    def sample_program_configs(self):
        def generate_input1(dtype, attrs: list[dict[str, Any]]):
            if dtype == -1 or dtype == 5:
                return np.random.random([1, 3, 64, 64]).astype(np.float32)
            elif dtype == 2:
                return np.random.random([1, 3, 64, 64]).astype(np.int32)
            elif dtype == 0:
                return np.random.random([1, 3, 64, 64]).astype(np.bool_)
            elif dtype == 3:
                return np.random.random([1, 3, 64, 64]).astype(np.int64)
            elif dtype == 6:
                return np.random.random([1, 3, 64, 64]).astype(np.float64)

        for keep_dim in [True, False]:
            for dim in [
                [],
                [1],
                [0],
                [0, 1],
                [1, 2, 3],
                [-2, 0, 3],
                [-3],
                [-4, 1],
                [3, 4, 5],
            ]:
                for reduce_all in [True, False]:
                    for out_dtype in [-1, 0, 2, 5, 3]:
                        if out_dtype != 0:
                            reduce_type_list = [
                                "reduce_max",
                                "reduce_min",
                                "reduce_mean",
                                "reduce_sum",
                                "reduce_prod",
                            ]
                        else:
                            reduce_type_list = [
                                "reduce_all",
                                "reduce_any",
                            ]

                        for op_type in reduce_type_list:
                            dics = [
                                {
                                    "keep_dim": keep_dim,
                                    "dim": dim,
                                    "reduce_all": reduce_all,
                                    "out_dtype": out_dtype,
                                    "in_dtype": out_dtype,
                                },
                                {},
                            ]

                            ops_config = [
                                {
                                    "op_type": op_type,
                                    "op_inputs": {"X": ["input_data"]},
                                    "op_outputs": {
                                        "Out": ["reduce_output_data"]
                                    },
                                    "op_attrs": dics[0],
                                }
                            ]
                            if op_type in ["reduce_any", "reduce_all"]:
                                ops_config[0]["outputs_dtype"] = {
                                    "reduce_output_data": np.bool_
                                }
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={},
                                inputs={
                                    "input_data": TensorConfig(
                                        data_gen=partial(
                                            generate_input1, out_dtype, dics
                                        )
                                    )
                                },
                                outputs=["reduce_output_data"],
                            )

                            if not self.is_program_valid(program_config):
                                continue

                            yield program_config

    def generate_dynamic_shape(self):
        self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                if (not attrs[0]['keep_dim']) and attrs[0]['reduce_all']:
                    return 0, 3
                else:
                    return 1, 2
            else:
                if 0 in attrs[0]['dim'] or attrs[0]['reduce_all']:
                    return 0, 3
                else:
                    return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        if not run_pir:
            self.trt_param.precision = paddle_infer.PrecisionType.Float32
            program_config.set_input_type(np.float32)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), (1e-5, 1e-5)
            self.trt_param.precision = paddle_infer.PrecisionType.Half
            program_config.set_input_type(np.float16)
            yield self.create_inference_config(), generate_trt_nodes_num(
                attrs, False
            ), (1e-3, 1e-3)

        # for dynamic_shape
        self.generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()

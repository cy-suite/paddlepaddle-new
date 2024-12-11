#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import os
import time
import unittest

import numpy as np

import paddle
import paddle.inference as paddle_infer

model_dir = os.getenv('OPENVINO_MODEL_DIR')
print("model_dir is : ", model_dir)
paddle.device.set_device("cpu")


class TestOpenVINOMobilenetModelBatch2(unittest.TestCase):
    def setUp(self):
        self.model_dir = model_dir
        self.x_shape = [2, 3, 224, 224]
        self.target_tensor = 'save_infer_model/scale_0.tmp_0'
        self.input_tensor = 'x'

    def get_paddle_inference_result(self, model_dir, data):
        config = paddle_infer.Config(
            model_dir + '/inference.pdmodel', model_dir + '/inference.pdiparams'
        )
        config.disable_gpu()
        config.switch_ir_optim(False)
        self.paddle_predictor_base = paddle_infer.create_predictor(config)
        data = paddle.Tensor(data)
        results = self.paddle_predictor_base.run([data])
        get_tensor = self.paddle_predictor_base.get_output_handle(
            self.target_tensor
        ).copy_to_cpu()
        return get_tensor

    def apply_test(self):
        config = paddle_infer.Config(
            self.model_dir + '/inference.pdmodel',
            self.model_dir + '/inference.pdiparams',
        )
        config.disable_gpu()
        config.enable_openvino_engine(paddle_infer.PrecisionType.Float32)
        config.set_cpu_math_library_num_threads(2)
        cache_dir = os.path.join(self.model_dir, '__cache__')
        config.set_optim_cache_dir(cache_dir)
        start = time.time()
        self.paddle_predictor_openvino = paddle_infer.create_predictor(config)
        end1 = time.time()
        print("first load model time is: %.3f sec" % (end1 - start))

        x_data = np.random.random(self.x_shape).astype("float32")
        data = paddle.Tensor(x_data)
        self.paddle_predictor_openvino.run([data])
        for i in range(10):
            self.paddle_predictor_openvino.run([data])

        repeat = 10
        end4 = time.perf_counter()
        for i in range(repeat):
            self.paddle_predictor_openvino.run([data])
        end5 = time.perf_counter()
        print(
            "Repeat %d times, average Executor.run() time is: %.3f ms"
            % (repeat, (end5 - end4) * 1000 / repeat)
        )

        openvino_result = self.paddle_predictor_openvino.get_output_handle(
            self.target_tensor
        ).copy_to_cpu()
        target_result = self.get_paddle_inference_result(self.model_dir, x_data)
        print("result in test_model: \n")
        print(target_result)
        openvino_result = openvino_result.reshape(-1)
        target_result = target_result.reshape(-1)
        for i in range(0, min(openvino_result.shape[0], 200)):
            if np.abs(openvino_result[i] - target_result[i]) > 1e-3:
                print(
                    "Error! ",
                    i,
                    "-th data has diff with target data:\n",
                    openvino_result[i],
                    " vs: ",
                    target_result[i],
                    ". Diff is: ",
                    openvino_result[i] - target_result[i],
                )
        np.testing.assert_allclose(openvino_result, target_result, atol=1e-3)

    def test_model(self):
        self.apply_test()


if __name__ == "__main__":
    unittest.main()

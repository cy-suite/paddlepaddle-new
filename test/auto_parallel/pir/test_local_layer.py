# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import collective.test_communication_api_base as test_base


class TestLocalLayer(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=300,
        )
        self._default_envs = {"dtype": "float32", "seed": "2023"}
        self._changeable_envs = {"backend": ["gpu"]}

    def test_local_layer(self):
        envs_list = test_base.gen_product_envs_list(
            {"dtype": "float32", "seed": "2023"}, {"backend": ["gpu"]}
        )
        # {"DISTRIBUTED_TRAINER_ENDPOINTS", "10.67.188.11:8544"}
        # self._log_dir.name = "./log"
        for envs in envs_list:
            self.run_test_case(
                "local_layer_demo.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()

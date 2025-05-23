#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from legacy_test.test_collective_api_base import TestDistBase

import paddle

paddle.enable_static()


class TestCollectiveBarrierAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_barrier_nccl(self):
        self.check_with_place("collective_barrier_api.py", "barrier", "nccl")

    def test_barrier_gloo(self):
        self.check_with_place(
            "collective_barrier_api.py", "barrier", "gloo", "5"
        )

    def test_barrier_flagcx(self):
        if paddle.base.core.is_compiled_with_flagcx():
            self.check_with_place(
                "collective_barrier_api.py",
                "barrier",
                "flagcx",
                static_mode="0",
            )


if __name__ == '__main__':
    unittest.main()

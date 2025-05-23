#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestCollectiveGatherAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_gather_nccl_dygraph(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_gather_api_dygraph.py",
                "gather",
                "nccl",
                static_mode="0",
                dtype=dtype,
            )

    def test_gather_flagcx_dygraph(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ]
        if paddle.base.core.is_compiled_with_flagcx():
            for dtype in dtypes_to_test:
                self.check_with_place(
                    "collective_gather_api_dygraph.py",
                    "gather",
                    "flagcx",
                    static_mode="0",
                    dtype=dtype,
                )


if __name__ == "__main__":
    unittest.main()

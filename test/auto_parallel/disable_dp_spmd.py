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


import numpy as np

import paddle
import paddle.distributed as dist


class TestDisableDPSpmd:
    def __init__(self):
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_disable_dp_spmd_case1(self):
        a = paddle.to_tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="float32"
        )
        a = dist.shard_tensor(a, self._mesh, [dist.Shard(0)])
        b = paddle.sum(a, axis=0)
        a_local_value = a._local_value()[0].numpy()
        b_local_value = b._local_value().numpy()

        assert b.placements[0] == dist.Shard(0)
        assert np.array_equal(a_local_value, b_local_value)

    def test_disable_dp_spmd_case2(self):
        a = paddle.to_tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="float32"
        )
        linear = paddle.nn.Linear(2, 2, bias_attr=False)
        a = dist.shard_tensor(a, self._mesh, [dist.Shard(0)])
        linear.weight = dist.shard_tensor(
            linear.weight, self._mesh, [dist.Replicate()]
        )
        c = linear(a)
        c.backward()
        assert linear.weight.batch_dim == -1
        assert linear.weight.placements[0] == dist.Replicate()

    def run_test_case(self):
        self.test_disable_dp_spmd_case1()
        self.test_disable_dp_spmd_case2()


if __name__ == "__main__":
    test = TestDisableDPSpmd()
    test.run_test_case()

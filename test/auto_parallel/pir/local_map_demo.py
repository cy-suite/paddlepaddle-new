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

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.distributed import ProcessMesh


def custom_function(x):
    mask = paddle.zeros_like(x)
    if dist.get_rank() == 0:
        mask[1:3] = 1
    else:
        mask[4:7] = 1

    x = x * mask
    mask_sum = paddle.sum(x)
    mask_sum = mask_sum / mask.sum()
    return mask_sum


class TestLocalMap(unittest.TestCase):
    def test_local_map(self):

        dist.init_parallel_env()

        mesh = ProcessMesh([0, 1], dim_names=["x"])

        local_input = paddle.arange(0, 10, dtype="float32")
        local_input = local_input + dist.get_rank()

        input_dist = dist.auto_parallel.api.dtensor_from_local(
            local_input, mesh, [dist.Shard(0)]
        )

        wrapped_func = dist.local_map(
            custom_function,
            out_placements=[dist.Partial(dist.ReduceType.kRedSum)],
            in_placements=(dist.Shard(0),),
            process_mesh=mesh,
        )

        output_dist = wrapped_func(input_dist)

        local_value = output_dist._local_value()
        gathered_values: list[Tensor] = []
        dist.all_gather(gathered_values, local_value)

        expected_rank0 = 1.5
        expected_rank1 = 6.0
        expected_global = 7.5

        self.assertAlmostEqual(
            gathered_values[0].item(),
            expected_rank0,
            delta=1e-6,
            msg=f"Rank 0 value mismatch: got {gathered_values[0].item()}, expected {expected_rank0}",
        )
        self.assertAlmostEqual(
            gathered_values[1].item(),
            expected_rank1,
            delta=1e-6,
            msg=f"Rank 1 value mismatch: got {gathered_values[1].item()}, expected {expected_rank1}",
        )
        self.assertAlmostEqual(
            output_dist.item(),
            expected_global,
            delta=1e-6,
            msg=f"Global value mismatch: got {output_dist.item()}, expected {expected_global}",
        )


if __name__ == "__main__":
    unittest.main()

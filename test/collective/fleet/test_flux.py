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

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core
from paddle.distributed import fleet
from paddle.distributed.utils.nccl_utils import check_nccl_version_for_bf16
from paddle.nn.functional import all_gather_gemm, flux, gemm_reduce_scatter


def all_gather(input, group=None, axis=0):
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    if axis == 0:
        output_shape[axis] = output_shape[axis] * parallelism
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        dist.stream.all_gather(output, input, group=group, use_calc_stream=True)
        return output
    outputs = [
        paddle.empty(output_shape, dtype=input.dtype)
        for _ in range(parallelism)
    ]
    dist.stream.all_gather(outputs, input, group=group, use_calc_stream=True)
    output = paddle.concat(outputs, axis=axis)
    return output


def reduce_scatter(input, group=None):
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    assert (
        input.shape[0] % parallelism == 0
    ), f"Input sequence length {input.shape[0]} can't be divided exactly by sequence parallelism {parallelism}"
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    dist.stream.reduce_scatter(
        output, input, op=dist.ReduceOp.SUM, group=group, use_calc_stream=True
    )
    return output


def naive_all_gather_gemm(input, weight, group):
    input_parallel = all_gather(input, group)
    output = paddle.matmul(input_parallel, weight)
    return output, input_parallel


def naive_gemm_reduce_scatter(input, weight, group):
    gemm_output = paddle.matmul(input, weight)
    output = reduce_scatter(gemm_output, group)
    return output


def is_flux_supported():
    is_sm8x = (
        core.is_compiled_with_cuda()
        and paddle.device.cuda.get_device_capability()[0] == 8
        and paddle.device.cuda.get_device_capability()[1] >= 0
    )

    is_sm90 = (
        core.is_compiled_with_cuda()
        and paddle.device.cuda.get_device_capability()[0] == 9
        and paddle.device.cuda.get_device_capability()[1] == 0
    )

    is_sm_supported = is_sm8x or is_sm90

    if not core.is_compiled_with_cuda() or not is_sm_supported:
        return False
    return True


@unittest.skipIf(
    not is_flux_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestAllGatherGemmAPI(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        self.THRESHOLD_MAP = {
            "float16": 1e-2,
            "bfloat16": 1e-2,
        }

    def _rand(self, shape, dtype, rank):
        return (-2 * paddle.rand(shape, dtype=dtype) + 1) / 100 * (rank + 1)

    def test_all_gather_gemm(self):
        m_dim = 1024
        n_dim = 1024
        k_dim = 1024
        transpose_weight = False
        rank_id = dist.get_rank()
        num_trainer = dist.get_world_size()

        check_group = dist.new_group(list(range(num_trainer)))
        for dtype in ('bfloat16', 'float16'):
            input = self._rand(shape=[m_dim, k_dim], dtype=dtype, rank=rank_id)
            if transpose_weight:
                weight = self._rand(
                    shape=[k_dim, n_dim], dtype=dtype, rank=rank_id
                )
            else:
                weight = self._rand(
                    shape=[n_dim, k_dim], dtype=dtype, rank=rank_id
                )

            output, input_parallel = all_gather_gemm(
                input, weight, check_group, deepcopy_input_parallel=True
            )

            output_ref, input_parallel_ref = naive_all_gather_gemm(
                input, weight, check_group
            )

            if dtype == 'bfloat16':
                output = paddle.cast(output, 'float32')
                input_parallel = paddle.cast(input_parallel, 'float32')

                output_ref = paddle.cast(output_ref, 'float32')
                input_parallel_ref = paddle.cast(input_parallel_ref, 'float32')

                np.testing.assert_allclose(
                    output_ref.numpy(),
                    output.numpy(),
                    rtol=self.THRESHOLD_MAP[dtype],
                    atol=self.THRESHOLD_MAP[dtype],
                )

                np.testing.assert_allclose(
                    input_parallel_ref.numpy(),
                    input_parallel.numpy(),
                    rtol=self.THRESHOLD_MAP[dtype],
                    atol=self.THRESHOLD_MAP[dtype],
                )

    def test_all_gather_gemm_can_implement(self):
        cases = [
            {'m_dim': 1024, 'n_dim': 1024, 'k_dim': 1024, 'res': True},
            {'m_dim': 1024, 'n_dim': 1025, 'k_dim': 1024, 'res': False},
        ]
        transpose_weight = False
        rank_id = dist.get_rank()
        num_trainer = dist.get_world_size()

        check_group = dist.new_group(list(range(num_trainer)))
        for dtype in ('bfloat16', 'float16'):
            for case in cases:
                m_dim = case['m_dim']
                n_dim = case['n_dim']
                k_dim = case['k_dim']
                input = self._rand(
                    shape=[m_dim, k_dim], dtype=dtype, rank=rank_id
                )
                if transpose_weight:
                    weight = self._rand(
                        shape=[k_dim, n_dim], dtype=dtype, rank=rank_id
                    )
                else:
                    weight = self._rand(
                        shape=(n_dim, k_dim), dtype=dtype, rank=rank_id
                    )

                res = flux.all_gather_gemm_can_implement(
                    input, weight, check_group
                )
                assert res == case['res']


@unittest.skipIf(
    not is_flux_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestGemmReduceScatterAPI(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        self.THRESHOLD_MAP = {
            "float16": 1e-2,
            "bfloat16": 2e-2,
        }

    def _rand(self, shape, dtype, rank):
        return (-2 * paddle.rand(shape, dtype=dtype) + 1) / 100 * (rank + 1)

    def test_gemm_reduce_scatter(self):
        m_dim = 1024
        n_dim = 1024
        k_dim = 1024
        transpose_weight = False
        rank_id = dist.get_rank()
        num_trainer = dist.get_world_size()

        check_group = dist.new_group(list(range(num_trainer)))
        for dtype in ('bfloat16', 'float16'):
            input = self._rand(shape=[m_dim, k_dim], dtype=dtype, rank=rank_id)
            if transpose_weight:
                weight = self._rand(
                    shape=[k_dim, n_dim], dtype=dtype, rank=rank_id
                )
            else:
                weight = self._rand(
                    shape=(n_dim, k_dim), dtype=dtype, rank=rank_id
                )

            output = gemm_reduce_scatter(input, weight, check_group)

            output_ref = naive_gemm_reduce_scatter(input, weight, check_group)

            if dtype == 'bfloat16':
                output = paddle.cast(output, 'float32')

                output_ref = paddle.cast(output_ref, 'float32')

                np.testing.assert_allclose(
                    output_ref.numpy(),
                    output.numpy(),
                    rtol=self.THRESHOLD_MAP[dtype],
                    atol=self.THRESHOLD_MAP[dtype],
                )

    def test_gemm_reduce_scatter_can_implement(self):
        cases = [
            {'m_dim': 1024, 'n_dim': 1024, 'k_dim': 1024, 'res': True},
            {'m_dim': 1024, 'n_dim': 1025, 'k_dim': 1024, 'res': False},
        ]
        transpose_weight = False
        rank_id = dist.get_rank()
        num_trainer = dist.get_world_size()

        check_group = dist.new_group(list(range(num_trainer)))
        for dtype in ('bfloat16', 'float16'):
            for case in cases:
                m_dim = case['m_dim']
                n_dim = case['n_dim']
                k_dim = case['k_dim']
                input = self._rand(
                    shape=[m_dim, k_dim], dtype=dtype, rank=rank_id
                )
                if transpose_weight:
                    weight = self._rand(
                        shape=[k_dim, n_dim], dtype=dtype, rank=rank_id
                    )
                else:
                    weight = self._rand(
                        shape=(n_dim, k_dim), dtype=dtype, rank=rank_id
                    )

                res = flux.gemm_reduce_scatter_can_implement(
                    input, weight, check_group
                )
                assert res == case['res']


if __name__ == "__main__":
    if check_nccl_version_for_bf16() and is_flux_supported():
        unittest.main()

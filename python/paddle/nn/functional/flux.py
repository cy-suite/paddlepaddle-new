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

import paddle
from paddle import _C_ops


def gemm_reduce_scatter_launcher(input, weight, group, check_can_implement):
    transpose_weight = input.shape[1] == weight.shape[0]
    global_M = input.shape[0]
    global_N = weight.shape[1] if transpose_weight else weight.shape[0]
    fuse_reduction = False
    nnodes = 1
    ring_id = group.id
    nranks = group.nranks
    bias = None
    input_scale = None
    weight_scale = None
    output_scale = None

    return _C_ops.gemm_reduce_scatter(
        input,
        weight,
        bias,
        input_scale,
        weight_scale,
        output_scale,
        nnodes,
        global_M,
        global_N,
        transpose_weight,
        fuse_reduction,
        check_can_implement,
        ring_id,
        nranks,
    )


def gemm_reduce_scatter_can_implement(input, weight, group):
    output = gemm_reduce_scatter_launcher(input, weight, group, True)
    assert (
        paddle.numel(output) == 1
    ), "The check result should have exactly one element."
    assert (
        output.dtype == paddle.bool
    ), "The check result is not of boolean type."
    return output.item()


def gemm_reduce_scatter(input, weight, group):
    return gemm_reduce_scatter_launcher(input, weight, group, False)


def all_gather_gemm_launcher(
    input, weight, group, deepcopy_input_parallel, check_can_implement
):
    nnodes = 1
    transpose_weight = input.shape[1] == weight.shape[0]
    full_m = input.shape[0] * group.nranks
    k_dim = input.shape[1]
    n_dim = weight.shape[1] if transpose_weight else weight.shape[0]
    ring_id = group.id
    fast_accum = False
    local_copy = False
    bias = None
    input_scale = None
    weight_scale = None
    output_scale = None
    return _C_ops.all_gather_gemm(
        input,
        weight,
        bias,
        input_scale,
        weight_scale,
        output_scale,
        nnodes,
        full_m,
        n_dim,
        k_dim,
        ring_id,
        fast_accum,
        deepcopy_input_parallel,
        transpose_weight,
        check_can_implement,
    )


def all_gather_gemm_can_implement(input, weight, group):
    output, _ = all_gather_gemm_launcher(input, weight, group, False, True)
    assert (
        paddle.numel(output) == 1
    ), "The check result should have exactly one element."
    assert (
        output.dtype == paddle.bool
    ), "The check result is not of boolean type."
    return output.item()


def all_gather_gemm(input, weight, group, deepcopy_input_parallel):
    return all_gather_gemm_launcher(
        input, weight, group, deepcopy_input_parallel, False
    )

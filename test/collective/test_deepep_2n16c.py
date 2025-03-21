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

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

try:
    from paddle.distributed.communication import deep_ep

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

assert HAVE_DEEP_EP, "DeepEP is necessary for this test."

BASE_SEED = 42
SEQ_LEN = 4096
HIDDEN_SIZE = 7168
NUM_EXPERTS = 64
TOP_K = 8

_BUFFER = None
TEST_NVL_BYTES = 2**30
TEST_RDMA_BYTES = 2**30


def generate_inputs():
    seed = BASE_SEED + dist.get_rank()
    paddle.seed(seed)
    hidden_states = paddle.rand((SEQ_LEN, HIDDEN_SIZE), dtype=paddle.bfloat16)
    token_probs_all = paddle.rand((SEQ_LEN, NUM_EXPERTS), dtype=paddle.float32)
    token_probs, token_indices = paddle.topk(
        token_probs_all, TOP_K, axis=-1, sorted=True
    )
    return hidden_states, token_indices, token_probs


def get_buffer(group):
    global _BUFFER
    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if _BUFFER is None:
        _BUFFER = deep_ep.Buffer(group, TEST_NVL_BYTES, TEST_RDMA_BYTES)

    return _BUFFER


def fused_dispatch_forward_func(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """Forward pass of fused dispatch."""
    buffer = get_buffer(group)
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event_,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    assert token_probs.dtype == paddle.float32
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive,
    # so this is not compatible with CUDA graph
    (
        deepep_dispatch_output,
        recv_token_indices,
        recv_token_probs,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    states = {}
    states["dispatched_indices"] = recv_token_indices
    states["tokens_per_expert"] = num_recv_tokens_per_expert_list
    states["handle"] = handle

    return deepep_dispatch_output, recv_token_probs, states, event


def fused_combine_forward_func(
    x,
    group,
    states,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """Forward pass of fused combine."""
    handle = states["handle"]
    buffer = get_buffer(group)
    deepep_combine_output, _, event = buffer.combine(
        x,
        handle=handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return deepep_combine_output


def get_raw_ops_result(hidden_states, token_indices, token_probs, moe_group):
    hidden_size, topk = token_indices.shape
    # notify
    exp_counts = paddle.bincount(
        token_indices.reshape([-1]), minlength=NUM_EXPERTS
    )
    exp_counts_list = []
    dist.all_gather(
        exp_counts_list, exp_counts.unsqueeze(axis=0), group=moe_group
    )
    experts_per_rank = NUM_EXPERTS // moe_group.nranks
    start_indice = moe_group.rank * experts_per_rank
    exp_counts_current_rank = paddle.concat(exp_counts_list, axis=0).slice(
        starts=start_indice, ends=(start_indice + experts_per_rank), axes=[1]
    )
    out_split_sizes = exp_counts_current_rank.sum(axis=1)
    in_split_sizes = paddle.bincount(
        token_indices.reshape([-1]).floor_divide(
            paddle.to_tensor([experts_per_rank])
        ),
        minlength=moe_group.nranks,
    )

    # permute
    in_tensor = (
        hidden_states.unsqueeze(1)
        .expand([-1, topk, -1])
        .reshape([-1, hidden_states.shape[1]])
    )
    permuted_token_indices_list = []
    tmp_indices = paddle.concat(
        [paddle.to_tensor([0]), exp_counts[:-1]]
    ).cumsum()
    for i in range(SEQ_LEN):
        permuted_token_indices_list.append(
            paddle.take_along_axis(tmp_indices, token_indices[i], axis=0)
        )
        tmp_indices.put_along_axis_(token_indices[i], 1, 0, "add")
    combine_token_indices = paddle.concat(permuted_token_indices_list)
    dispatch_token_indices = paddle.argsort(combine_token_indices)
    in_tensor = paddle.index_select(
        in_tensor, axis=0, index=dispatch_token_indices
    )

    # do dispatch
    out_tensor = paddle.zeros(
        [out_split_sizes.sum().item(), hidden_size], dtype=hidden_states.dtype
    )
    dist.alltoall_single(
        out_tensor,
        in_tensor,
        out_split_sizes=out_split_sizes,
        in_split_sizes=in_split_sizes,
        group=moe_group,
    )
    dispatch_result = out_tensor

    # do combine
    dist.alltoall_single(
        in_tensor,
        out_tensor,
        out_split_sizes=in_split_sizes,
        in_split_sizes=out_split_sizes,
        group=moe_group,
    )
    unpermuted_in_tensor = paddle.index_select(
        in_tensor, axis=0, index=combine_token_indices
    )
    token_weights = (
        token_probs.astype(hidden_states.dtype).flatten().reshape((-1, 1))
    )
    combine_result = (
        (unpermuted_in_tensor * token_weights)
        .unsqueeze(1)
        .reshape((-1, topk, hidden_size))
        .sum(axis=1)
    )

    return dispatch_result, combine_result


def test_deepep():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 16,
    }
    fleet.init(is_collective=True, strategy=strategy)

    moe_group = dist.get_group()
    print(f"check moe_group: {moe_group}")
    check_tensor = paddle.to_tensor([0])
    dist.all_reduce(check_tensor, group=moe_group)
    hidden_states, token_indices, token_probs = generate_inputs()
    raw_dispatch_output, raw_combine_output = get_raw_ops_result(
        hidden_states, token_indices, token_probs, moe_group
    )
    print(
        f"raw dispatch results: {raw_dispatch_output._md5sum()} {raw_dispatch_output.shape}"
    )
    print(
        f"raw combine results: {raw_combine_output._md5sum()} {raw_combine_output.shape}"
    )
    deepep_dispatch_output, recv_token_probs, states, event = (
        fused_dispatch_forward_func(
            hidden_states, token_indices, token_probs, NUM_EXPERTS, moe_group
        )
    )
    print(
        f"deepep dispatch results: {deepep_dispatch_output._md5sum()} {deepep_dispatch_output.shape}"
    )
    deepep_combine_output = fused_combine_forward_func(
        deepep_dispatch_output, moe_group, states
    )
    print(
        f"deepep combine results: {deepep_combine_output._md5sum()} {deepep_combine_output.shape}"
    )


if __name__ == "__main__":
    test_deepep()

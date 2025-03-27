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

import logging

from paddle.base import core

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import (
    _pir_program_for_vpp,
    _pir_split_matmul_grad_to_matmul,
)
from .pipeline_pass_base import PipelinePassBase

FORWARD = "forward"
BACKWARD = "backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_VPP")
class PipelineVirtualPipelinePass(PipelinePassBase):
    def __init__(self):
        super().__init__()
        self._real_overlap_sharding_reduce = False
        self.reduce_comm_suffix = "_reduce"
        self._forward_micro_step_counter = {}
        self._backward_micro_step_counter = {}

    def _record_fwd_micro_step(self, virtual_pp_rank):
        real_micro_step = self._forward_micro_step_counter[virtual_pp_rank]
        self._forward_micro_step_counter[virtual_pp_rank] += 1
        return real_micro_step

    def _record_bwd_micro_step(self, virtual_pp_rank):
        real_micro_step = self._backward_micro_step_counter[virtual_pp_rank]
        self._backward_micro_step_counter[virtual_pp_rank] += 1
        return real_micro_step

    def _create_job_list(self):
        accumulate_steps = self.get_attr("num_micro_batches")
        stage_id = self.get_attr("pp_stage")
        num_stages = self.get_attr("pp_degree")
        num_model_chunks = self.get_attr("vpp_degree")
        split_backward = self.get_attr("split_backward", False)
        remainder = accumulate_steps % num_stages
        for i in range(num_model_chunks):
            self._forward_micro_step_counter[i] = 0
            self._backward_micro_step_counter[i] = 0

        assert accumulate_steps >= num_stages

        def _get_virtual_pp_rank(micro_step, forward):
            virtual_pp_stage = micro_step % (num_stages * num_model_chunks)
            if micro_step <= (accumulate_steps // num_stages) * (
                num_stages * num_model_chunks
            ):
                virtual_pp_stage = virtual_pp_stage // num_stages
            else:
                virtual_pp_stage = virtual_pp_stage // remainder
            if not forward:
                virtual_pp_stage = num_model_chunks - virtual_pp_stage - 1
            return virtual_pp_stage

        total_num_steps = accumulate_steps * num_model_chunks
        if accumulate_steps == num_stages:
            warmup_steps = total_num_steps
        else:
            warmup_steps = (num_stages - stage_id - 1) * 2
            warmup_steps += (num_model_chunks - 1) * num_stages
            warmup_steps = min(warmup_steps, total_num_steps)

        steady_steps = total_num_steps - warmup_steps
        real_split_backward = (
            accumulate_steps == num_stages
        ) and split_backward

        job_list = []
        for micro_step in range(warmup_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=True)
            micro_batch_id = self._record_fwd_micro_step(virtual_pp_rank)
            fw_job = core.Job(FORWARD + str(virtual_pp_rank))
            fw_job.set_micro_batch_id(micro_batch_id)
            job_list.append(fw_job)

        for micro_step in range(steady_steps):
            fwd_micro_step = micro_step + warmup_steps
            fwd_virtual_pp_rank = _get_virtual_pp_rank(
                fwd_micro_step, forward=True
            )
            fwd_micro_batch_id = self._record_fwd_micro_step(
                fwd_virtual_pp_rank
            )
            fwd_job = core.Job(FORWARD + str(fwd_virtual_pp_rank))
            fwd_job.set_micro_batch_id(fwd_micro_batch_id)
            job_list.append(fwd_job)

            bw_micro_step = micro_step
            bwd_virtual_pp_rank = _get_virtual_pp_rank(
                bw_micro_step, forward=False
            )
            bwd_micro_batch_id = self._record_bwd_micro_step(
                bwd_virtual_pp_rank
            )
            if real_split_backward:
                bwd_job = core.Job(BACKWARD + "_b" + str(bwd_virtual_pp_rank))
            else:
                bwd_job = core.Job(BACKWARD + str(bwd_virtual_pp_rank))
            bwd_job.set_micro_batch_id(bwd_micro_batch_id)
            job_list.append(bwd_job)

        for micro_step in range(steady_steps, total_num_steps):
            virtual_pp_rank = _get_virtual_pp_rank(micro_step, forward=False)
            micro_batch_id = self._record_bwd_micro_step(virtual_pp_rank)
            if real_split_backward:
                bwd_job = core.Job(BACKWARD + "_b" + str(virtual_pp_rank))
            else:
                bwd_job = core.Job(BACKWARD + str(virtual_pp_rank))
            bwd_job.set_micro_batch_id(micro_batch_id)
            job_list.append(bwd_job)
            # TODO(lizhiyu): Inserting 'backward_b' and 'backward_w' interleavedly can decrease the memory,
            #                but it reduces the speed. We should find the better way to use the code here.
            # next_virtual_pp_rank = _get_virtual_pp_rank(micro_step + 1, forward=False)
            # if next_virtual_pp_rank != virtual_pp_rank:
            #     for micro_batch_id in range(0, accumulate_steps):
            #         w_job = core.Job(BACKWARD + "_w" + str(virtual_pp_rank))
            #         w_job.set_micro_batch_id(micro_batch_id)
            #         job_list.append(w_job)

        if real_split_backward:
            for chunk_id in range(num_model_chunks - 1, -1, -1):
                for micro_batch_id in range(0, accumulate_steps):
                    if (
                        self._real_overlap_sharding_reduce
                        and micro_batch_id == accumulate_steps - 1
                    ):
                        w_job = core.Job(
                            BACKWARD
                            + "_w"
                            + str(chunk_id)
                            + self.reduce_comm_suffix
                        )
                    else:
                        w_job = core.Job(BACKWARD + "_w" + str(chunk_id))
                    w_job.set_micro_batch_id(micro_batch_id)
                    job_list.append(w_job)
        job_types = [job.type() for job in job_list]
        logger.debug(f"The VPP job list: {job_types}")
        opt_job = core.Job(OPT)
        job_list.append(opt_job)
        return job_list

    def _pir_split_matmul_grad_ops_to_matmul(self, program):
        for block in program.blocks:
            matmul_grad_op_idx = []
            ops = block.ops
            for i, op_i in enumerate(ops):
                if (
                    op_i.name() == "pd_op.matmul_grad"
                    and not op_i.has_attr("trans_x")
                    and not op_i.has_attr("trans_y")
                ):
                    matmul_grad_op_idx.append(i)

            for matmul_grad_id in reversed(matmul_grad_op_idx):
                _pir_split_matmul_grad_to_matmul(block, matmul_grad_id)
      
    def _partial_programs(self, program):
        raise RuntimeError("Not support old IR for VPP")

    def _partial_pir_programs(self, program):
        num_model_chunks = self.get_attr("vpp_degree")
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        split_backward = self.get_attr("split_backward", False)
        accumulate_steps = self.get_attr("num_micro_batches")
        num_stages = self.get_attr("pp_degree")

        if accumulate_steps != num_stages:
            split_backward = False

        assert (
            not enable_send_recv_overlap
        ), "PIR does not support VPP with enable_send_recv_overlap yet."

        if split_backward:
            self._pir_split_matmul_grad_ops_to_matmul(program)

        types, sub_program_list = _pir_program_for_vpp(
            program, num_model_chunks, split_backward, enable_send_recv_overlap
        )

        for i in range(len(types)):
            logger.debug(
                f"type = {types[i]}, sub_programs = {sub_program_list[i]}\n"
            )

        return types, sub_program_list

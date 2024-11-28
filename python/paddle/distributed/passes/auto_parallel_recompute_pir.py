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

import paddle
from paddle.base import core

OpRole = core.op_proto_and_checker_maker.OpRole

from paddle.autograd import backward_utils

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()


@register_pass("auto_parallel_recompute_pir")
class AutoParallelRecomputePIRPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _check_user(self, value):
        pass

    def get_fwd_bwd_ops(self, program):
        fwd_ops = []
        bwd_ops = []
        for op in program.global_block().ops:
            if op.op_role == int(OpRole.Forward):
                fwd_ops.append(op)
            elif op.op_role == int(OpRole.Backward):
                bwd_ops.append(op)
        return fwd_ops, bwd_ops

    def get_first_bwd_used_op(self, fwd_op, bwd_ops):
        first_op = bwd_ops[-1]
        for res in fwd_op.results():
            for user_op in res.all_used_ops():
                if user_op in bwd_ops and first_op.id() > user_op.id():
                    first_op = user_op
        return first_op

    def is_seed_used_by_dropout(self, seed_op):
        if seed_op.name() != "seed":
            return False
        seed_value = seed_op.results()[0]
        dropout_ops = ["pd_op.dropout", "pd_op.fused_dropout_add"]
        return any(
            True
            for used_op in seed_value.all_used_ops()
            if used_op.name() in dropout_ops
        )

    def get_checkpoints(self, program):
        segment_beg = {}
        segment_end = {}
        max_op_id = len(program.global_block().ops)
        for idx, op in enumerate(program.global_block().ops):
            if not op.has_attr("recompute_id"):
                continue
            rc_id = op.attrs()["recompute_id"]
            if rc_id not in segment_beg:
                segment_beg[rc_id] = max_op_id
                segment_end[rc_id] = 0
            segment_beg[rc_id] = min(segment_beg[rc_id], idx)
            segment_end[rc_id] = max(segment_end[rc_id], idx)

        checkpoints = []
        assert len(segment_beg.keys()) == len(segment_end.keys())
        for segment_id, beg_id in segment_beg.items():
            assert segment_id in segment_end.keys()
            end_id = segment_end[segment_id]
            assert beg_id <= end_id
            checkpoints.append([beg_id, end_id])

        checkpoints.sort()

        # TODO: add check for checkpoints
        # pre_seg_end_id < nxt_seg_beg_id

        return checkpoints

    def _apply_single_impl(self, main_program, startup_program, context=None):
        checkpoints = self.get_checkpoints(main_program)
        print("xxx checkpoints: ", checkpoints)
        if len(checkpoints) == 0:
            logger.info("No recompute found.")
            return

        fwd_ops, bwd_ops = self.get_fwd_bwd_ops(main_program)

        segments = []
        for segment in checkpoints:
            assert len(segment) == 2
            beg_op_idx = segment[0]
            end_op_idx = segment[1]
            beg_op = main_program.global_block().ops[beg_op_idx]
            end_op = main_program.global_block().ops[end_op_idx]
            if beg_op not in fwd_ops or end_op not in fwd_ops:
                continue
            segments.append(
                main_program.global_block().ops[beg_op_idx:end_op_idx]
            )

        input_value = main_program.list_vars()
        value_map = paddle.pir.IrMapping()
        for val in input_value:
            value_map.add(val, val)

        segment_id = 1
        for segment in segments:
            first_bwd_used_op = bwd_ops[-1]
            for op in segment:
                bwd_used_op = self.get_first_bwd_used_op(op, bwd_ops)
                if first_bwd_used_op.id() > bwd_used_op.id():
                    first_bwd_used_op = bwd_used_op

            ori_segment_outputs = backward_utils.ValueSet()
            paddle.pir.set_insertion_point(first_bwd_used_op)
            for op in segment:
                ori_segment_outputs.update(op.results())
                op.set_int_attr("forward_recompute_segment_id", segment_id)
                if self.is_seed_used_by_dropout(op):
                    print("xxx seed op: ", op)
                    continue
                rc_op = op.clone(
                    value_map, paddle.pir.CloneOptions(False, True, True)
                )
                # rc_op.set_bool_attr("is_recompute_bw_op", True)
                rc_op.set_int_attr("backward_recompute_segment_id", segment_id)
                if first_bwd_used_op.has_attr('op_role'):
                    rc_op.set_int_attr("op_role", first_bwd_used_op.op_role)

                if first_bwd_used_op.has_attr('chunk_id'):
                    rc_op.set_int_attr("chunk_id", first_bwd_used_op.chunk_id)
                # print("xxx clone op: ", rc_op, "\n")
                # TODO: whethere delete attrs about recompute
            segment_id += 1

            for ori_value in ori_segment_outputs:
                rc_value = value_map.look_up(ori_value)
                ori_value.replace_grad_users_with(rc_value, set(bwd_ops))
        # print(main_program)

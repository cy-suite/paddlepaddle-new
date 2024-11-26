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
        self.set_attr("checkpoints", None)
        self.set_attr("loss", None)
        self.set_attr("dist_context", None)
        self.set_attr("no_grad_set", None)

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

    def _apply_single_impl(self, main_program, startup_program, context=None):
        print(main_program)
        checkpoints = self.get_attr("checkpoints")
        loss = self.get_attr("loss")
        self.dist_context = self.get_attr("dist_context")
        print("xxx self.loss: ", loss)
        print("xxx self.checkpoints: ", checkpoints)
        print("xxx self.dist_context: ", self.dist_context)
        fwd_ops, bwd_ops = self.get_fwd_bwd_ops(main_program)
        checkpoints.sort()
        print("xxx checkpoints: ", checkpoints)

        segments = []
        for idx in range(0, len(checkpoints), 2):
            if idx + 1 >= len(checkpoints):
                break
            beg_op_idx = checkpoints[idx]
            end_op_idx = checkpoints[idx + 1] - 1
            beg_op = main_program.global_block().ops[beg_op_idx]
            end_op = main_program.global_block().ops[end_op_idx]
            if beg_op not in fwd_ops or end_op not in fwd_ops:
                continue
            segments.append(
                main_program.global_block().ops[beg_op_idx : end_op_idx + 1]
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
            print("xxx first_backward_use_op: ", first_bwd_used_op)
            paddle.pir.set_insertion_point(first_bwd_used_op)
            for op in segment:
                ori_segment_outputs.update(op.results())
                # op.set_bool_attr("is_recompute_op", True)
                op.set_int_attr("forward_recompute_segment_id", segment_id)
                # print("xxx op: ", op)
                rc_op = op.clone(
                    value_map, paddle.pir.CloneOptions(False, True, True)
                )
                # rc_op.set_bool_attr("is_recompute_bw_op", True)
                rc_op.set_int_attr("backward_recompute_segment_id", segment_id)
                print("xxx rc_op: ", rc_op)
                if first_bwd_used_op.has_attr(
                    'op_role'
                ) and first_bwd_used_op.has_attr('chunk_id'):
                    rc_op.set_int_attr("op_role", first_bwd_used_op.op_role)
                    rc_op.set_int_attr("chunk_id", first_bwd_used_op.chunk_id)
                segment_id += 1

            # rc_segment_outputs = backward_utils.ValueSet()
            for ori_value in ori_segment_outputs:
                rc_value = value_map.look_up(ori_value)
                ori_value.replace_grad_users_with(rc_value, set(bwd_ops))
                # rc_segment_outputs.add(rc_value)
        print(main_program)

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

# from .common import copy_op_without_infer_shape
from paddle import pir
from paddle.autograd.backward_utils import ValueDict, ValueSet
from paddle.base import core

from .pass_utils import ProgramStats, _find_op_path

OpRole = core.op_proto_and_checker_maker.OpRole

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()


class RecomputeState(ProgramStats):
    def __init__(self, block, ops):
        super().__init__(block=block, ops=ops)
        self._block = block
        self._ops = ops
        self.value_op_deps = ValueDict()

    def init_value_name(self):
        pass

    def build_stats(self):
        for i, op in enumerate(self._ops):
            for value in op.operands_source():
                # if _some_in_set([value], self.value_op_deps) :
                if value in self.value_op_deps:
                    self.value_op_deps[value]["var_as_input_ops"].extend([i])
                else:
                    self.value_op_deps[value] = {}
                    self.value_op_deps[value]["var_as_input_ops"] = [i]
                    self.value_op_deps[value]["var_as_output_ops"] = []

            for value in op.results():
                # if _some_in_set([value], self.value_op_deps) :
                if value in self.value_op_deps:
                    self.value_op_deps[value]["var_as_output_ops"].extend([i])
                else:
                    self.value_op_deps[value] = {}
                    self.value_op_deps[value]["var_as_input_ops"] = []
                    self.value_op_deps[value]["var_as_output_ops"] = [i]

        print("xxx self.value_op_deps: ", self.value_op_deps)

    def get_recompute_segments(self, checkpoints):
        """get recompute segments from checkpoints"""
        print("xxx enter get_recompute_segments ")
        segments = []
        start_idx = -1
        pre_segment_end_idx = -1
        while start_idx + 1 < len(checkpoints):
            if start_idx == -1:
                ckpt_value = checkpoints[start_idx + 1]
                if ckpt_value not in self.value_op_deps:
                    start_idx += 1
                    continue
                op_idx_list = self.value_op_deps[ckpt_value][
                    "var_as_output_ops"
                ]
                if op_idx_list:
                    segments.append([0, max(op_idx_list) + 1])
            else:
                flag, min_idx, max_idx = self.is_subgraph(
                    [checkpoints[start_idx]], [checkpoints[start_idx + 1]]
                )
                if flag:
                    min_idx = self._update_segment_start(
                        min_idx, pre_segment_end_idx
                    )
                    segments.append([min_idx, max_idx + 1])
                else:
                    logging.info(
                        f"Could not recompute op range [{min_idx}] - [{max_idx + 1}] "
                    )
            start_idx += 1

        for i, (idx1, idx2) in enumerate(segments):
            logging.info(f"recompute segment[{i}]")
            logging.info(
                f"segment start op: [{self._ops[idx1].name()}]: [{self._ops[idx1].operands_source()}] [{self._ops[idx1].results()}]"
            )
            logging.info(
                f"segment end op: [{self._ops[idx2 - 1].name()}]: [{self._ops[idx2 - 1].operands_source()}] [{self._ops[idx2 - 1].results()}]"
            )

        return segments

    def modify_forward_desc_for_recompute(self, dist_context):
        """
        If program's forward part has 'dropout' op, this function will insert
        a seed op before it to guarantee that two dropout op have the same outputs.
        """
        op_names = [op.name() for op in self._ops]
        if "dropout" not in op_names:
            return

        op_idx = 0
        while op_idx < len(self._ops):
            cur_op = self._ops[op_idx]
            if "_grad" in cur_op.name():
                break
            if cur_op.name() != "dropout":
                op_idx += 1
                continue
            has_seed = False
            for value in cur_op.operands_source():
                input_op = value.get_defined_op()
                if input_op.name() == "seed":
                    has_seed = True
                    break
            if has_seed:
                op_idx += 1
                continue

        pass

        # cur_op_dist_attr = dist_context.get_dist_op_for_program(cur_op)
        # insert seed op to guarantee that two dropout op have the same outputs
        # op_unique_name = unique_name.generate("seed")
        # var_unique_name = unique_name.generate_with_ignorable_key(".".join(
        #     [op_unique_name, 'tmp']))
        # seed_var = self._block.create_var(
        #     name=var_unique_name,
        #     dtype='int32',
        #     type=core.VarDesc.VarType.LOD_TENSOR,
        #     persistable=False,
        #     stop_gradient=False)


#             # set new seed_var's dist_attr
#             ref_dims_mapping = [-1]
#             ref_process_mesh = cur_op_dist_attr.process_mesh
#             seed_var_dist_attr = set_var_dist_attr(
#                 dist_context, seed_var, ref_dims_mapping, ref_process_mesh)

#             seed = 0 if cur_op.attr("fix_seed") is False else int(
#                 cur_op.attr("seed"))
#             seed_op = self._block._insert_op_without_sync(
#                 index=cur_op.idx,
#                 type="seed",
#                 inputs={},
#                 outputs={"Out": seed_var},
#                 attrs={"seed": seed,
#                        "force_cpu": True})
#             # set new seed op's dist_attr
#             naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
#                 seed_op, ref_process_mesh, ref_dims_mapping, dist_context)

#             # modify dropout op's desc
#             self._ops.insert(op_idx, seed_op)
#             cur_op.desc.set_input("Seed", [var_unique_name])
#             cur_op.desc.remove_attr("fix_seed")
#             cur_op.desc.remove_attr("seed")
#             cur_op_dist_attr.set_input_dist_attr(seed_var.name,
#                                                  seed_var_dist_attr)
#             self._block._sync_with_cpp()
#             op_idx += 2


def _get_stop_gradients(program):
    no_grad_values = []
    # for v in no_grad_set:
    #     no_grad_set_name.append(v.name)
    for value in program.list_vars():
        if value.stop_gradient:
            print("xxx no gradient: ", value.index(), value)
            no_grad_values.append(value)
    # _append_grad_suffix_ 生成反向 value 的 name
    return no_grad_values


def find_index_in_list(value, values):
    for i, v in enumerate(values):
        if value.is_same(v):
            return i
    return -1


def _add_needed_descs_to_block(
    descs, block, main_block, vars_should_be_hold, dist_context
):
    """
    Get the recomputed ops which will insert the backward part
    """
    if len(descs) == 0:
        return []

    pir.set_insertion_point(descs[-1])
    # for op in

    result_descs = []
    for desc in descs:
        # if isinstance(desc, framework.Operator):
        if isinstance(desc, pir.Operation):
            desc = desc.attrs
        if isinstance(desc, tuple):
            desc = desc[0]
        is_needed = False
        for value in desc.results():
            if value.persistable:
                continue
            if value not in vars_should_be_hold:
                is_needed = True
        if is_needed:
            block.append()
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(desc)
            # set_dist_op_desc_original_id(new_op_desc, desc, dist_context)
            new_op_desc._set_attr(OP_ROLE_KEY, OpRole.Backward)
            result_descs.append(new_op_desc)
    return result_descs


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

    def get_fwd_bwd_ops(program):
        fwd_ops = []
        bwd_ops = []
        for op in program.global_block().ops:
            if op.op_role == int(OpRole.Forward):
                fwd_ops.append(op)
            elif op.op_role == int(OpRole.Backward):
                bwd_ops.append(op)
        return fwd_ops, bwd_ops

    def get_first_backward_use_op(self, fwd_op, bwd_ops):
        # user_op = bwd_ops[-1]
        for user_op in fwd_op.results()[0].all_used_ops():
            if user_op in bwd_ops:
                return user_op

    def _apply_single_impl(self, main_program, startup_program, context=None):
        # checkpoints = self.get_attr(
        #     "checkpoints"
        # )  # checkpoints (list): List of Variable
        # loss = self.get_attr("loss")
        # self.dist_context = self.get_attr("dist_context")
        # print("xxx self.loss: ", loss)
        # print("xxx self.checkpoints: ", checkpoints)
        # print("xxx self.dist_context: ", self.dist_context)
        # fwd_ops, bwd_ops = self.get_fwd_bwd_ops(main_program)
        # sort(checkpoints)
        # segments = []
        # for idx in range(0, len(checkpoints), 2):
        #     if idx + 1 >= len(checkpoints):
        #         break
        #     beg_op_idx = checkpoints[idx]
        #     end_op_idx = checkpoints[idx + 1]
        #     beg_op = main_program.global_block().ops[beg_op_idx]
        #     end_op = main_program.global_block().ops[end_op_idx]
        #     if beg_op not in fwd_ops or end_op not in fwd_ops:
        #         continue
        #     segments.append(main_program.global_block().ops[beg_op_idx : end_op_idx + 1])

        # mid_hold_values = analyze_mid_hold_values(
        #     program,
        #     saved_values,
        #     inputs,
        #     outputs,
        #     fwd_op_end_idx,
        #     backward_op_start_idx,
        # )
        # recompute_forward_subgraph = (
        #         _extract_forward_recompute_subgraph_for_backward(
        #             saved_values, mid_values
        #         )
        #     )
        # origin_subgraph_inputs = recompute_forward_subgraph["inputs"]
        # value_map = paddle.pir.IrMapping()
        # for input_value in origin_subgraph_inputs:
        #     value_map.add(input_value, input_value)

        # for segment in segments:

        #     for op in segment:
        #         # print("xxx op: ", op)

        #         first_backward_use_op = self.get_first_backward_use_op(op, bwd_ops)
        #         paddle.pir.set_insertion_point(first_backward_use_op)
        #         new_op = op.clone(
        #             value_map, paddle.pir.CloneOptions(False, True, True)
        #         )
        #         if (
        #             first_backward_use_op is not None
        #             and first_backward_use_op.has_attr('op_role')
        #             and first_backward_use_op.has_attr('chunk_id')
        #         ):
        #             new_op.set_int_attr("op_role", first_backward_use_op.op_role)
        #             new_op.set_int_attr("chunk_id", first_backward_use_op.chunk_id)
        # cloned_ops.append(new_op)

        checkpoints = self.get_attr(
            "checkpoints"
        )  # checkpoints (list): List of Variable
        loss = self.get_attr("loss")
        self.dist_context = self.get_attr("dist_context")
        print("xxx self.loss: ", loss)
        print("xxx self.checkpoints: ", checkpoints)
        print("xxx self.dist_context: ", self.dist_context)

        main_block = main_program.global_block()
        no_grad_values = _get_stop_gradients(main_program)
        op_path = _find_op_path(main_block, [loss], [], no_grad_values)
        for op in op_path:
            print("xxx op_path: ", op)

        # step 1: build recompute state
        rc_state = RecomputeState(main_block, op_path)
        # assump no drop op now
        # rc_state.modify_forward_desc_for_recompute(self._dist_context)
        rc_state.build_stats()
        checkpoints = rc_state.sort_checkpoints(checkpoints)
        segments = rc_state.get_recompute_segments(checkpoints)
        for seg in segments:
            print("xxx segment: ", seg)
        if segments == []:
            return

        # step 2: get vars_should_be_hold
        vars_should_be_hold = ValueSet()
        checkpoints = ValueSet(checkpoints)
        for segment in segments:
            vars_should_be_hold.update(
                rc_state.get_out_of_subgraph_vars(segment[0], segment[1])
            )
        cross_vars = set(vars_should_be_hold) - set(checkpoints)
        print("xxx cross_vars: ", cross_vars)
        logging.info(
            f"found [{len(cross_vars)}] vars which cross recompute segment: [{cross_vars}],"
            "better checkpoints might be set to reduce those vars"
        )
        vars_should_be_hold.update(rc_state.get_reserved_vars())
        vars_should_be_hold.update(rc_state.get_input_nodes())
        print("xxx vars_should_be_hold: ", vars_should_be_hold)
        vars_in_memory = vars_should_be_hold
        vars_in_memory.update(ValueSet(checkpoints))
        print("xxx vars_in_memory: ", vars_in_memory)

        # step 3: get recomputed fwd ops desc
        var_name_dict = ValueDict()
        ckpt_ops_dict = ValueDict()
        # buffer_block = main_block.program._create_block()
        # print("xxx buffer_block: ", buffer_block)
        for i, segment in enumerate(segments[::-1]):
            fwd_ops = op_path[segment[0] : segment[1]]
            # var_suffix = ".subprog_%d" % i
            for op in fwd_ops:
                input_and_output_values = ValueSet()
                input_and_output_values.update(op.operands_source())
                input_and_output_values.update(op.results())
                cur_op_dist_attr = op.dist_attr
                assert cur_op_dist_attr is not None
                for value in input_and_output_values:
                    if value.persistable or value in checkpoints:
                        continue
                    if value in vars_should_be_hold:
                        continue
                    if value not in var_name_dict:
                        ref_process_mesh = cur_op_dist_attr.process_mesh
                        if value in ValueSet(op.operands_source()):
                            pos = find_index_in_list(
                                value, op.operands_source()
                            )
                            ref_dims_mapping = (
                                cur_op_dist_attr.operand(pos)
                                .as_tensor_dist_attr()
                                .dims_mapping
                            )
                        else:
                            pos = find_index_in_list(value, op.results())
                            ref_dims_mapping = (
                                cur_op_dist_attr.result(pos)
                                .as_tensor_dist_attr()
                                .dims_mapping
                            )
                        # record recomputed var's old_name and new_name (old_name.subprog_XXX)
                        # create new var with new name
                        # ref_op = value.get_defineing_op()
                        # rc_op = copy_op_without_infer_shape(ref_op, main_block, self.dist_context)
                        # var_name_dict[value] = new value # name + var_suffix
                        # ref_var = value
                        # rc_var = main_block.create_var(
                        #     name=var_name_dict[name],
                        #     shape=ref_var.shape,
                        #     dtype=ref_var.dtype,
                        #     type=ref_var.type,
                        #     persistable=ref_var.persistable,
                        #     stop_gradient=ref_var.stop_gradient)
                        # # set new recomputed var's dist attr
                        # set_var_dist_attr(self._dist_context, rc_var,
                        #                   ref_dims_mapping, ref_process_mesh)
            # get recomputed segment's descs
            segment_descs = _add_needed_descs_to_block(
                fwd_ops,
                main_block,
                main_block,
                vars_in_memory,
                self._dist_context,
            )
            # rename recomputed ops' input and output var name
            # for key in var_name_dict:
            #     _rename_arg_(segment_descs, key, var_name_dict[key])

            # NOTE: one forward op could be correspond to multiple xxx_grad op.
            # When traversing all grad_ops in reverse, need to set a flag to indicate
            # whether the ckpt and its segment_descs can be used.
            ckpt_op = op_path[segment[1] - 1]
            ckpt_ops_dict[ckpt_op.id()] = [True, segment_descs]

        print("-----------------there is recompute pass")

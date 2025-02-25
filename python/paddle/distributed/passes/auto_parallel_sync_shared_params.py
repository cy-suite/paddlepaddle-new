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
import paddle.distributed as dist
from paddle.base.framework import (
    auto_complete_op_role,
)
from paddle.distributed.auto_parallel.static.process_group import (
    new_process_group,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.static.pir_io import get_pir_parameters

from .pass_base import PassBase, register_pass


@register_pass("auto_parallel_sync_shared_params")
class AutoParallelSyncSharedParamsPass(PassBase):
    def __init__(self):
        super().__init__()
        self.params_maybe_shared = []
        self.src_ranks = []
        self.dst_ranks = []

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _build_new_src_dist_attr(self, src_dist_attr, dst_dist_attr):
        new_src_dist_attr = (
            paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                dst_dist_attr.process_mesh,
                src_dist_attr.dims_mapping,
                src_dist_attr.partial_status,
            )
        )
        return new_src_dist_attr

    def pre_analysis_test(self, main_program, startup_program):

        params, _ = get_pir_parameters(main_program)
        for param in params:

            users = param.all_used_ops()
            # shared param has 3 user op  at least:
            #   1. stage_1_op
            #   2. stage_1_grad_op
            #   3. reshard(diff mesh) -> stage_n_op ...
            if len(users) < 2:
                continue
            user_name = None
            for reshard_op in users:
                if reshard_op.name() == "dist_op.reshard":
                    dist_attr = reshard_op.dist_attr
                    src_dist_attr = dist_attr.operand(0).as_tensor_dist_attr()
                    dst_dist_attr = dist_attr.result(0).as_tensor_dist_attr()
                    src_mesh = src_dist_attr.process_mesh
                    dst_mesh = dst_dist_attr.process_mesh
                    # reshard(diff mesh)

                    if src_mesh == dst_mesh:
                        continue
                    param_name = param.get_defining_op().str_attr(
                        'parameter_name'
                    )

                    # add builtin.parameter
                    insert_pos = None
                    for op in main_program.global_block().ops:
                        if op.name() == "builtin.parameter":
                            now_param_name = op.str_attr('parameter_name')
                            if now_param_name == param_name:
                                insert_pos = (
                                    main_program.global_block().ops.index(op)
                                    + 1
                                )
                                break

                    with auto_complete_op_role(main_program, OpRole.Forward):
                        with paddle.static.program_guard(
                            main_program, startup_program
                        ):
                            shared_param = paddle.pir.core.create_parameter(
                                dtype=param.dtype,
                                shape=param.shape,
                                name="shared_" + param_name,
                                process_mesh=dst_mesh,
                                placements=src_dist_attr.placements,
                                initializer=paddle.nn.initializer.Constant(
                                    value=0
                                ),
                            )
                    main_program.global_block().move_op(
                        shared_param.get_defining_op(), insert_pos
                    )
                    main_program.set_parameters_from(startup_program)

                    new_src_dist_attr = (
                        self._build_new_src_dist_attr(
                            src_dist_attr, dst_dist_attr
                        ),
                    )  # not sure
                    print("xxxx  src_mesh: ", type(src_mesh))
                    print("xxxx  dst_mesh: ", type(dst_mesh))
                    print("xxxx  new_src_dist_attr: ", new_src_dist_attr[0])
                    print("xxxx  dst_dist_attr: ", type(dst_dist_attr))
                    self.params_maybe_shared.append(
                        {
                            'src_mesh': src_mesh,
                            'dst_mesh': dst_mesh,
                            'shape': param.shape,
                            'dtype': param.dtype,
                            'src_dist_attr': src_dist_attr,  # not sure
                            'new_src_dist_attr': self._build_new_src_dist_attr(
                                src_dist_attr, dst_dist_attr
                            ),  # not sure,
                            'dst_dist_attr': dst_dist_attr,  # not sure
                            'param_name': param_name,
                        }
                    )

                    # modify reshard op dist attr
                    reshard_op.dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            dst_mesh,
                            [new_src_dist_attr[0]],
                            [dst_dist_attr],
                            -1,
                        )
                    )
                    reshard_op.operand(0).set_source(shared_param)

                    self.src_ranks.extend(src_mesh.process_ids)
                    self.dst_ranks.extend(dst_mesh.process_ids)

                    for nex_op in reshard_op.result(0).all_used_ops():
                        nex_op.set_str_attr(
                            "shared_parameter_name", "shared_" + param_name
                        )
                    user_name = param_name
            if user_name is None:
                continue
            for user in users:
                if user.name() == "dist_op.reshard":
                    continue
                user.set_str_attr("shared_parameter_name", param_name)

        if len(self.params_maybe_shared) == 0:
            return

        # prepare comm group
        for idx in range(len(self.src_ranks)):
            group = new_process_group(
                sorted([self.src_ranks[idx], self.dst_ranks[idx]])
            )
        all_group = new_process_group(
            sorted(src_mesh.process_ids + dst_mesh.process_ids)
        )
        return

    def apply_src_test(self, main_program, startup_program):
        for param_mess in self.params_maybe_shared:
            param_name = param_mess['param_name']
            src_mesh_ids = param_mess['src_mesh'].process_ids
            dst_mesh_ids = param_mess['dst_mesh'].process_ids
            src_dist_attr = param_mess['src_dist_attr']
            shape = param_mess['shape']
            dtype = param_mess['dtype']

            # startup program
            set_param_op = None
            for op in startup_program.global_block().ops:
                if op.name() == "builtin.set_parameter":
                    now_param_name = op.str_attr('parameter_name')
                    if now_param_name == param_name:
                        set_param_op = op
                        break
            idx = src_mesh_ids.index(self.cur_rank)
            peer_rank = dst_mesh_ids[idx]
            group = new_process_group(sorted([self.cur_rank, peer_rank]))
            param_value = set_param_op.operand_source(0)
            paddle.pir.set_insertion_point(set_param_op)

            bc_value = paddle._C_ops.broadcast(param_value, group.id, 0)
            bc_value.update_dist_attr(param_value.dist_attr())

            set_param_op.operand(0).set_source(bc_value)

            # dist program
            param = main_program.get_parameter_value_by_name(param_name)
            for grad_op in param.all_used_ops():
                if grad_op.op_role != 1:
                    continue
                grad_value = grad_op.result(0)
                idx = src_mesh_ids.index(self.cur_rank)
                peer_rank = dst_mesh_ids[idx]
                ar_group = new_process_group(
                    sorted(src_mesh_ids + dst_mesh_ids)
                )

                insert_pos = grad_value.all_used_ops()[0]
                if insert_pos.id() > grad_value.all_used_ops()[1].id():
                    insert_pos = grad_value.all_used_ops()[1]
                paddle.pir.set_insertion_point(insert_pos)

                allreduce_val = paddle._C_ops.all_reduce(
                    grad_value,
                    ar_group.id,
                    dist.ReduceOp.SUM,
                )
                allreduce_val.update_dist_attr(grad_value.dist_attr())
                allreduce_op = allreduce_val.get_defining_op()
                allreduce_op.op_role = grad_op.op_role

                allreduce_pos = (
                    main_program.global_block().ops.index(allreduce_op) - 1
                )
                main_program.global_block().move_op(allreduce_op, allreduce_pos)

                for user in grad_value.all_used_ops():
                    if user.name() == "pd_op.all_reduce":
                        continue
                    for idx, operand in enumerate(user.operands()):
                        if user.operand_source(idx).is_same(grad_value):
                            user.operand(idx).set_source(allreduce_val)

    def apply_dst_test(self, main_program, startup_program):
        for param_mess in self.params_maybe_shared:
            param_name = param_mess['param_name']
            src_mesh_ids = param_mess['src_mesh'].process_ids
            dst_mesh_ids = param_mess['dst_mesh'].process_ids
            src_dist_attr = param_mess['src_dist_attr']
            new_src_dist_attr = param_mess['new_src_dist_attr']
            shape = param_mess['shape']
            dtype = param_mess['dtype']

            # startup program
            set_param_op = None
            for op in startup_program.global_block().ops:
                if op.name() == "builtin.set_parameter":
                    now_param_name = op.str_attr('parameter_name')
                    if now_param_name == "shared_" + param_name:
                        set_param_op = op
                        break
            idx = dst_mesh_ids.index(self.cur_rank)
            peer_rank = src_mesh_ids[idx]
            group = new_process_group(sorted([self.cur_rank, peer_rank]))
            param_value = set_param_op.operand_source(0)
            paddle.pir.set_insertion_point(set_param_op)

            bc_value = paddle._C_ops.broadcast(param_value, group.id, 0)
            bc_value.update_dist_attr(param_value.dist_attr())

            # dist program
            param = main_program.get_parameter_value_by_name(
                "shared_" + param_name
            )
            slice_op = None
            for user in param.all_used_ops():
                if user.op_role == 0:
                    slice_op = user
                    break
            for grad_op in slice_op.result(0).all_used_ops():
                if grad_op.op_role != 1:
                    continue
                grad_value = grad_op.result(1).all_used_ops()[0].result(0)
                idx = dst_mesh_ids.index(self.cur_rank)
                peer_rank = src_mesh_ids[idx]
                ar_group = new_process_group(
                    sorted(src_mesh_ids + dst_mesh_ids)
                )

                insert_pos = grad_value.all_used_ops()[0]
                if insert_pos.id() > grad_value.all_used_ops()[1].id():
                    insert_pos = grad_value.all_used_ops()[1]
                paddle.pir.set_insertion_point(insert_pos)

                allreduce_val = paddle._C_ops.all_reduce(
                    grad_value,
                    ar_group.id,
                    dist.ReduceOp.SUM,
                )
                allreduce_val.update_dist_attr(grad_value.dist_attr())
                allreduce_op = allreduce_val.get_defining_op()
                allreduce_op.op_role = grad_op.op_role

                for user in grad_value.all_used_ops():
                    if user.name() == "pd_op.all_reduce":
                        continue
                    for idx, operand in enumerate(user.operands()):
                        if user.operand_source(idx).is_same(grad_value):
                            user.operand(idx).set_source(allreduce_val)

    def _apply_single_impl(self, main_program, startup_program, context):
        if len(self.params_maybe_shared) == 0:
            return

        assert len(self.params_maybe_shared) == 1
        self.cur_rank = paddle.distributed.get_rank()

        if self.cur_rank in self.src_ranks:
            self.apply_src_test(main_program, startup_program)
        if self.cur_rank in self.dst_ranks:
            self.apply_dst_test(main_program, startup_program)

        return

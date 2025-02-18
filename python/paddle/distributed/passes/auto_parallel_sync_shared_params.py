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
from paddle.base.framework import auto_complete_op_role
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
        self.optimizer = None
        self.hahah = None

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _get_sec_dst_mesh_from_reshard(self, reshard_op):
        assert reshard_op.name() == "dist_op.reshard"
        dist_attr = reshard_op.dist_attr
        src_dist_attr = dist_attr.operand(0).as_tensor_dist_attr()
        dst_dist_attr = dist_attr.result(0).as_tensor_dist_attr()
        src_mesh = src_dist_attr.process_mesh
        dst_mesh = dst_dist_attr.process_mesh
        return src_mesh, dst_mesh

    def _get_outptut_index_by_input_index(self, grad_op, input_value):
        input_idx = -1
        for idx, operand in enumerate(grad_op.operands_source()):
            if operand.is_same(input_value):
                input_idx = idx
                break
        assert input_idx != -1
        # hack some grad_op
        if grad_op.name() == "pd_op.matmul_grad":
            return input_idx
        if grad_op.name() == "dist_op.reshard":
            return 0
        return 0

    def _set_shared_attr_for_dst_mesh_op(
        self, operand, op, param_name, ori_src_mesh, ori_dst_mesh
    ):
        # reshard(src_mesh, dst_mesh) -> stage_n_op -> stage_n_grad_op -> reshard(dst_mesh, src_mesh)
        # to find reshard(now_src_mesh, now_dst_mesh)
        # require 1 & 2:
        #   1. ori_src_mesh == now_dst_mesh
        #   2. ori_dst_mesh == now_src_mesh
        if op.name() == "dist_op.reshard":
            now_src_mesh, now_dst_mesh = self._get_sec_dst_mesh_from_reshard(op)
            if ori_dst_mesh == now_src_mesh and ori_src_mesh == now_dst_mesh:
                op.set_str_attr("shared_parameter_name", param_name)
                # not sure
                # op.set_int_array_attr("shared_parameter_src_mesh", src_mesh.process_ids)
                # op.set_int_array_attr("shared_parameter_dst_mesh", dst_mesh.process_ids)
                return True

        # hack input_index -> grad_op -> outptut_index !!!
        output_idx = self._get_outptut_index_by_input_index(op, operand)

        find_reshard = False
        result = op.result(output_idx)
        for user in result.all_used_ops():
            # need grad_op
            if user.op_role != 1:
                continue
            if self._set_shared_attr_for_dst_mesh_op(
                result, user, param_name, ori_src_mesh, ori_dst_mesh
            ):
                op.set_str_attr("shared_parameter_name", param_name)
                # not sure
                # op.set_int_array_attr("shared_parameter_src_mesh", src_mesh.process_ids)
                # op.set_int_array_attr("shared_parameter_dst_mesh", dst_mesh.process_ids)
                # print("xxx set attr ok!")
                find_reshard = True
                # usually, only one reshard(diff mesh), it can return True directly
        return find_reshard

    def _build_new_src_dist_attr(self, src_dist_attr, dst_dist_attr):
        new_src_dist_attr = (
            paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                dst_dist_attr.process_mesh,
                src_dist_attr.dims_mapping,
                src_dist_attr.partial_status,
            )
        )
        return new_src_dist_attr

    def pre_analysis(self, main_program, startup_program, params_grads):
        params, _ = get_pir_parameters(main_program)
        for param in params:
            users = param.all_used_ops()
            # shared param has 3 user op  at least:
            #   1. stage_1_op
            #   2. stage_1_grad_op
            #   3. reshard(diff mesh) -> stage_n_op ...
            if len(users) < 3:
                continue
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
                    self.params_maybe_shared.append(
                        {
                            'src_mesh': src_mesh,
                            'dst_mesh': dst_mesh,
                            'shape': param.shape,
                            'dtype': param.dtype,
                            'src_dist_attr': src_dist_attr,  # not sure
                            'new_src_dist_attr': self._build_new_src_dist_attr(
                                src_dist_attr, dst_dist_attr
                            ),  # not sure
                            'dst_dist_attr': dst_dist_attr,  # not sure
                            'param_name': param_name,
                            'new_param': None,
                            'new_param_grad': None,
                            # 'mp_reshard_name': None,
                            'mp_reshard_operand_dist_attr': None,
                            'mp_reshard_result_dist_attr': None,
                        }
                    )
                    self.src_ranks.extend(src_mesh.process_ids)
                    self.dst_ranks.extend(dst_mesh.process_ids)

                    # set shared param attr of dst mesh ops
                    # reshard(src_mesh, dst_mesh) -> stage_n_op -> stage_n_grad_op -> reshard(dst_mesh, src_mesh)
                    self._set_shared_attr_for_dst_mesh_op(
                        param, reshard_op, param_name, src_mesh, dst_mesh
                    )

        if len(self.params_maybe_shared) == 0:
            return
        print("xxx self.params_maybe_shared  ", self.params_maybe_shared)
        # hack embedding shared param !!!!!!
        assert len(self.params_maybe_shared) == 1
        assert len(self.src_ranks) == len(self.dst_ranks)
        return

    def _is_grad_recv_combine_sum(self, grad_op, main_program):
        # 1、param -> grad   -> combine -> sum
        # 2、recv -> reshard -> combine -> sum
        if grad_op.op_role != 1 or len(grad_op.result(0).all_used_ops()) != 1:
            return None, None
        combine_op = grad_op.result(0).all_used_ops()[0]

        if (
            combine_op.name() != "builtin.combine"
            or len(combine_op.result(0).all_used_ops()) != 1
        ):
            return None, None
        sum_op = combine_op.result(0).all_used_ops()[0]

        if combine_op.num_operands() != 2:
            return None, None

        reshard_val = combine_op.operand_source(0)
        if reshard_val.is_same(grad_op.result(0)):
            reshard_val = combine_op.operand_source(1)
        reshard_op = reshard_val.get_defining_op()
        # if len(reshard_op.num_operands()) != 1:
        #     return None, None
        recv_op = reshard_op.operand_source(0).get_defining_op()
        if recv_op.name() != "pd_op.recv_v2":
            return None, None
        return recv_op, sum_op

    def _get_parameter_value_by_name_in_main_prog(
        self, main_program, param_name
    ):
        for block in main_program.blocks:
            for op in block.ops:
                if op.name() != "builtin.parameter":
                    continue
                if op.str_attr("parameter_name") != param_name:
                    continue
                return op.result(0)

    def _get_parameter_value_by_name_in_startup_prog(
        self, startup_program, param_name
    ):
        for block in startup_program.blocks:
            for op in block.ops:
                if op.name() != "builtin.set_parameter":
                    continue
                if op.str_attr("parameter_name") != param_name:
                    continue
                return op.operand_source(0)

    def _apply_single_impl_stage_src(self, main_program, startup_program):
        print("xxx apply single impl stage src")
        for param_mess in self.params_maybe_shared:
            param_name = param_mess['param_name']
            param_in_main = self._get_parameter_value_by_name_in_main_prog(
                main_program, param_name
            )

            # param has 3 user at least: send/stage_1_op/op_grad/
            send_op = None
            recv_op = None
            sum_op = None
            grad_op = None

            # 1、param -> send
            # 2、param -> grad \
            # recv -> reshard -> combine -> sum
            for user in param_in_main.all_used_ops():
                if user.name() == "pd_op.send_v2":
                    send_op = user
                elif user.op_role == 1:
                    grad_op = user
                    recv_op, sum_op = self._is_grad_recv_combine_sum(
                        user, main_program
                    )
            if send_op is None or recv_op is None or sum_op is None:
                return

            # broadcast + set_parameter
            # op -> set_parameter("para")
            # op -> broadcast -> set_parameter("shared_para")
            param_in_startup = (
                self._get_parameter_value_by_name_in_startup_prog(
                    startup_program, param_name
                )
            )
            param_op = param_in_startup.get_defining_op()
            paddle.pir.set_insertion_point_after(param_op)
            src_mesh_ids = param_mess['src_mesh'].process_ids
            dst_mesh_ids = param_mess['dst_mesh'].process_ids

            src_rank = paddle.distributed.get_rank()
            idx = src_mesh_ids.index(src_rank)
            dst_rank = dst_mesh_ids[idx]

            # bc_group = new_process_group(sorted(src_mesh_ids + dst_mesh_ids))
            # bc_value = paddle._C_ops.broadcast(
            #     op.operand_source(0), bc_group.id, 0
            # )
            # paddle._pir_ops.set_parameter(
            #     bc_value, "shared_" + var_name
            # )

            comm_group = new_process_group(
                [src_rank, dst_rank], group_type="p2p"
            )
            paddle._C_ops.send_v2(
                param_in_startup,
                comm_group.id,
                comm_group.ranks.index(dst_rank),
                True,
                False,
            )

            main_program.set_parameters_from(startup_program)

            # all_reduce
            # param -> grad  \
            # recv -> reshard -> combine -> sum
            # =>
            # param -> grad -> all_reduce
            paddle.pir.set_insertion_point_after(grad_op)
            ar_group = new_process_group(sorted(src_mesh_ids + dst_mesh_ids))
            allreduce_val = paddle._C_ops.all_reduce(
                grad_op.result(0),
                ar_group.id,
                dist.ReduceOp.SUM,
            )
            allreduce_op = allreduce_val.get_defining_op()
            allreduce_op.op_role = grad_op.op_role
            allreduce_op.chunk_id = grad_op.chunk_id
            allreduce_val.update_dist_attr(param_mess['dst_dist_attr'])

            sum_op.result(0).replace_all_uses_with(allreduce_val)

            combine_op = sum_op.operand_source(0).get_defining_op()
            reshard_op = recv_op.result(0).all_used_ops()[0]

            # hack!!!
            reshard_dist_attr = reshard_op.dist_attr
            reshard_operand_dist_attr = reshard_dist_attr.operand(
                0
            ).as_tensor_dist_attr()
            reshard_result_dist_attr = reshard_dist_attr.result(
                0
            ).as_tensor_dist_attr()
            param_mess['mp_reshard_operand_dist_attr'] = (
                reshard_operand_dist_attr
            )
            param_mess['mp_reshard_result_dist_attr'] = reshard_result_dist_attr

            send_op.erase()
            sum_op.erase()
            combine_op.erase()
            reshard_op.erase()
            recv_op.erase()
        return

    def _is_forward_recv_backward_send_dst(
        self, main_program, recv_op, src_mesh
    ):
        assert recv_op.name() == "pd_op.recv_v2"
        if (
            recv_op.has_attr("peer")
            or recv_op.attr("peer") not in src_mesh.process_ids
            or recv_op.op_role != 0
        ):
            return False

        peer = recv_op.has_attr("peer")
        send_op = None

        def is_send_dfs(op):
            if op.name() == "pd_op.send_v2":
                if (
                    op.has_attr("peer")
                    and op.attr("peer") == peer
                    and op.op_role == 1
                ):
                    send_op = op
                    return True
            for res in op.results():
                for user in res.all_used_ops():
                    if is_send_dfs(user):
                        return True
            return False

        is_send_dfs(recv_op)
        return send_op

    def _find_new_param(self, param_name):
        # print("xxx find_new_param: ", param_name)
        for param_mess in self.params_maybe_shared:
            # print("xxx param_mess: ", param_mess)
            if param_mess['param_name'] == param_name:
                if param_mess["new_param"] is None:
                    print("xxx error")
                return param_mess['new_param']

    def _find_parameter_insert_pos(self, main_program):
        pos = None
        for block in main_program.blocks:
            for op in block.ops:
                if op.name() == "builtin.parameter":
                    pos = op
                elif pos is None:
                    continue
                else:
                    return pos

    def _apply_single_impl_stage_dst(self, main_program, startup_program):
        print("xxx apply single impl stage dst ")
        # print("xxx self.xuexixi: ", self.xuexixi)

        # broadcast/recv + set_parameter
        for param_mess in self.params_maybe_shared:
            # param = param_mess['param']
            src_mesh_ids = param_mess['src_mesh'].process_ids
            dst_mesh_ids = param_mess['dst_mesh'].process_ids
            shape = param_mess['shape']
            dtype = param_mess['dtype']
            print("xxx add param_mess: ", param_mess)
            # broadcast + set_parameter
            # op -> set_parameter("para")
            # op -> broadcast -> set_parameter("shared_para")
            print("xxx add bc")
            param_name = param_mess['param_name']
            print("xxx param_name: ", param_name)

            paddle.pir.set_insertion_point_after(
                startup_program.global_block().ops[0]
            )
            dst_rank = paddle.distributed.get_rank()
            index = dst_mesh_ids.index(dst_rank)
            src_rank = src_mesh_ids[index]

            comm_group = new_process_group(
                [src_rank, dst_rank], group_type="p2p"
            )
            recv_value = paddle._C_ops.recv_v2(
                shape,
                dtype,
                comm_group.ranks.index(src_rank),
                comm_group.id,
                True,
                False,
            )
            recv_value.update_dist_attr(param_mess['new_src_dist_attr'])
            recv_op = recv_value.get_defining_op()
            print(type(param_mess['dst_mesh']))
            print(type(param_mess['new_src_dist_attr']))
            recv_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    param_mess['dst_mesh'],
                    [],
                    [param_mess['new_src_dist_attr']],
                    -1,
                )
            )

            print("xxx recv_v2 : ", recv_op)

            paddle._pir_ops.set_parameter(recv_value, "shared_" + param_name)
            main_program.set_parameters_from(startup_program)

            # print("xxx startup_program: ", startup_program)

            # builtin.parameter in main_program
            insert_op = self._find_parameter_insert_pos(main_program)
            paddle.pir.set_insertion_point_after(insert_op)
            new_param = paddle._pir_ops.parameter("shared_" + param_name)

            base_param = insert_op.result(0)

            new_param.is_distributed = base_param.is_distributed
            new_param.is_parameter = base_param.is_parameter
            new_param.need_clip = base_param.need_clip
            new_param.persistable = base_param.persistable
            new_param.trainable = base_param.trainable

            new_param.update_dist_attr(param_mess['new_src_dist_attr'])

            new_param_op = new_param.get_defining_op()
            new_param_op.op_role = 0
            new_param_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    param_mess['dst_mesh'],
                    [],
                    [param_mess['new_src_dist_attr']],
                    -1,
                )
            )
            param_mess["new_param"] = new_param
            print("xxx new_param: ", new_param)
            print("xxx new_param_op: ", new_param_op)

        # print("xxx main_program : ", main_program)

        del_ops = []
        # recv(param_name) -> slice/shared_data/..(son)  -> op(sun)
        #                                                -> grad_op(sun)
        # new_param -> slice/shared_data/..(son)  -> op(sun)
        #                                         -> grad_op(sun)
        for block in main_program.blocks:
            for recv_op in block.ops:
                if recv_op.name() == "pd_op.recv_v2":
                    # print("xxx recv_v2 op: ", recv_op)
                    param_name = None
                    son = recv_op.result(0)
                    son_op = son.all_used_ops()[0]
                    sun = son_op.result(0)
                    for sun_op in sun.all_used_ops():
                        # print("xxx user: ", sun_op)
                        if sun_op.has_attr("shared_parameter_name"):
                            if param_name is None:
                                param_name = sun_op.attrs()[
                                    "shared_parameter_name"
                                ]
                            elif (
                                param_name
                                != sun_op.attrs()["shared_parameter_name"]
                            ):
                                print(
                                    "xxx recv_v2 son have two diff shared_parameter_name"
                                )
                                return
                    if param_name is None:
                        # print("xxx recv_v2 son have no shared_parameter_name")
                        continue
                    new_param = self._find_new_param(param_name)
                    assert new_param is not None
                    son.replace_all_uses_with(new_param)
                    print("xxx old recv_v2 : ", recv_op)
                    print("xxx new new_param : ", new_param.get_defining_op())
                    del_ops += [recv_op]

        # param_grad -> send
        # allreduce(param_grad)
        for block in main_program.blocks:
            for send_op in block.ops:
                if send_op.name() == "pd_op.send_v2":
                    # print("xxx send_v2 op: ", send_op)
                    operand = send_op.operand_source(0)
                    ring_id = send_op.attrs()["ring_id"]
                    define_op = operand.get_defining_op()
                    if not define_op.has_attr("shared_parameter_name"):
                        # print("xxx send_v2 son have no shared_parameter_name")
                        continue
                    param_name = define_op.attrs()["shared_parameter_name"]
                    paddle.pir.set_insertion_point_after(define_op)

                    # shared_data mp reshard
                    tmp_param_mess = self.params_maybe_shared[0]
                    dst_type = tmp_param_mess['dtype']

                    share_data_value = paddle._C_ops.share_data_(operand)
                    share_data_value.set_type(dst_type)

                    share_data_op = share_data_value.get_defining_op()
                    src_dist_attr = tmp_param_mess[
                        'mp_reshard_operand_dist_attr'
                    ]
                    dst_dist_attr = tmp_param_mess[
                        'mp_reshard_result_dist_attr'
                    ]

                    new_src_dist_attr = self._build_new_src_dist_attr(
                        src_dist_attr, tmp_param_mess['dst_dist_attr']
                    )
                    new_dst_dist_attr = self._build_new_src_dist_attr(
                        dst_dist_attr, tmp_param_mess['dst_dist_attr']
                    )
                    share_data_op.dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            tmp_param_mess['dst_mesh'],
                            [new_src_dist_attr],
                            [new_dst_dist_attr],
                            send_op.chunk_id,
                        )
                    )

                    print("xxx share_data_op: ", share_data_op)

                    allreduce_value = paddle._C_ops.all_reduce(
                        share_data_value,
                        ring_id,
                        dist.ReduceOp.SUM,
                    )
                    allreduce_op = allreduce_value.get_defining_op()
                    allreduce_op.op_role = send_op.op_role
                    allreduce_op.chunk_id = send_op.chunk_id
                    print("xxx allreduce_op: ", allreduce_op)
                    # allreduce_value.update_dist_attr(param_mess['dst_dist_attr'])

                    # set new_param_grad
                    for param_mess in self.params_maybe_shared:
                        if param_mess['param_name'] == param_name:
                            param_mess['new_param_grad'] = allreduce_value
                            break

                    del_ops += [send_op]

        # del send & recv op
        for op in del_ops:
            op.erase()

        print("xxx startup_program : ", startup_program)
        print("xxx main_program : ", main_program)

        # optimizer
        for param_mess in self.params_maybe_shared:
            new_param = param_mess['new_param']
            new_param_grad = param_mess['new_param_grad']

            new_params_grads = [
                (p1, p2) for p1, p2 in zip([new_param], [new_param_grad])
            ]
            print("xxx new_param: ", type(new_param), new_param)
            print("xxx new_param_grad: ", type(new_param_grad), new_param_grad)
            with auto_complete_op_role(main_program, OpRole.Optimize):
                with paddle.static.program_guard(main_program, startup_program):
                    self.optimizer._apply_optimize(
                        self.loss,
                        startup_program,
                        params_grads=new_params_grads,
                    )
            print("xxx 1234")

    def _apply_single_impl(self, main_program, startup_program, context):
        if len(self.params_maybe_shared) == 0:
            return

        self.optimizer = self.get_attr("optimizer")
        self.loss = self.get_attr("loss")
        print("xxx apply _apply_single_impl ")

        # hack embedding shared param !!!!!!
        assert len(self.params_maybe_shared) == 1
        assert len(self.src_ranks) == 1
        assert len(self.dst_ranks) == 1
        cur_rank = paddle.distributed.get_rank()
        # print("xxx cur_rank: ", cur_rank)
        # print("xxx src_ranks: ", self.src_ranks[0])

        if cur_rank in self.src_ranks:
            self._apply_single_impl_stage_src(main_program, startup_program)
        if cur_rank in self.dst_ranks:
            self._apply_single_impl_stage_dst(main_program, startup_program)

        # print("xxx tmp startup_program: ", startup_program)

        # TODO!!!!!
        return

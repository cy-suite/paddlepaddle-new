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
from paddle.distributed.auto_parallel.process_mesh import merge_process_meshes
from paddle.distributed.passes.pass_utils import find_var_used_op_chunk_id
from paddle.distributed.utils.stream_utils import ExecutionStreamType

from ..process_group import new_process_group
from .base_reshard_func import ReshardFunction, copy_dist_attr_with_new_member


class SameStatusReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if src_dist_attr.dims_mapping != dst_dist_attr.dims_mapping:
            return False
        if src_dist_attr.partial_dims != dst_dist_attr.partial_dims:
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh == out_mesh:
            return False
        if in_mesh.shape != out_mesh.shape:
            return False
        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_mesh = src_dist_attr.process_mesh
        dst_mesh = dst_dist_attr.process_mesh

        cur_global_rank = paddle.distributed.get_rank()

        for src, dst in zip(src_mesh.process_ids, dst_mesh.process_ids):
            if src != dst:
                new_process_group([src, dst])
                new_process_group([dst, src])

        is_send = True
        for src, dst in zip(src_mesh.process_ids, dst_mesh.process_ids):
            if cur_global_rank != src and cur_global_rank != dst:
                continue
            if src == cur_global_rank:
                chunk_id = -1
                if (
                    src_value.get_defining_op().name() == "pd_op.add_n"
                    and src_value.get_defining_op()
                    .operand_source(0)
                    .get_defining_op()
                    .name()
                    == "builtin.combine"
                ):
                    add_n_op = src_value.get_defining_op()
                    combine_op = add_n_op.operand_source(0).get_defining_op()
                    combine_op_chunk_id_list = []
                    for input in combine_op.operands():
                        if input.source().get_defining_op().dist_attr:
                            combine_op_chunk_id_list.append(
                                input.source()
                                .get_defining_op()
                                .dist_attr.chunk_id
                            )
                        else:
                            combine_op_chunk_id_list.append(-1)
                    # check combine_op operands chunk_id equal
                    assert all(
                        x == combine_op_chunk_id_list[0]
                        for x in combine_op_chunk_id_list
                    ), "combine_op's operands has different chunk_id."
                    chunk_id = combine_op_chunk_id_list[0]
                    # reset add_n chunk_id
                    add_n_op.dist_attr = (
                        paddle.base.libpaddle.pir.create_op_dist_attribute(
                            add_n_op.dist_attr.process_mesh,
                            add_n_op.dist_attr.operands(),
                            add_n_op.dist_attr.results(),
                            chunk_id,
                        )
                    )
                else:
                    if src_value.get_defining_op().dist_attr:
                        chunk_id = (
                            src_value.get_defining_op().dist_attr.chunk_id
                        )

                # the root rank will broadcast the src_value to the dst rank
                tmp_value = paddle._C_ops.share_data_(src_value)
                value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    src_value.type(), src_value.dist_attr()
                )
                tmp_value.set_type(value_type)
                op = tmp_value.get_defining_op()
                op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        src_mesh, [src_dist_attr], [src_dist_attr], chunk_id
                    )
                )

            elif dst == cur_global_rank:
                is_send = False
                all_used_ops = src_value.all_used_ops()
                chunk_id = -1
                for used_op in all_used_ops:
                    var = used_op.result(0)
                    if var.dist_attr().process_mesh == dst_mesh:
                        chunk_id = find_var_used_op_chunk_id(var)

                assert (
                    -1 not in dst_type.shape
                ), "dynamic shape is not supported by pir-auto parallel yet."

                # create the buffer on other ranks for receving the data
                value_dist_attr = copy_dist_attr_with_new_member(
                    dst_dist_attr, new_process_mesh=dst_mesh
                )
                tmp_value = paddle.zeros(dst_type._local_shape, dst_type.dtype)
                tmp_value.set_type(dst_type)
                op = tmp_value.get_defining_op()
                op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        dst_mesh, [], [value_dist_attr], chunk_id
                    )
                )

            comm_group = new_process_group([src, dst])
            broadcast_value = paddle._C_ops.broadcast(
                tmp_value,
                comm_group.id,
                comm_group.ranks.index(src),
            )
            broadcast_value.set_type(dst_type)
            broadcast_op = broadcast_value.get_defining_op()
            broadcast_op.set_execution_stream(
                ExecutionStreamType.DefaultStream.value
            )
            bcast_mesh = merge_process_meshes([src_mesh, dst_mesh])
            broadcast_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    bcast_mesh, [src_dist_attr], [dst_dist_attr], chunk_id
                )
            )
        if is_send:
            # fake var will be removed in remove_other_rank_op_pass.
            fake_var = paddle._C_ops.reshard_v2(src_value, dst_dist_attr)
            fake_var.set_type(dst_type)
            return fake_var
        else:
            return broadcast_value

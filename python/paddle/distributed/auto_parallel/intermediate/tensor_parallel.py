#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import re

import paddle
import paddle.distributed as dist

from .parallel_base import ParallelModel, ParallelOptimizer


class PlanBase:
    def apply(self, param, process_mesh, shard_weight, shard_bias):
        raise NotImplementedError


class ColWiseParallel(PlanBase):
    """
    TODO(yaliu): DOC
    """

    def __init__(self):
        super().__init__()

    def apply(self, layer, process_mesh, shard_weight=True, shard_bias=True):
        index = process_mesh.dim_names.index('mp')
        size = len(process_mesh.shape)
        placement = [dist.Replicate() for _ in range(size)]
        assert isinstance(layer, paddle.nn.Layer)
        if not isinstance(layer, (paddle.nn.Linear, paddle.nn.Embedding)):
            logging.warning(
                f"ColWiseParallel is designed to handle Linear and Embedding. "
                f"But got {layer.__class__.__name__}. "
                f"Will try to shard weight and bias if the layer contains one."
            )
        if (
            hasattr(layer, "weight")
            and layer.weight is not None
            and shard_weight
        ):
            placement[index] = dist.Shard(1)
            assert len(layer.weight.shape) == 2
            layer.weight = dist.shard_tensor(
                layer.weight,
                process_mesh,
                placement,
            )
        if hasattr(layer, "bias") and layer.bias is not None and shard_bias:
            placement[index] = dist.Shard(0)
            assert len(layer.bias.shape) == 1
            layer.bias = dist.shard_tensor(layer.bias, process_mesh, placement)


class RowWiseParallel(PlanBase):
    """
    TODO(yaliu): DOC
    """

    def __init__(self):
        super().__init__()

    def apply(self, layer, process_mesh, shard_weight=True, shard_bias=False):
        index = process_mesh.dim_names.index('mp')
        size = len(process_mesh.shape)
        placement = [dist.Replicate() for _ in range(size)]
        placement[index] = dist.Shard(0)
        assert isinstance(layer, paddle.nn.Layer)
        if not isinstance(layer, (paddle.nn.Linear, paddle.nn.Embedding)):
            logging.warning(
                f"RowWiseParallel is designed to handle Linear and Embedding. "
                f"But got {layer.__class__.__name__}. "
                f"Will try to shard weight if the layer contains one."
            )
        if (
            hasattr(layer, "weight")
            and layer.weight is not None
            and shard_weight
        ):
            assert len(layer.weight.shape) == 2
            layer.weight = dist.shard_tensor(
                layer.weight,
                process_mesh,
                placement,
            )


class TensorParallel(ParallelModel):
    """
    TODO(yaliu): DOC
    """

    def __init__(self, model, parallelize_plan=None):
        super().__init__(model)
        if parallelize_plan is not None:
            assert isinstance(parallelize_plan, dict)
            for key, plan in parallelize_plan.items():
                assert isinstance(
                    key, str
                ), "The key of the parallelize plan should be a string."
                assert isinstance(
                    plan, PlanBase
                ), "The value the the parallelize plan should be a instance of PlanBase."

            self.global_mesh = dist.auto_parallel.get_mesh()
            self.parallelize_plan = parallelize_plan
            self.tp_parallelizer = self.tensor_parallelizer_fn

    def get_mesh(self):
        # TODO(yaliu): fit pp
        assert "mp" in self.global_mesh.dim_names
        if "pp" in self.global_mesh.dim_names:
            assert (
                self.global_mesh.get_dim_size("pp") == 1
            ), "Not support pp with mp for now."
            mesh = self.global_mesh.get_mesh_with_dim("pp")[0]
        else:
            mesh = self.global_mesh
        assert len(mesh.shape) in [1, 2]
        return mesh

    def parse_layer(self, name):
        for key, plan in self.parallelize_plan.items():
            shard_weight = True
            shard_bias = True
            if key.endswith(".weight"):
                key = key.replace(".weight", "")
                shard_bias = False
            elif key.endswith(".bias"):
                key = key.replace(".bias", "")
                shard_weight = False
            re_find = re.match(key, name)
            if key == name or (re_find is not None and re_find.string == name):
                return plan, shard_weight, shard_bias
        return None, None, None

    def tensor_parallelizer_fn(self, model):
        if self.parallelize_plan is None:
            return
        for name, layer in model.named_sublayers():
            if len(layer.sublayers()) == 0:
                plan, shard_weight, shard_bias = self.parse_layer(name)
                if plan is not None:
                    plan.apply(layer, self.get_mesh(), shard_weight, shard_bias)
        return model


def tensor_parallel(model, parallelize_plan=None, optimizer=None):
    """
    TODO(yaliu): DOC
    """
    if parallelize_plan is None:
        # Do nothing if no plan.
        logging.warning(
            "No parallelize plan, tensor parallel won't do anything."
        )
        return model, optimizer

    global_mesh = dist.auto_parallel.get_mesh()

    assert (
        global_mesh is not None
    ), "global mesh must not be None, please call fleet.auto.set_mesh(global_mesh) firstly"
    assert (
        "mp" in global_mesh.dim_names
    ), "mp must in the mesh dim_names when use tensor_parallel"

    model = TensorParallel(model, parallelize_plan)
    if optimizer is not None:
        optimizer = ParallelOptimizer(optimizer)

    return model, optimizer

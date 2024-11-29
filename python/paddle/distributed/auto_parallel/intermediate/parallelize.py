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
import warnings

from .parallel_base import ParallelOptimizer, parallelize_model_and_optimizer
from .pipeline_parallel import pipeline_parallel
from .sharded_data_parallel import sharded_data_parallel
from .tensor_parallel import tensor_parallel


def parallelize(
    model, optimizer, dp_config=None, mp_config=None, pp_config=None
):
    """
    `parallelize` will parallelize both model and optimizer.

    Args:
        model (paddle.nn.Layer): The single card model to be parallelized
        optimizer (paddle.optimizer.Optimizer): The optimizer to be parallelized
        dp_config (dict): config for data/sharding parallel
        {
            "sharding_level": 0, # can be chosen from 0/1/2/3. 0 for data parallel. 1/2/3 for different shrading stage.
            "offload": False,   # offload or not, not supported for now
            "exclude_layer": None,  # layer not doing sharding, not supported for now
        }
        mp_config (dict): config for tensor parallel
        {
            "parallelize_plan": dict(), # plan to parallelize the model
        }
        An example for the parallelize_plan is:
        ```
        plan = {
            "llama.embed_tokens": ColWiseParallel(gather_output=True),
            "llama.layers.*.self_attn.q_proj": ColWiseParallel(),
            "llama.layers.*.self_attn.k_proj": ColWiseParallel(),
            "llama.layers.*.self_attn.v_proj": ColWiseParallel(),
            "llama.layers.*.self_attn.o_proj": RowWiseParallel(),
            "llama.layers.*.mlp.gate_proj": ColWiseParallel(),
            "llama.layers.*.mlp.up_proj": ColWiseParallel(),
            "llama.layers.*.mlp.down_proj": RowWiseParallel(),
            "lm_head.weight": ColWiseParallel(),
        }
        ```
        pp_config (dict): config for pipeline parallel
        {
            "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point
                if split_spec is a string or list, such as "llama.layer" or ["llama.layerA", "llama.layerB"], Then the layer with same prefix a will be divided equally according to the size of pipeline degree.
                if split_spec is a OrderedDict|dict, key is the layer name, and the value is the split position that can be SplitPoint.BEGINNING or SplitPoint.END, the order of the keys is the order of the pipeline stage.
                NOTE: dict is also ordered after python3.7, so use dict at this time
            "global_spec": str|list(str), make the output tensor of specific layers on global mesh
        }

    Returns:
        model: (paddle.nn.Layer) the model after parallelize.
        optimizer: (paddle.optimizer.Optimizer) the optimizer after parallelize.
    """
    if pp_config is not None:
        assert isinstance(pp_config, dict)
        model, optimizer = pipeline_parallel(
            model,
            optimizer,
            pp_config,
        )
    if mp_config is not None:
        assert isinstance(mp_config, dict)
        model, optimizer = tensor_parallel(model, optimizer, mp_config)
    if dp_config is not None:
        assert isinstance(dp_config, dict)
        if 'sharding_level' not in dp_config.keys():
            warnings.warn(
                "The dp_config doesn't contain sharding_level, will run under dp."
            )
        model, optimizer = sharded_data_parallel(
            model,
            optimizer,
            config=dp_config,
        )
    model, optimizer = parallelize_model_and_optimizer(model, optimizer)
    return model, optimizer


has_parallelized_model = False


def parallelize_model(model, dp_config=None, mp_config=None, pp_config=None):
    global has_parallelized_model
    has_parallelized_model = True
    model, _ = parallelize(model, None, dp_config, mp_config, pp_config)
    return model


def parallelize_optimizer(
    optimizer, dp_config=None, mp_config=None, pp_config=None
):
    global has_parallelized_model
    assert (
        has_parallelized_model
    ), "Please parallelize the model before parallelize optimizer."
    param_list = optimizer._parameter_list
    if isinstance(param_list[0], dict):
        for param_group in param_list:
            for param in param_group['params']:
                assert (
                    param.is_dist()
                ), "Please use model after parallelize to create optimizer."
    else:
        for param in param_list:
            assert (
                param.is_dist()
            ), "Please use model after parallelize to create optimizer."

    level = None
    sharding_mesh_dim = None
    if dp_config is not None:
        if 'sharding_level' not in dp_config.keys():
            warnings.warn(
                "The dp_config doesn't contain sharding_level, will run under dp."
            )
        level = dp_config.get('sharding_level')
        sharding_mesh_dim = dp_config.get('sharding_mesh_dim', "dp")
    optimizer = ParallelOptimizer(optimizer, level, sharding_mesh_dim)
    optimizer = optimizer.parallelize()
    return optimizer

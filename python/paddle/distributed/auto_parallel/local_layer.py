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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import paddle
import paddle.distributed as dist
from paddle.nn import Layer

if TYPE_CHECKING:
    from paddle.distributed import Placement
    from paddle.distributed.auto_parallel.process_mesh import ProcessMesh


class LocalLayer(Layer):
    def __init__(
        self, out_dist_attrs: list[tuple[ProcessMesh, list[Placement]]]
    ):
        super().__init__()
        self.out_dist_attrs = out_dist_attrs

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        inputs = list(inputs)
        for i in range(len(inputs)):
            if inputs[i].is_dist():
                inputs[i] = dist.auto_parallel.api.dtensor_to_local(inputs[i])
        outputs = Layer.__call__(self, *inputs, **kwargs)
        list_outs = paddle.utils.flatten(outputs)
        for idx in range(len(list_outs)):
            list_outs[i] = dist.auto_parallel.api.dtensor_from_local(
                list_outs[i],
                self.out_dist_attrs[i][0],
                self.out_dist_attrs[i][1],
            )
        return paddle.utils.pack_sequence_as(outputs, list_outs)

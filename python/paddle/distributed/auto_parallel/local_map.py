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
from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Union

import paddle
import paddle.distributed as dist
from paddle.utils import flatten, pack_sequence_as

if TYPE_CHECKING:
    from paddle.distributed import ProcessMesh

__all__ = ["local_map"]

PlacementType = Sequence[dist.Placement] | None
InputPlacements = tuple[PlacementType, ...] | None
OutputPlacements = Union[PlacementType, tuple[PlacementType, ...]]


def local_map(
    func: Callable,
    out_placements: OutputPlacements,
    in_placements: InputPlacements = None,
    process_mesh: ProcessMesh | None = None,
    *,
    redistribute_inputs: bool = False,
):
    """
    :meth:`local_map` is an experimental API that allows users to pass dist_tensors
    to a function that is written to be applied on ``paddle.Tensor`` s. It works by extracting
    the local components of dist_tensors, calling the function, and wrapping the outputs
    as dist_tensors according to the ``out_placements``.

    Args:
        func (Callable): The function to be applied on each local shard of dist_tensors.
        out_placements (Union[`PlacementType`, Tuple[`PlacementType`, ...]]):
            The desired placements of the dist_tensors in ``func``'s flattened output.
            If the flattened ``output`` is a single value, the ``out_placements`` should be
            of type `PlacementType`. Otherwise if the flattened ``output`` has multiple
            values, the ``out_placements`` should be a tuple of `PlacementType` values 1:1
            mapping to the flattened ``output``.
            For tensor output, we use `PlacementType` as its placements (a sequence of
            `Placement` values). For non-tensor output, the `PlacementType` should be `None`.
            Note that when no dist_tensor argument is passed in, even if `out_placements`
            is not `None`, the result function will ignore the desired placements because
            the function is not running with dist_tensors.

        in_placements (Tuple[`PlacementType`, ...], optional):
            The required placements of the dist_tensors in the flattened inputs of ``func``.
            If ``in_placements`` is specified, :meth:`local_map` will examine whether the
            placements of each dist_tensor argument matches the required placements.
            If the placements don't match and ``redistribute_inputs`` is ``False``, an
            exception will be raised. If ``redistribute_inputs`` is ``True``, the argument
            will be first redistributed to the required placements before passing its local
            tensor to ``func``.
            The only exception is when required placements are not ``None`` and the argument
            is a regular ``paddle.Tensor``. In this case, the placements check will be
            skipped and the argument will be directly passed to ``func``.
            If ``in_placements`` is ``None``, no placements check will be performed.
            Default: None

        process_mesh (:class:`ProcessMesh`, optional):
            The process mesh that all dist_tensors are placed on. If not specified,
            this will be inferred from the input dist_tensors' process mesh.
            local_map requires all dist_tensors to be placed on the same process mesh.
            Default: None

        redistribute_inputs (bool, optional):
            Whether to redistribute the input dist_tensors when their placements
            don't match the required input placements. If this value is ``False`` and
            some dist_tensor input has different placements, an exception will
            be raised. Default: False

    Returns:
        A ``Callable`` that applies ``func`` to each local shard of the input dist_tensors
        and returns a dist_tensor constructed from the return value of ``func``.

    Raises:
        AssertionError: If the input dist_tensor is not placed on the same process
            mesh, or if they are placed on a different process mesh than the ``process_mesh``
            argument passed in.

        AssertionError: For any non-tensor output, we require its corresponding output
            placement in ``out_placements`` be None. An AssertionError will be raised if
            this is not the case.

        ValueError: If ``redistribute_inputs=False`` but the input dist_tensor needs
            a redistribution according to ``in_placements``.

    Example:
        >>> from __future__ import annotations
        >>> import paddle
        >>> import paddle.distributed as dist
        >>> from paddle import Tensor
        >>> from paddle.distributed import ProcessMesh
        >>>
        >>> def custom_function(x):
        >>>     mask = paddle.zeros_like(x)
        >>>     if dist.get_rank() == 0:
        >>>         mask[1:3] = 1
        >>>     else:
        >>>         mask[4:7] = 1
        >>>     x = x * mask
        >>>     mask_sum = paddle.sum(x)
        >>>     mask_sum = mask_sum / mask.sum()
        >>>     return mask_sum
        >>>
        >>> # Initialize distributed environment
        >>> dist.init_parallel_env()
        >>> mesh = ProcessMesh([0, 1], dim_names=["x"])
        >>>
        >>> # Create input data
        >>> local_input = paddle.arange(0, 10, dtype="float32")
        >>> local_input = local_input + dist.get_rank()
        >>>
        >>> # Convert to distributed tensor
        >>> input_dist = dist.auto_parallel.api.dtensor_from_local(
        >>>     local_input, mesh, [dist.Shard(0)]
        >>> )
        >>>
        >>> # Wrap function with local_map
        >>> wrapped_func = dist.local_map(
        >>>     custom_function,
        >>>     out_placements=[dist.Partial(dist.ReduceType.kRedSum)],
        >>>     in_placements=(dist.Shard(0),),
        >>>     process_mesh=mesh
        >>> )
        >>>
        >>> # Apply function to distributed tensor
        >>> output_dist = wrapped_func(input_dist)
        >>>
        >>> # Collect and print results
        >>> local_value = output_dist._local_value()
        >>> gathered_values: list[Tensor] = []
        >>> dist.all_gather(gathered_values, local_value)
        >>>
        >>> print(f"[Rank 0] local_value={gathered_values[0].item()}")
        [Rank 0] local_value=1.5
        >>> print(f"[Rank 1] local_value={gathered_values[1].item()}")
        [Rank 1] local_value=6.0
        >>> print(f"global_value (distributed)={output_dist.item()}")
        global_value (distributed)=7.5
    """

    def wrapped(process_mesh: ProcessMesh | None, *args, **kwargs):
        # Process input arguments
        flat_args = flatten(args)
        if in_placements is not None:
            assert len(in_placements) == len(flat_args), (
                f"in_placements length {len(in_placements)} does not match "
                f"number of input args {len(flat_args)}!"
            )

        # Assume all dist_tensors are on the same process mesh
        flat_local_args = []
        seen_dist_tensor = False

        for idx, arg in enumerate(flat_args):
            if _is_distributed_tensor(arg):
                # TODO: the current code doesn't consider the uneven sharding case
                # Need to think about what the consequence is when the input DTensor
                # is uneven sharded.
                if process_mesh is None:
                    process_mesh = arg.process_mesh

                seen_dist_tensor = True

                assert arg.process_mesh == process_mesh, (
                    f"Mismatched process mesh for arg {arg}: "
                    f"got {arg.process_mesh} but expected {process_mesh}!"
                )

                if in_placements is not None:
                    spec = in_placements[idx]
                    assert spec is not None, (
                        f"Expected placements for dist_tensor input {arg} "
                        f"but got {spec}!"
                    )

                    if not isinstance(spec, list):
                        spec = [spec]

                    if arg.placements != spec:
                        if redistribute_inputs:
                            # Redistribute to input placements
                            arg = arg.redistribute(process_mesh, spec)
                        else:
                            raise ValueError(
                                f"Mismatched placements for arg {arg}: "
                                f"got {arg.placements} but required {spec}! "
                                "Set redistribute_inputs=True if redistribution "
                                "is needed."
                            )

                local_arg = dist.auto_parallel.api.dtensor_to_local(
                    arg, process_mesh, spec
                )
                flat_local_args.append(local_arg)
            else:
                if in_placements is not None and not isinstance(
                    arg, paddle.Tensor
                ):
                    spec = in_placements[idx]
                    assert spec is None, (
                        f"Expected None placements for non-tensor input {arg} "
                        f"but got {spec}!"
                    )
                flat_local_args.append(arg)

        local_args = pack_sequence_as(args, flat_local_args)
        out = func(*local_args, **kwargs)

        if seen_dist_tensor:
            flat_out = flatten(out)
            out_placements_tuple = (
                out_placements
                if isinstance(out_placements, tuple)
                else (out_placements,)
            )

            assert len(flat_out) == len(out_placements_tuple), (
                "local_map requires one PlacementType for each output value, "
                f"got {len(out_placements_tuple)} placements but expected "
                f"{len(flat_out)}!"
            )

            flat_dist_out = []
            for out, spec in zip(flat_out, out_placements_tuple):
                if isinstance(out, paddle.Tensor):
                    assert not _is_distributed_tensor(
                        out
                    ), f"Expected dense tensor output but got {type(out)}: {out}"

                    flat_dist_out.append(
                        dist.auto_parallel.api.dtensor_from_local(
                            out, process_mesh, spec
                        )
                    )
                else:
                    assert spec is None, (
                        f"Expected None placements for non-tensor output {out} "
                        f"but got {spec}!"
                    )
                    flat_dist_out.append(out)

            return pack_sequence_as(out, flat_dist_out)
        else:
            return out

    def _is_distributed_tensor(tensor) -> bool:
        """
        Check if an input is a dist_tensor.

        Args:
            tensor: The input to check

        Returns:
            bool: True if the input is a dist_tensor, False otherwise
        """
        return (
            isinstance(tensor, paddle.Tensor)
            and hasattr(tensor, 'is_dist')
            and tensor.is_dist()
        )

    return functools.partial(wrapped, process_mesh)

/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/batched_tensor.h"
#include "paddle/utils/small_vector.h"
#include "paddle/utils/test_macros.h"
#include "vector"

// This file contains abstractions used for transforming *logical* vmap
// arguments into *physical* arguments. (Keep reading for definitions of these
// terms).

// NOTE: [Logical vs physical args]
// Consider the following vmap.
//   vmap(vmap(func, in_dims=(2,)), in_dims=(0,))(torch.ones(2, 3, 4))
// This would produce a BatchedTensor wrapping a paddle::Tensor of size [2, 3,
// 4], with batch dims 0 and 2:
//   BatchedTensor(ones(2, 3, 4), bdims=[(lvl=1,dim=0),(lvl=2,dim=2)])
//
// We say the *logical* view of the tensor has size [3] -- tensors inside
// `func` appear to have size [3].
// However, the *physical* underlying tensor (the one passed to vmap) has size
// [2, 3, 4].
//
// This notion of logical vs physical also extends to non-tensor arguments.
// Consider the previous tensor; let's assume the user called
// `torch.sum(tensor, dim=0)` inside of `func`. Then the logical
// dimension they are reducing over is dim 0 but the physical dim is dim 1
// (the first non-batch dimension)
// Overload for explicit function and ArrayRef
template <class F, class T>
inline auto fmap(const T& inputs, const F& fn)
    -> std::vector<decltype(fn(*inputs.begin()))> {
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  for (const auto& input : inputs) r.push_back(fn(input));
  return r;
}

// C++ forbids taking an address of a constructor, so here's a workaround...
// Overload for constructor (R) application
template <typename R, typename T>
inline std::vector<R> fmap(const T& inputs) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for (auto& input : inputs) r.push_back(R(input));
  return r;
}

// Forward declared; see NOTE: [What is a VmapPhysicalView?]
struct VmapPhysicalView;

// Most PyTorch operators take 4 or fewer inputs.
constexpr int64_t kVmapTransformStaticInputSize = 4;
using VmapPhysicalViewVec =
    paddle::small_vector<VmapPhysicalView, kVmapTransformStaticInputSize>;

// Pytorch generally advertises good performance for <= 5 dims.
// (see ATen/core/DimVector.h). We add a few extra dims (~3) for vmap
// dimensions to get 8. Adjust this number as necessary
constexpr int64_t kVmapStaticDimVecSize = 8;
using VmapDimVector = paddle::small_vector<int64_t, kVmapStaticDimVecSize>;

// NOTE: [What is an VmapTransform?]
// An *VmapTransform* converts logical views of tensors to physical views.
//
// Batching rules use vmap_transforms to convert logical arguments to
// physical arguments, then call one or more at:: operator that handles the
// physical arguments, and then converts the physical result back to a logical
// argument.

// VmapTransform for operators that take tensors with multiple batch dims.
// Given one or more logical views on Tensors, `logicalToPhysical`
// permutes all of the batch dims to the front of the tensor, aligns
// and expands the batch dims to match each other (according to their `level`),
// and returns a VmapPhysicalView on the tensor(s).
struct TEST_API MultiBatchVmapTransform {
  static VmapPhysicalView logicalToPhysical(
      const paddle::Tensor& logical_tensor);
  static VmapPhysicalViewVec logicalToPhysical(
      const std::vector<paddle::Tensor>& logical_tensors);
};

// VmapTransform for operators that broadcast all inputs.
// Given some logical views on Tensors, `logicalToPhysical`:
// - permutes all of the batch dims to the front of the tensors
// - aligns all the batch dims to the collective levels of all of the tensors.
//   If a tensor does not have a batch dim for a vmap level, then it receives
//   a size-one dimension for said level.
// - aligns the non-batch dims to have the same dimensionality, adding extra
//   size-1 dimensions in between the batch dimensions and the non-batch
//   dimensions so that the batch dimensions are lined up from the right.
//
// For example: given inputs of size (B, 2) and (B, 3, 2) where B is the batch
// dimension, BroadcastingVmapTransform returns VmapPhysicalViews that wrap
// tensors of size (B, 1, 2) and (B, 3, 2).
//
// Given inputs of size (B, 2) and (2,), BroadcastingVmapTransform returns
// VmapPhysicalViews wrapping tensors of size (B, 2) and (1, 2). We don't
// actually *need* to return a tensor of size (1, 2) for the second tensor
// because the broadcasting operation takes care of that for us, but we do
// it anyways to keep things simple.
struct TEST_API BroadcastingVmapTransform {
  static VmapPhysicalViewVec logicalToPhysical(
      std::vector<paddle::Tensor> logical_tensors);
};

// Forward declared, if you're reading this file head to toe, don't worry about
// it yet.
struct VmapPhysicalToLogicalMap;

// NOTE: [What is a VmapPhysicalView?]
// VmapPhysicalView represents a physical view on a paddle::Tensor.
//
// One can use it to further convert logical dimension indices, logical shapes,
// and more to their physical variants, or convert a new (physical) tensor into
// a logical BatchedTensor. (TODO(rzou): some of these are not yet implemented).
//
// VmapPhysicalView stores a physical tensor with all of its batch dimensions at
// the front and some levels that correspond to said batch dimensions.
//
// The levels bitset specifies which vmap levels correspond to the batch
// dimensions at the front of the tensor. In particular, the number of set bits
// corresponds to the number of batch dimensions on `tensor` and the rightmost
// bit of `levels` specifies the maximum number of nested vmaps we are in at
// this point in time.
// For example, given:
//   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5, 6), levels={1, 3})
//
// Rightmost bit of `levels` is 3 indicating the number of nested vmaps less
// than or equal to 3.
//   bitset: 010100
//              ^
//              |
//   levels: 012345
struct TEST_API VmapPhysicalView {
  VmapPhysicalView(paddle::Tensor&& tensor,
                   std::bitset<phi::kVmapNumLevels> levels)
      : levels_(levels), tensor_(std::move(tensor)) {
    PD_CHECK(!phi::isBatchedTensor(tensor_));
  }

  paddle::Tensor& tensor() { return tensor_; }
  const paddle::Tensor& tensor() const { return tensor_; }

  // Maps logical dim indices to physical dim indices. Also does dim wrapping.
  //
  // For example, given:
  //   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5), levels={1, 3})
  //
  // Then physical_view.getPhysicalDims({0, 1}) returns {2, 3}.
  // This is because the size of levels tell us that the first two dimensions
  // of `tensor_` are batch dimensions, so a logical dim of `n` is actually
  // a physical dim of `n + 2`.
  VmapDimVector getPhysicalDims(const std::vector<int64_t>& logical_dims) const;
  int64_t getPhysicalDim(int64_t logical_dim) const;

  // Returns a VmapPhysicalToLogicalMap object. This can be used for
  // mapping a physical tensor to a new logical tensor (BatchedTensor)
  VmapPhysicalToLogicalMap getPhysicalToLogicalMap() const;

  // Maps a logical shape to a physical shape by pre-pending the batch
  // sizes to the logical shape.
  VmapDimVector getPhysicalShape(std::vector<int64_t> logical_shape) const;

  int64_t numBatchDims() const;

 private:
  int64_t numLogicalDims() const;

  std::bitset<phi::kVmapNumLevels> levels_;
  paddle::Tensor tensor_;
};

// Convenience struct used for mapping a physical tensor (a non-BatchedTensor)
// to a logical one (BatchedTensor). It holds some levels that are used to do
// the mapping and assumes that the batch dimensions in the physical tensor all
// occur at the front of the tensor.
struct TEST_API VmapPhysicalToLogicalMap {
  VmapPhysicalToLogicalMap(std::bitset<phi::kVmapNumLevels> levels)
      : levels_(levels) {}

  // Maps a physical tensor to a new logical tensor (BatchedTensor).
  // Assumes that all of the "batch dimensions" are at the front
  // of the physical tensor. For example, given:
  // - x = rank-4 paddle::Tensor with size 2, 3, 5, 7
  // - levels = (2, 4)
  // Returns:
  // - BatchedTensor(x, bdims=[(dim=0,lvl=2), (dim=1, lvl=4)])
  paddle::Tensor apply(const paddle::Tensor& physical_tensor) const;

  // Given a vector of physical tensors,
  // 1. maps each tensor to a new logical tensor. Assumes that all of the
  //    "batch dimensions" are at the front of the physical tensors.
  // 2. stores the new logical tensors back into the passed-in vector. This is
  //    to avoid additional dynamic allocations.
  void applyInplace(std::vector<paddle::Tensor>& physical_tensors) const;

  std::bitset<phi::kVmapNumLevels> levels_;
};

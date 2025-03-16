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

#include "paddle/fluid/eager/vectorize/vmap_transforms.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/batched_tensor.h"

using BatchDimsRef = phi::BatchDimsRef;
using Tensor = paddle::Tensor;

// Checks if the batch dims in `bdims` appear at the front of the tensor.
static bool areBdimsAtFrontInOrder(BatchDimsRef bdims) {
  for (auto idx = 0; idx < static_cast<int64_t>(bdims.size()); ++idx) {
    if (bdims[idx].dim() != idx) {
      return false;
    }
  }
  return true;
}

// Takes a BatchedTensorImpl, permutes all of the batch dims to the front,
// and then returns a physical version of the Tensor.
static Tensor permuteBatchDimsToFront(phi::BatchedTensor* batched) {
  auto bdims = batched->bdims();
  const Tensor& physical_tensor = batched->value();
  if (areBdimsAtFrontInOrder(bdims)) {
    return physical_tensor;
  }
  const auto sizes = common::vectorize(physical_tensor.dims());
  std::vector<int> permutation(sizes.size(), 0);
  permutation.reserve(sizes.size());
  const auto is_bdim = phi::createBatchDimBitset(bdims);
  int64_t idx = 0;
  for (const auto& bdim : bdims) {
    permutation[idx++] = bdim.dim();
  }
  for (size_t ptr = 0; ptr < sizes.size(); ++ptr) {
    if (is_bdim[ptr]) {
      continue;
    }
    permutation[idx++] = static_cast<int64_t>(ptr);
  }
  return transpose_ad_func(physical_tensor, permutation);
}

VmapPhysicalView MultiBatchVmapTransform::logicalToPhysical(
    const Tensor& logical_tensor) {
  auto* batched = phi::maybeGetBatchedImpl(logical_tensor);
  PD_CHECK(batched != nullptr,
           "logicalToPhysical(tensor) should only be passed a BatchedTensor");
  return {permuteBatchDimsToFront(batched),
          phi::createVmapLevelsBitset(batched->bdims())};
}

int64_t VmapPhysicalView::numBatchDims() const {
  return static_cast<int64_t>(levels_.count());
}

int64_t VmapPhysicalView::numLogicalDims() const {
  return /*physical*/ tensor_.dims().size() - numBatchDims();
}

VmapDimVector VmapPhysicalView::getPhysicalDims(
    const std::vector<int64_t>& opt_logical_dims) const {
  auto logical_ndim = numLogicalDims();
  // NB: fmap doesn't have a SmallVector variant, so we don't use it here.
  VmapDimVector result;
  result.reserve(logical_ndim);
  if (!opt_logical_dims.empty()) {
    auto logical_dims = opt_logical_dims;
    for (auto dim : logical_dims) {
      result.push_back(phi::normalize_axis(dim, logical_ndim) + numBatchDims());
    }
  } else {
    for (int64_t dim = 0; dim < logical_ndim; dim++) {
      result.push_back(dim + numBatchDims());
    }
  }
  return result;
}

int64_t VmapPhysicalView::getPhysicalDim(int64_t logical_dim) const {
  auto logical_ndim = numLogicalDims();
  return phi::normalize_axis(logical_dim, logical_ndim) + numBatchDims();
}

VmapDimVector VmapPhysicalView::getPhysicalShape(
    std::vector<int64_t> logical_shape) const {
  VmapDimVector result;
  result.reserve(logical_shape.size() + numBatchDims());
  auto tensor_sizes = common::vectorize(tensor_.dims());
  result.insert(result.end(),
                tensor_sizes.begin(),
                tensor_sizes.begin() + numBatchDims());
  result.insert(result.end(), logical_shape.begin(), logical_shape.end());
  return result;
}

static phi::BatchDims computeFrontBatchDimsFromLevels(
    std::bitset<phi::kVmapNumLevels> levels_bitset) {
  phi::BatchDims bdims;
  int64_t dim = 0;
  for (auto level = 0; level < phi::kVmapNumLevels; ++level) {
    if (!levels_bitset[level]) {
      continue;
    }
    bdims.emplace_back(level, dim++);
  }
  return bdims;
}

// Given a Tensor or a BatchedTensor, returns the underlying physical tensor
// with all vmapped dimensions permuted to the front, if they exist, and a
// bitset of vmap levels that were present in the tensor.
static std::pair<paddle::Tensor, std::bitset<phi::kVmapNumLevels>>
getPhysicalTensorAndLevels(const Tensor& self) {
  auto* batched = phi::maybeGetBatchedImpl(self);
  if (batched) {
    return {permuteBatchDimsToFront(batched),
            phi::createVmapLevelsBitset(batched->bdims())};
  }
  return {self, 0};
}

// Given a Tensor or a BatchedTensor, creates a physical view of the tensor
// such that it has a batch dimension for each level in `requested_levels`
// and `requested_example_dim` number of non-batch-dimensions.
//
// This function is useful in preparing physical views on tensors that can
// then be passed into broadcasting operations. For example, when adding
// two BatchedTensors of sizes [B0, 3] and [B0, B1, 2, 3], where the Bi are the
// batch dimensions, we must align the batch dimensions and non-batch-dimensions
// (henceforth referred to as the "example" dimensions) separately to produce
// tensors of size [B0, 1, 1, 3] and [B0, B1, 2, 3] so that they can be added.
//
// Here's a direct example of using alignBatchDimsAtFront on the above two
// tensors.
//
// 1) alignBatchDimsAtFront([B0, 3], requested_levels={0, 1},
// requested_example_dim=2) returns a physical view of size [B0, 1, 1, 3] by
// adding an extra dimension for level 1 and another extra dimension to pad the
// example dimensions to 2.
//
// 2) alignBatchDimsAtFront([B0, B1, 2, 3], requested_levels={0, 1},
// requested_example_dim=2) returns a physical view of size [B0, B1, 2, 3]
static Tensor alignBatchDimsAtFront(
    const Tensor& self,
    std::bitset<phi::kVmapNumLevels> requested_levels,
    int64_t requested_example_dim) {
  auto [physical_tensor, tensor_levels] = getPhysicalTensorAndLevels(self);

  PD_CHECK((tensor_levels | requested_levels) == requested_levels,
           "`requested_levels` must be a superset of `self`'s levels");

  auto physical_sizes = common::vectorize(physical_tensor.dims());

  const auto tensor_example_dim =
      (static_cast<int64_t>(physical_sizes.size()) -
       /*num_batch_dims*/ static_cast<int64_t>(tensor_levels.count()));
  PD_CHECK(tensor_example_dim <= requested_example_dim);

  if (tensor_levels == requested_levels &&
      tensor_example_dim == requested_example_dim) {
    // Optimization: no need to do another view if the physical tensor is
    // already the correct shape
    return physical_tensor;
  }

  std::vector<int64_t> aligned_sizes(
      requested_levels.count() + requested_example_dim, 1);

  // align the example dims (non-bdims dims) first
  // aligned_sizes[-tensor_example_dim:] = tensor_sizes[-tensor_example_dim:]
  std::copy(physical_sizes.rbegin(),
            physical_sizes.rbegin() + tensor_example_dim,
            aligned_sizes.rbegin());

  // align the bdims
  int64_t level = 0;
  int64_t tensor_dim = 0;
  for (size_t bdim = 0; bdim < requested_levels.count(); ++bdim) {
    // Determine the level of the bdim
    while (!requested_levels[level]) level++;
    if (tensor_levels[level]) {
      aligned_sizes[bdim] = physical_sizes[tensor_dim++];
    }
    level++;
  }
  return view_shape_ad_func(physical_tensor, aligned_sizes);
}

// The algorithm is as follows:
// 1. Figure out what all of the collective levels in `logical_tensors` is.
// 2. Move all batch dims to the front of the tensors and add extra dims
//    of size 1. At this point, every tensor will have a dimension for
//    each of the collective levels.
// 3. Compute the batch_sizes.
// 4. Expand each physical tensor so that they have output batch size equal
//    to `batch_sizes`
VmapPhysicalViewVec MultiBatchVmapTransform::logicalToPhysical(
    const std::vector<paddle::Tensor>& logical_tensors) {
  // Figure out all of the collective vmap levels in `logical_tensors`.
  std::bitset<phi::kVmapNumLevels> collective_levels;
  for (const auto& logical_tensor : logical_tensors) {
    auto* batched = phi::maybeGetBatchedImpl(logical_tensor);
    if (batched) {
      collective_levels |= phi::createVmapLevelsBitset(batched->bdims());
    }
  }

  // Populate physical_tensors.
  // This contains a list of regular (non-Batched) Tensors where all of the
  // batch dims have been moved to the front of the tensor. Any previously
  // non-existing batch dims get added to the tensors as new dimensions of
  // size 1.
  std::vector<Tensor> physical_tensors;
  auto num_batch_dims = collective_levels.count();
  for (const auto& logical_tensor : logical_tensors) {
    auto requested_example_dim = /*logical_dim*/ logical_tensor.dims().size();
    auto physical_tensor = alignBatchDimsAtFront(
        logical_tensor, collective_levels, requested_example_dim);
    physical_tensors.push_back(std::move(physical_tensor));
  }

  // Compute batch_sizes
  VmapDimVector batch_sizes(num_batch_dims, 1);
  for (const auto& physical_tensor : physical_tensors) {
    auto physical_sizes = common::vectorize(physical_tensor.dims());
    for (size_t dim = 0; dim < num_batch_dims; ++dim) {
      if (physical_sizes[dim] != 1) {
        batch_sizes[dim] = physical_sizes[dim];
      }
    }
  }

  // Expand each physical_tensor so that it has batch sizes `batch_sizes`
  VmapPhysicalViewVec result;
  for (const auto& physical_tensor : physical_tensors) {
    std::vector<int64_t> expanded_size(batch_sizes.begin(), batch_sizes.end());
    auto physical_sizes = common::vectorize(physical_tensor.dims());
    expanded_size.insert(expanded_size.end(),
                         physical_sizes.begin() + num_batch_dims,
                         physical_sizes.end());
    result.emplace_back(expand_ad_func(physical_tensor, expanded_size),
                        collective_levels);
  }
  return result;
}

static std::pair<std::bitset<phi::kVmapNumLevels>, int64_t>
getLevelsAndLargestLogicalDim(std::vector<paddle::Tensor> logical_tensors) {
  PD_CHECK(!logical_tensors.empty());
  std::bitset<phi::kVmapNumLevels> levels;
  int64_t largest_logical_dim = -1;
  for (const auto& tensor : logical_tensors) {
    auto* batched = phi::maybeGetBatchedImpl(tensor);
    if (batched) {
      levels = levels | phi::createVmapLevelsBitset(batched->bdims());
    }
    auto tensor_logical_dim = /*logical dim*/ tensor.dims().size();
    if (tensor_logical_dim > largest_logical_dim) {
      largest_logical_dim = tensor_logical_dim;
    }
  }
  return {levels, largest_logical_dim};
}

VmapPhysicalViewVec BroadcastingVmapTransform::logicalToPhysical(
    std::vector<paddle::Tensor> logical_tensors) {
  PD_CHECK(logical_tensors.size() == 2,
           "This function has only been tested for two tensors. Please add "
           "more tests ",
           "before removing this check ");

  VmapPhysicalViewVec result;

  auto [levels, largest_logical_dim] =
      getLevelsAndLargestLogicalDim(logical_tensors);

  for (const auto& tensor : logical_tensors) {
    // NB: It's possible that we didn't actually need to align `tensor`.
    // For example, when adding two tensors of size (B, 2), and (3, 2), where
    // the first Tensor is a BatchedTensor with batch dim B and the second is
    // a regular Tensor, we will return views of size (B, 1, 2) and (1, 3, 2).
    // However, the view on the second tensor is unnecessary: broadcasting
    // semantics allow for the addition of two tensors of size (B, 1, 2) and (3,
    // 2)!
    //
    // If this unnecessary view is a problem, consider optimizing it away in
    // the future. This may involve creating a new type of VmapPhysicalView
    auto aligned = alignBatchDimsAtFront(tensor, levels, largest_logical_dim);
    result.emplace_back(std::move(aligned), levels);
  }
  return result;
}

VmapPhysicalToLogicalMap VmapPhysicalView::getPhysicalToLogicalMap() const {
  return VmapPhysicalToLogicalMap(levels_);
}

Tensor VmapPhysicalToLogicalMap::apply(const Tensor& physical_tensor) const {
  return phi::makeBatched(physical_tensor,
                          computeFrontBatchDimsFromLevels(levels_));
}

void VmapPhysicalToLogicalMap::applyInplace(
    std::vector<Tensor>& physical_tensors) const {
  for (auto& physical_tensor : physical_tensors) {
    physical_tensor = apply(physical_tensor);
  }
}

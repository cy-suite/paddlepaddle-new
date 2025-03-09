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

#include "paddle/phi/core/batched_tensor.h"
#include "bitset"
#include "paddle/common/ddim.h"
#include "paddle/common/dim.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

BatchedTensor::BatchedTensor(paddle::Tensor value, BatchDims bdims) {
  this->value_ = std::move(value);
  this->bdims_ = std::move(bdims);

  PD_CHECK(value_.has_allocation(), "BatchedTensor value must have allocation");
  const int64_t public_rank = value_.dims().size() - bdims_.size();
  const common::DDim value_dims = value_.dims();
  const common::DDim value_strides = value_.strides();

  std::cout << "public_rank: " << public_rank << std::endl;
  this->meta_.dims =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
  this->meta_.strides =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);

  for (int64_t dim = 0; dim < public_rank; ++dim) {
    int64_t actual_dim = BatchedTensor::actualDim(dim, /*wrap_dim*/ false);
    this->meta_.dims.at(dim) = value_dims.at(actual_dim);
    this->meta_.strides.at(dim) = value_strides.at(actual_dim);
  }
  std::cout << "meta_.dims: " << this->meta_.dims << std::endl;
  std::cout << "meta_.strides: " << this->meta_.strides << std::endl;
}

// BatchedTensor::BatchedTensor(const paddle::Tensor& value, const BatchDims&
// bdims) {
//   this->value_ = value;
//   this->bdims_ = bdims;

//   PD_CHECK(value.has_allocation(), "BatchedTensor value must have
//   allocation");

//   const int64_t public_rank = value_.dims().size() - bdims_.size();
//   const common::DDim value_dims = value_.dims();
//   const common::DDim value_strides = value_.strides();
//   this->meta_.dims =
//       common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
//   this->meta_.strides =
//       common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
//   for (int64_t dim = 0; dim < public_rank; ++dim) {
//     int64_t actual_dim = BatchedTensor::actualDim(dim, /*wrap_dim*/ false);
//     this->meta_.dims.at(dim) = value_dims.at(actual_dim);
//     this->meta_.strides.at(dim) = value_strides.at(actual_dim);
//   }
// }

// BatchedTensor::BatchedTensor(const paddle::Tensor& value, BatchDims bdims) {
//   this->value_ = value;
//   this->bdims_ = std::move(bdims);

//   PD_CHECK(value.has_allocation(), "BatchedTensor value must have
//   allocation");

//   const int64_t public_rank = value_.dims().size() - bdims_.size();
//   const common::DDim value_dims = value_.dims();
//   const common::DDim value_strides = value_.strides();
//   this->meta_.dims =
//       common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
//   this->meta_.strides =
//       common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
//   for (int64_t dim = 0; dim < public_rank; ++dim) {
//     int64_t actual_dim = BatchedTensor::actualDim(dim, /*wrap_dim*/ false);
//     this->meta_.dims.at(dim) = value_dims.at(actual_dim);
//     this->meta_.strides.at(dim) = value_strides.at(actual_dim);
//   }
// }

int64_t BatchedTensor::actualDim(int64_t dim, bool wrap_dim) const {
  if (wrap_dim) {
    const auto ndim = this->meta_.dims.size();
    dim = normalize_axis(dim, static_cast<int64_t>(ndim));
  }
  auto is_bdim = createBatchDimBitset(bdims_);

  // Example: assume dim = 3, and is_bdim = 10010011000...
  // The 1's are batch dims and 0's are normal dims of the underlying value_
  // paddle::Tensor. actualDim gives us the index of `dim` in the `value_`
  // paddle::Tensor, which is equivalent to asking "where does the 3rd
  // (0-indexed) zero occur in the bitset?". The answer to that is index 5.
  //
  // TODO(rzou): the PDEP instruction does exactly this
  // (https://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int)
  // but it might require newer (>= ~2015) CPUs. We should clean this up
  // if/when we have dropped support for older CPUs.
  int64_t non_bdim_count = 0;
  for (int actual_dim = 0; actual_dim < kVmapMaxTensorDims; ++actual_dim) {
    if (is_bdim[actual_dim]) {
      continue;
    }
    if (non_bdim_count == dim) {
      return actual_dim;
    }
    non_bdim_count++;
  }
  // If we hit this assert, then that means
  // `non_bdim_count` + #num_bdims > kVmapMaxTensorDims. We restrict the number
  // of dims a BatchedTensorImpl can have to kVmapMaxTensorDims so this should
  // never be hit.
  PD_CHECK(false,
           "`non_bdim_count(%d) + #num_bdims(%d) > kVmapMaxTensorDims(%d)",
           non_bdim_count,
           bdims_.size(),
           kVmapMaxTensorDims);
}

void* BatchedTensor::AllocateFrom(Allocator* allocator,
                                  DataType dtype,
                                  size_t requested_size,
                                  bool fake_alloc) {
  PD_CHECK(false);
}

void BatchedTensor::checkInvariants() const {
  int64_t prev_level = -1;
  for (const auto& bdim : bdims_) {
    PD_CHECK(bdim.level() > prev_level, "BatchDims must be sorted by level");
    prev_level = bdim.level();
  }
}

paddle::Tensor makeBatched(const paddle::Tensor& tensor, BatchDims bdims) {
  PD_CHECK(!isBatchedTensor(tensor),
           "Given tensor should not be a BatchedTensor");
  auto tensor_ndim = tensor.dims().size();
  PD_CHECK(tensor_ndim <= kVmapMaxTensorDims,
           "vmap only supports tensors of dimensionality up to %d"
           "but got a tensor with dim %d",
           kVmapMaxTensorDims,
           tensor_ndim);
  PD_CHECK(std::all_of(bdims.begin(),
                       bdims.end(),
                       [](const BatchDim& bdim) {
                         return bdim.level() < phi::kVmapNumLevels;
                       }),
           "We only support up to %d nested vmaps",
           phi::kVmapNumLevels);
  return paddle::Tensor(std::make_shared<BatchedTensor>(tensor, bdims));
}

paddle::Tensor addBatchDim(const paddle::Tensor& tensor,
                           int64_t level,
                           int64_t dim) {
  const auto* batched = phi::maybeGetBatchedImpl(tensor);
  if (!batched) {
    BatchDims bdims;
    bdims.emplace_back(level, dim);
    return paddle::Tensor(
        std::make_shared<BatchedTensor>(tensor, std::move(bdims)));
  }
  BatchDims new_bdims(batched->bdims().begin(), batched->bdims().end());
  auto actual_bdim = batched->actualDim(dim, /*wrap_dim=*/true);
  new_bdims.emplace_back(level, actual_bdim);
  return makeBatched(batched->value(), std::move(new_bdims));
}

}  // namespace phi

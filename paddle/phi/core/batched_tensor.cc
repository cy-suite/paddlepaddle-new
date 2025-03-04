/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

inline std::bitset<kVmapMaxTensorDims> createBatchDimBitset(
    BatchDimsRef bdims) {
  std::bitset<kVmapMaxTensorDims> is_bdim;
  for (const auto& bdim : bdims) {
    is_bdim.set(bdim.dim());
  }
  return is_bdim;
}

inline int64_t maybe_wrap_dim(int64_t dim, int64_t ndim) {
  PD_CHECK(-ndim <= dim && dim < ndim,
           "Expected -%lld(-ndim) <= %lld(dim) < %lld(ndim)",
           ndim,
           dim,
           ndim);
  // if (-ndim <= dim && dim < ndim) {
  if (dim < 0) {
    return dim + ndim;
  }
  return dim;
  // }
}

// // inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
// //   if (tensors.empty()) {
// //     // can't wrap empty TensorList; rely on underlying implementation to
// throw
// //     // error if necessary.
// //     return dim;
// //   }
// //   return maybe_wrap_dim(dim, tensors[0].dim());
// // }

// inline int64_t maybe_wrap_dim(
//     int64_t dim,
//     const std::vector<std::vector<int64_t>>& tensor_sizes) {
//   if (tensor_sizes.empty()) {
//     // can't wrap empty list; rely on underlying implementation to throw
//     error
//     // if necessary
//     return dim;
//   }
//   return maybe_wrap_dim(dim, tensor_sizes[0].size());
// }

BatchedTensor::BatchedTensor(Tensor value, BatchDims bdims) {
  this->value_ = std::move(value);
  this->bdims_ = std::move(bdims);

  PD_CHECK(value.has_allocation(), "BatchedTensor value must have allocation");

  const int64_t public_rank = value_.dims().size() - bdims_.size();
  const common::DDim value_dims = value_.dims();
  const common::DDim value_strides = value_.strides();
  this->meta_.dims =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
  this->meta_.strides =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
  for (int64_t dim = 0; dim < public_rank; ++dim) {
    int64_t actual_dim = BatchedTensor::actualDim(dim, /*wrap_dim*/ false);
    this->meta_.dims.at(dim) = value_dims.at(actual_dim);
    this->meta_.strides.at(dim) = value_strides.at(actual_dim);
  }
  // this->holder_ = this->value_.;
}

BatchedTensor::BatchedTensor(const Tensor& value, const BatchDims& bdims) {
  this->value_ = value;
  this->bdims_ = bdims;

  PD_CHECK(value.has_allocation(), "BatchedTensor value must have allocation");

  const int64_t public_rank = value_.dims().size() - bdims_.size();
  const common::DDim value_dims = value_.dims();
  const common::DDim value_strides = value_.strides();
  this->meta_.dims =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
  this->meta_.strides =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
  for (int64_t dim = 0; dim < public_rank; ++dim) {
    int64_t actual_dim = BatchedTensor::actualDim(dim, /*wrap_dim*/ false);
    this->meta_.dims.at(dim) = value_dims.at(actual_dim);
    this->meta_.strides.at(dim) = value_strides.at(actual_dim);
  }
  // this->holder_ = this->value_.;
}

BatchedTensor::BatchedTensor(const Tensor& value, BatchDims bdims) {
  this->value_ = value;
  this->bdims_ = std::move(bdims);

  PD_CHECK(value.has_allocation(), "BatchedTensor value must have allocation");

  const int64_t public_rank = value_.dims().size() - bdims_.size();
  const common::DDim value_dims = value_.dims();
  const common::DDim value_strides = value_.strides();
  this->meta_.dims =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
  this->meta_.strides =
      common::DDim(std::vector<int64_t>(public_rank, 0).data(), public_rank);
  for (int64_t dim = 0; dim < public_rank; ++dim) {
    int64_t actual_dim = BatchedTensor::actualDim(dim, /*wrap_dim*/ false);
    this->meta_.dims.at(dim) = value_dims.at(actual_dim);
    this->meta_.strides.at(dim) = value_strides.at(actual_dim);
  }
  // this->holder_ = this->value_.;
}

int64_t BatchedTensor::actualDim(int64_t dim, bool wrap_dim) const {
  if (wrap_dim) {
    const int64_t ndim = this->meta_.dims.size();
    dim = maybe_wrap_dim(dim, ndim);
  }
  auto is_bdim = createBatchDimBitset(bdims_);

  int64_t non_bdim_count = 0;
  for (int64_t actual_dim = 0; actual_dim < kVmapMaxTensorDims; ++actual_dim) {
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
  // of dims a BatchedTensor can have to kVmapMaxTensorDims so this should
  // never be hit.
  PD_CHECK(false, "Should not reach here");
}

void BatchedTensor::checkInvariants() const {
  int64_t prev_level = -1;
  for (const auto& bdim : bdims_) {
    PD_CHECK(bdim.level() > prev_level, "BatchDims must be sorted by level");
    prev_level = bdim.level();
  }
}

Tensor makeBatched(const Tensor& tensor, BatchDims bdims) {
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
                         return bdim.level() < kVmapNumLevels;
                       }),
           "We only support up to %d nested vmaps",
           kVmapNumLevels);
  return Tensor(std::make_shared<BatchedTensor>(tensor, bdims));
}

Tensor addBatchDim(const Tensor& tensor, int64_t level, int64_t dim) {
  const auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    BatchDims bdims;
    bdims.emplace_back(level, dim);
    return Tensor(std::make_shared<BatchedTensor>(tensor, std::move(bdims)));
  }
  BatchDims new_bdims(batched->bdims().begin(), batched->bdims().end());
  auto actual_bdim = batched->actualDim(dim, /*wrap_dim=*/true);
  new_bdims.emplace_back(level, actual_bdim);
  return makeBatched(batched->value(), std::move(new_bdims));
}

}  // namespace phi

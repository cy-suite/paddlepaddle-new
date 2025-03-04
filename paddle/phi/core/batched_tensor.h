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

#pragma once

#include "bitset"
#include "memory"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kmap_cache.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/utils/small_vector.h"

namespace phi {

class DenseTensorUtils;

using Tensor = paddle::Tensor;

// We assume this in a few other places in the codebase,
// but there isn't a centralized definition.
constexpr int64_t kVmapMaxTensorDims = 8;

// The valid vmap levels range from [0, 8). This effectively means that we
// support a maximum of 8 nested vmaps.
constexpr int64_t kVmapNumLevels = 8;

// Store this number of elements of BatchDims on the stack. Most people will
// probably use <= 5 nested vmaps, but adjust this number as necessary.
constexpr int64_t kBatchDimsStackSize = 5;

// A BatchedTensor holds an underlying Tensor and a single batch dim
// NB: We use the term "BatchedTensor" to mean a Tensor that is backed with a
// BatchedTensor.
//
// The batch dimensions are treated as being "private"; they are not
// user-visible. For example, in the following Tensor,
//    bt = BatchedTensor(ones(2, 3, 5, 7), lvl=1, dim=0)
// dimension 0 is batch dimension.
//
// bt.sizes() returns (5, 7); bt.sum(0) performs a reduction over the (public)
// dim 0, which is equivalent to dim 3 in the underlying ones(2, 3, 5, 7)
// tensor.

struct BatchDim {
  BatchDim(int64_t level, int64_t dim) : dim_(dim), level_(level) {}
  int64_t dim() const { return dim_; }
  int64_t level() const { return level_; }

 private:
  int64_t dim_;
  int64_t level_;
};

using BatchDims = paddle::small_vector<BatchDim, kBatchDimsStackSize>;
using BatchDimsRef = const paddle::small_vector<BatchDim, kBatchDimsStackSize>&;

class BatchedTensor : public TensorBase,
                      public TypeInfoTraits<TensorBase, BatchedTensor> {
 public:
  explicit BatchedTensor(Tensor value, BatchDims bdims);
  explicit BatchedTensor(const Tensor& value, const BatchDims& bdims);
  explicit BatchedTensor(const Tensor& value, BatchDims bdims);

  // Returns a reference to BatchDims that represent which dimensions of this
  // tensor are private.
  BatchDimsRef bdims() const { return bdims_; }

  const Tensor& value() const { return value_; }

  int64_t actualDim(int64_t dim, bool wrap_dim = true) const;

  virtual ~BatchedTensor() = default;
  /// \brief Return the number of elements contained in original dense tensor
  /// \return The number of elements contained in original dense tensor
  int64_t numel() const override { return product(meta_.dims); }

  /// \brief Returns the dims of the original dense tensor.
  /// \return The dims of the original dense tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return meta_.dtype; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return meta_.layout; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override { return this->holder_->place(); }

  /// \brief Test whether the holder is created.
  /// \return Whether the holder is created.
  bool has_allocation() const override { return holder_ != nullptr; }

  /// \brief Test whether the non_zero_elements_ metadata is valid.
  /// \return Whether the non_zero_elements_ metadata is valid.
  bool valid() const noexcept override { return this->has_allocation(); }

  /// \brief Test whether the allocation is allocated.
  /// return Whether the allocation is allocated.
  bool initialized() const override { return holder_ && holder_->ptr(); }

  /// \brief This function is not recommended
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) {
    PD_CHECK(value_.is_dense_tensor());
    return value_.impl()->AllocateFrom(
        allocator, dtype, requested_size, fake_alloc);
  }

 private:
  // see NOTE: [BatchedTensor levels invariant]
  void checkInvariants() const;

  Tensor value_;

  // Note: [BatchedTensor levels invariant]
  // There is an invariant that the BatchDims must be stored in increasing
  // `level` order. That is, for i < j, bdims_[i].level must be less than
  // bdims_[j].level.
  BatchDims bdims_;

 protected:
  BatchedTensorMeta meta_;
  std::shared_ptr<phi::Allocation> holder_;
};

// NB: We use the term "BatchedTensor" to mean a Tensor that is backed with a
// BatchedTensor.
inline bool isBatchedTensor(const Tensor& tensor) {
  return tensor.is_batched_tensor();
}

// It is unsafe to call this on a Tensor that is not backed by a
// BatchedTensor. Please use `maybeGetBatchedImpl` whenever possible.
inline BatchedTensor* unsafeGetBatchedImpl(const Tensor& tensor) {
  return static_cast<BatchedTensor*>(tensor.impl().get());
}

inline BatchedTensor* maybeGetBatchedImpl(const Tensor& tensor) {
  if (!isBatchedTensor(tensor)) {
    return nullptr;
  }
  return unsafeGetBatchedImpl(tensor);
}

// Use this to construct a BatchedTensor from a regular Tensor
TEST_API Tensor makeBatched(const Tensor& tensor, BatchDims bdims);

// Adds a batch dim to `tensor`, returning a BatchedTensor
TEST_API Tensor addBatchDim(const Tensor& tensor, int64_t level, int64_t dim);

// // Checks if an inplace operation on self and other is "vmap compatible".
// // See NOTE: [vmap-incompatible in-place operations] for the definition of
// this. TEST_API bool inplaceIsVmapCompatible(const Tensor& self, const Tensor&
// other);

}  // namespace phi

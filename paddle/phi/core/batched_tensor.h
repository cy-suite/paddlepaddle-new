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

// using paddle::Tensor = paddle::paddle::Tensor;

// We assume this in a few other places in the codebase,
// but there isn't a centralized definition.
constexpr int64_t kVmapMaxTensorDims = 64;

// The valid vmap levels range from [0, 64). This effectively means that we
// support a maximum of 64 nested vmaps.
constexpr int64_t kVmapNumLevels = 64;

// Store this number of elements of BatchDims on the stack. Most people will
// probably use <= 5 nested vmaps, but adjust this number as necessary.
constexpr int64_t kBatchDimsStackSize = 5;

// a BatchDim represents a "private" dimension on a paddle::Tensor created
// inside of vmap. It is a (level, dim) tuple, with the `dim` indicating which
// dimension is being vmap'ed over and the `level` being an identifier for which
// vmap said dimension was created inside. The `dim` corresponds to a "physical
// dim" - it is a dimension index on the underlying physical tensor that is
// being vmapped over.

struct BatchDim {
  BatchDim(int64_t level, int64_t dim) : level_(level), dim_(dim) {}
  int64_t level() const { return level_; }
  int64_t dim() const { return dim_; }

 private:
  int64_t level_;
  int64_t dim_;
};

using BatchDims = paddle::small_vector<BatchDim, kBatchDimsStackSize>;
using BatchDimsRef = const paddle::small_vector<BatchDim, kBatchDimsStackSize>&;

// A BatchedTensorImpl holds an underlying paddle::Tensor and a list of BatchDim
// NB: We use the term "BatchedTensor" to mean a paddle::Tensor that is backed
// with a BatchedTensorImpl.
//
// The batch dimensions are treated as being "private"; they are not
// user-visible. For example, in the following paddle::Tensor,
//    bt = BatchedTensorImpl(ones(2, 3, 5, 7), [(lvl=1, dim=0), (lvl=2, dim=1)])
// dimensions 0 and 1 are batch dimensions.
//
// bt.sizes() returns (5, 7); bt.sum(0) performs a reduction over the (public)
// dim 0, which is equivalent to dim 3 in the underlying ones(2, 3, 5, 7)
// tensor.
class BatchedTensor : public TensorBase,
                      public TypeInfoTraits<TensorBase, BatchedTensor> {
 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "BatchedTensor"; }

  explicit BatchedTensor(paddle::Tensor value, BatchDims bdims);
  // explicit BatchedTensor(const paddle::Tensor& value, const BatchDims&
  // bdims); explicit BatchedTensor(const paddle::Tensor& value, BatchDims
  // bdims);

  // Returns a reference to BatchDims that represent which dimensions of this
  // tensor are private.
  BatchDimsRef bdims() const { return bdims_; }

  // BatchedTensorImpl wraps a paddle::Tensor
  const paddle::Tensor& value() const { return value_; }

  int64_t actualDim(int64_t dim, bool wrap_dim = true) const;

  /// \brief Return the number of elements contained in original dense tensor
  /// \return The number of elements contained in original dense tensor
  int64_t numel() const override { return value_.numel(); }

  /// \brief Returns the dims of the original dense tensor.
  /// \return The dims of the original dense tensor.
  const DDim& dims() const noexcept override { return meta_.dims; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override { return value_.dtype(); }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept override { return meta_.layout; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override { return value_.place(); }

  /// \brief Test whether the holder is created.
  /// \return Whether the holder is created.
  bool has_allocation() const override { return value_.has_allocation(); }

  /// \brief Test whether the allocation is allocated.
  /// return Whether the allocation is allocated.
  bool initialized() const override { return value_.initialized(); }

  /// \brief Returns the stride of the tensor.
  /// \return The stride of the tensor.
  const DDim& strides() const noexcept { return meta_.strides; }

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept override { return meta_.valid(); }

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

  /// \brief Sets the stride of the tensor.
  /// \param meta The stride of the tensor.
  void set_strides(const DDim& strides) { meta_.strides = strides; }

  /// \brief Sets the dims of the tensor.
  /// \param meta The dims of the tensor.
  void set_dims(const DDim& dims) { meta_.dims = dims; }

  /// \brief Sets the meta information of the tensor. Only when the original
  /// attribute of Tensor is incomplete, can it be reset.
  /// \param meta The meta information of the tensor.
  void set_meta(BatchedTensorMeta&& meta);

  void set_meta(const BatchedTensorMeta& meta);

  /// \brief Returns the meta information of the tensor.
  /// \return The meta information of the tensor.
  const BatchedTensorMeta& meta() const noexcept { return meta_; }

  /// \brief Returns the mutable tensor value in batched tensor.
  /// \return The mutable pointer of DenseTensor value
  DenseTensor* unsafe_mutable_value() {
    return std::static_pointer_cast<DenseTensor>(value_.impl()).get();
  }

 private:
  friend class DenseTensorUtils;

  void checkInvariants() const;

  paddle::Tensor value_ = paddle::Tensor();

  BatchedTensorMeta meta_;
  // Note: [BatchedTensor levels invariant]
  // There is an invariant that the BatchDims must be stored in increasing
  // `level` order. That is, for i < j, bdims_[i].level must be less than
  // bdims_[j].level.
  BatchDims bdims_;
};

// NB: We use the term "BatchedTensor" to mean a paddle::Tensor that is backed
// with a BatchedTensor.
inline bool isBatchedTensor(const paddle::Tensor& tensor) {
  return tensor.is_batched_tensor();
}

inline bool isBatchedTensor(const std::vector<paddle::Tensor>& tensors) {
  return std::any_of(
      tensors.begin(), tensors.end(), [](const paddle::Tensor& t) {
        return t.is_batched_tensor();
      });
}

// It is unsafe to call this on a paddle::Tensor that is not backed by a
// BatchedTensor. Please use `maybeGetBatchedImpl` whenever possible.
inline BatchedTensor* unsafeGetBatchedImpl(const paddle::Tensor& tensor) {
  return static_cast<BatchedTensor*>(tensor.impl().get());
}

inline BatchedTensor* maybeGetBatchedImpl(const paddle::Tensor& tensor) {
  if (!isBatchedTensor(tensor)) {
    return nullptr;
  }
  return unsafeGetBatchedImpl(tensor);
}

inline std::vector<BatchedTensor*> maybeGetBatchedImpl(
    const std::vector<paddle::Tensor>& tensors) {
  std::vector<BatchedTensor*> results(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    results[i] = maybeGetBatchedImpl(tensors[i]);
  }
  return results;
}

// Creates a bitset for all of the levels present in `bdims`
inline std::bitset<kVmapNumLevels> createVmapLevelsBitset(BatchDimsRef bdims) {
  std::bitset<kVmapNumLevels> result;
  for (const auto& bdim : bdims) {
    result.set(bdim.level());
  }
  return result;
}

inline std::ostream& operator<<(std::ostream& out, const BatchDim& bdim) {
  out << "(lvl=" << bdim.level() << ", dim=" << bdim.dim() << ")";
  return out;
}

// Use this to construct a BatchedTensor from a regular paddle::Tensor
TEST_API paddle::Tensor makeBatched(const paddle::Tensor& tensor,
                                    BatchDims bdims);

// Adds a batch dim to `tensor`, returning a BatchedTensor
TEST_API paddle::Tensor addBatchDim(const paddle::Tensor& tensor,
                                    int64_t level,
                                    int64_t dim);

inline int64_t normalize_axis(int64_t dim, int64_t ndim) {
  PD_CHECK(-ndim <= dim,
           "dim(%lld) should be larger than or equal to -ndim(%lld)",
           dim,
           ndim);
  PD_CHECK(
      dim < ndim, "dim(%lld) should be smaller than ndim(%lld)", dim, ndim);
  if (dim < 0) return dim + ndim;
  return dim;
}

// Returns a bitset. If bit i is set, then that means dim i is a batchdim.
inline std::bitset<kVmapMaxTensorDims> createBatchDimBitset(
    BatchDimsRef bdims) {
  std::bitset<kVmapMaxTensorDims> is_bdim;
  for (const auto& bdim : bdims) {
    is_bdim.set(bdim.dim());
  }
  return is_bdim;
}

// Checks if an inplace operation on self and other is "vmap compatible".
// See NOTE: [vmap-incompatible in-place operations] for the definition of this.
// TEST_API bool inplaceIsVmapCompatible(const paddle::Tensor& self, const
// paddle::Tensor& other);

}  // namespace phi

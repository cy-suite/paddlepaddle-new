// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

namespace paddle {
namespace distributed {

template <typename DeviceContext, typename T>
struct ConcatDenseTensorByNumel {
  void operator()(const DeviceContext &context,
                  const std::vector<phi::DenseTensor> &in,
                  phi::DenseTensor *out) {
    auto out_dims = common::vectorize(out->dims());
    auto flattened_out_dims = {out->numel()};
    std::vector<phi::DenseTensor> in_flatten;
    std::vector<std::vector<int64_t>> origin_in_dims;

    phi::DenseTensor out_flatten(out->Holder(), out->meta());
    out_flatten.Resize(flattened_out_dims);

    int64_t in_numel_sum = 0;
    for (auto &tensor : in) {
      phi::DenseTensor tensor_flatten(tensor.Holder(), tensor.meta());
      tensor_flatten.Resize({tensor.numel()});
      in_flatten.push_back(tensor_flatten);

      in_numel_sum += tensor.numel();
    }
    PADDLE_ENFORCE_EQ(
        out->numel(),
        in_numel_sum,
        common::errors::Unimplemented("Numel of in and out must be equal"));

    phi::funcs::ConcatFunctor<DeviceContext, T> concat_functor;
    concat_functor(context, in_flatten, 0, &out_flatten);
  }
};

template <typename DeviceContext>
void ConcatDenseTensorByNumelWithType(
    const DeviceContext &dev_ctx,
    const std::vector<phi::DenseTensor> &t_list,
    phi::DenseTensor *p_out,
    phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      ConcatDenseTensorByNumel<DeviceContext, bool>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::UINT8:
      ConcatDenseTensorByNumel<DeviceContext, uint8_t>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT8:
      ConcatDenseTensorByNumel<DeviceContext, int8_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT32:
      ConcatDenseTensorByNumel<DeviceContext, int32_t>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT64:
      ConcatDenseTensorByNumel<DeviceContext, int64_t>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT16:
      ConcatDenseTensorByNumel<DeviceContext, phi::dtype::float16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::BFLOAT16:
      ConcatDenseTensorByNumel<DeviceContext, phi::dtype::bfloat16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT32:
      ConcatDenseTensorByNumel<DeviceContext, float>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT64:
      ConcatDenseTensorByNumel<DeviceContext, double>()(dev_ctx, t_list, p_out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors.", type));
  }
}

template <>
void ConcatDenseTensorByNumelWithType(
    const phi::XPUContext &dev_ctx,
    const std::vector<phi::DenseTensor> &t_list,
    phi::DenseTensor *p_out,
    phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      ConcatDenseTensorByNumel<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::BFLOAT16:
      ConcatDenseTensorByNumel<phi::XPUContext, phi::dtype::bfloat16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT32:
      ConcatDenseTensorByNumel<phi::XPUContext, float>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT32:
      ConcatDenseTensorByNumel<phi::XPUContext, int32_t>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT64:
      ConcatDenseTensorByNumel<phi::XPUContext, int64_t>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::UINT8:
      ConcatDenseTensorByNumel<phi::XPUContext, uint8_t>()(
          dev_ctx, t_list, p_out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors.", type));
  }
}

template <typename DeviceContext, typename T>
struct SplitDenseTensorByNumel {
  void operator()(const DeviceContext &context,
                  const phi::DenseTensor &in,
                  std::vector<phi::DenseTensor> *out) {
    phi::DenseTensor in_flatten(in.Holder(), in.meta());
    in_flatten.Resize({in.numel()});

    std::vector<phi::DenseTensor> out_flatten;
    std::vector<const phi::DenseTensor *> shape_refer;
    std::vector<phi::DenseTensor *> out_p_list;

    int64_t out_numel_sum = 0;

    for (auto &tensor : *out) {
      phi::DenseTensor tensor_flatten(tensor.Holder(), tensor.meta());
      tensor_flatten.Resize({tensor.numel()});
      out_flatten.push_back(tensor_flatten);
      out_numel_sum += tensor.numel();
    }
    for (auto &tensor : out_flatten) {
      shape_refer.push_back(&tensor);
      out_p_list.push_back(&tensor);
    }

    PADDLE_ENFORCE_EQ(
        in.numel(),
        out_numel_sum,
        common::errors::Unimplemented("Numel of in and out must be equal"));

    phi::funcs::SplitFunctor<DeviceContext, T> split_functor;
    split_functor(context, in_flatten, shape_refer, 0, &out_p_list);
  }
};

template <typename DeviceContext>
void SplitDenseTensorByNumelWithType(const DeviceContext &dev_ctx,
                                     const phi::DenseTensor &t_in,
                                     std::vector<phi::DenseTensor> *t_list,
                                     phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      SplitDenseTensorByNumel<DeviceContext, bool>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::UINT8:
      SplitDenseTensorByNumel<DeviceContext, uint8_t>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::INT8:
      SplitDenseTensorByNumel<DeviceContext, int8_t>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::INT32:
      SplitDenseTensorByNumel<DeviceContext, int32_t>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::INT64:
      SplitDenseTensorByNumel<DeviceContext, int64_t>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::FLOAT16:
      SplitDenseTensorByNumel<DeviceContext, phi::dtype::float16>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::BFLOAT16:
      SplitDenseTensorByNumel<DeviceContext, phi::dtype::bfloat16>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::FLOAT32:
      SplitDenseTensorByNumel<DeviceContext, float>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::FLOAT64:
      SplitDenseTensorByNumel<DeviceContext, double>()(dev_ctx, t_in, t_list);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors.", type));
  }
}

template <>
void SplitDenseTensorByNumelWithType(const phi::XPUContext &dev_ctx,
                                     const phi::DenseTensor &t_in,
                                     std::vector<phi::DenseTensor> *t_list,
                                     phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      SplitDenseTensorByNumel<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::BFLOAT16:
      SplitDenseTensorByNumel<phi::XPUContext, phi::dtype::bfloat16>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::FLOAT32:
      SplitDenseTensorByNumel<phi::XPUContext, float>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::INT32:
      SplitDenseTensorByNumel<phi::XPUContext, int32_t>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::INT64:
      SplitDenseTensorByNumel<phi::XPUContext, int64_t>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::UINT8:
      SplitDenseTensorByNumel<phi::XPUContext, uint8_t>()(
          dev_ctx, t_in, t_list);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors.", type));
  }
}

void ConcatTensorByNumel(const phi::DeviceContext &dev_ctx,
                         const std::vector<phi::DenseTensor> &tensor_list,
                         phi::DenseTensor *tensor) {
  const auto &place = dev_ctx.GetPlace();
  if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    ConcatDenseTensorByNumelWithType(
        static_cast<const phi::XPUContext &>(dev_ctx),
        tensor_list,
        tensor,
        tensor->dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not support XPU, please "
        "recompile or reinstall Paddle with XPU support."));
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Concat tensor by numel not supported on place (%s)", place));
  }
}

void SplitTensorByNumel(const phi::DeviceContext &dev_ctx,
                        const phi::DenseTensor &tensor,
                        std::vector<phi::DenseTensor> *tensor_list) {
  const auto &place = dev_ctx.GetPlace();
  if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    SplitDenseTensorByNumelWithType(
        static_cast<const phi::XPUContext &>(dev_ctx),
        tensor,
        tensor_list,
        tensor.dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with XPU, "
        "please recompile or reinstall Paddle with XPU support."));
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Split tensor by numel not supported on place (%s)", place));
  }
}

}  // namespace distributed
}  // namespace paddle

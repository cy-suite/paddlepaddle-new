// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/clip_tensor_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  DenseTensor ex_x;
  DenseTensor ex_min;
  DenseTensor ex_max;
  std::vector<int> real_target_shape = common::vectorize<int>(out->dims());
  if (x.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, x, real_target_shape, &ex_x);
  } else {
    ex_x = x;
  }
  if (min.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, min, real_target_shape, &ex_min);
  } else {
    ex_min = min;
  }
  if (max.dims() != out->dims()) {
    phi::ExpandKernel<T, Context>(dev_ctx, max, real_target_shape, &ex_max);
  } else {
    ex_max = max;
  }

  phi::CastKernel<T, Context>(dev_ctx, ex_min, ex_x.dtype(), &ex_min);
  phi::CastKernel<T, Context>(dev_ctx, ex_max, ex_x.dtype(), &ex_max);

  const auto& onednn_engine = dev_ctx.GetEngine();
  auto& astream = OneDNNContext::tls().get_stream();

  DenseTensor* tem_out;
  auto* non_const_x = &ex_x;
  auto* non_const_min = &ex_min;
  auto* non_const_max = &ex_max;

  funcs::BinaryOneDNNHandler<T> MAXhandler(dnnl::algorithm::binary_max,
                                           -1,
                                           onednn_engine,
                                           dev_ctx.GetPlace(),
                                           non_const_x,
                                           non_const_min,
                                           tem_out,
                                           1.0f,
                                           1.0f,
                                           1.0f,
                                           true);

  auto src_memory_p_x = MAXhandler.AcquireSrcMemory(non_const_x);
  auto src_memory_p_min = MAXhandler.AcquireSecondSrcMemory(non_const_min);
  auto dst_memory_p = MAXhandler.AcquireDstMemory(tem_out);
  auto activation_p = MAXhandler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, *src_memory_p_x},
      {DNNL_ARG_SRC_1, *src_memory_p_min},
      {DNNL_ARG_DST, *dst_memory_p}};

  if (MAXhandler.Has_SRC_0_Scale()) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                 MAXhandler.Get_SRC_0_Scale_Memory()});
  }

  if (MAXhandler.Has_SRC_1_Scale()) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                 MAXhandler.Get_SRC_1_Scale_Memory()});
  }

  activation_p->execute(astream, args);

  funcs::BinaryOneDNNHandler<T> MINhandler(dnnl::algorithm::binary_min,
                                           -1,
                                           onednn_engine,
                                           dev_ctx.GetPlace(),
                                           tem_out,
                                           non_const_max,
                                           out,
                                           1.0f,
                                           1.0f,
                                           1.0f,
                                           true);

  auto src_memory_p_x2 = MINhandler.AcquireSrcMemory(tem_out);
  auto src_memory_p_max2 = MINhandler.AcquireSecondSrcMemory(non_const_max);
  auto dst_memory_p2 = MINhandler.AcquireDstMemory(out);
  auto activation_p2 = MINhandler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args2 = {
      {DNNL_ARG_SRC_0, *src_memory_p_x2},
      {DNNL_ARG_SRC_1, *src_memory_p_max2},
      {DNNL_ARG_DST, *dst_memory_p2}};

  if (MINhandler.Has_SRC_0_Scale()) {
    args2.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                  MINhandler.Get_SRC_0_Scale_Memory()});
  }

  if (MINhandler.Has_SRC_1_Scale()) {
    args2.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                  MINhandler.Get_SRC_1_Scale_Memory()});
  }

  activation_p2->execute(astream, args2);

  astream.wait();

  out->set_mem_desc(dst_memory_p2->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(clip_tensor,
                   OneDNN,
                   ONEDNN,
                   phi::ClipTensorKernel,
                   float,
                   phi::dtype::bfloat16) {}

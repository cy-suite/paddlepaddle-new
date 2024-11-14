// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/clip_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ClipTensorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& min,
                      const DenseTensor& max,
                      DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();
  auto& astream = OneDNNContext::tls().get_stream();

  DenseTensor* tem_out;
  auto* non_const_x = &x;
  auto* non_const_min = &min;
  auto* non_const_max = &max;
  
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

  std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC_0, *src_memory_p_x},
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

  std::unordered_map<int, dnnl::memory> args2 = {{DNNL_ARG_SRC_0, *src_memory_p_x2},
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

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& min,
                const Scalar& max,
                DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  funcs::ClipOneDNNHandler<T> handler(
      min, max, onednn_engine, dev_ctx.GetPlace(), &x);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);
  auto activation_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_p->execute(
      astream, {{DNNL_ARG_FROM, *src_memory_p}, {DNNL_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(
    clip_tensor, OneDNN, ONEDNN, phi::ClipTensorKernel, float, phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    clip, OneDNN, ONEDNN, phi::ClipKernel, float, phi::dtype::bfloat16) {}

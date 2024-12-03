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

#include "paddle/phi/kernels/clip_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ClipTensorGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& min,
                    const DenseTensor& max,
                    const DenseTensor& out_grad,
                    DenseTensor* x_grad) {

  const auto& onednn_engine = dev_ctx.GetEngine();
  auto& astream = OneDNNContext::tls().get_stream();

  DenseTensor* tem_min_mask;
  DenseTensor* tem_max_mask;
  DenseTensor* tem_zero_mask;
  auto* non_const_x = &x;
  auto* non_const_min = &min;
  auto* non_const_max = &max;
  auto* non_const_out_grad = &out_grad;

  funcs::BinaryOneDNNHandler<T> Lesshandler(dnnl::algorithm::binary_lt,
                                        -1,
                                        onednn_engine,
                                        dev_ctx.GetPlace(),
                                        non_const_min,
                                        non_const_out_grad,
                                        tem_min_mask,
                                        1.0f,
                                        1.0f,
                                        1.0f,
                                        true);

  auto src_memory_p_min1 = Lesshandler.AcquireSrcMemory(non_const_min);
  auto src_memory_p_out_grad1 = Lesshandler.AcquireSecondSrcMemory(non_const_out_grad);
  auto dst_memory_p1 = Lesshandler.AcquireDstMemory(tem_min_mask);
  auto activation_p1 = Lesshandler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args1 = {{DNNL_ARG_SRC_0, *src_memory_p_min1},
                                                {DNNL_ARG_SRC_1, *src_memory_p_out_grad1},
                                                {DNNL_ARG_DST, *dst_memory_p1}};

  if (Lesshandler.Has_SRC_0_Scale()) {
    args1.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                 Lesshandler.Get_SRC_0_Scale_Memory()});
  }

  if (Lesshandler.Has_SRC_1_Scale()) {
    args1.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                 Lesshandler.Get_SRC_1_Scale_Memory()});
  }

  activation_p1->execute(astream, args1);

  funcs::BinaryOneDNNHandler<T> Grahandler(dnnl::algorithm::binary_gt,
                                        -1,
                                        onednn_engine,
                                        dev_ctx.GetPlace(),
                                        non_const_max,
                                        non_const_out_grad,
                                        tem_max_mask,
                                        1.0f,
                                        1.0f,
                                        1.0f,
                                        true);

  auto src_memory_p_max2 = Grahandler.AcquireSrcMemory(non_const_max);
  auto src_memory_p_out_grad2 = Grahandler.AcquireSecondSrcMemory(non_const_out_grad);
  auto dst_memory_p2 = Grahandler.AcquireDstMemory(tem_max_mask);
  auto activation_p2 = Grahandler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args2 = {{DNNL_ARG_SRC_0, *src_memory_p_max2},
                                                {DNNL_ARG_SRC_1, *src_memory_p_out_grad2},
                                                {DNNL_ARG_DST, *dst_memory_p2}};

  if (Grahandler.Has_SRC_0_Scale()) {
    args2.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                 Grahandler.Get_SRC_0_Scale_Memory()});
  }

  if (Grahandler.Has_SRC_1_Scale()) {
    args2.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                 Grahandler.Get_SRC_1_Scale_Memory()});
  }

  activation_p2->execute(astream, args2);

  funcs::BinaryOneDNNHandler<T> Mulhandler1(dnnl::algorithm::binary_mul,
                                        -1,
                                        onednn_engine,
                                        dev_ctx.GetPlace(),
                                        tem_min_mask,
                                        tem_max_mask,
                                        tem_zero_mask,
                                        1.0f,
                                        1.0f,
                                        1.0f,
                                        true);

  auto src_memory_p_min3 = Mulhandler1.AcquireSrcMemory(tem_min_mask);
  auto src_memory_p_max3 = Mulhandler1.AcquireSecondSrcMemory(tem_max_mask);
  auto dst_memory_p3 = Mulhandler1.AcquireDstMemory(tem_zero_mask);
  auto activation_p3 = Mulhandler1.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args3 = {{DNNL_ARG_SRC_0, *src_memory_p_min3},
                                                {DNNL_ARG_SRC_1, *src_memory_p_max3},
                                                {DNNL_ARG_DST, *dst_memory_p3}};

  if (Mulhandler1.Has_SRC_0_Scale()) {
    args3.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                 Mulhandler1.Get_SRC_0_Scale_Memory()});
  }

  if (Mulhandler1.Has_SRC_1_Scale()) {
    args3.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                 Mulhandler1.Get_SRC_1_Scale_Memory()});
  }

  activation_p3->execute(astream, args3);

  funcs::BinaryOneDNNHandler<T> Mulhandler2(dnnl::algorithm::binary_mul,
                                        -1,
                                        onednn_engine,
                                        dev_ctx.GetPlace(),
                                        tem_zero_mask,
                                        non_const_x,
                                        x_grad,
                                        1.0f,
                                        1.0f,
                                        1.0f,
                                        true);

  auto src_memory_p_zero4 = Mulhandler2.AcquireSrcMemory(tem_zero_mask);
  auto src_memory_p_x4 = Mulhandler2.AcquireSecondSrcMemory(non_const_x);
  auto dst_memory_p4 = Mulhandler2.AcquireDstMemory(x_grad);
  auto activation_p4 = Mulhandler2.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> args4 = {{DNNL_ARG_SRC_0, *src_memory_p_zero4},
                                                {DNNL_ARG_SRC_1, *src_memory_p_x4},
                                                {DNNL_ARG_DST, *dst_memory_p4}};

  if (Mulhandler2.Has_SRC_0_Scale()) {
    args4.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                 Mulhandler2.Get_SRC_0_Scale_Memory()});
  }

  if (Mulhandler2.Has_SRC_1_Scale()) {
    args4.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                 Mulhandler2.Get_SRC_1_Scale_Memory()});
  }

  activation_p4->execute(astream, args4);

  astream.wait();

  x_grad->set_mem_desc(dst_memory_p4->get_desc());
}

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const Scalar& min,
                    const Scalar& max,
                    DenseTensor* x_grad) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  funcs::ClipOneDNNHandler<T> handler(
      min, max, onednn_engine, dev_ctx.GetPlace(), &x, &out_grad);

  auto src_memory_p = handler.AcquireBackwardSrcMemory(&x);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(&out_grad);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(x_grad);
  auto activation_backward_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_backward_p->execute(astream,
                                 {{DNNL_ARG_SRC, *src_memory_p},
                                  {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                  {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  x_grad->set_mem_desc(diff_dst_memory_p->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(clip_grad,
                   OneDNN,
                   ONEDNN,
                   phi::ClipGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(clip_tensor_grad,
                   OneDNN,
                   ONEDNN,
                   phi::ClipTensorGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

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

#include "paddle/phi/kernels/conv_transpose_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/xpu/conv_utils_xpu.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
#ifdef PADDLE_WITH_XPU_XRE5
#include "xpudnn/xpudnn.h"
namespace xpudnn = baidu::xpu::xpudnn;
#endif
namespace phi {

// target_len == 2 || target_len == 4
inline std::vector<int> vector_extend(const std::vector<int>& src,
                                      int target_len) {
  if (target_len == 2 && src.size() == 1) {
    return {src[0], src[0]};
  }
  if (target_len == 4 && src.size() == 1) {
    return {src[0], src[0], src[0], src[0]};
  }
  if (target_len == 4 && src.size() == 2) {
    return {src[0], src[0], src[1], src[1]};
  }
  return src;
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      data_format == "NHWC" || data_format == "NDHWC",
      false,
      errors::InvalidArgument(
          ("XPU do support data_format is NCHW in conv_transpose op.")));

  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  DDim filter_data_dims = slice_ddim(filter.dims(), 2, filter.dims().size());

#ifdef PADDLE_WITH_XPU_XRE5
  std::vector<int64_t> ksize = common::vectorize<int64_t>(filter_data_dims);
  std::vector<int64_t> paddings_ =
      std::vector<int64_t>(paddings.begin(), paddings.end());
  std::vector<int64_t> dilations_ =
      std::vector<int64_t>(dilations.begin(), dilations.end());
  std::vector<int64_t> strides_ =
      std::vector<int64_t>(strides.begin(), strides.end());
  UpdatePaddingAndDilation(&paddings_,
                           &dilations_,
                           padding_algorithm,
                           in_data_dims,
                           strides_,
                           ksize);

  const int64_t batch_size = static_cast<int64_t>(x.dims()[0]);
  const int64_t img_yc = static_cast<int64_t>(x.dims()[1]);
  const int64_t img_xc = static_cast<int64_t>(out->dims()[1]);
  const int64_t img_xh = static_cast<int64_t>(out->dims()[2]);
  const int64_t img_xw = static_cast<int64_t>(out->dims()[3]);

  int fc_calc_type = FCCalcType<XPUType>();
  if (fc_calc_type == XPUFCCalcType::FC_INT32) {
    int r = xpudnn::conv2d_transpose_fusion_v2<float, float, float, int32_t>(
        ctx.x_context(),
        x.data<float>(),
        filter.data<float>(),
        out->data<float>(),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides_,
        paddings_,
        dilations_,
        static_cast<int64_t>(groups),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        xpu::Activation_t::LINEAR,
        true,
        nullptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_fusion_v2");
  } else if (fc_calc_type == XPUFCCalcType::FC_FLOAT) {
    int r = xpudnn::conv2d_transpose_fusion_v2<float, float, float, float>(
        ctx.x_context(),
        x.data<float>(),
        filter.data<float>(),
        out->data<float>(),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides_,
        paddings_,
        dilations_,
        static_cast<int64_t>(groups),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        xpu::Activation_t::LINEAR,
        true,
        nullptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_fusion_v2");
  } else if (fc_calc_type == XPUFCCalcType::FC_INT32_WITH_LL) {
    if (output_size.size()) {
      VLOG(4) << "int_with_ll quantization is not supported when output_size "
                 "is specified, "
              << "use int31 instead";
      int r = xpudnn::conv2d_transpose_fusion_v2<float, float, float, int32_t>(
          ctx.x_context(),
          x.data<float>(),
          filter.data<float>(),
          out->data<float>(),
          batch_size,
          img_yc,
          img_xh,
          img_xw,
          img_xc,
          ksize,
          strides_,
          paddings_,
          dilations_,
          static_cast<int64_t>(groups),
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          xpu::Activation_t::LINEAR,
          true,
          nullptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_fusion_v2");
    } else {
      // xpu::conv2d_transpose_v2 do not support int_with_ll now
      // use xpu::conv2d_transpose
      int64_t img_yh = static_cast<int64_t>(x.dims()[2]);
      int64_t img_yw = static_cast<int64_t>(x.dims()[3]);
      int r = xpudnn::
          conv2d_transpose_fusion_v2<float, float, float, int_with_ll_t>(
              ctx.x_context(),
              x.data<float>(),
              filter.data<float>(),
              out->data<float>(),
              batch_size,
              img_yc,
              img_yh,
              img_yw,
              img_xc,
              ksize,
              strides_,
              paddings_,
              dilations_,
              static_cast<int64_t>(groups),
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              xpu::Activation_t::LINEAR,
              true,
              nullptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose");
    }
  } else {
    int r =
        xpudnn::conv2d_transpose_fusion_v2<XPUType, XPUType, XPUType, int16_t>(
            ctx.x_context(),
            reinterpret_cast<const XPUType*>(x.data<T>()),
            reinterpret_cast<const XPUType*>(filter.data<T>()),
            reinterpret_cast<XPUType*>(out->data<T>()),
            batch_size,
            img_yc,
            img_xh,
            img_xw,
            img_xc,
            ksize,
            strides_,
            paddings_,
            dilations_,
            static_cast<int64_t>(groups),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            xpu::Activation_t::LINEAR,
            true,
            nullptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_fusion_v2");
  }
#else
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(x.dims()[0]);
  const int img_yc = static_cast<int>(x.dims()[1]);
  const int img_xc = static_cast<int>(out->dims()[1]);
  const int img_xh = static_cast<int>(out->dims()[2]);
  const int img_xw = static_cast<int>(out->dims()[3]);

  int fc_calc_type = FCCalcType<XPUType>();
  if (fc_calc_type == XPUFCCalcType::FC_INT32) {
    int r = xpu::conv2d_transpose_v2<float, float, float, int32_t>(
        ctx.x_context(),
        x.data<float>(),
        filter.data<float>(),
        out->data<float>(),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
  } else if (fc_calc_type == XPUFCCalcType::FC_FLOAT) {
    int r = xpu::conv2d_transpose_v2<float, float, float, float>(
        ctx.x_context(),
        x.data<float>(),
        filter.data<float>(),
        out->data<float>(),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
  } else if (fc_calc_type == XPUFCCalcType::FC_INT32_WITH_LL) {
    if (output_size.size()) {
      VLOG(4) << "int_with_ll quantization is not supported when output_size "
                 "is specified, "
              << "use int31 instead";
      int r = xpu::conv2d_transpose_v2<float, float, float, int32_t>(
          ctx.x_context(),
          x.data<float>(),
          filter.data<float>(),
          out->data<float>(),
          batch_size,
          img_yc,
          img_xh,
          img_xw,
          img_xc,
          ksize,
          strides,
          paddings_,
          dilations_,
          groups,
          nullptr,
          nullptr,
          nullptr,
          true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
    } else {
      // xpu::conv2d_transpose_v2 do not support int_with_ll now
      // use xpu::conv2d_transpose
      int img_yh = static_cast<int>(x.dims()[2]);
      int img_yw = static_cast<int>(x.dims()[3]);
      int r = xpu::conv2d_transpose<float, float, float, int_with_ll_t>(
          ctx.x_context(),
          x.data<float>(),
          filter.data<float>(),
          out->data<float>(),
          batch_size,
          img_yc,
          img_yh,
          img_yw,
          img_xc,
          ksize,
          strides,
          paddings_,
          dilations_,
          groups,
          nullptr,
          nullptr,
          nullptr,
          true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose");
    }
  } else {
    int r = xpu::conv2d_transpose_v2<XPUType, XPUType, XPUType, int16_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(filter.data<T>()),
        reinterpret_cast<XPUType*>(out->data<T>()),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
  }
#endif
}

template <typename T, typename Context>
void Conv3dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const std::vector<int>& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(out);

  // data_format follow the legacy interface:
  // https://github.com/PaddlePaddle/Paddle/blob/639abfd4/python/paddle/nn/functional/conv.py#L1726
  PADDLE_ENFORCE_EQ(
      data_format == "NCHW" || data_format == "NHWC",
      true,
      errors::InvalidArgument(
          ("XPU only support data_format is NCHW(in Python, it is specified as "
           "NCDHW) or NHWC(in Python, it is specified as NDHWC) in "
           "conv3d_transpose op.")));

  phi::DDim in_data_dims;
  if (data_format == "NHWC") {
    in_data_dims = common::slice_ddim(x.dims(), 1, x.dims().size() - 1);
  } else {
    in_data_dims = common::slice_ddim(x.dims(), 2, x.dims().size());
  }
  phi::DDim filter_data_dims =
      common::slice_ddim(filter.dims(), 2, filter.dims().size());

  std::vector<int64_t> ksize = common::vectorize<int64_t>(filter_data_dims);
  std::vector<int64_t> paddings_ =
      std::vector<int64_t>(paddings.begin(), paddings.end());
  std::vector<int64_t> dilations_ =
      std::vector<int64_t>(dilations.begin(), dilations.end());
  std::vector<int64_t> strides_ =
      std::vector<int64_t>(strides.begin(), strides.end());
  UpdatePaddingAndDilation(&paddings_,
                           &dilations_,
                           padding_algorithm,
                           in_data_dims,
                           strides_,
                           ksize);

  for (int64_t dilation : dilations_) {
    PADDLE_ENFORCE_LE(
        dilation,
        1,
        errors::Unimplemented(
            "XPU do not support dilation > 1 in conv3d_transpose."));
  }

  int64_t batch_size = x.dims()[0];
  int64_t img_yc = x.dims()[1];
  int64_t img_yd = x.dims()[2];
  int64_t img_yh = x.dims()[3];
  int64_t img_yw = x.dims()[4];
  int64_t img_xc = out->dims()[1];

  bool is_ndhwc = false;
  if (data_format == "NHWC") {
    img_yc = x.dims()[4];
    img_yd = x.dims()[1];
    img_yh = x.dims()[2];
    img_yw = x.dims()[3];
    img_xc = out->dims()[4];
    is_ndhwc = true;

    PADDLE_ENFORCE_LE(
        groups,
        1,
        errors::Unimplemented("XPU do not support group > 1 when data_format "
                              "is NHWC(in Python, it is specified as NDHWC) "
                              "in conv3d_transpose."));
  }

  const XPUType* filter_data =
      reinterpret_cast<const XPUType*>(filter.data<T>());

  int fc_calc_type = GetConvCalcType<XPUType>();
  PD_VISIT_XPU_CONV_TYPES(XPUType, fc_calc_type, "conv3d_transpose", [&] {
    using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
    using RealTGEMM = std::conditional_t<
        (
            // 如果 XPUType 是 XPUTypeFP16 且 TGEMM 不是 FP16 或 int16
            (std::is_same_v<XPUType, XPUTypeFP16> &&
             !std::is_same_v<TGEMM, XPUTypeFP16> &&
             !std::is_same_v<TGEMM, int16_t>) ||

            // 如果 XPUType 是 float 且 TGEMM 不是 int32、int16 或 tfloat32
            (std::is_same_v<XPUType, float> &&
             !std::is_same_v<TGEMM, int32_t> &&
             !std::is_same_v<TGEMM, int16_t> &&
             !std::is_same_v<TGEMM, tfloat32>)),
        std::conditional_t<std::is_same_v<XPUType, XPUTypeFP16>,
                           XPUTypeFP16,
                           tfloat32>,
        TGEMM>;

    int ret = xpudnn::conv3d_transpose<XPUType, XPUType, XPUType, RealTGEMM>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        filter_data,  // filter的shape固定为[yc, xc, fd, fh, fw],
        reinterpret_cast<XPUType*>(out->data<T>()),
        batch_size,
        img_yc,
        img_yd,
        img_yh,
        img_yw,
        img_xc,
        ksize,
        strides_,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        is_ndhwc);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "conv3d_transpose");
  });
}

template <typename T, typename Context>
void DepthwiseConv2dTransposeKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& filter,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    const std::vector<int>& output_padding,
                                    const IntArray& output_size,
                                    const std::string& padding_algorithm,
                                    int groups,
                                    const std::vector<int>& dilations,
                                    const std::string& data_format,
                                    DenseTensor* out) {
  Conv2dTransposeKernel<T, Context>(ctx,
                                    x,
                                    filter,
                                    strides,
                                    paddings,
                                    output_padding,
                                    output_size,
                                    padding_algorithm,
                                    groups,
                                    dilations,
                                    data_format,
                                    out);
}

}  // namespace phi
PD_REGISTER_KERNEL(depthwise_conv2d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dTransposeKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv2d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeKernel,
                   float,
                   phi::dtype::float16) {}

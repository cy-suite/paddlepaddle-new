/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/layernorm_shift_partition_op.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

class LayerNormShiftPartitionOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a layernorm_shift_partition op to tensorrt "
               "layernorm_shift_partition plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
    const int begin_norm_axis =
        op_desc.HasAttr("begin_norm_axis")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
            : 1;
    const float eps = op_desc.HasAttr("epsilon")
                          ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
    const int window_size =
        PADDLE_GET_CONST(int, op_desc.GetAttr("window_size"));
    const int shift_size = PADDLE_GET_CONST(int, op_desc.GetAttr("shift_size"));
    const int input_resolution =
        PADDLE_GET_CONST(int, op_desc.GetAttr("input_resolution"));
    // int shift_size = window_size / 2;
    // shift_size = (input_resolution <= window_size) ? 0 : shift_size;
    // int shift_size = 0;

    PADDLE_ENFORCE_NOT_NULL(
        Bias_v,
        common::errors::InvalidArgument(
            "Input(Bias) of layer_norm should not be null."));
    PADDLE_ENFORCE_NOT_NULL(
        Scale_v,
        common::errors::InvalidArgument(
            "Input(Scale) of layer_norm should not be null."));
    PADDLE_ENFORCE_EQ(
        begin_norm_axis,
        2,
        common::errors::InvalidArgument(
            "The begin_norm_axis of LayernormShiftPartition should be %d",
            begin_norm_axis));

    auto* Bias_t = Bias_v->GetMutable<phi::DenseTensor>();
    auto* Scale_t = Scale_v->GetMutable<phi::DenseTensor>();

    auto bias_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Bias").front(), *Bias_t);
    auto scale_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Scale").front(), *Scale_t);
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    PADDLE_ENFORCE_EQ(bias_weight.get().count,
                      scale_weight.get().count,
                      common::errors::InvalidArgument(
                          "The num between bias_weight and scale_weight should "
                          "be equal. (%d vs %d)",
                          bias_weight.get().count,
                          scale_weight.get().count));
    nvinfer1::ILayer* layernorm_layer = nullptr;
    plugin::LayernormShiftPartitionPluginDynamic* plugin =
        new plugin::LayernormShiftPartitionPluginDynamic(
            static_cast<const float*>(scale_weight.get().values),
            static_cast<const float*>(bias_weight.get().values),
            bias_weight.get().count,
            shift_size,
            window_size,
            input_resolution,
            eps,
            with_fp16);
    layernorm_layer = engine_->AddDynamicPlugin(&X, 1, plugin);

    auto output_name = op_desc.Output("Y").front();
    ReplenishLayerAndOutput(
        layernorm_layer, "layernorm_shift_partition", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(layernorm_shift_partition,
                          LayerNormShiftPartitionOpConverter);

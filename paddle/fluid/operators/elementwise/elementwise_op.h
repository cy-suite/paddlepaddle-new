/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>  // for max
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/elementwise/elementwise_op_function.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

namespace paddle {
namespace operators {

class ElementwiseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ElementwiseOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ElementwiseOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ElementwiseOp");

    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Y").front(),
        framework::proto::VarType::DENSE_TENSOR,
        common::errors::InvalidArgument(
            "The input var's type should be phi::DenseTensor, but the "
            "received is %s [%s].",
            ctx->GetInputsVarType("Y").front(),
            ctx->Inputs("Y").front()));

    if (ctx->GetInputsVarType("X").front() ==
        framework::proto::VarType::SELECTED_ROWS) {
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Y").size(),
          1u,
          common::errors::InvalidArgument(
              "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
              "), Y must be scalar, the size of Y should be 1. "
              "But received the size of Y = %s.",
              ctx->GetInputDim("Y").size()));
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Y")[0],
          1,
          common::errors::InvalidArgument(
              "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
              "), Y must be scalar, the first dimension of Y should be 1. "
              "But received the first dimension of Y = %s.",
              ctx->GetInputDim("Y")[0]));
    } else if (ctx->GetInputsVarType("X").front() !=
               framework::proto::VarType::DENSE_TENSOR) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Input X's type[%s] is not supported by elementwise_op. Please set "
          "its type to DENSE_TENSOR.",
          ctx->GetInputsVarType("X").front()));
    }

    if (ctx->GetInputDim("X") == ctx->GetInputDim("Y")) {
      ctx->ShareDim("X", /*->*/ "Out");
      ctx->ShareLoD("X", /*->*/ "Out");
    } else {
      auto x_dims = ctx->GetInputDim("X");
      auto y_dims = ctx->GetInputDim("Y");
      int max_dim = std::max(x_dims.size(), y_dims.size());
      int axis = ctx->Attrs().Get<int>("axis");
      if (x_dims.size() == y_dims.size()) {
        PADDLE_ENFORCE_EQ((axis == -1) || (axis == 0),
                          true,
                          common::errors::InvalidArgument(
                              "axis should be -1 or 0 while the dimension of "
                              "tensor X (%s) is equal to the dimension of "
                              "tensor Y (%s), but received axis: %s",
                              x_dims.size(),
                              y_dims.size(),
                              axis));
      }
      PADDLE_ENFORCE_EQ((axis >= (-1 * max_dim)) && (axis < max_dim),
                        true,
                        common::errors::InvalidArgument(
                            "The axis range must be [%s, %s), but axis is %s. "
                            "Please set the axis again.",
                            -1 * max_dim,
                            max_dim,
                            axis));
      axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                       : axis);
      std::vector<int> x_dims_array(max_dim);
      std::vector<int> y_dims_array(max_dim);
      std::vector<int> out_dims_array(max_dim);
#ifdef PADDLE_WITH_DNNL
      // Broadcasting of dims has to be done on Paddle shapes (NHWC)
      // if model is using NHWC and any of shapes in at least 3D
      bool should_rotate =
          ctx->IsRunMKLDNNKernel() &&
          (phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
           phi::DataLayout::kNHWC) &&
          (x_dims.size() >= 3 || y_dims.size() >= 3);
      if (should_rotate) {
        // Pick bigger shape and rotate this one
        bool x_over_y = (x_dims.size() > y_dims.size());
        auto vdims = x_over_y ? common::vectorize<int>(x_dims)
                              : common::vectorize<int>(y_dims);
        std::rotate(vdims.begin() + 1, vdims.begin() + 2, vdims.end());
        if (x_over_y) {
          x_dims = common::make_ddim(vdims);
        } else {
          y_dims = common::make_ddim(vdims);
        }
      }
#endif

      phi::funcs::GetBroadcastDimsArrays(x_dims,
                                         y_dims,
                                         x_dims_array.data(),
                                         y_dims_array.data(),
                                         out_dims_array.data(),
                                         max_dim,
                                         axis);
#ifdef PADDLE_WITH_DNNL
      // Now rotate shape back if needed (NHWC -> NCHW)
      if (should_rotate) {
        std::rotate(out_dims_array.begin() + 1,
                    out_dims_array.end() - 1,
                    out_dims_array.end());
      }
#endif
      ctx->SetOutputDim("Out", common::make_ddim(out_dims_array));
      // to do
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name UNUSED,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs's types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
#ifdef PADDLE_WITH_DNNL
      // When elementwise is first oneDNN op (there was some non oneDNN op
      // previously)
      // then we also need to rotate shape NHWC -> NCWH
      if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
          (tensor.layout() != phi::DataLayout::ONEDNN) &&
          phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
              phi::DataLayout::kNHWC) {
        return phi::KernelKey(tensor.place(),
                              phi::DataLayout::kNHWC,
                              expected_kernel_type.dtype());
      }
#endif
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class ElementwiseOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class ElementwiseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInputX();
    AddInputY();
    AddOpOutput();
    AddAttr<int>("axis",
                 "(int, default -1). If X.dimension != Y.dimension,"
                 "Y.dimension must be a subsequence of x.dimension. And axis "
                 "is the start dimension index "
                 "for broadcasting Y onto X. ")
        .SetDefault(-1);
    AddOpComment();
  }

 protected:
  virtual void AddInputX() {
    AddInput("X", "(Tensor), The first input tensor of elementwise op.");
  }
  virtual void AddInputY() {
    AddInput("Y", "(Tensor), The second input tensor of elementwise op.");
  }
  virtual void AddOpOutput() {
    AddOutput("Out",
              "N-dimension tensor. A location into which the result is stored. "
              "It's dimension "
              "equals with x");
  }
  virtual void AddOpComment() { AddComment(GetCommentExamples()); }

  virtual std::string GetOpFunctionality() const { return ""; }

  virtual std::string GetName() const = 0;
  virtual std::string GetEquation() const = 0;

  std::string GetCommentExamples() const {
    return string::Sprintf(R"DOC(
Elementwise %s Operator.

%s

The equation is:

$$%s$$

- $X$: a tensor of any dimension.
- $Y$: a tensor whose dimensions must be less than or equal to the dimensions of $X$.

There are two cases for this operator:

1. The shape of $Y$ is the same with $X$.
2. The shape of $Y$ is a continuous subsequence of $X$.

For case 2:

1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index
   for broadcasting $Y$ onto $X$.
2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of
   subsequence, such as shape(Y) = (2, 1) => (2).

For example:

  .. code-block:: text

    shape(X) = (2, 3, 4, 5), shape(Y) = (,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

)DOC",
                           GetName(),
                           GetOpFunctionality(),
                           GetEquation());
  }
};

class ElementwiseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ElementwiseOpGrad");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name),
                   "Input",
                   out_grad_name,
                   "ElementwiseOpGrad");
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", /*->*/ x_grad_name);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", /*->*/ y_grad_name);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name UNUSED,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs's types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class ElementwiseOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", x_grad_name);
      ctx->ShareLoD("X", x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", y_grad_name);
      ctx->ShareLoD("Y", y_grad_name);
    }
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
      ctx->ShareLoD("DOut", "DDOut");
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DOut");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name UNUSED,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs's types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class ElementwiseOpDoubleGradWithoutDXDY
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DOut", "DDOut");
      ctx->ShareLoD("DOut", "DDOut");
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type input_data_type;
    if (ctx.HasInput("DDX") == false) {
      OP_INOUT_CHECK(ctx.HasInput("DDY"),
                     "Input",
                     "DDY",
                     "ElementwiseOpDoubleGradWithoutDXDY");
      input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DDY");
    } else if (ctx.HasInput("DDY") == false) {
      OP_INOUT_CHECK(ctx.HasInput("DDX"),
                     "Input",
                     "DDX",
                     "ElementwiseOpDoubleGradWithoutDXDY");
      input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "DDX");
    } else {
      input_data_type =
          OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "DDX", "DDY");
    }
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name UNUSED,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs's types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

class ElementwiseOpTripleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    if (ctx->HasOutput("D_DDX")) {
      ctx->ShareDim("DDX", "D_DDX");
      ctx->ShareLoD("DDX", "D_DDX");
    }
    if (ctx->HasOutput("D_DDY")) {
      ctx->ShareDim("DDY", "D_DDY");
      ctx->ShareLoD("DDY", "D_DDY");
    }
    if (ctx->HasOutput("D_X")) {
      ctx->ShareDim("X", "D_X");
      ctx->ShareLoD("X", "D_X");
    }
    if (ctx->HasOutput("D_Y")) {
      ctx->ShareDim("Y", "D_Y");
      ctx->ShareLoD("Y", "D_Y");
    }
    if (ctx->HasOutput("D_DOut")) {
      ctx->ShareDim("DOut", "D_DOut");
      ctx->ShareLoD("DOut", "D_DOut");
    }
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::proto::VarType::Type input_data_type;
    input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "D_DDOut");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name UNUSED,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.dtype())) {
      // only promote inputs's types when contains complex input
      return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
    } else {
      return phi::KernelKey(
          tensor.place(), tensor.layout(), expected_kernel_type.dtype());
    }
  }
};

template <typename T>
class ElemwiseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *dx = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto &dout =
        *context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    phi::funcs::ElementwiseGradPreProcess(dout, dx);
  }
};

DECLARE_INPLACE_OP_INFERER(ElementwiseOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(ElementwiseGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_INPLACE_OP_INFERER(ElementwiseDoubleGradOpInplaceInferer,
                           {"DDX", "DDOut"});

DECLARE_INPLACE_OP_INFERER(ElementwiseTripleGradOpInplaceInferer,
                           {"D_DDOut", "D_DDX"});

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseGradNoBufVarsInferer, "X", "Y");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseDoubleGradNoBufVarsInferer,
                                    "Y",
                                    "DOut");
DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseTripleGradNoBufVarsInferer,
                                    "DDX",
                                    "DDY");

}  // namespace operators
}  // namespace paddle
#define REGISTER_ELEMWISE_GRAD_MAKER(kernel_type, op_name)              \
  template <typename T>                                                 \
  class kernel_type##GradMaker                                          \
      : public paddle::framework::SingleGradOpMaker<T> {                \
   public:                                                              \
    using ::paddle::framework::SingleGradOpMaker<T>::SingleGradOpMaker; \
                                                                        \
   protected:                                                           \
    void Apply(::paddle::framework::GradOpPtr<T> op) const override {   \
      op->SetType(#kernel_type "_grad");                                \
      op->SetInput("X", this->Input("X"));                              \
      op->SetInput("Y", this->Input("Y"));                              \
      op->SetInput(::paddle::framework::GradVarName("Out"),             \
                   this->OutputGrad("Out"));                            \
      op->SetAttrMap(this->Attrs());                                    \
      op->SetOutput(::paddle::framework::GradVarName("X"),              \
                    this->InputGrad("X"));                              \
      op->SetOutput(::paddle::framework::GradVarName("Y"),              \
                    this->InputGrad("Y"));                              \
    }                                                                   \
  }

#define REGISTER_ELEMWISE_EXPLICIT_OP_WITHOUT_GRAD(op_type, op_name)    \
  REGISTER_OPERATOR(op_type,                                            \
                    ::paddle::operators::ElementwiseOp,                 \
                    ::paddle::operators::Elementwise##op_name##OpMaker, \
                    ::paddle::operators::ElementwiseOpInferVarType,     \
                    op_type##GradMaker<::paddle::framework::OpDesc>,    \
                    op_type##GradMaker<::paddle::imperative::OpBase>,   \
                    ::paddle::operators::ElementwiseOpInplaceInferer);

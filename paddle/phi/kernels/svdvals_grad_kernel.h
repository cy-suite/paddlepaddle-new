#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void SvdvalsGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& s,
                       const paddle::optional<DenseTensor>& s_grad,
                       DenseTensor* x_grad);
}  // namespace phi

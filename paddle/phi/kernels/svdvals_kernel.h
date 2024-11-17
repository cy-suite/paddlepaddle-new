#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SvdvalsKernel(const Context& dev_ctx,
                   const DenseTensor& X,
                   DenseTensor* S);

}  // namespace phi

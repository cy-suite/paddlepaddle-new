#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/svdvals_grad_kernel.h"

PD_REGISTER_KERNEL(
    svdvals_grad, CPU, ALL_LAYOUT, phi::SvdvalsGradKernel, float, double) {}
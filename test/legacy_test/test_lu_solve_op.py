#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest

import numpy as np
import scipy.linalg

sys.path.append("..")
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


def ref_lu_factor(A):
    # Support broadcasting by handling each batch separately
    orig_shape = A.shape
    batch_shape = orig_shape[:-2]
    n = orig_shape[-1]

    # Reshape to 2D arrays
    A_flat = A.reshape(-1, n, n)

    # Process each matrix in batch
    lu_list = []
    piv_list = []
    for i in range(A_flat.shape[0]):
        lu, piv = scipy.linalg.lu_factor(A_flat[i])
        lu_list.append(lu)
        piv_list.append(piv)

    # Stack results and reshape back to original batch dimensions
    lu_result = np.stack(lu_list).reshape(*batch_shape, n, n)
    piv_result = np.stack(piv_list).reshape(*batch_shape, n)

    return lu_result, piv_result


def ref_lu_solve(lu_piv, b):
    lu, piv = lu_piv

    # Get shapes
    lu_batch_shape = lu.shape[:-2]  # batch dims of lu
    b_batch_shape = b.shape[:-2]  # batch dims of b
    n = lu.shape[-1]  # matrix dimension
    nrhs = b.shape[-1]  # number of right hand sides

    # Calculate broadcast shape
    try:
        broadcast_shape = np.broadcast_shapes(lu_batch_shape, b_batch_shape)
    except ValueError:
        raise ValueError(
            f"Incompatible batch shapes for broadcasting: {lu_batch_shape} vs {b_batch_shape}"
        )

    # Expand arrays to broadcast dimensions
    lu_expanded = np.broadcast_to(lu, (*broadcast_shape, n, n))
    piv_expanded = np.broadcast_to(piv, (*broadcast_shape, n))
    b_expanded = np.broadcast_to(b, (*broadcast_shape, n, nrhs))

    # Reshape to 2D arrays
    lu_flat = lu_expanded.reshape(-1, n, n)
    piv_flat = piv_expanded.reshape(-1, n)
    b_flat = b_expanded.reshape(-1, n, nrhs)

    # Process each system in batch
    x_list = []
    for i in range(b_flat.shape[0]):
        x = scipy.linalg.lu_solve((lu_flat[i], piv_flat[i]), b_flat[i])
        x_list.append(x)

    # Stack results and reshape back to broadcast dimensions
    x_result = np.stack(x_list).reshape(*broadcast_shape, n, nrhs)

    return x_result


class TestLUSolveOp(OpTest):
    """
    Test case 1:
    A is a square matrix, and B is the corresponding right-hand side.
    """

    def config(self):
        self.A_shape = [25, 25]
        self.b_shape = [25, 5]
        self.dtype = np.float64

    def generate_well_conditioned_matrix(self):
        """
        Generate a well-conditioned matrix for higher dimensions by:
        1. Starting with a random matrix
        2. Adding a scaled identity matrix
        3. Using matrix balancing
        4. Ensuring diagonal dominance
        """
        batch_shape = self.A_shape[:-2]
        n = self.A_shape[-1]

        # Initialize the final array
        A = np.zeros(self.A_shape, dtype=self.dtype)

        # Generate matrices for each batch element
        for idx in np.ndindex(*batch_shape):
            # Create random matrix
            matrix = np.random.random((n, n)).astype(self.dtype)

            # Add scaled identity to improve conditioning
            matrix += np.eye(n, dtype=self.dtype) * n

            # Ensure diagonal dominance
            diag_abs = np.abs(matrix.diagonal())
            row_sums = np.sum(np.abs(matrix), axis=1) - diag_abs
            scale = 2.0 * row_sums / diag_abs
            matrix[range(n), range(n)] *= scale

            # Assign to the corresponding batch position
            A[idx] = matrix

        return A

    def set_output(self):
        np.random.seed(2025)
        # LU and pivots are obtained from matrix A
        lu = self.inputs['lu']
        pivots = self.inputs['pivots']
        self.output = ref_lu_solve((lu, pivots), self.inputs['b'])

    def setUp(self):
        self.op_type = "lu_solve"
        self.python_api = paddle.linalg.lu_solve
        self.config()
        # Generate a non-singular matrix A: random matrix plus identity matrix
        A = self.generate_well_conditioned_matrix()
        # Compute LU factorization of A
        lu, pivots = ref_lu_factor(A)
        b = np.random.random(self.b_shape).astype(self.dtype)
        self.inputs = {
            "lu": lu,
            "pivots": pivots,  # pivots is an integer array
            "b": b,
        }
        self.set_output()
        self.outputs = {"out": self.output}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        # Only check gradients with respect to the right-hand side B,
        # because LU factors and pivots do not participate in backpropagation.
        self.check_grad(["b"], ["out"], max_relative_error=1e-5, check_pir=True)


class TestLUSolveOpBroadcast(TestLUSolveOp):
    """
    Test case 2 (broadcast/batch scenario)
    """

    def config(self):
        self.A_shape = [1, 2, 3, 15, 15]
        self.b_shape = [1, 3, 15, 5]
        self.dtype = np.float64


class TestLUSolveAPI(unittest.TestCase):
    def config(self):
        self.A_shape = [15, 15]
        self.b_shape = [15, 5]
        self.dtype = "float32"

    def generate_well_conditioned_matrix(self):
        """
        Generate a well-conditioned matrix for higher dimensions by:
        1. Starting with a random matrix
        2. Adding a scaled identity matrix
        3. Using matrix balancing
        4. Ensuring diagonal dominance
        """
        batch_shape = self.A_shape[:-2]
        n = self.A_shape[-1]

        # Initialize the final array
        A = np.zeros(self.A_shape, dtype=self.dtype)

        # Generate matrices for each batch element
        for idx in np.ndindex(*batch_shape):
            # Create random matrix
            matrix = np.random.random((n, n)).astype(self.dtype)

            # Add scaled identity to improve conditioning
            matrix += np.eye(n, dtype=self.dtype) * n

            # Ensure diagonal dominance
            diag_abs = np.abs(matrix.diagonal())
            row_sums = np.sum(np.abs(matrix), axis=1) - diag_abs
            scale = 2.0 * row_sums / diag_abs
            matrix[range(n), range(n)] *= scale

            # Assign to the corresponding batch position
            A[idx] = matrix

        return A

    def setUp(self):
        np.random.seed(2025)
        self.config()
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # Define input data
            lu = paddle.static.data(
                name="lu", shape=self.A_shape, dtype=self.dtype
            )
            pivots = paddle.static.data(
                name="pivots", shape=self.A_shape[:-1], dtype="int32"
            )
            b = paddle.static.data(
                name="b", shape=self.b_shape, dtype=self.dtype
            )
            out = paddle.linalg.lu_solve(lu, pivots, b)

            # Generate matrix A and compute its LU factorization
            A = self.generate_well_conditioned_matrix()
            b_np = np.random.random(self.b_shape).astype(self.dtype)

            lu, pivots = ref_lu_factor(A)
            ref_out = ref_lu_solve((lu, pivots), b_np)

            # Convert to 1-based indexing for paddle
            pivots = pivots + 1

            exe = base.Executor(place)
            fetches = exe.run(
                feed={"lu": lu, "pivots": pivots.astype("int32"), "b": b_np},
                fetch_list=[out],
            )
            np.testing.assert_allclose(fetches[0], ref_out, rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            # Generate random matrix and ensure it's well-conditioned
            A = self.generate_well_conditioned_matrix()
            b_np = np.random.random(self.b_shape).astype(self.dtype)

            # Compute LU factorization
            lu, pivots = ref_lu_factor(A)
            ref_out = ref_lu_solve((lu, pivots), b_np)

            # Convert to paddle tensors (pivots need to be 1-based)
            lu_tensor = paddle.to_tensor(lu)
            pivots_tensor = paddle.to_tensor(pivots.astype("int32") + 1)
            b_tensor = paddle.to_tensor(b_np)

            out = paddle.linalg.lu_solve(lu_tensor, pivots_tensor, b_tensor)
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestLUSolveAPIBroadcast(TestLUSolveAPI):
    def config(self):
        self.dtype = "float64"
        self.A_shape = [1, 2, 3, 15, 15]
        self.b_shape = [1, 3, 15, 5]


if __name__ == "__main__":
    unittest.main()

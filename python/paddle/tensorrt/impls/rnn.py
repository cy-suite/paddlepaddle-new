# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import tensorrt as trt

from paddle.tensorrt.register import converter_registry


@converter_registry.register("rnn", trt_version="8.x")
def rnn_converter(network, paddle_op, inputs):
    """
    Converter function to transform PaddlePaddle RNN (LSTM) operations into TensorRT layers.

    Args:
        network (trt.INetworkDefinition): The TensorRT network.
        paddle_op (paddle.fluid.proto.framework_pb2.OpDesc): The PaddlePaddle operation descriptor.
        inputs (dict): Dictionary of input tensors.

    Returns:
        trt.ITensor: The output tensor after applying the RNN layer.
    """
    # Retrieve inputs
    input_tensor = inputs.get("Input")
    pre_state_h = inputs.get("PreState_H")  # Previous hidden state
    pre_state_c = inputs.get("PreState_C")  # Previous cell state

    # Extract attributes
    num_layers = paddle_op.attr["num_layers"]
    hidden_size = paddle_op.attr["hidden_size"]
    input_size = paddle_op.attr["input_size"]
    is_bidirec = paddle_op.attr["is_bidirec"]
    K = 2 if is_bidirec else 1

    # Validate dimensions
    if input_tensor.shape.nb_dims != 3:
        raise ValueError(
            f"RNN(LSTM)'s input must be 3 dimensions [seq_len, batch, input_size], but got {input_tensor.shape.nb_dims} dimensions."
        )
    if pre_state_h.shape.nb_dims != 3:
        raise ValueError(
            f"RNN(LSTM)'s PreState_H must be 3 dimensions [num_layers, batch, hidden_size], but got {pre_state_h.shape.nb_dims} dimensions."
        )
    if pre_state_c.shape.nb_dims != 3:
        raise ValueError(
            f"RNN(LSTM)'s PreState_C must be 3 dimensions [num_layers, batch, hidden_size], but got {pre_state_c.shape.nb_dims} dimensions."
        )

    # Extract weights and biases
    weight_bias_vec = []
    weight_list = paddle_op.input("WeightList")
    scope = paddle_op.scope

    for layer_id in range(num_layers):
        if is_bidirec:

            def extract_and_combine_weight(start):
                for k in range(K):
                    var0_name = weight_list[k + start]
                    var1_name = weight_list[k + 2 + start]
                    var0 = scope.find_var(var0_name).get_tensor()
                    var1 = scope.find_var(var1_name).get_tensor()
                    data0_ptr = var0.data().numpy().flatten()
                    data1_ptr = var1.data().numpy().flatten()
                    combined = np.concatenate([data0_ptr, data1_ptr]).astype(
                        np.float32
                    )
                    weight_bias_vec.append(combined)

            extract_and_combine_weight(4 * layer_id)
            extract_and_combine_weight(4 * layer_id + 4 * num_layers)
        else:

            def extract_weight(start):
                for k in range(2 * K):
                    var_name = weight_list[k + start]
                    var = scope.find_var(var_name).get_tensor()
                    data_ptr = var.data().numpy().flatten().astype(np.float32)
                    weight_bias_vec.append(data_ptr)

            extract_weight(2 * layer_id)  # Filter weights
            extract_weight(2 * num_layers + 2 * layer_id)  # Biases

    # Create an identity layer for input
    identity_layer = network.add_identity(input_tensor)
    current_input = identity_layer.get_output(0)

    for layer_id in range(num_layers):
        # Loop layer over sequence length
        loop = network.add_loop()

        # Extract sequence length and batch size
        input_shape = network.add_shape(current_input).get_output(0)
        seq_len = network.add_slice(
            input_shape, start=(0,), size=(1,), stride=(1,)
        ).get_output(0)
        batch_size = network.add_slice(
            input_shape, start=(1,), size=(1,), stride=(1,)
        ).get_output(0)
        input_size_tensor = trt.ITensor.create_constant(
            network, np.array([input_size], dtype=np.int32)
        )

        # Add trip limit
        loop.set_trip_limit(seq_len)

        # Add iterators
        iter_forward = loop.add_iter(name='iter_forward').get_output(
            0
        )  # [batch, input_size]
        if is_bidirec:
            iter_reverse = loop.add_iter(
                name='iter_reverse', reverse=True
            ).get_output(0)
            # Concatenate forward and reverse
            iter_input = network.add_concatenation([iter_forward, iter_reverse])
            iter_input.axis = 0  # Concatenate on the first dimension (K)
            iter_input = iter_input.get_output(0)
        else:
            iter_input = iter_forward

        # Reshape iter_input to [K, batch, input_size]
        shuffle_layer = network.add_shuffle(iter_input)
        shuffle_layer.reshape_dims = (K, -1, input_size)
        iter_input_reshaped = shuffle_layer.get_output(0)

        # Gather previous hidden and cell states
        gather_h = network.add_gather(
            pre_state_h, loop.current_iteration, axis=0
        ).get_output(0)
        gather_c = network.add_gather(
            pre_state_c, loop.current_iteration, axis=0
        ).get_output(0)

        # Recurrence layers for hidden and cell states
        recurrence_h = loop.add_recurrence(gather_h)
        recurrence_c = loop.add_recurrence(gather_c)

        # Compute gates
        def run_matmul_bias(k, is_input=True):
            h_dim = 4 * hidden_size
            w_dim = input_size if is_input else hidden_size
            if is_input and K > 1:
                w_dim = K * hidden_size

            weight = weight_bias_vec[k].reshape((K, h_dim, w_dim))
            bias = weight_bias_vec[k + 2].reshape((K, 1, h_dim))

            # Create weights as constant tensors
            weight_tensor = network.add_constant(
                weight.shape, trt.Weights(weight.flatten())
            ).get_output(0)
            bias_tensor = network.add_constant(
                bias.shape, trt.Weights(bias.flatten())
            ).get_output(0)

            iter_tensor = (
                recurrence_h.get_output(0) if k % 2 else iter_input_reshaped
            )

            matmul = network.add_matrix_multiply(
                iter_tensor,
                trt.MatrixOperation.NONE,
                weight_tensor,
                trt.MatrixOperation.TRANSPOSE,
            ).get_output(0)
            matmul_bias = network.add_elementwise(
                matmul, bias_tensor, trt.ElementWiseOperation.SUM
            ).get_output(0)
            return matmul_bias

        iter_input_w_b = run_matmul_bias(layer_id * 4, is_input=True)
        iter_hidden_w_b = run_matmul_bias(layer_id * 4 + 1, is_input=False)
        iter_input_hidden_add = network.add_elementwise(
            iter_input_w_b, iter_hidden_w_b, trt.ElementWiseOperation.SUM
        ).get_output(0)

        # Split gates
        split_layers = []
        for i in range(4):
            slice_layer = network.add_slice(
                iter_input_hidden_add,
                start=(i * hidden_size,),
                size=(hidden_size,),
                stride=(1,),
            )
            split = slice_layer.get_output(0)
            activation = (
                trt.ActivationType.SIGMOID
                if i in [0, 1, 3]
                else trt.ActivationType.TANH
            )
            activated = network.add_activation(split, activation).get_output(0)
            split_layers.append(activated)

        i_gate, f_gate, c_gate, o_gate = split_layers

        # Cell state computation: C_t = i_gate * c_gate + f_gate * C_{t-1}
        ic_prod = network.add_elementwise(
            i_gate, c_gate, trt.ElementWiseOperation.PROD
        ).get_output(0)
        fCt1_prod = network.add_elementwise(
            f_gate, recurrence_c.get_output(0), trt.ElementWiseOperation.PROD
        ).get_output(0)
        Ct = network.add_elementwise(
            ic_prod, fCt1_prod, trt.ElementWiseOperation.SUM
        ).get_output(0)
        recurrence_c.set_input(Ct)

        # Hidden state computation: H_t = tanh(C_t) * o_gate
        tanh_ct = network.add_activation(
            Ct, trt.ActivationType.TANH
        ).get_output(0)
        Ht = network.add_elementwise(
            tanh_ct, o_gate, trt.ElementWiseOperation.PROD
        ).get_output(0)
        recurrence_h.set_input(Ht)

        # Handle bidirectional output
        if is_bidirec:
            slice_forward = network.add_slice(
                Ht, start=(0, 0, 0), size=(1, 0, 0), stride=(1, 1, 1)
            ).get_output(0)
            slice_reverse = network.add_slice(
                Ht, start=(1, 0, 0), size=(0, 0, 0), stride=(1, 1, 1)
            ).get_output(0)
            loop_output_forward = loop.add_output(
                slice_forward, trt.LoopOutput.CONCATENATE
            )
            loop_output_reverse = loop.add_output(
                slice_reverse, trt.LoopOutput.REVERSE
            )
            concat_output = network.add_concatenation(
                [loop_output_forward, loop_output_reverse]
            )
            concat_output.axis = 3  # Concatenate on the last dimension
            final_tensor = concat_output.get_output(0)
        else:
            loop_output = loop.add_output(Ht, trt.LoopOutput.CONCATENATE)
            final_tensor = loop_output.get_output(0)

        # Reshape final_tensor to [seq_len, batch, K * hidden_size]
        shuffle_final = network.add_shuffle(final_tensor)
        shuffle_final.reshape_dims = (
            seq_len.shape,
            batch_size.shape,
            K * hidden_size,
        )
        current_input = shuffle_final.get_output(0)

    # Set the output
    output = current_input  # Assuming single output for simplicity
    return output

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest
from io import BytesIO

import numpy as np
from test_imperative_base import new_program_scope

import paddle
import paddle.optimizer as opt
from paddle import base, nn
from paddle.base import core, framework
from paddle.framework import in_pir_mode
from paddle.framework.io_utils import get_value, is_pir_fetch_var, set_value
from paddle.optimizer import Adam
from paddle.optimizer.lr import LRScheduler

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4
SEED = 10

IMAGE_SIZE = 784
CLASS_NUM = 10

LARGE_PARAM = 2**26


def random_batch_reader():
    def _get_random_inputs_and_labels():
        np.random.seed(SEED)
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.randint(
            0,
            CLASS_NUM - 1,
            (
                BATCH_SIZE,
                1,
            ),
        ).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_inputs_and_labels()
            batch_image = paddle.to_tensor(batch_image)
            batch_label = paddle.to_tensor(batch_label)
            yield batch_image, batch_label

    return __reader__


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)


class LayerWithLargeParameters(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._l = paddle.nn.Linear(10, LARGE_PARAM)

    def forward(self, x):
        y = self._l(x)
        return y


def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()


class TestSaveLoadLargeParameters(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_large_parameters_paddle_save(self):
        # enable dygraph mode
        paddle.disable_static()
        paddle.set_device("cpu")
        # create network
        layer = LayerWithLargeParameters()
        save_dict = layer.state_dict()

        path = os.path.join(
            self.temp_dir.name,
            "test_paddle_save_load_large_param_save",
            "layer.pdparams",
        )
        protocol = 4
        paddle.save(save_dict, path, protocol=protocol)
        dict_load = paddle.load(path, return_numpy=True)
        # compare results before and after saving
        for key, value in save_dict.items():
            np.testing.assert_array_equal(dict_load[key], value.numpy())


class TestSaveLoadPickle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pickle_protocol(self):
        # enable dygraph mode
        paddle.disable_static()
        # create network
        layer = LinearNet()
        save_dict = layer.state_dict()

        path = os.path.join(
            self.temp_dir.name,
            "test_paddle_save_load_pickle_protocol",
            "layer.pdparams",
        )

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 2.0)

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 1)

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 5)

        protocols = [2, 3, 4]
        for protocol in protocols:
            paddle.save(save_dict, path, pickle_protocol=protocol)
            dict_load = paddle.load(path)
            # compare results before and after saving
            for key, value in save_dict.items():
                np.testing.assert_array_equal(
                    dict_load[key].numpy(), value.numpy()
                )


class TestSaveLoadAny(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def set_zero(self, prog, place, scope=None):
        if scope is None:
            scope = base.global_scope()
        for var in prog.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                if is_pir_fetch_var(var):
                    continue
                ten = scope.find_var(var.name).get_tensor()
                if ten is not None:
                    ten.set(np.zeros_like(np.array(ten)), place)
                    new_t = np.array(scope.find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

    def replace_static_save(self, program, model_path, pickle_protocol=2):
        scope = base.global_scope()
        with self.assertRaises(TypeError):
            program.state_dict(1)
        with self.assertRaises(TypeError):
            program.state_dict(scope=1)
        with self.assertRaises(ValueError):
            program.state_dict('x', scope)
        state_dict_param = program.state_dict('param', scope)
        paddle.save(state_dict_param, model_path + '.pdparams')
        state_dict_opt = program.state_dict('opt', scope)
        paddle.save(state_dict_opt, model_path + '.pdopt')
        state_dict_all = program.state_dict('all', scope)
        paddle.save(state_dict_opt, model_path + '.pdall')

    def replace_static_load(self, program, model_path):
        with self.assertRaises(TypeError):
            program.set_state_dict(1)
        scope = base.global_scope()
        state_dict_param = paddle.load(model_path + '.pdparams')
        if not in_pir_mode():
            state_dict_param['fake_var_name.@@'] = np.random.randn(1, 2)
            state_dict_param['static_x'] = 'UserWarning'
            program.set_state_dict(state_dict_param)
            state_dict_param['static_x'] = np.random.randn(1, 2)
            program.set_state_dict(state_dict_param)
        program.set_state_dict(state_dict_param, scope)
        state_dict_opt = paddle.load(model_path + '.pdopt')
        program.set_state_dict(state_dict_opt, scope)

    def test_replace_static_save_load(self):
        paddle.enable_static()
        with new_program_scope():
            x = paddle.static.data(
                name="static_x", shape=[None, IMAGE_SIZE], dtype='float32'
            )
            z = paddle.static.nn.fc(x, 10)
            z = paddle.static.nn.fc(z, 10, bias_attr=False)
            loss = paddle.mean(z)
            opt = Adam(learning_rate=1e-3)
            opt.minimize(loss)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            fake_inputs = np.random.randn(2, IMAGE_SIZE).astype('float32')
            exe.run(prog, feed={'static_x': fake_inputs}, fetch_list=[loss])
            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_map[var.name] = t
            path = os.path.join(
                self.temp_dir.name, "test_replace_static_save_load", "model"
            )
            # paddle.save, legacy paddle.base.load
            self.replace_static_save(prog, path)
            self.set_zero(prog, place)
            paddle.static.load(prog, path)
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, np.array(base_t))
            # legacy paddle.base.save, paddle.load
            paddle.static.save(prog, path)
            self.set_zero(prog, place)
            self.replace_static_load(prog, path)
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            # test for return tensor
            path_vars = 'test_replace_save_load_return_tensor_static/model'
            for var in prog.list_vars():
                if var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    tensor = base.global_scope().find_var(var.name).get_tensor()
                    paddle.save(
                        tensor,
                        os.path.join(self.temp_dir.name, path_vars, var.name),
                    )

            # Pir value currently does not have .set_value() and .get_value()
            # Instead, use new functions to replace them
            with self.assertRaises(TypeError):
                get_value(var, 'base.global_scope()')
            # Pir get_value() currently does not raise ValueError
            # Maybe fix it later
            if not in_pir_mode():
                with self.assertRaises(ValueError):
                    x.get_value()
            with self.assertRaises(TypeError):
                set_value(x, '1')
            fake_data = np.zeros([3, 2, 1, 2, 3])
            with self.assertRaises(TypeError):
                set_value(x, fake_data, '1')
            with self.assertRaises(ValueError):
                set_value(x, fake_data)
            with self.assertRaises(ValueError):
                set_value(var, fake_data)
            # set var to zero
            self.set_zero(prog, place)
            for var in prog.list_vars():
                if var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    tensor = paddle.load(
                        os.path.join(self.temp_dir.name, path_vars, var.name),
                        return_numpy=False,
                    )
                    set_value(var, tensor)
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

    def test_paddle_save_load_v2(self):
        paddle.disable_static()

        class StepDecay(LRScheduler):
            def __init__(
                self,
                learning_rate,
                step_size,
                gamma=0.1,
                last_epoch=-1,
                verbose=False,
            ):
                self.step_size = step_size
                self.gamma = gamma
                super().__init__(learning_rate, last_epoch, verbose)

            def get_lr(self):
                i = self.last_epoch // self.step_size
                return self.base_lr * (self.gamma**i)

        layer = LinearNet()
        inps = paddle.randn([2, IMAGE_SIZE])
        adam = opt.Adam(
            learning_rate=StepDecay(0.1, 1), parameters=layer.parameters()
        )
        y = layer(inps)
        y.mean().backward()
        adam.step()
        state_dict = adam.state_dict()
        path = os.path.join(
            self.temp_dir.name, 'paddle_save_load_v2/model.pdparams'
        )
        with self.assertRaises(TypeError):
            paddle.save(state_dict, path, use_binary_format='False')
        # legacy paddle.save, paddle.load
        paddle.framework.io._legacy_save(state_dict, path)
        load_dict_tensor = paddle.load(path, return_numpy=False)
        # legacy paddle.load, paddle.save
        paddle.save(state_dict, path)
        load_dict_np = paddle.framework.io._legacy_load(path)
        for k, v in state_dict.items():
            if isinstance(v, dict):
                self.assertTrue(v == load_dict_tensor[k])
            else:
                np.testing.assert_array_equal(
                    v.numpy(), load_dict_tensor[k].numpy()
                )
                if not np.array_equal(v.numpy(), load_dict_np[k]):
                    print(v.numpy())
                    print(load_dict_np[k])
                np.testing.assert_array_equal(v.numpy(), load_dict_np[k])

    def test_single_pickle_var_dygraph(self):
        # enable dygraph mode
        paddle.disable_static()
        layer = LinearNet()
        path = os.path.join(
            self.temp_dir.name, 'paddle_save_load_v2/var_dygraph'
        )
        tensor = layer._linear.weight
        with self.assertRaises(ValueError):
            paddle.save(tensor, path, pickle_protocol='3')
        with self.assertRaises(ValueError):
            paddle.save(tensor, path, pickle_protocol=5)
        paddle.save(tensor, path)
        t_dygraph = paddle.load(path)
        np_dygraph = paddle.load(path, return_numpy=True)
        self.assertTrue(
            isinstance(
                t_dygraph,
                paddle.base.core.eager.Tensor,
            )
        )
        np.testing.assert_array_equal(tensor.numpy(), np_dygraph)
        np.testing.assert_array_equal(tensor.numpy(), t_dygraph.numpy())
        paddle.enable_static()
        lod_static = paddle.load(path)
        np_static = paddle.load(path, return_numpy=True)
        self.assertTrue(isinstance(lod_static, paddle.base.core.DenseTensor))
        np.testing.assert_array_equal(tensor.numpy(), np_static)
        np.testing.assert_array_equal(tensor.numpy(), np.array(lod_static))

    def test_single_pickle_var_static(self):
        # enable static graph mode
        paddle.enable_static()
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="x", shape=[None, IMAGE_SIZE], dtype='float32'
            )
            z = paddle.static.nn.fc(x, 128)
            loss = paddle.mean(z)
            place = (
                base.CPUPlace()
                if not paddle.base.core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            for var in prog.list_vars():
                if list(var.shape) == [IMAGE_SIZE, 128]:
                    tensor = get_value(var)
                    break
            scope = base.global_scope()
        origin_tensor = np.array(tensor)
        path = os.path.join(
            self.temp_dir.name, 'test_single_pickle_var_static/var'
        )
        paddle.save(tensor, path)
        self.set_zero(prog, place, scope)
        # static load
        lod_static = paddle.load(path)
        np_static = paddle.load(path, return_numpy=True)
        # set_tensor(np.ndarray)
        set_value(var, np_static, scope)
        np.testing.assert_array_equal(origin_tensor, np.array(tensor))
        # set_tensor(DenseTensor)
        self.set_zero(prog, place, scope)
        set_value(var, lod_static, scope)
        np.testing.assert_array_equal(origin_tensor, np.array(tensor))
        # enable dygraph mode
        paddle.disable_static()
        var_dygraph = paddle.load(path)
        np_dygraph = paddle.load(path, return_numpy=True)
        np.testing.assert_array_equal(np.array(tensor), np_dygraph)
        np.testing.assert_array_equal(np.array(tensor), var_dygraph.numpy())

    def test_dygraph_save_static_load_pir(self):
        inps = np.random.randn(1, IMAGE_SIZE).astype('float32')
        path = os.path.join(
            self.temp_dir.name,
            'test_dygraph_save_static_load/dy-static.pdparams',
        )
        paddle.disable_static()
        with paddle.utils.unique_name.guard():
            layer = LinearNet()
            state_dict_dy = layer.state_dict()
            paddle.save(state_dict_dy, path)
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            with new_program_scope():
                layer = LinearNet()
                data = paddle.static.data(
                    name='x_static_save',
                    shape=(None, IMAGE_SIZE),
                    dtype='float32',
                )
                y_static = layer(data)
                program = paddle.static.default_main_program()
                place = (
                    base.CPUPlace()
                    if not paddle.base.core.is_compiled_with_cuda()
                    else base.CUDAPlace(0)
                )
                exe = paddle.static.Executor(paddle.CPUPlace())
                exe.run(paddle.static.default_startup_program())
                state_dict = paddle.load(path, keep_name_table=True)
                paddle.pir.core.set_state_dict(
                    program, state_dict, paddle.static.global_scope()
                )
                state_dict_param = program.state_dict(
                    "param", paddle.static.global_scope()
                )
                for name, tensor in state_dict_dy.items():
                    np.testing.assert_array_equal(
                        tensor.numpy(), np.array(state_dict_param[tensor.name])
                    )

    def test_save_load_complex_object_dygraph_save(self):
        paddle.disable_static()
        layer = paddle.nn.Linear(3, 4)
        state_dict = layer.state_dict()
        obj1 = [
            paddle.randn([3, 4], dtype='float32'),
            np.random.randn(5, 6),
            ('fake_weight', np.ones([7, 8], dtype='float32')),
        ]
        obj2 = {'k1': obj1, 'k2': state_dict, 'epoch': 123}
        obj3 = (
            paddle.randn([5, 4], dtype='float32'),
            np.random.randn(3, 4).astype("float32"),
            {"state_dict": state_dict, "opt": state_dict},
        )
        obj4 = (np.random.randn(5, 6), (123,))

        path1 = os.path.join(
            self.temp_dir.name, "test_save_load_any_complex_object_dygraph/obj1"
        )
        path2 = os.path.join(
            self.temp_dir.name, "test_save_load_any_complex_object_dygraph/obj2"
        )
        path3 = os.path.join(
            self.temp_dir.name, "test_save_load_any_complex_object_dygraph/obj3"
        )
        path4 = os.path.join(
            self.temp_dir.name, "test_save_load_any_complex_object_dygraph/obj4"
        )
        paddle.save(obj1, path1)
        paddle.save(obj2, path2)
        paddle.save(obj3, path3)
        paddle.save(obj4, path4)

        load_tensor1 = paddle.load(path1, return_numpy=False)
        load_tensor2 = paddle.load(path2, return_numpy=False)
        load_tensor3 = paddle.load(path3, return_numpy=False)
        load_tensor4 = paddle.load(path4, return_numpy=False)

        np.testing.assert_array_equal(load_tensor1[0].numpy(), obj1[0].numpy())
        np.testing.assert_array_equal(load_tensor1[1], obj1[1])
        np.testing.assert_array_equal(load_tensor1[2].numpy(), obj1[2][1])
        for i in range(len(load_tensor1)):
            self.assertTrue(
                type(load_tensor1[i]) == type(load_tensor2['k1'][i])
            )
        for k, v in state_dict.items():
            np.testing.assert_array_equal(
                v.numpy(), load_tensor2['k2'][k].numpy()
            )
        self.assertTrue(load_tensor2['epoch'] == 123)

        np.testing.assert_array_equal(load_tensor3[0].numpy(), obj3[0].numpy())
        np.testing.assert_array_equal(np.array(load_tensor3[1]), obj3[1])

        for k, v in state_dict.items():
            np.testing.assert_array_equal(
                load_tensor3[2]['state_dict'][k].numpy(), v.numpy()
            )

        for k, v in state_dict.items():
            np.testing.assert_array_equal(
                load_tensor3[2]['opt'][k].numpy(), v.numpy()
            )

        np.testing.assert_array_equal(load_tensor4[0].numpy(), obj4[0])

        load_array1 = paddle.load(path1, return_numpy=True)
        load_array2 = paddle.load(path2, return_numpy=True)
        load_array3 = paddle.load(path3, return_numpy=True)
        load_array4 = paddle.load(path4, return_numpy=True)

        np.testing.assert_array_equal(load_array1[0], obj1[0].numpy())
        np.testing.assert_array_equal(load_array1[1], obj1[1])
        np.testing.assert_array_equal(load_array1[2], obj1[2][1])
        for i in range(len(load_array1)):
            self.assertTrue(type(load_array1[i]) == type(load_array2['k1'][i]))
        for k, v in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), load_array2['k2'][k])
        self.assertTrue(load_array2['epoch'] == 123)

        np.testing.assert_array_equal(load_array3[0], obj3[0].numpy())
        np.testing.assert_array_equal(load_array3[1], obj3[1])

        for k, v in state_dict.items():
            np.testing.assert_array_equal(
                load_array3[2]['state_dict'][k], v.numpy()
            )

        for k, v in state_dict.items():
            np.testing.assert_array_equal(load_array3[2]['opt'][k], v.numpy())

        np.testing.assert_array_equal(load_array4[0], obj4[0])

        # static graph mode
        paddle.enable_static()

        load_tensor1 = paddle.load(path1, return_numpy=False)
        load_tensor2 = paddle.load(path2, return_numpy=False)
        load_tensor3 = paddle.load(path3, return_numpy=False)
        load_tensor4 = paddle.load(path4, return_numpy=False)

        np.testing.assert_array_equal(
            np.array(load_tensor1[0]), obj1[0].numpy()
        )
        np.testing.assert_array_equal(np.array(load_tensor1[1]), obj1[1])
        np.testing.assert_array_equal(np.array(load_tensor1[2]), obj1[2][1])

        for i in range(len(load_tensor1)):
            self.assertTrue(
                type(load_tensor1[i]) == type(load_tensor2['k1'][i])
            )
        for k, v in state_dict.items():
            np.testing.assert_array_equal(
                v.numpy(), np.array(load_tensor2['k2'][k])
            )
        self.assertTrue(load_tensor2['epoch'] == 123)

        self.assertTrue(
            isinstance(load_tensor3[0], paddle.base.core.DenseTensor)
        )
        np.testing.assert_array_equal(
            np.array(load_tensor3[0]), obj3[0].numpy()
        )
        np.testing.assert_array_equal(np.array(load_tensor3[1]), obj3[1])

        for k, v in state_dict.items():
            self.assertTrue(
                isinstance(
                    load_tensor3[2]["state_dict"][k],
                    paddle.base.core.DenseTensor,
                )
            )
            np.testing.assert_array_equal(
                np.array(load_tensor3[2]['state_dict'][k]), v.numpy()
            )

        for k, v in state_dict.items():
            self.assertTrue(
                isinstance(
                    load_tensor3[2]["opt"][k], paddle.base.core.DenseTensor
                )
            )
            np.testing.assert_array_equal(
                np.array(load_tensor3[2]['opt'][k]), v.numpy()
            )

        self.assertTrue(load_tensor4[0], paddle.base.core.DenseTensor)
        np.testing.assert_array_equal(np.array(load_tensor4[0]), obj4[0])

        load_array1 = paddle.load(path1, return_numpy=True)
        load_array2 = paddle.load(path2, return_numpy=True)
        load_array3 = paddle.load(path3, return_numpy=True)
        load_array4 = paddle.load(path4, return_numpy=True)

        np.testing.assert_array_equal(load_array1[0], obj1[0].numpy())
        np.testing.assert_array_equal(load_array1[1], obj1[1])
        np.testing.assert_array_equal(load_array1[2], obj1[2][1])
        for i in range(len(load_array1)):
            self.assertTrue(type(load_array1[i]) == type(load_array2['k1'][i]))
        for k, v in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), load_array2['k2'][k])
        self.assertTrue(load_array2['epoch'] == 123)

        self.assertTrue(isinstance(load_array3[0], np.ndarray))
        np.testing.assert_array_equal(load_array3[0], obj3[0].numpy())
        np.testing.assert_array_equal(load_array3[1], obj3[1])

        for k, v in state_dict.items():
            np.testing.assert_array_equal(
                load_array3[2]['state_dict'][k], v.numpy()
            )

        for k, v in state_dict.items():
            np.testing.assert_array_equal(load_array3[2]['opt'][k], v.numpy())

        np.testing.assert_array_equal(load_array4[0], obj4[0])

    def test_save_load_complex_object_static_save(self):
        paddle.enable_static()
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="x", shape=[None, IMAGE_SIZE], dtype='float32'
            )
            z = paddle.static.nn.fc(x, 10, bias_attr=False)
            z = paddle.static.nn.fc(z, 128, bias_attr=False)
            loss = paddle.mean(z)
            place = (
                base.CPUPlace()
                if not paddle.base.core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())

            state_dict = prog.state_dict('all', base.global_scope())
            keys = list(state_dict.keys())
            obj1 = [
                state_dict[keys[0]],
                np.random.randn(5, 6),
                ('fake_weight', np.ones([7, 8], dtype='float32')),
            ]
            obj2 = {'k1': obj1, 'k2': state_dict, 'epoch': 123}
            obj3 = (
                state_dict[keys[0]],
                np.ndarray([3, 4], dtype="float32"),
                {"state_dict": state_dict, "opt": state_dict},
            )
            obj4 = (np.ndarray([3, 4], dtype="float32"),)

            path1 = os.path.join(
                self.temp_dir.name,
                "test_save_load_any_complex_object_static/obj1",
            )
            path2 = os.path.join(
                self.temp_dir.name,
                "test_save_load_any_complex_object_static/obj2",
            )
            path3 = os.path.join(
                self.temp_dir.name,
                "test_save_load_any_complex_object_static/obj3",
            )
            path4 = os.path.join(
                self.temp_dir.name,
                "test_save_load_any_complex_object_static/obj4",
            )
            paddle.save(obj1, path1)
            paddle.save(obj2, path2)
            paddle.save(obj3, path3)
            paddle.save(obj4, path4)

            load_tensor1 = paddle.load(path1, return_numpy=False)
            load_tensor2 = paddle.load(path2, return_numpy=False)
            load_tensor3 = paddle.load(path3, return_numpy=False)
            load_tensor4 = paddle.load(path4, return_numpy=False)

            np.testing.assert_array_equal(
                np.array(load_tensor1[0]), np.array(obj1[0])
            )
            np.testing.assert_array_equal(np.array(load_tensor1[1]), obj1[1])
            np.testing.assert_array_equal(np.array(load_tensor1[2]), obj1[2][1])
            for i in range(len(load_tensor1)):
                self.assertTrue(
                    type(load_tensor1[i]) == type(load_tensor2['k1'][i])
                )
            for k, v in state_dict.items():
                np.testing.assert_array_equal(
                    np.array(v), np.array(load_tensor2['k2'][k])
                )
            self.assertTrue(load_tensor2['epoch'] == 123)

            self.assertTrue(isinstance(load_tensor3[0], base.core.DenseTensor))
            np.testing.assert_array_equal(np.array(load_tensor3[0]), obj3[0])
            self.assertTrue(isinstance(load_tensor3[1], base.core.DenseTensor))
            np.testing.assert_array_equal(np.array(load_tensor3[1]), obj3[1])

            for k, v in state_dict.items():
                self.assertTrue(
                    isinstance(
                        load_tensor3[2]["state_dict"][k], base.core.DenseTensor
                    )
                )
                np.testing.assert_array_equal(
                    np.array(load_tensor3[2]['state_dict'][k]), np.array(v)
                )

            for k, v in state_dict.items():
                self.assertTrue(
                    isinstance(load_tensor3[2]["opt"][k], base.core.DenseTensor)
                )
                np.testing.assert_array_equal(
                    np.array(load_tensor3[2]['opt'][k]), np.array(v)
                )

            self.assertTrue(isinstance(load_tensor4[0], base.core.DenseTensor))
            np.testing.assert_array_equal(np.array(load_tensor4[0]), obj4[0])

            load_array1 = paddle.load(path1, return_numpy=True)
            load_array2 = paddle.load(path2, return_numpy=True)
            load_array3 = paddle.load(path3, return_numpy=True)
            load_array4 = paddle.load(path4, return_numpy=True)

            np.testing.assert_array_equal(load_array1[0], np.array(obj1[0]))
            np.testing.assert_array_equal(load_array1[1], obj1[1])
            np.testing.assert_array_equal(load_array1[2], obj1[2][1])
            for i in range(len(load_array1)):
                self.assertTrue(
                    type(load_array1[i]) == type(load_array2['k1'][i])
                )
            for k, v in state_dict.items():
                np.testing.assert_array_equal(np.array(v), load_array2['k2'][k])
            self.assertTrue(load_array2['epoch'] == 123)

            np.testing.assert_array_equal(load_array3[0], np.array(obj3[0]))
            np.testing.assert_array_equal(load_array3[1], obj3[1])

            for k, v in state_dict.items():
                np.testing.assert_array_equal(
                    load_array3[2]['state_dict'][k], np.array(v)
                )

            for k, v in state_dict.items():
                np.testing.assert_array_equal(
                    load_array3[2]['opt'][k], np.array(v)
                )

            np.testing.assert_array_equal(load_array4[0], obj4[0])

            # dygraph mode
            paddle.disable_static()

            load_tensor1 = paddle.load(path1, return_numpy=False)
            load_tensor2 = paddle.load(path2, return_numpy=False)
            load_tensor3 = paddle.load(path3, return_numpy=False)
            load_tensor4 = paddle.load(path4, return_numpy=False)

            np.testing.assert_array_equal(
                np.array(load_tensor1[0]), np.array(obj1[0])
            )
            np.testing.assert_array_equal(np.array(load_tensor1[1]), obj1[1])
            np.testing.assert_array_equal(load_tensor1[2].numpy(), obj1[2][1])
            for i in range(len(load_tensor1)):
                self.assertTrue(
                    type(load_tensor1[i]) == type(load_tensor2['k1'][i])
                )
            for k, v in state_dict.items():
                np.testing.assert_array_equal(
                    np.array(v), np.array(load_tensor2['k2'][k])
                )
            self.assertTrue(load_tensor2['epoch'] == 123)

            self.assertTrue(
                isinstance(
                    load_tensor3[0],
                    base.core.eager.Tensor,
                )
            )
            np.testing.assert_array_equal(load_tensor3[0].numpy(), obj3[0])
            self.assertTrue(
                isinstance(
                    load_tensor3[1],
                    base.core.eager.Tensor,
                )
            )
            np.testing.assert_array_equal(load_tensor3[1].numpy(), obj3[1])

            for k, v in state_dict.items():
                self.assertTrue(
                    isinstance(
                        load_tensor3[2]["state_dict"][k],
                        base.core.eager.Tensor,
                    )
                )
                np.testing.assert_array_equal(
                    load_tensor3[2]['state_dict'][k].numpy(), np.array(v)
                )

            for k, v in state_dict.items():
                self.assertTrue(
                    isinstance(
                        load_tensor3[2]["opt"][k],
                        base.core.eager.Tensor,
                    )
                )
                np.testing.assert_array_equal(
                    load_tensor3[2]['opt'][k].numpy(), np.array(v)
                )

            self.assertTrue(
                isinstance(
                    load_tensor4[0],
                    base.core.eager.Tensor,
                )
            )
            np.testing.assert_array_equal(load_tensor4[0].numpy(), obj4[0])

            load_array1 = paddle.load(path1, return_numpy=True)
            load_array2 = paddle.load(path2, return_numpy=True)
            load_array3 = paddle.load(path3, return_numpy=True)
            load_array4 = paddle.load(path4, return_numpy=True)

            np.testing.assert_array_equal(load_array1[0], np.array(obj1[0]))
            np.testing.assert_array_equal(load_array1[1], obj1[1])
            np.testing.assert_array_equal(load_array1[2], obj1[2][1])
            for i in range(len(load_array1)):
                self.assertTrue(
                    type(load_array1[i]) == type(load_array2['k1'][i])
                )
            for k, v in state_dict.items():
                np.testing.assert_array_equal(np.array(v), load_array2['k2'][k])
            self.assertTrue(load_array2['epoch'] == 123)

            np.testing.assert_array_equal(load_array3[0], np.array(obj3[0]))
            np.testing.assert_array_equal(load_array3[1], obj3[1])

            for k, v in state_dict.items():
                np.testing.assert_array_equal(
                    load_array3[2]['state_dict'][k], np.array(v)
                )

            for k, v in state_dict.items():
                np.testing.assert_array_equal(
                    load_array3[2]['opt'][k], np.array(v)
                )

            self.assertTrue(isinstance(load_array4[0], np.ndarray))
            np.testing.assert_array_equal(load_array4[0], obj4[0])

    def test_varbase_binary_var(self):
        paddle.disable_static()
        varbase = paddle.randn([3, 2], dtype='float32')
        path = os.path.join(
            self.temp_dir.name,
            'test_paddle_save_load_varbase_binary_var/varbase',
        )
        paddle.save(varbase, path, use_binary_format=True)
        load_array = paddle.load(path, return_numpy=True)
        load_tensor = paddle.load(path, return_numpy=False)
        origin_array = varbase.numpy()
        load_tensor_array = load_tensor.numpy()
        if paddle.base.core.is_compiled_with_cuda():
            base.core._cuda_synchronize(paddle.CUDAPlace(0))
        np.testing.assert_array_equal(origin_array, load_array)
        np.testing.assert_array_equal(origin_array, load_tensor_array)


class TestSaveLoadToMemory(unittest.TestCase):
    def test_dygraph_save_to_memory(self):
        paddle.disable_static()
        linear = LinearNet()
        state_dict = linear.state_dict()
        byio = BytesIO()
        paddle.save(state_dict, byio)
        tensor = paddle.randn([2, 3], dtype='float32')
        paddle.save(tensor, byio)
        byio.seek(0)
        # load state_dict
        dict_load = paddle.load(byio, return_numpy=True)
        for k, v in state_dict.items():
            np.testing.assert_array_equal(v.numpy(), dict_load[k])
        # load tensor
        tensor_load = paddle.load(byio, return_numpy=True)
        np.testing.assert_array_equal(tensor_load, tensor.numpy())

        with self.assertRaises(ValueError):
            paddle.save(4, 3)
        with self.assertRaises(ValueError):
            paddle.save(state_dict, '')
        with self.assertRaises(ValueError):
            paddle.framework.io_utils._open_file_buffer('temp', 'b')


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()

        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_and_train_model(self):
        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()

        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

        # create data loader
        # TODO: using new DataLoader cause unknown Timeout on windows, replace it
        loader = random_batch_reader()

        # train
        train(layer, loader, loss_fn, adam)

        return layer, adam

    def check_load_state_dict(self, orig_dict, load_dict):
        for var_name, value in orig_dict.items():
            load_value = (
                load_dict[var_name].numpy()
                if hasattr(load_dict[var_name], 'numpy')
                else np.array(load_dict[var_name])
            )
            np.testing.assert_array_equal(value.numpy(), load_value)

    def test_save_load(self):
        paddle.disable_static()
        layer, opt = self.build_and_train_model()

        # save
        layer_save_path = os.path.join(
            self.temp_dir.name, "test_paddle_save_load.linear.pdparams"
        )
        opt_save_path = os.path.join(
            self.temp_dir.name, "test_paddle_save_load.linear.pdopt"
        )
        layer_state_dict = layer.state_dict()
        opt_state_dict = opt.state_dict()

        paddle.save(layer_state_dict, layer_save_path)
        paddle.save(opt_state_dict, opt_save_path)

        # load
        load_layer_state_dict = paddle.load(layer_save_path)
        load_opt_state_dict = paddle.load(opt_save_path)

        self.check_load_state_dict(layer_state_dict, load_layer_state_dict)
        self.check_load_state_dict(opt_state_dict, load_opt_state_dict)

        # test save load in static graph mode
        paddle.enable_static()
        static_save_path = os.path.join(
            self.temp_dir.name,
            "static_mode_test/test_paddle_save_load.linear.pdparams",
        )
        paddle.save(layer_state_dict, static_save_path)
        load_static_state_dict = paddle.load(static_save_path)
        self.check_load_state_dict(layer_state_dict, load_static_state_dict)

        # error test cases, some tests relay base test above
        # 1. test save obj not dict error
        test_list = [1, 2, 3]

        # 2. test save path format error
        with self.assertRaises(ValueError):
            paddle.save(
                layer_state_dict,
                os.path.join(
                    self.temp_dir.name, "test_paddle_save_load.linear.model/"
                ),
            )

        # 3. test load path not exist error
        with self.assertRaises(ValueError):
            paddle.load(
                os.path.join(
                    self.temp_dir.name, "test_paddle_save_load.linear.params"
                )
            )

        # 4. test load old save path error
        with self.assertRaises(ValueError):
            paddle.load(
                os.path.join(self.temp_dir.name, "test_paddle_save_load.linear")
            )


class TestAsyncSaveLoad(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()

        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_and_train_model(self):
        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()

        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

        # create data loader
        # TODO: using new DataLoader cause unknown Timeout on windows, replace it
        loader = random_batch_reader()

        # train
        train(layer, loader, loss_fn, adam)

        return layer, adam

    def check_load_state_dict(self, orig_dict, load_dict):
        for var_name, value in orig_dict.items():
            load_value = (
                load_dict[var_name].numpy()
                if hasattr(load_dict[var_name], 'numpy')
                else np.array(load_dict[var_name])
            )
            np.testing.assert_array_equal(value.numpy(), load_value)

    def test_async_save_load(self):
        layer, opt = self.build_and_train_model()

        # save
        layer_save_path = os.path.join(
            self.temp_dir.name, "test_paddle_async_save_load.linear.pdparams"
        )
        opt_save_path = os.path.join(
            self.temp_dir.name, "test_paddle_async_save_load.linear.pdopt"
        )
        layer_state_dict = layer.state_dict()
        opt_state_dict = opt.state_dict()

        paddle.async_save(
            layer_state_dict, layer_save_path, sync_other_task=True
        )
        paddle.async_save(opt_state_dict, opt_save_path)
        paddle.clear_async_save_task_queue()

        # load
        load_layer_state_dict = paddle.load(layer_save_path)
        load_opt_state_dict = paddle.load(opt_save_path)

        self.check_load_state_dict(layer_state_dict, load_layer_state_dict)
        self.check_load_state_dict(opt_state_dict, load_opt_state_dict)

        # test assertion on illegal object
        some_tuple_obj = (1, 2, 3)
        tuple_save_path = os.path.join(
            self.temp_dir.name, "test_paddle_async_save_load.tuple.pdparams"
        )
        with self.assertRaises(TypeError):
            paddle.async_save(some_tuple_obj, tuple_save_path)

        # test assertion on static graph
        paddle.enable_static()
        static_save_path = os.path.join(
            self.temp_dir.name,
            "static_mode_test/test_paddle_async_save_load.linear.pdparams",
        )
        with self.assertRaises(ValueError):
            paddle.async_save(layer_state_dict, static_save_path)


class TestSaveLoadProgram(unittest.TestCase):
    def test_save_load_program_pir(self):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            with new_program_scope():
                layer = LinearNet()
                data = paddle.static.data(
                    name='x_static_save',
                    shape=(None, IMAGE_SIZE),
                    dtype='float32',
                )
                y_static = layer(data)
                main_program = paddle.static.default_main_program()
                startup_program = paddle.static.default_startup_program()
                path1 = os.path.join(
                    temp_dir.name,
                    "test_paddle_save_load_program/main_program.json",
                )
                path2 = os.path.join(
                    temp_dir.name,
                    "test_paddle_save_load_program/startup_program.json",
                )
                paddle.save(main_program, path1)
                paddle.save(startup_program, path2)

            with new_program_scope():
                load_main = paddle.load(path1)
                load_startup = paddle.load(path2)
                self.assertTrue(
                    len(main_program.global_block().ops)
                    == len(load_main.global_block().ops)
                )
                self.assertTrue(
                    len(startup_program.global_block().ops)
                    == len(load_startup.global_block().ops)
                )
            temp_dir.cleanup()


class TestSaveLoadLayer(unittest.TestCase):
    def test_save_load_layer(self):
        paddle.disable_static()
        temp_dir = tempfile.TemporaryDirectory()
        inps = paddle.randn([1, IMAGE_SIZE], dtype='float32')
        layer1 = LinearNet()
        layer2 = LinearNet()
        layer1.eval()
        layer2.eval()
        origin_layer = (layer1, layer2)
        origin = (layer1(inps), layer2(inps))
        path = os.path.join(
            temp_dir.name, "test_save_load_layer_/layer.pdmodel"
        )
        with self.assertRaises(ValueError):
            paddle.save(origin_layer, path)
        temp_dir.cleanup()


class TestSaveLoadRngState(unittest.TestCase):
    def test_save_load_layer(self):
        paddle.disable_static()
        paddle.set_device('cpu')
        paddle.seed(42)
        temp_dir = tempfile.TemporaryDirectory()
        rand_a = paddle.rand([2, 2])
        checkpoint_rng_state = {
            "cpu": paddle.framework.core.default_cpu_generator().get_state()
        }
        rand_b = paddle.rand([2, 2])
        path = os.path.join(temp_dir.name, "test_save_load_rng_/rng_state.pth")
        paddle.save(checkpoint_rng_state, path)
        checkpoint_rng_state = paddle.load(path, return_numpy=True)
        core.default_cpu_generator().set_state(checkpoint_rng_state["cpu"])
        rand_c = paddle.rand([2, 2])
        np.testing.assert_array_equal(rand_b.numpy(), rand_c.numpy())
        temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()

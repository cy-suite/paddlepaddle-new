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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import BatchSampler, DataLoader


class Config:
    def __init__(self):
        self.batch_num = 5
        self.batch_size = 4
        self.seq_len = 8
        self.input_size = 32
        self.hidden_size = 16
        self.class_num = 10
        self.run_ep = False
        self.mesh = dist.ProcessMesh([0, 1])
        self.expert_mesh_list = []
        self.expert_mesh_list.append(dist.ProcessMesh([0]))
        self.expert_mesh_list.append(dist.ProcessMesh([1]))
        self.layer_type = "DemoLayer"


class Config_shared:
    def __init__(self):
        self.batch_num = 5
        self.batch_size = 4
        self.input_size = 32
        self.hidden_size = 16
        self.shared_hidden_size = 32
        self.seq_len = 8
        self.class_num = 10
        self.run_ep = False
        self.mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.num_devices = 2
        self.num_experts = 4
        self.layer_type = "DemoSharedLayer"
        self.expert_mesh_list = []
        for i in range(self.num_devices):
            for j in range(self.num_experts // self.num_devices):
                self.expert_mesh_list.append(
                    dist.ProcessMesh([i], dim_names=["x"])
                )


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples, return_dict=False):
        self.images = images
        self.labels = labels
        self.num_samples = self.images.shape[0]
        self.return_dict = return_dict

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "image": self.images[idx],
                "label": self.labels[idx],
            }
        else:
            return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class MLP(nn.Layer):
    def __init__(self, config, is_shared=False):
        super().__init__()
        self.config = config
        if is_shared:
            self.input_size = config.input_size * config.num_experts
        else:
            self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.class_num = config.class_num
        self.down_proj = nn.Linear(
            self.hidden_size, self.class_num, bias_attr=False
        )

    def redistribute_expert(self, mesh, placements):
        # place the experts on different devices
        self.down_proj.weight = dist.shard_tensor(
            self.down_proj.weight, mesh, placements
        )

    def forward(self, x):
        return self.down_proj(x)


class DemoLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.gate.weight = dist.shard_tensor(
            self.gate.weight, config.mesh, [dist.Replicate()]
        )

        self.experts = nn.LayerList()
        self.experts.append(MLP(config))
        self.experts.append(MLP(config))
        if config.run_ep:
            for i, expert in enumerate(self.experts):
                expert.redistribute_expert(
                    config.expert_mesh_list[i], [dist.Replicate()]
                )

    def forward(self, x):
        h = self.gate(x)
        if self.config.run_ep:
            local_val_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
                h, self.config.mesh, 0, [dist.Shard(0)]
            )
        else:
            local_val_list = paddle.split(h, num_or_sections=2, axis=0)
        expert_out_list = []
        for i, expert in enumerate(self.experts):
            local_val = local_val_list[i]
            expert_out_list.append(expert(local_val))
        if self.config.run_ep:
            out = dist.auto_parallel.api.moe_global_mesh_tensor(
                expert_out_list, self.config.mesh, [dist.Shard(0)], 0
            )
        else:
            out = paddle.stack(expert_out_list, axis=0)
            out = out.reshape((-1, self.config.seq_len, self.config.class_num))
        return out


class DemoSharedLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.gate.weight = dist.shard_tensor(
            self.gate.weight, config.mesh, [dist.Replicate()]
        )
        self.shared_gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.shared_gate.weight = dist.shard_tensor(
            self.shared_gate.weight, config.mesh, [dist.Replicate()]
        )
        self.shared_expert = MLP(config, is_shared=True)
        self.experts = nn.LayerList()
        for i in range(self.config.num_experts):
            self.experts.append(MLP(config))
        if config.run_ep:
            self.shared_expert.redistribute_expert(
                self.config.mesh, [dist.Replicate()]
            )
            for i, expert in enumerate(self.experts):
                expert.redistribute_expert(
                    config.expert_mesh_list[i], [dist.Replicate()]
                )

    def forward(self, x):
        h = self.gate(x)
        y = self.shared_gate(x)

        if self.config.run_ep:
            local_val_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
                h, self.config.mesh, 0, [dist.Shard(0)]
            )
        else:
            local_val_list = paddle.split(
                h, num_or_sections=self.config.num_experts, axis=0
            )
        expert_out_list = []
        if self.config.run_ep:
            for i in range(self.config.num_devices):
                device_input = paddle.split(
                    local_val_list[i],
                    num_or_sections=self.config.num_experts
                    // self.config.num_devices,
                    axis=0,
                )
                device_out = []
                for j in range(
                    self.config.num_experts // self.config.num_devices
                ):
                    local_val = device_input[j]
                    device_out.append(
                        self.experts[
                            i
                            * self.config.num_experts
                            // self.config.num_devices
                            + j
                        ](local_val)
                    )
                expert_out_list.append(paddle.stack(device_out, axis=0))
        else:
            for i, expert in enumerate(self.experts):
                local_val = local_val_list[i]
                expert_out_list.append(expert(local_val))
        z = self.shared_expert(y)
        if self.config.run_ep:
            out = dist.auto_parallel.api.moe_global_mesh_tensor(
                expert_out_list, self.config.mesh, [dist.Shard(0)], 0
            )
        else:
            out = paddle.stack(expert_out_list, axis=0)
            out = out.reshape((-1, self.config.seq_len, self.config.class_num))
        out = paddle.squeeze(out)
        return out + z


class DemoAll2AllLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.input_size, config.hidden_size, bias_attr=False
        )
        self.gate.weight = dist.shard_tensor(
            self.gate.weight, config.mesh, [dist.Replicate()]
        )

        param_initializer = paddle.nn.initializer.Constant(value=0.0)
        b_shape = [config.batch_size, config.seq_len, config.input_size]
        self.b = dist.shard_tensor(
            paddle.create_parameter(
                shape=b_shape,
                dtype="float32",
                default_initializer=param_initializer,
            ),
            config.mesh,
            [dist.Shard(1)],
        )

        self.experts = nn.LayerList()
        self.experts.append(MLP(config))
        self.experts.append(MLP(config))
        if config.run_ep:
            for i, expert in enumerate(self.experts):
                expert.redistribute_expert(
                    config.expert_mesh_list[i], [dist.Replicate()]
                )

    def forward(self, x):
        x = x + self.b
        dispatched_input = self.gate(x)
        dispatched_input = dist.reshard(
            dispatched_input, self.config.mesh, [dist.Shard(0)]
        )
        local_val_list = dist.auto_parallel.api.moe_sub_mesh_tensors(
            dispatched_input, self.config.mesh, 0, [dist.Shard(0)]
        )

        expert_out_list = []
        for i, expert in enumerate(self.experts):
            local_val = local_val_list[i]
            expert_out_list.append(expert(local_val))

        out = dist.auto_parallel.api.moe_global_mesh_tensor(
            expert_out_list, self.config.mesh, [dist.Shard(0)], 0
        )

        return out


class Criterion(nn.Layer):
    def __init__(self):
        super().__init__()
        self.loss_func = paddle.nn.MSELoss()

    def forward(self, logits, labels):
        loss = self.loss_func(logits, labels)
        return loss


class TestSimpleNetForEP:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))

        paddle.set_device(self._backend)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_optimizer(self, model, lr_scheduler=None):
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        )
        return optimizer

    def create_data_loader(self, config):
        nsamples = config.batch_size * config.batch_num
        images = np.random.rand(
            nsamples, config.seq_len, config.input_size
        ).astype('float32')
        labels = np.random.rand(
            nsamples, config.seq_len, config.class_num
        ).astype('float32')
        train_dataset = RandomDataset(images, labels, config.batch_size)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        return train_dataloader

    def build(self, config):
        model_classes = {
            "DemoLayer": DemoLayer,
            "DemoSharedLayer": DemoSharedLayer,
            "DemoAll2AllLayer": DemoAll2AllLayer,
        }

        if config.layer_type not in model_classes:
            raise ValueError(f"Unsupported layer type: {config.layer_type}")

        model = model_classes[config.layer_type](config)
        dataloader = self.create_data_loader(config)
        optimizer = self.create_optimizer(model)
        criterion = Criterion()
        return model, dataloader, criterion, optimizer

    def build_shared(self, config):
        model = DemoSharedLayer(config)
        dataloader = self.create_data_loader(config)
        optimizer = self.create_optimizer(model)
        criterion = Criterion()
        return model, dataloader, criterion, optimizer

    def train(self, config, model, train_dataloader, criterion, optimizer):
        tr_loss = float(0)
        global_step = 0
        model.train()

        losses = []
        for step, inputs in enumerate(train_dataloader()):
            inputs, labels = inputs
            logits = model(inputs)
            tr_loss = criterion(logits, labels)

            tr_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(tr_loss.numpy())

        return losses

    def train_dy2st(
        self, config, model, train_dataloader, criterion, optimizer
    ):
        dist_model = dist.to_static(
            model, train_dataloader, criterion, optimizer
        )
        dist_model.train()

        loss_list = []
        for batch_id, data in enumerate(train_dataloader()):
            if isinstance(data, dict):
                image = data['image']
                label = data['label']
            else:
                image, label = data
            loss = dist_model(image, label)
            loss_list.append(loss)

        paddle.disable_static()
        global_mesh = dist.ProcessMesh([0, 1])
        pd_loss_dy2st = paddle.to_tensor(np.array(loss_list))
        pd_loss_dy2st = dist.auto_parallel.api.dtensor_from_local(
            pd_loss_dy2st,
            global_mesh,
            [dist.Partial(dist.ReduceType.kRedAvg)],
        )
        pd_loss_dy2st = dist.reshard(
            pd_loss_dy2st, global_mesh, [dist.Replicate()]
        )
        return pd_loss_dy2st.numpy()

    def test_ep_shared_demo_net(self):
        self.set_seed(self._seed)
        config = Config_shared()
        config.run_ep = True
        model, train_dataloader, criterion, optimizer = self.build(config)
        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims="x"
        )
        ep_loss = self.train(
            config, model, dist_dataloader, criterion, optimizer
        )

        self.set_seed(self._seed)
        config.run_ep = False
        model, train_dataloader, criterion, optimizer = self.build(config)
        replicate_loss = self.train(
            config, model, train_dataloader, criterion, optimizer
        )
        np.testing.assert_allclose(ep_loss, replicate_loss, rtol=1e-6)

    def test_ep_demo_net(self):
        self.set_seed(self._seed)
        config = Config()
        config.run_ep = False

        # train without expert parallel
        model, train_dataloader, criterion, optimizer = self.build(config)
        replicate_loss = self.train(
            config, model, train_dataloader, criterion, optimizer
        )

        # train with expert parallel
        self.set_seed(self._seed)
        config.run_ep = True
        model, train_dataloader, criterion, optimizer = self.build(config)
        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims=0
        )
        ep_loss = self.train(
            config, model, dist_dataloader, criterion, optimizer
        )
        np.testing.assert_allclose(ep_loss, replicate_loss, rtol=1e-6)

        # expert parallel dy2st
        self.set_seed(self._seed)
        model, train_dataloader, criterion, optimizer = self.build(config)
        dist_dataloader = dist.shard_dataloader(
            train_dataloader, config.mesh, shard_dims=0
        )
        dy2st_loss = self.train_dy2st(
            config, model, dist_dataloader, criterion, optimizer
        )
        np.testing.assert_equal(ep_loss, dy2st_loss)

    def test_all2all_demo(self):
        self.set_seed(self._seed)
        config = Config()
        config.layer_type = "DemoAll2AllLayer"
        config.mesh = dist.ProcessMesh([[0], [1]])
        config.expert_mesh_list[0] = dist.ProcessMesh([[0]])
        config.expert_mesh_list[1] = dist.ProcessMesh([[1]])
        config.run_ep = True

        model, train_dataloader, criterion, optimizer = self.build(config)
        dy_loss = self.train(
            config, model, train_dataloader, criterion, optimizer
        )

        self.set_seed(self._seed)
        model, train_dataloader, criterion, optimizer = self.build(config)
        dy2st_loss = self.train_dy2st(
            config, model, train_dataloader, criterion, optimizer
        )

        np.testing.assert_equal(dy_loss, dy2st_loss)

    def run_test_case(self):
        self.test_ep_demo_net()
        self.test_ep_shared_demo_net()
        self.test_all2all_demo()


if __name__ == "__main__":
    TestSimpleNetForEP().run_test_case()

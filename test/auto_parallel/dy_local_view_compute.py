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

import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet, get_rank
from paddle.distributed.auto_parallel.api import (
    dtensor_from_local,
    dtensor_to_local,
)
from paddle.distributed.auto_parallel.local_layer import LocalLayer
from paddle.io import DataLoader, DistributedBatchSampler

base_lr = 0.01  # Learning rate
l2_decay = 1e-4  # Weight decay
clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

epoch = 5  # Number of training epochs
batch_num = 100  # Number of batches per epoch
batch_size = 32  # Batch size for training
class_dim = 10


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([256]).astype('float32')
        label = np.random.randint(0, class_dim - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleNet(paddle.nn.Layer):
    def __init__(self, input_size, inner_size, output_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(input_size, inner_size)
        self.linear2 = paddle.nn.Linear(inner_size, input_size)
        self.linear3 = paddle.nn.Linear(input_size, output_size)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.relu(x)
        return x


def masked_lm_loss_func(pred, label, lossmask):
    pred_sub = pred[:, 0:1]  # shape [B,1]
    label_float = paddle.cast(label, 'float32')  # shape [B,1]

    raw_loss = paddle.abs(pred_sub - label_float)

    lossmask_ = lossmask.reshape([-1]).cast('float32')
    raw_loss_flat = raw_loss.reshape([-1]).cast('float32')

    masked_lm_loss_sum = paddle.sum(raw_loss_flat * lossmask_)
    valid_count = paddle.sum(lossmask_)

    loss = masked_lm_loss_sum / (valid_count + 1e-8)
    return loss


class LocalViewMaskLoss(LocalLayer):
    def __init__(self, out_dist_attrs):
        super().__init__(out_dist_attrs)
        self.local_loss = None

    def forward(self, pred, label, lossmask):
        loss = masked_lm_loss_func(pred, label, lossmask)
        self.local_loss = loss
        return loss


class TestLocalViewCompute:
    def __init__(self):
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def set_random_seed(self):
        np.random.seed(2025)
        paddle.seed(2025)
        random.seed(2025)

    def run_test_cases(self):
        self.run_dy_hand()

    def run_dy_hand(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.hybrid_configs = {
            "dp_degree": 2,
            "mp_degree": 1,
            "pp_degree": 1,
        }

        fleet.init(is_collective=True, strategy=dist_strategy)
        model = SimpleNet(
            input_size=256, inner_size=102400, output_size=class_dim
        )
        optimizer = paddle.optimizer.AdamW(
            learning_rate=base_lr,
            weight_decay=l2_decay,
            parameters=model.parameters(),
            grad_clip=clip,
        )

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        dataset = RandomDataset(batch_num * batch_size)
        sampler = DistributedBatchSampler(
            dataset,
            rank=get_rank(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=1)

        model.train()
        loss_list = []
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            out = model(img)

            lossmask = (label != -100).astype('float32')
            avg_loss = masked_lm_loss_func(out, label, lossmask)

            avg_loss.backward()
            optimizer.step()
            model.clear_gradients()

            loss_list.append(avg_loss.numpy())
        return loss_list

    def run_dy_semi_auto(self):
        pass

    def local_view_compute(self, local_pred, local_label):
        # do not use dist.shard_tensor here
        local_pred = local_pred + 1
        local_loss = masked_lm_loss_func(
            local_pred, local_label, ignored_idx=-100
        )

        return local_loss

    def test_local_view_compute(self):
        dist.init_parallel_env()
        cur_rank = dist.get_rank()

        # prepare data and label for mask_lm_loss
        if cur_rank == 0:
            pred = paddle.to_tensor([[1.0, 2.0], [4.0, 4.0]], dtype='float32')
            label = paddle.to_tensor([[1], [3]], dtype='int64')
        elif cur_rank == 1:
            pred = paddle.to_tensor([[2.0, 2.0], [7.0, 8.0]], dtype='float32')
            label = paddle.to_tensor([[2], [-100]], dtype='int64')

        local_result = self.local_view_compute(pred.clone(), label.clone())

        dist_pred = dist.shard_tensor(pred, self._mesh, [dist.Replicate()])
        dist_label = dist.shard_tensor(label, self._mesh, [dist.Replicate()])

        local_pred = dtensor_to_local(dist_pred)
        local_label = dtensor_to_local(dist_label)

        local_pred = local_pred + 1
        local_loss = masked_lm_loss_func(
            local_pred, local_label, ignored_idx=-100
        )

        assert local_result == local_loss, "local_result != local_loss"

        tensor_list = []
        dist.all_gather(tensor_list, local_loss)
        loss_sum = paddle.sum(paddle.stack(tensor_list))
        dist_loss = dtensor_from_local(
            local_loss, self._mesh, [dist.Partial(dist.ReduceType.kRedSum)]
        )

        assert loss_sum == dist_loss, "loss_sum != dist_loss"


if __name__ == '__main__':
    TestLocalViewCompute().run_test_cases()

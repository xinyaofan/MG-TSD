import time
from typing import Optional, Union

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from gluonts.transform import Transformation
from gluonts.dataset.common import Dataset
import wandb


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device
        self.total_step = 0
        self.log_metrics = kwargs.get('log_metrics')
        print(f'self.log_metrics: {self.log_metrics}')

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
        validation_dataset: Optional[Dataset] = None,
        transformation: Transformation = None,
        estimator=None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        for epoch_no in range(self.epochs):
            self.total_step += 1
            if self.log_metrics == True:
                wandb.log({"train/epoch": epoch_no}, step=self.total_step)
            # mark epoch start time
            tic = time.time()
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            # training loop
            net.train()
            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()

                    inputs = [v.to(self.device) for v in data_entry.values()]
                    output = net(*inputs)

                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )
                    if self.log_metrics == True:
                        wandb.log({'train/loss': avg_epoch_loss},
                                  step=self.total_step)
                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(
                            net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break
                it.close()

            # validation loop
            if validation_iter is not None:
                net.eval()
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total=total, colour="green") as it:

                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [v.to(self.device)
                                  for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )
                        if self.log_metrics == True:
                            wandb.log({'val/loss': avg_epoch_loss_val},
                                      step=self.total_step)
                        if self.num_batches_per_epoch == batch_no:
                            break
                it.close()
                if self.log_metrics == True:
                    print('log_metrics')
                    res = estimator.get_metric(
                        transformation, net, device, validation_dataset, prefix="val/")
                    wandb.log(res, step=self.total_step)

            # mark epoch end time and log time cost of current epoch
            toc = time.time()

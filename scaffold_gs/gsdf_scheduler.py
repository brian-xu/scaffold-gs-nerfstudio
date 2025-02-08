# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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


"""Scheduler Classes"""

from dataclasses import dataclass, field
from typing import Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


@dataclass
class DelayedCosineDecaySchedulerConfig(SchedulerConfig):
    """Config for cosine decay schedule"""

    _target: Type = field(default_factory=lambda: DelayedCosineDecayScheduler)
    """target class to instantiate"""
    pretrain_steps: int = 15000
    """Iteration number where pretrain ends"""
    warm_up_end: int = 5000
    """Iteration number where warmup ends"""
    learning_rate_alpha: float = 0.05
    """Learning rate alpha value"""
    max_steps: int = 300000
    """The maximum number of steps."""


class DelayedCosineDecayScheduler(Scheduler):
    """Cosine decay scheduler with linear warmup"""

    config: DelayedCosineDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        def func(step):
            if step < self.config.pretrain_steps:
                return 0
            step -= self.config.pretrain_steps
            if step < self.config.warm_up_end:
                learning_factor = step / self.config.warm_up_end
            else:
                alpha = self.config.learning_rate_alpha
                progress = (step - self.config.warm_up_end) / (
                    self.config.max_steps - self.config.warm_up_end
                )
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                    1 - alpha
                ) + alpha
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler

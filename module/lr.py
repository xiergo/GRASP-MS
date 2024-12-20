# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""learning rate"""

import math
import numpy as np


def cos_decay_lr(start_step, lr_init, lr_min, lr_max, decay_steps, warmup_steps, total_steps):
    """cosine decay learning rate"""
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        elif i < decay_steps:
            lr = lr_min + 0.5 * (lr_max-lr_min) * (1 + math.cos((i - warmup_steps) / (decay_steps - warmup_steps) * math.pi))
        else:
            lr = lr_min
        
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step[start_step:]

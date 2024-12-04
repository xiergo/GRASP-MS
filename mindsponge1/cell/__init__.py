# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""init"""
from .basic import Attention, GlobalAttention
from .msa import MSARowAttentionWithPairBias, MSAColumnAttention, MSAColumnGlobalAttention
from .triangle import TriangleAttention, TriangleMultiplication, OuterProductMean
from .equivariant import InvariantPointAttention
from .transition import Transition
# from .dense import ProcessSBR, AddInterface
from .interface import AddInterface
from .sbr import ProcessSBR

__all__ = ['Attention', 'GlobalAttention', 'MSARowAttentionWithPairBias',
           'MSAColumnAttention', 'MSAColumnGlobalAttention',
           'TriangleAttention', 'TriangleMultiplication', 'OuterProductMean',
           'InvariantPointAttention', 'Transition', 'AddInterface', 'ProcessSBR']

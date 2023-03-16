# SPDX-License-Identifier: Apache-2.0
# Copyright 2020 Blue Cheetah Analog Design Inc.
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

# -*- coding: utf-8 -*-

from typing import Dict, Any, Mapping, List, Tuple

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__vco_cnter_latch(Module):
    """Module for library bag3_sync_sar_adc cell vco_cnter_latch.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'vco_cnter_latch.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length',
            seg_dict='transistor segments dictionary.',
            w_dict='transistor width dictionary.',
            th_dict='transistor threshold dictionary.',
            dum_info='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(dum_info=None)

    def design(self, lch: int, seg_dict: Mapping[str, int], w_dict: Mapping[str, int], th_dict: Mapping[str, str],
               dum_info: List[Tuple]) -> None:

        dev_list = ['nin', 'pin', 'ntail', 'ptail', 'nfb', 'pfb']

        for name in dev_list:
            uname = name.upper()
            w = w_dict[name]
            nf = seg_dict[name]
            intent = th_dict[name]
            if 'tail' in name:
                self.instances[f'X{uname}'].design(l=lch, w=w, nf=nf, intent=intent)
            else:
                self.instances[f'X{uname}_P'].design(l=lch, w=w, nf=nf, intent=intent)
                self.instances[f'X{uname}_N'].design(l=lch, w=w, nf=nf, intent=intent)

        if dum_info:
            self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')
        else:
            self.remove_instance('XDUM')

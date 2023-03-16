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

from typing import Dict, Any, Mapping

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__comp_strongarm_core(Module):
    """Module for library bag3_sync_sar_adc cell comp_strongarm_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'comp_strongarm_core.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length',
            seg_dict='transistor segments dictionary.',
            w_dict='transistor width dictionary.',
            th_dict='transistor threshold dictionary.',
            has_bridge='True to add bridge switch.',
            has_ofst='True to add bridge switch.',
            stack_br='Number of stacks in bridge switch.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(has_bridge=False, has_ofst=False, stack_br=1)

    def design(self, lch: int, seg_dict: Mapping[str, int], w_dict: Mapping[str, int], has_ofst: bool,
               th_dict: Mapping[str, str], has_bridge: bool, stack_br: int) -> None:

        for name in ['in', 'tail', 'nfb', 'pfb', 'swo', 'swm']:
            uname = name.upper()
            w = w_dict[name]
            nf = seg_dict[name]
            intent = th_dict[name]
            if name == 'tail':
                inst_name = 'XTAIL'
                self.instances[inst_name].design(l=lch, w=w, nf=nf, intent=intent, stack=1)
                self.reconnect_instance_terminal(inst_name, 'g', 'clk')
            elif name == 'swo':
                self.instances['XSWOP'].design(l=lch, w=w, nf=nf, intent=intent)
                self.instances['XSWON'].design(l=lch, w=w, nf=nf, intent=intent)
                self.reconnect_instance('XSWOP', [('D', 'outp'), ('G', 'clk'),
                                                  ('S', 'VDD'), ('B', 'VDD')])
                self.reconnect_instance('XSWON', [('D', 'outn'), ('G', 'clk'),
                                                  ('S', 'VDD'), ('B', 'VDD')])
            else:
                self.instances[f'X{uname}P'].design(l=lch, w=w, nf=nf, intent=intent)
                self.instances[f'X{uname}N'].design(l=lch, w=w, nf=nf, intent=intent)

        if has_bridge:
            w = w_dict['br']
            seg = seg_dict['br']
            intent = th_dict['br']
            self.instances['XBRM'].design(lch=lch, w=w, seg=seg, intent=intent, stack=stack_br)
            if stack_br == 1:
                self.reconnect_instance_terminal('XBRM', 'g', 'clk')
            else:
                self.reconnect_instance_terminal('XBRM', f'g<{stack_br - 1}:0>', 'clk')
        else:
            self.remove_instance('XBRM')
        self.remove_instance('XBRO')

        if has_ofst:
            w = w_dict['os']
            nf = seg_dict['os']
            intent = th_dict['os']
            self.instances['XOSP'].design(l=lch, w=w, nf=nf, intent=intent)
            self.instances['XOSN'].design(l=lch, w=w, nf=nf, intent=intent)
        else:
            self.delete_instance('XOSP')
            self.delete_instance('XOSN')
            self.remove_pin('osp')
            self.remove_pin('osn')


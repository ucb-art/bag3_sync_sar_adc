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

from typing import Dict, Any, List, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__cap_unit(Module):
    """Module for library bag3_sync_sar_adc cell cap_unit.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'cap_unit.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            res_minus='Parameters for metal resistor on minus terminal',
            res_plus='Parameters for metal resistor on plus terminal',
            cap='Parameters for momcap schematic value',
            minus_term='Plus term name',
            plus_term='Plus term name',
            m='Number of parallel cap',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            plus_term='plus',
            minus_term='minus',
            cap=None,
            m=1,
        )

    def design(self, res_minus: Dict[str, int], res_plus: Dict[str, int], m: int, plus_term: str, minus_term: str,
               cap: Optional[int]) -> None:
        print(minus_term)
        print(plus_term)
        print(m)
        if res_plus is not None and res_minus is not None:
            self.instances['XRES_MINUS'].design(**res_minus)
            self.instances['XRES_PLUS'].design(**res_plus)

            if m > 1:
                self.rename_instance('XRES_MINUS', f'XRES_MINUS<{m-1}:0>')
                self.rename_instance('XRES_PLUS', f'XRES_PLUS<{m-1}:0>')
                self.reconnect_instance_terminal(f'XRES_MINUS<{m-1}:0>', 'PLUS', minus_term)
                self.reconnect_instance_terminal(f'XRES_MINUS<{m-1}:0>', 'MINUS', minus_term) #assuming same net
                self.reconnect_instance_terminal(f'XRES_PLUS<{m-1}:0>', 'PLUS', plus_term)
                self.reconnect_instance_terminal(f'XRES_PLUS<{m-1}:0>', 'MINUS', plus_term)
                # If it's schematic-only, add cap for simulation
            else:
                self.reconnect_instance_terminal('XRES_MINUS', 'PLUS', minus_term)
                self.reconnect_instance_terminal('XRES_MINUS', 'MINUS', minus_term) #assuming same net
                self.reconnect_instance_terminal('XRES_PLUS', 'PLUS', plus_term)
                self.reconnect_instance_terminal('XRES_PLUS', 'MINUS', plus_term)
        else:
            self.delete_instance('XRES_MINUS')
            self.delete_instance('XRES_PLUS')

            print("metal resistors deleted")

        if cap:
            src_load_list = [dict(type='cap', lib='analogLib', value=cap,
                                  conns=dict(PLUS=plus_term, MINUS=minus_term)) for idx in range(m)]
            self.design_sources_and_loads(src_load_list, default_name='XCAP')
            self.instances['C0'].set_param('lvsignore', 'True')
        else:
            self.delete_instance('XCAP')

        if plus_term != 'plus':
            self.rename_pin('plus', plus_term)
        if minus_term != 'minus':
            self.rename_pin('minus', minus_term)

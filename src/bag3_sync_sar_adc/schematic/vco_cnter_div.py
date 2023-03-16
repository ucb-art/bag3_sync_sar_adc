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

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__vco_cnter_div(Module):
    """Module for library bag3_sync_sar_adc cell vco_cnter_div.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'vco_cnter_div.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            latch_params_list='unit inverter parameters',
            num_stages='number of stage in RO',
            clkbuf='Has clock bufer',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            clkbuf=None
        )

    def design(self, latch_params_list, num_stages, clkbuf):
        if clkbuf:
            self.instances['XCLKBUF'].design(**clkbuf)
            self.rename_instance('XCLKBUF', 'XCLKBUF<1:0>', [('VDD', 'VDD'), ('VSS', 'VSS'),
                                                             ('in', 'clkn,clkp'), ('out', 'clkp_int,clkn_int')])
        else:
            self.remove_instance('XCLKBUF')
        name_list = [f'XL<{idx}>' for idx in range(num_stages)] if num_stages > 1 else 'XL'
        clkp_name = ['clkp_int', 'clkn_int'] * (num_stages // 2) if clkbuf else ['clkp', 'clkn'] * (num_stages // 2)
        clkn_name = ['clkn_int', 'clkp_int'] * (num_stages // 2) if clkbuf else ['clkn', 'clkp'] * (num_stages // 2)

        out_name = [f'outp<{idx}>' for idx in range(num_stages)]
        out_b_name = [f'outn<{idx}>' for idx in range(num_stages)]

        inp_name_shift = [f'outn<{num_stages-1}>']+[f'outp<{idx}>' for idx in range(num_stages-1)]
        inn_name_shift = [f'outp<{num_stages-1}>']+[f'outn<{idx}>' for idx in range(num_stages-1)]
        term_list = [{'outp': out_name[idx], 'outn': out_b_name[idx],
                      'dn': inn_name_shift[idx], 'd': inp_name_shift[idx],
                      'clkn': clkn_name[idx], 'clkp': clkp_name[idx]} for idx in range(num_stages)]

        self.array_instance('XL', name_list, term_list)

        for idx in range(num_stages):
            self.instances[f'XL<{idx}>'].design(**latch_params_list[idx])
        self.rename_pin('out', f'outp<0:{num_stages-1}>')
        self.rename_pin('outn', f'outn<0:{num_stages-1}>')

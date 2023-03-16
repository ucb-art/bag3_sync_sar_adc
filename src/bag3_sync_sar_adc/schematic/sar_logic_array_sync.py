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

from typing import Dict, Any, List
import copy 

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__sar_logic_array_sync(Module):
    """Module for library bag3_sync_sar_adc cell sar_logic_array_sync.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sar_logic_array_sync.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            nbits='Number of bits in SAR',
            buf_list='List of buffer segments',
            buf_clk='Parameters for clk buffer (for retimer)',
            buf_out='Parameters for clk buffer (for output)',
            logic='Parameters for sar logic unit',
            logic_list='Parameters list for sar logic unit',
            flop='Parameters for flop unit',
            has_pmos_sw='True if CDAC has pmos switch, need differential logic',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            logic=None,
            logic_list=[],
            buf_list=[],
            has_pmos_sw=False
        )

    def design(self, nbits: int, buf_list: List[int], logic_list: List[Param], buf_clk: Param, buf_out: Param,
               logic: Param, flop: Param, has_pmos_sw: bool) -> None:
        # Rename pins
        for pname in ['dm', 'dn', 'dp', 'state', 'data_out']:
            self.rename_pin(pname, f"{pname}<{nbits-1}:0>")

        if has_pmos_sw:
            for pname in ['dn_b', 'dp_b']:
                self.rename_pin(pname, f"{pname}<{nbits - 1}:0>")
        else:
            self.remove_pin('dn_b')
            self.remove_pin('dp_b')

        # Design instances
        self.instances['XBUF_CLK'].design(**buf_clk)
        self.instances['XBUF_OUT'].design(**buf_out)
        self.instances["XFLOP_OUT"].design(**flop)

        # Array logic units
        logic_term_list = []
        for idx in range(nbits):
            _name = f'XLOGIC{idx}'
            bit_conn = f"state<{nbits-idx}>" if idx else 'clk_mid_b'
            if has_pmos_sw:
                _term = [('bit', bit_conn), ('bit_nxt', f"state<{nbits - idx - 1}>"),
                         ('dm', f'dm<{nbits - idx - 1}>'), ('dp', f'dp<{nbits - idx - 1}>'),
                         ('dn', f'dn<{nbits - idx - 1}>'), ('dp_b', f'dp_b<{nbits - idx - 1}>'),
                         ('dn_b', f'dn_b<{nbits - idx - 1}>'), ('out_ret', f"out_ret<{nbits - idx - 1}>")]
            else:
                _term = [('bit', bit_conn), ('bit_nxt', f"state<{nbits-idx-1}>"),
                         ('dm', f'dm<{nbits-idx-1}>'), ('dp', f'dp<{nbits-idx-1}>'), ('dn', f'dn<{nbits-idx-1}>'),
                         ('out_ret', f"out_ret<{nbits-idx-1}>")]
            logic_term_list.append((_name, _term))

        self.array_instance('XLOGIC', inst_term_list=logic_term_list, dx=2*self.instances['XLOGIC'].width)

        if logic_list:
            for idx, _params in enumerate(logic_list):
                self.instances[f'XLOGIC{idx}'].design(**_params)
        else:
            logic_unit_params = logic.to_dict()
            for idx in range(nbits):
                _params = copy.deepcopy(logic_unit_params)
                _params.update(buf_seg=buf_list[idx])
                self.instances[f'XLOGIC{idx}'].design(**_params)

        # Array retimer units
        retimer_conn = [('in', f"out_ret<{nbits-1}:0>"), ('out', f"data_out<{nbits-1}:0>"), ('outb', f"outn<{nbits-1}:0>")]
        self.rename_instance('XFLOP_OUT', f"XFLOP_OUT<{nbits-1}:0>", retimer_conn)
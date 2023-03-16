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
class bag3_sync_sar_adc__sar_slice_bot(Module):
    """Module for library bag3_sync_sar_adc cell sar_slice_bot.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sar_slice_bot.yaml')))

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
            nbits='Number of bits',
            comp='Parameters of comparator',
            logic='Parameters of sar logic block',
            cdac='Parameters of cdac',
            ideal_switch='True to put ideal switch in front of SAR for sch simulation',
            tri_sa='True to enable tri-tail comparator',
            has_pmos_sw='True to have pmos switch in cdac',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ideal_switch=True,
            tri_sa=False,
            has_pmos_sw=False
        )

    def design(self, nbits: int, comp: Param, logic: Param, cdac: Param, ideal_switch: bool, has_pmos_sw: bool,
               tri_sa: bool) -> None:

        for pname in ['dm', 'dn', 'dp', 'data_out']:
            self.rename_pin(pname, f"{pname}<{nbits - 1}:0>")
        for pname in ['bot_p', 'bot_n']:
            self.rename_pin(pname, f"{pname}<{nbits - 2}:0>")
        self.rename_pin('vref', 'vref<2:0>')
        self.remove_pin('clk_sel')

        if tri_sa:
            self.replace_instance_master('XCOMP', 'bag3_sync_sar_adc', 'strongarm_tri')
            comp_conn = [('VDD', 'VDD'), ('VSS', 'VSS'),
                         ('inp', 'top_p'), ('inn', 'top_n'), ('osn', 'osn'), ('osp', 'osp'),
                         ('outp', 'comp_p'), ('outn', 'comp_n'), ('outp_m', 'comp_p_m'), ('outn_m', 'comp_n_m'),
                         ('clk', 'comp_clk'), ('clkb', 'comp_clkb'), ]
            self.instances['XCOMP'].design(**comp)
            for con_pair in comp_conn:
                self.reconnect_instance_terminal('XCOMP', con_pair[0], con_pair[1])
        else:
            self.instances['XCOMP'].design(**comp)
        self.instances['XLOGIC'].design(**logic)
        [self.instances[inst].design(**cdac) for inst in ['XDACN', 'XDACP']]

        logic_conn = [(f"state<{nbits - 1}:0>", f"state<{nbits - 1}:0>"),
                      (f"data_out<{nbits - 1}:0>", f"data_out<{nbits - 1}:0>"),
                      (f"dm<{nbits - 1}:0>", f"dm<{nbits - 1}:0>"),
                      (f"dn<{nbits - 1}:0>", f"dn<{nbits - 1}:0>"),
                      (f"dp<{nbits - 1}:0>", f"dp<{nbits - 1}:0>"), ]
        if has_pmos_sw:
            logic_conn.extend([(f"dn_b<{nbits - 1}:0>", f"dn_b<{nbits - 1}:0>"),
                               (f"dp_b<{nbits - 1}:0>", f"dp_b<{nbits - 1}:0>"), ])
        self.instances['XLOGIC'].design(**logic)
        for con_pair in logic_conn:
            self.reconnect_instance_terminal('XLOGIC', con_pair[0], con_pair[1])

        dac_conn_p = [(f"vref<2:0>", f"vref<2:0>"),
                      (f"ctrl_m<{nbits - 2}:0>", f"dm<{nbits - 1}:1>"),
                      (f"ctrl_p<{nbits - 2}:0>", f"dp<{nbits - 1}:1>"),
                      (f"ctrl_n<{nbits - 2}:0>", f"dn_b<{nbits - 1}:1>" if has_pmos_sw else f"dn<{nbits - 1}:1>"),
                      (f"bot<{nbits - 2}:0>", f"bot_p<{nbits - 2}:0>"), ]

        dac_conn_n = [(f"vref<2:0>", f"vref<2:0>"),
                      (f"ctrl_m<{nbits - 2}:0>", f"dm<{nbits - 1}:1>"),
                      (f"ctrl_p<{nbits - 2}:0>", f"dn<{nbits - 1}:1>"),
                      (f"ctrl_n<{nbits - 2}:0>", f"dp_b<{nbits - 1}:1>" if has_pmos_sw else f"dp<{nbits - 1}:1>"),
                      (f"bot<{nbits - 2}:0>", f"bot_n<{nbits - 2}:0>"), ]

        for con_pair in dac_conn_n:
            self.reconnect_instance_terminal('XDACN', con_pair[0], con_pair[1])
        for con_pair in dac_conn_p:
            self.reconnect_instance_terminal('XDACP', con_pair[0], con_pair[1])

        self.reconnect_instance_terminal('XDACN', 'sam', 'clk_e')
        self.reconnect_instance_terminal('XDACP', 'sam', 'clk_e')

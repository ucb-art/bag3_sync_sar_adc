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

from typing import Dict, Any, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__sar_logic_unit_bot_sync(Module):
    """Module for library bag3_sync_sar_adc cell sar_logic_unit_bot_sync.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sar_logic_unit_bot_sync.yaml')))

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
            oai='Parameters for oai gate',
            oai_fb='Parameters for oai output middle inv',
            buf='Parameters for output buffers template',
            buf_np='Parameters for output buffers template',
            buf_seg='Segment for buffer',
            buf_ratio='Buffer chain ratio',
            nor='Parameters for nor gate',
            rflop='Parameters for reset flop',
            flop = 'Parameters for output flop',
            nand_done='Nand gate for done signal',
            nand_state='Nand gate for state signal',
            inv_done='Inverter for done signal',
            inv_clk='Inverter for clk signal',
            pg='Passgate parameters, used when has_pmos_sw = True',
            has_pmos_sw='True if CDAC has pmos switch, need differential logic',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            buf_ratio=2,
            buf_seg=-1,
            has_pmos_sw=False,
            pg=None,
        )

    def design(self, nand_done: Param, nand_state: Param, inv_done: Param, inv_clk: Param,
                oai: Param, oai_fb: Param, buf: Param, buf_np: Param, 
                nor: Param, rflop: Param, flop: Param, pg: Optional[Param],
               has_pmos_sw: bool, buf_seg: int, buf_ratio: int) -> None:

        for gate_type in ['N', 'P']:
            self.instances[f"XOAI_{gate_type}"].design(**oai)
            self.instances[f"XINV_{gate_type}_MID"].design(**oai_fb)

        # if buf_seg > 0:
        #     buf_params = buf.to_dict()
        #     buf_m = copy.deepcopy(buf_params)
        #     buf_m.update(seg=buf_seg)
        #     buf_out = copy.deepcopy(buf_params)
        #     buf_out.update(seg=buf_ratio*buf_seg)
        #     buf_chain = dict(
        #         dual_output=False,
        #         inv_params=[
        #             buf_m,
        #             buf_out
        #         ]
        #     )
        # else:
        #     buf_chain = buf

        self.instances['XRFLOP'].design(**rflop)
        self.instances['XFLOP'].design(**flop)
        self.instances['XBUF_M'].design(**buf)
        self.instances['XBUF_P'].design(**buf_np)
        self.instances['XBUF_N'].design(**buf_np)

        self.instances['XINV_DONE'].design(**inv_done)
        self.instances['XINV_CLK'].design(**inv_clk)

        self.reconnect_instance_terminal('XNOR', 'in<2:0>', 'rst,dn_mid,dp_mid')
        self.instances['XNOR'].design(**nor)
        self.instances['XNAND_DONE'].design(**nand_done)
        self.instances['XNAND_STATE'].design(**nand_state)
        if has_pmos_sw:
            self.reconnect_instance_terminal('XBUF_N', 'out', 'dn_m')
            self.reconnect_instance_terminal('XBUF_N', 'outb', 'dn_b')
            self.reconnect_instance_terminal('XBUF_P', 'out', 'dp_m')
            self.reconnect_instance_terminal('XBUF_P', 'outb', 'dp_b')
            self.reconnect_instance_terminal('XFLOP', 'in', 'dp_m')
            self.instances['XPG_P'].design(**pg)
            self.instances['XPG_N'].design(**pg)
        else:
            self.delete_instance('XPG_N')
            self.delete_instance('XPG_P')
            self.remove_pin('dn_b')
            self.remove_pin('dp_b')

# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# -*- coding: utf-8 -*-

from typing import Mapping, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__sar_sync_counter(Module):
    """Module for library bag3_sync_sar_adc cell sar_sync_counter.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sar_sync_counter.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Mapping[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            inv_div='Parameters for inv in divider flip flop',
            flop_div='Parameters for flop divider',
            nand='Parameters for reset nand logic',
            rflop='Parameters for reset flip flop',
            nor='Parameters for reset nor',
            buf_out="Parameters for output buf",
            buf_in="Parameters for input buf",
            buf_comp_clk="Parameters for clk buf",
            total_cycles="total clock cycles"
        )

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(
            
        )

    def design(self, inv_div: Param, flop_div: Param, nand: Param, rflop: Param,
                 nor: Param, buf_out: Param, buf_in: Param, buf_comp_clk: Param,
                 total_cycles: int) -> None:
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        bin_code = [int(b) for b in bin(total_cycles-2)[2:]]
        max_bit = len(bin_code)
        padding = [0 for i in range(4-len(bin_code))]
        bin_code = padding + bin_code

        self.instances['XBUF_IN'].design(**buf_in)
        self.instances['XBUF_COMPCLK'].design(**buf_comp_clk)
        self.instances['XFLOP_DIV0'].design(**flop_div)
        self.instances['XINV_DIV0'].design(**inv_div)
        self.instances['XFLOP_DIV1'].design(**flop_div)
        self.instances['XINV_DIV1'].design(**inv_div)
        self.instances['XFLOP_DIV2'].design(**flop_div)
        self.instances['XINV_DIV2'].design(**inv_div)
        self.instances['XFLOP_DIV3'].design(**flop_div)
        self.instances['XINV_DIV3'].design(**inv_div)
        self.instances['XBUF_CLKDIV'].design(**buf_out)
        self.reconnect_instance_terminal('XBUF_CLKDIV', 'in', 'rst')
        self.reconnect_instance_terminal('XBUF_CLKDIV', 'out', 'clk_out_b')
        self.reconnect_instance_terminal('XBUF_CLKDIV', 'outb', 'clk_out')

        self.instances['XNOR1'].design(**nor)
        self.instances['XNOR2'].design(**nor)

        str0 = 'xb16' if bin_code[0] else 'x16'
        str1 = 'xb8' if bin_code[1] else 'x8'
        self.reconnect_instance_terminal('XNOR2', 'in<1:0>', str1+','+str0)
        
        str2 = 'xb4' if bin_code[2] else 'x4'
        str3 = 'xb2' if bin_code[3] else 'x2'
        self.reconnect_instance_terminal('XNOR1', 'in<1:0>', str3+','+str2)

        self.instances['XNAND'].design(**nand)
        self.instances['XINV_RST'].design(**inv_div)
        self.instances['XFLOP_RST'].design(**rflop)

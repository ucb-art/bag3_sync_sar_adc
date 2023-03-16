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

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param
from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_sync_sar_adc__sar_slice_wsamp(Module):
    """Module for library bag3_sync_sar_adc cell sar_slice_wsamp.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sar_slice_wsamp.yaml')))

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
            slice_params='',
            sampler_params='',
            sync=False,
            bootstrap=False
        )

    def design(self, sampler_params, slice_params, sync, bootstrap) -> None:
        if sync:
            self.replace_instance_master('XSAR', 'bag3_sync_sar_adc', 
                        'sar_slice_bot_sync', keep_connections=True)
            #self.add_pin('clk16', TermType.output)
        if bootstrap:
            self.replace_instance_master('XSAM', 'bag3_sync_sar_adc', 
                    'sampler_top')
            # self.add_pin('vcm', TermType.input)
            # self.reconnect_instance_terminal('XSAM', 'vcm', 'vref<1>')
            # self.reconnect_instance_terminal('XSAM', 'bot_n_bot', 'top_n')
            # self.reconnect_instance_terminal('XSAM', 'out_p_bot', 'top_p')
        self.instances['XSAM'].design(**sampler_params)
        self.instances['XSAR'].design(**slice_params)
        sar_pins = list(self.instances['XSAR'].master.pins.keys())
        sam_pins = list(self.instances['XSAM'].master.pins.keys())
        sar_conn_list = [(p,p) for p in sar_pins]
        sam_conn_list = [(p,p) for p in sam_pins]
        sam_conn_list_new = []
        for pin, ppin in sam_conn_list:
            if 'out' in pin and not('bot' in pin) :
                ppin=ppin.replace('out', 'bot')
            if 'sam' in pin:
                if sync:
                    ppin=ppin.replace('sam', 'clk16')
                else:
                    ppin=ppin.replace('sam', 'clk')
            sam_conn_list_new.append((pin, ppin))
        self.reconnect_instance('XSAR', sar_conn_list)
        self.reconnect_instance('XSAM', sam_conn_list_new)

        if bootstrap:
            self.reconnect_instance_terminal('XSAM', 'vcm', 'vref<1>')
            self.reconnect_instance_terminal('XSAM', 'out_n_bot', 'top_n')
            self.reconnect_instance_terminal('XSAM', 'out_p_bot', 'top_p')
            self.reconnect_instance_terminal('XSAM', 'sig_n', 'in_n')
            self.reconnect_instance_terminal('XSAM', 'sig_p', 'in_p')
        nbits=slice_params['nbits']
        for pname in ['dm', 'dn', 'dp', 'data_out']:
            self.rename_pin(pname, f"{pname}<{nbits - 1}:0>")
        for pname in ['bot_p', 'bot_n']:
            self.rename_pin(pname, f"{pname}<{nbits - 2}:0>")
        self.rename_pin('vref', 'vref<2:0>')
        self.remove_pin('osn')
        self.remove_pin('osp')
        self.remove_pin('clk_e')
        self.remove_pin('done')
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

import pkg_resources
from pathlib import Path
from typing import Mapping, Any

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.util.immutable import Param


# noinspection PyPep8Naming
from pybag.enum import TermType


class bag3_sync_sar_adc__sampler_top(Module):
    """Module for library bag3_sync_sar_adc cell sampler_top.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sampler_top.yaml')))

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
            cm_sw_params='',
            sig_sampler_params='',
            vcm_sampler_params='',
        )

    def design(self, cm_sw_params, sig_sampler_params, vcm_sampler_params) -> None:

        self.instances['XN'].design(**sig_sampler_params)
        self.instances['XP'].design(**sig_sampler_params)
        if sig_sampler_params['no_sampler']:
            self.remove_pin('out_n')
            self.remove_pin('out_p')
            self.add_pin('vg_n', TermType.inout)
            self.add_pin('vg_p', TermType.inout)
        else:
            nouts = len(sig_sampler_params['dev_info']['XSAMPLE']['nf'])
            if nouts > 1: 
                self.reconnect_instance_terminal('XN', f'out<{nouts - 1}:0>', f'out_n<{nouts - 1}:0>')
                self.reconnect_instance_terminal('XP', f'out<{nouts - 1}:0>', f'out_p<{nouts - 1}:0>') 
                self.rename_pin('outp', f'out_p<{nouts - 1}:0>')
                self.rename_pin('outn', f'out_n<{nouts - 1}:0>')      

        # design common-mode sampler (nmos diff pair)
        self.instances['XCM'].design(**vcm_sampler_params)

        # common mode mid sw
        self.instances['XSW_CM_N'].design(**cm_sw_params['n'])
        self.delete_instance('XSW_CM_P')
        self.reconnect_instance_terminal('XCM', 'vg_n', 'vg_cm')
        self.reconnect_instance_terminal('XCM', 'vg_p', 'vg_cm')
        self.reconnect_instance_terminal('XSW_CM_N', 'g', 'vg_cm')
        self.reconnect_instance_terminal('XCM', 'sample', 'sam')
        self.reconnect_instance_terminal('XCM', 'sample_b', 'sam_b')

        self.remove_pin('sam_e')
        self.remove_pin('sam_e_b')
        

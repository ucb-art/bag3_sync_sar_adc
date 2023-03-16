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
class bag3_sync_sar_adc__vco_cnter_async(Module):
    """Module for library bag3_sync_sar_adc cell vco_cnter_async.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'vco_cnter_async.yaml')))

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
            div_params_list='list of divider parameters',
            ndivs='number of stage',
            nbits='number of bits in div'
        )

    def design(self, div_params_list, nbits, ndivs):

        # name_list = [f"XDIV<{num_stage-1}:0>"]
        name_list = [f"XDIV<{idx}>" for idx in range(ndivs)]

        div_stages = 2**nbits
        tot_out = div_stages * ndivs

        out_name = [f"outp<{div_stages*(idx-1)}:{div_stages*idx-1}>" for idx in range(1, ndivs+1)]
        out_b_name = [f"outn<{div_stages*(idx-1)}:{div_stages*idx-1}>" for idx in range(1, ndivs+1)]
        out_term_name = f"outp<0:{div_stages-1}>"
        out_b_term_name = f"outn<0:{div_stages-1}>"
        clk_name = ['clkp']+['outp<%d>' % (idx*div_stages-1) for idx in range(1, ndivs)]
        clk_b_name = ['clkn']+['outn<%d>' % (idx*div_stages-1) for idx in range(1, ndivs)]

        term_list = []
        for idx in range(ndivs):
            term_list.append({out_term_name: out_name[idx], out_b_term_name: out_b_name[idx],
                              'clkp': clk_name[idx], 'clkn': clk_b_name[idx]})

        self.array_instance('XDIV', name_list, term_list)
        for idx in range(ndivs):
            self.instances[f'XDIV<{idx}>'].design(**div_params_list[idx])
        self.rename_pin('out', f"outp<{tot_out-1}:0>")
        self.rename_pin('outn', f"outn<{tot_out-1}:0>")

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

from copy import deepcopy
from typing import Dict, List, Tuple, Any, Mapping

import pkg_resources
import warnings
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__bootstrap(Module):
    """Module for library bag3_sync_sar_adc cell bootstrap.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'bootstrap.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length of transistors',
            intent='device intent',
            dev_info='devices information including nf, w and stack',
            dum_info='dummy information including nf, w and stack',
            cap_params='capacitor parameters',
            cap_aux_params='capacitor parameters',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dum_info=[],
            cap_aux_params=None,
            cap_params=None,
        )

    def design(self, lch: int, intent: str, dev_info: Mapping[str, Any], dum_info: List[Tuple[Any]],
               cap_params: Mapping[str, Any], cap_aux_params: Mapping[str, Any]) -> None:
        sampler_info = dev_info['XSAMPLE']
        dev_info = dev_info.copy(remove=['XSAMPLE'])
        nout = len(sampler_info['nf'])
        if nout > 1:
            nbits = len(sampler_info['nf'])
            self.rename_pin('out', f'out<{nbits - 1}:0>')
            pname_term_list = []
            [pname_term_list.append((f'XSAMPLE<{idx}>', [('B', 'VSS'), ('S', 'in'), ('D', f'out<{idx}>'),
                                                    ('G', 'vg')])) for idx in range(nbits)]
            self.array_instance('XSAMPLE', inst_term_list=pname_term_list)

            for idx, m in enumerate(sampler_info['nf']):
                self.instances[f'XSAMPLE<{idx}>'].design(l=lch, w=sampler_info['w_n'], 
                                                        nf=sampler_info['seg_n']* m, intent=sampler_info['intent'])    
        else:
            self.instances['XSAMPLE'].design(l=lch, w=sampler_info['w_n'], 
                                                        nf=sampler_info['seg_n'], intent=sampler_info['intent'])    

        
        if 'XSAMPLE_DUM' not in dev_info.keys():
            warnings.warn("Doesn't implement dummy sampling sw")
            self.remove_pin('in_c')
            self.delete_instance('XSAMPLE_DUM')

        # if 'XCAP_P_AUX' not in dev_info.keys():
        #     self.remove_pin('cap_top_aux')
        #     self.delete_instance('XCAP_P_AUX')
        #     self.reconnect_instance_terminal('XCAP_P', 'B', 'cap_top')
        #     self.reconnect_instance_terminal('XON_P', 'B', 'cap_top')
        for key, _info in dev_info.items():
            _stack = _info.get('stack', 1)
            self.instances[key].design(l=lch, intent=intent, w=_info['w_n'], nf=_info['nf'], stack=_stack)
        if dum_info:
            self.design_dummy_transistors(dum_info, 'X_DUMMY', 'VDD', 'VSS')
        else:
            self.delete_instance('X_DUMMY')

        sch_params = deepcopy(cap_params.to_dict())
        sch_params['intent'] = cap_params['mim_type']
        if cap_params is None:
            self.delete_instance('X_CBOOT')
        else:
            self.instances['X_CBOOT'].design(**sch_params)
        # if cap_aux_params is None:
        #     self.delete_instance('X_CAUX')
        # else:
        #     self.instances['X_CAUX'].design(**cap_params)

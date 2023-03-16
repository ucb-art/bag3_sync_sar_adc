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

from typing import Dict, Any, Mapping

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__bootstrap_diff(Module):
    """Module for library bag3_sync_sar_adc cell bootstrap_diff.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'bootstrap_diff.yaml')))

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
            sampler_params='Sampler parameters',
            cdum='True to have xcp mos-cap between output and vg'
        )
    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(cdum=False)

    def design(self, sampler_params: Mapping[str, Any], cdum: bool):
        if sampler_params['break_outputs']:
            nout = sampler_params['dev_info']['XSAMPLE']['nf']//2
            self.rename_pin('out_p', f'out_p<{nout-1}:0>')
            self.rename_pin('out_n', f'out_n<{nout-1}:0>')
            self.reconnect_instance_terminal('XN', f'out<{nout-1}:0>', f'out_n<{nout-1}:0>')
            self.reconnect_instance_terminal('XP', f'out<{nout-1}:0>', f'out_p<{nout-1}:0>')
        self.instances['XN'].design(**sampler_params)
        self.instances['XP'].design(**sampler_params)
        if cdum:
            self.instances['XCDUM_N'].design(l=sampler_params['lch'], intent=sampler_params['intent'],
                                             nf=sampler_params['dev_info']['XSAMPLE']['nf']//4, w=4) #FIXME: get width

            self.instances['XCDUM_P'].design(l=sampler_params['lch'], intent=sampler_params['intent'],
                                             nf=sampler_params['dev_info']['XSAMPLE']['nf']//4, w=4)
        else:
            [self.delete_instance(inst) for inst in ['XCDUM_N', 'XCDUM_P']]


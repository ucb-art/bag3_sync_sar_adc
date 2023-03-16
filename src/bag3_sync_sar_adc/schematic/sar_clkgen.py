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
class bag3_sync_sar_adc__sar_clkgen(Module):
    """Module for library bag3_sync_sar_adc cell sar_clkgen.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sar_clkgen.yaml')))

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
            pulse='Parameters for pulse generator',
            delay='Parameters for variable delay',
            inv='Parameters for functional pins buffer',
            tri='Parameters for fast_mode tristate',
            nand='Parameters for nand done signal',
        )

    def design(self, pulse: Param, delay: Param, inv: Param, tri: Param, nand: Param) -> None:
        self.instances['XPULSE'].design(**pulse)
        self.instances['XDELAY'].design(**delay)
        self.instances['XTRI'].design(**tri)
        self.instances['XDONE'].design(**nand)
        [self.instances[inst].design(**inv) for inst in ['XINV<2:0>', 'XINV_FAST']]
        self.instances['XDONE_SUM'].design(nin=2)

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
class bag3_sync_sar_adc__cdac_unit(Module):
    """Module for library bag3_sync_sar_adc cell cdac_unit.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'cdac_unit.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            drv='cap switches parameters',
            cap='cap unit parameters',
            m='number of unit cap',
            sw='number of sw'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            m=1,
            sw=-1,
        )

    def design(self, drv: Param, cap: Param, m: int, sw: int) -> None:
        self.instances['XDRV'].design(**drv)
        self.instances['XCAP'].design(**cap)
        if m > 1:
            self.rename_instance('XCAP', f"XCAP<{m-1}:0>")
            m_sw = sw if sw > 0 else m
            self.rename_instance('XDRV', f"XDRV<{m_sw-1}:0>")

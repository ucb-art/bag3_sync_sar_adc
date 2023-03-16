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

from typing import Dict, Any, List, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_sync_sar_adc__cap_unit_dum(Module):
    """Module for library bag3_sync_sar_adc cell cap_unit_dum.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'cap_unit_dum.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cap='Parameters for momcap schematic value',
            minus_term='Plus term name',
            plus_term='Plus term name',
            m='Number of parallel cap',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            plus_term='plus',
            minus_term='minus',
            cap=None,
            m=1,
        )

    def design(self, plus_term: str, minus_term: str, cap: Optional[int], m: int) -> None:
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
        if cap:
            src_load_list = [dict(type='cap', lib='analogLib', value=cap,
                                  conns=dict(PLUS=plus_term, MINUS=minus_term)) for idx in range(m)]
            self.design_sources_and_loads(src_load_list, default_name='XCAP')
            self.instances['C0'].set_param('lvsignore', 'True')
        else:
            self.delete_instance('XCAP')

        if plus_term != 'plus':
            self.rename_pin('plus', plus_term)
        if minus_term != 'minus':
            self.rename_pin('minus', minus_term)



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
from typing import Any, Dict, List

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.util.immutable import Param
# noinspection PyPep8Naming
from bag3_liberty.enum import TermType


class bag3_sync_sar_adc__cdac_cap(Module):
    """Module for library bag3_sync_sar_adc cell cdac_cap.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'cdac_cap.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)
        self._has_bot = False

    def export_bot(self):
        return self._has_bot

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            cm_unit_params='Unit common-mode capacitor',
            unit_params_list='Parameters of unit capacitor + drv',
            bot_probe='True to export cap unit bottom plate',
            cm='Number of common-mode cap',
            cap_m_list='Number of capacitor',
            remove_cap='True to remove capacitor, use it when doesnt have rmetal',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            bot_probe=True,
            remove_cap=False,
        )

    def design(self, cm: int, cm_unit_params: Param, unit_params_list: List[Param], cap_m_list: List[int],
               bot_probe: bool, remove_cap: bool) -> None:
        remove_cap = self.params['remove_cap']
        nbits = len(unit_params_list)
        # check length of switch params and cap params list:
        cap_term_list = []

        # Remove pins first to avoid bus and scalar name conflict
        self.rename_pin('bot', f"bot<{nbits - 1}:0>")
        self.add_pin('bot_cm', TermType.inout)

        # List inst name and term connection
        for idx, cap_m in enumerate(cap_m_list):
            _cap_name = f'XCAP{idx}<{cap_m - 1}:0>' if cap_m > 1 else f'XCAP{idx}'
            cap_term_list.append((_cap_name, [('top', 'top'), ('bot', f'bot<{idx}>')]))

        # Design sar_sch array
        dx_max = 2 * self.instances['XCAP'].width
        self.array_instance('XCAP', inst_term_list=cap_term_list, dx=dx_max)
        for idx, (name, _) in enumerate(cap_term_list):
            self.instances[name].design(**unit_params_list[idx])
            # if remove_cap:
            #     self.remove_instance(name)

        # Design cm cap
        cm_name = f"<XCAP_CM{cm - 1}:0>" if cm > 1 else f"XCAP_CM"
        self.instances['XCAP_CM'].design(**cm_unit_params)
        if cm > 1:
            self.rename_instance('XCAP_CM', f'XCM{cm - 1:0}')
            self.reconnect_instance_terminal(cm_name, 'bot', 'bot_cm')
            self.reconnect_instance_terminal(cm_name, 'top', 'top')
        #     if remove_cap:
        #         self.remove_instance(f'XCM{cm - 1:0}')
        # elif remove_cap:
        #     self.remove_instance('XCAP_CM')
        else:
            self.reconnect_instance_terminal(cm_name, 'bot', 'bot_cm')
            self.reconnect_instance_terminal(cm_name, 'top', 'top')

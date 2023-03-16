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
class bag3_sync_sar_adc__strongarm(Module):
    """Module for library bag3_sync_sar_adc cell strongarm.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__, str(Path('netlist_info', 'strongarm.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            sa_params='Parameters of strongarm core',
            buf_params='Parameters of inv buffers',
        )

    def design(self, sa_params: Param, buf_params: Param) -> None:
        self.instances['XSA'].design(**sa_params)
        self.instances['XBUF<1:0>'].design(**buf_params)
        len_buf = len(buf_params['inv_params'])

        # swap output depending on number of stages in buffers
        if not len_buf & 1:
            self.reconnect_instance_terminal('XBUF<1:0>', 'out', 'outn,outp')

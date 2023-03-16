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


# noinspection PyPep8Naming
class bag3_sync_sar_adc__oai(Module):
    """Module for library bag3_sync_sar_adc cell oai.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'oai.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            lch='channel length',
            w_p='pmos width.',
            w_n='nmos width.',
            th_p='pmos threshold flavor.',
            th_n='nmos threshold flavor.',
            seg='segments of transistors',
            seg_pstack0='segments of stack input <3:2>',
            seg_pstack1='segments of stack input <1:0>',
            seg_n0='segments of nmos input <3:2>',
            seg_n1='segments of nmos input <1:0>',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_pstack0=-1,
            seg_pstack1=-1,
            seg_n0=-1,
            seg_n1=-1,
        )

    def design(self, seg: int, seg_pstack0: int, seg_pstack1: int, seg_n0: int, seg_n1: int,
               lch: int, w_p: int, w_n: int, th_p: str, th_n: str) -> None:
        if seg_pstack0 <= 0:
            seg_pstack0 = seg
        if seg_pstack1 <= 0:
            seg_pstack1 = seg
        if seg_n0 <= 0:
            seg_n0 = seg
        if seg_n1 <= 0:
            seg_n1 = seg

        pmos_inst_term_list = [('XP0', [('s', 'VDD'), ('b', 'VDD'), ('g<1:0>', 'in<1:0>'), ('d', 'out')]),
                               ('XP1', [('s', 'VDD'), ('b', 'VDD'), ('g<1:0>', 'in<3:2>'), ('d', 'out')])]
        self.array_instance('XP', inst_term_list=pmos_inst_term_list)
        self.instances['XP0'].design(w=w_p, lch=lch, seg=seg_pstack0, intent=th_n, stack=2)
        self.instances['XP1'].design(w=w_p, lch=lch, seg=seg_pstack1, intent=th_n, stack=2)

        nmos_inst_term_list = [('XN0', [('S', 'nmid'), ('B', 'VSS'), ('G', 'in<3>'), ('D', 'out')]),
                               ('XN1', [('S', 'nmid'), ('B', 'VSS'), ('G', 'in<2>'), ('D', 'out')]),
                               ('XN2', [('S', 'VSS'), ('B', 'VSS'), ('G', 'in<1>'), ('D', 'nmid')]),
                               ('XN3', [('S', 'VSS'), ('B', 'VSS'), ('G', 'in<0>'), ('D', 'nmid')])]
        self.array_instance('XN', inst_term_list=nmos_inst_term_list)
        [self.instances[inst].design(w=w_n, l=lch, nf=seg_n0, intent=th_n) for inst in ['XN0', 'XN1']]
        [self.instances[inst].design(w=w_n, l=lch, nf=seg_n1, intent=th_n) for inst in ['XN2', 'XN3']]




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
class bag3_sync_sar_adc__nmos_sampler_diff(Module):
    """Module for library bag3_sync_sar_adc cell nmos_sampler_diff.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'nmos_sampler_diff.yaml')))

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
            lch='',
            seg_p='',
            seg_n='',
            seg_dum_p='',
            seg_dum_n='',
            th_n='',
            th_p='',
            w_n='',
            w_p='',
            m_list='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg_p=0,
            seg_n=0,
            seg_dum_p=0,
            seg_dum_n=0,
            th_n='',
            th_p='',
            w_n=4,
            w_p=4,
            m_list=[1],
        )

    def design(self, lch, w_n, w_p, th_n, th_p, seg_n, seg_p, seg_dum_n, seg_dum_p, m_list) -> None:
        if len(m_list) == 1:
            m = m_list[0]
            if seg_p:
                self.instances['XPN'].design(l=lch, w=w_p, nf=seg_p, intent=th_p)
                self.instances['XPP'].design(l=lch, w=w_p, nf=seg_p, intent=th_p)
                if seg_dum_p:
                    self.instances['XP_DUM_N'].design(l=lch, w=w_p, nf=seg_dum_p, intent=th_p)
                    self.instances['XP_DUM_P'].design(l=lch, w=w_p, nf=seg_dum_p, intent=th_p)
                else:
                    self.remove_instance('XP_DUM_N')
                    self.remove_instance('XP_DUM_P')
                if m >1:
                    self.rename_instance('XPP', f'XPP<{m-1}:0>')
                    self.rename_instance('XPN', f'XPN<{m-1}:0>')
            else:
                self.remove_instance('XP_DUM_N')
                self.remove_instance('XP_DUM_P')
                self.remove_instance('XPP')
                self.remove_instance('XPN')
                self.remove_pin('sam_b')
                self.remove_pin('VDD')

            if seg_n:
                self.instances['XNN'].design(l=lch, w=w_n, nf=seg_n, intent=th_n)
                self.instances['XNP'].design(l=lch, w=w_n, nf=seg_n, intent=th_n)
                if seg_dum_n:
                    self.instances['XN_DUM_N'].design(l=lch, w=w_n, nf=seg_dum_n, intent=th_n)
                    self.instances['XN_DUM_P'].design(l=lch, w=w_n, nf=seg_dum_n, intent=th_n)
                else:
                    self.remove_instance('XN_DUM_N')
                    self.remove_instance('XN_DUM_P')
                if m>1:
                    self.rename_instance('XNN', f'XNN<{m-1}:0>')
                    self.rename_instance('XNP', f'XNP<{m-1}:0>')
            else:
                self.remove_instance('XN_DUM_N')
                self.remove_instance('XN_DUM_P')
                self.remove_instance('XNN')
                self.remove_instance('XNP')
                self.remove_pin('sam')
                self.remove_pin('VSS')
        else:
            nbits = len(m_list)
            self.rename_pin('out_n', f'out_n<{nbits-1}:0>')
            self.rename_pin('out_p', f'out_p<{nbits-1}:0>')
            if seg_p:
                pname_term_list_n, pname_term_list_p = [], []
                [pname_term_list_n.append((f'XPN<{idx}>', [('B', 'VDD'), ('S', 'in_n'), ('D', f'out_n<{idx}>'),
                                                          ('G', 'sam_b')])) for idx in range(nbits)]
                [pname_term_list_p.append((f'XPP<{idx}>', [('B', 'VDD'), ('S', 'in_p'), ('D', f'out_p<{idx}>'),
                                                           ('G', 'sam_b')])) for idx in range(nbits)]
                self.array_instance('XPN', inst_term_list=pname_term_list_n)
                self.array_instance('XPP', inst_term_list=pname_term_list_p)

                for idx, m in enumerate(m_list):
                    self.instances[f'XPN<{idx}>'].design(l=lch, w=w_p, nf=seg_p*m, intent=th_p)
                    self.instances[f'XPP<{idx}>'].design(l=lch, w=w_p, nf=seg_p*m, intent=th_p)
                if seg_dum_p:
                    pname_term_list_n, pname_term_list_p = [], []
                    [pname_term_list_n.append((f'XP_DUM_N<{idx}>', [('B', 'VDD'), ('S', 'in_n'), ('D', f'out_p<{idx}>'),
                                                               ('G', 'sam_b')])) for idx in range(nbits)]
                    [pname_term_list_p.append((f'XP_DUM_P<{idx}>', [('B', 'VDD'), ('S', 'in_p'), ('D', f'out_n<{idx}>'),
                                                               ('G', 'sam_b')])) for idx in range(nbits)]
                    self.array_instance('XP_DUM_N', inst_term_list=pname_term_list_n)
                    self.array_instance('XP_DUM_P', inst_term_list=pname_term_list_p)
                    for idx, m in enumerate(m_list):
                        self.instances[f'XP_DUM_N<{idx}>'].design(l=lch, w=w_p, nf=seg_dum_p*m, intent=th_p)
                        self.instances[f'XP_DUM_P<{idx}>'].design(l=lch, w=w_p, nf=seg_dum_p*m, intent=th_p)
                else:
                    self.remove_instance('XP_DUM_N')
                    self.remove_instance('XP_DUM_P')
            else:
                self.remove_instance('XPP')
                self.remove_instance('XPN')
                self.remove_instance('XP_DUM_N')
                self.remove_instance('XP_DUM_P')
                self.remove_pin('sam_b')
                self.remove_pin('VDD')

            if seg_n:
                pname_term_list_n, pname_term_list_p = [], []
                [pname_term_list_n.append((f'XNN<{idx}>', [('B', 'VSS'), ('S', 'in_n'), ('D', f'out_n<{idx}>'),
                                                          ('G', 'sam')])) for idx in range(nbits)]
                [pname_term_list_p.append((f'XNP<{idx}>', [('B', 'VSS'), ('S', 'in_p'), ('D', f'out_p<{idx}>'),
                                                           ('G', 'sam')])) for idx in range(nbits)]
                self.array_instance('XNN', inst_term_list=pname_term_list_n)
                self.array_instance('XNP', inst_term_list=pname_term_list_p)

                for idx, m in enumerate(m_list):
                    self.instances[f'XNN<{idx}>'].design(l=lch, w=w_n, nf=seg_n*m, intent=th_n)
                    self.instances[f'XNP<{idx}>'].design(l=lch, w=w_n, nf=seg_n*m, intent=th_n)
                if seg_dum_n:
                    pname_term_list_n, pname_term_list_p = [], []
                    [pname_term_list_n.append((f'XN_DUM_N<{idx}>', [('B', 'VSS'), ('S', 'in_n'), ('D', f'out_p<{idx}>'),
                                                               ('G', 'sam')])) for idx in range(nbits)]
                    [pname_term_list_p.append((f'XN_DUM_P<{idx}>', [('B', 'VSS'), ('S', 'in_p'), ('D', f'out_n<{idx}>'),
                                                               ('G', 'sam')])) for idx in range(nbits)]
                    self.array_instance('XN_DUM_N', inst_term_list=pname_term_list_n)
                    self.array_instance('XN_DUM_P', inst_term_list=pname_term_list_p)
                    for idx, m in enumerate(m_list):
                        self.instances[f'XN_DUM_N<{idx}>'].design(l=lch, w=w_n, nf=seg_dum_n*m, intent=th_n)
                        self.instances[f'XN_DUM_P<{idx}>'].design(l=lch, w=w_n, nf=seg_dum_n*m, intent=th_n)
                else:
                    self.remove_instance('XN_DUM_N')
                    self.remove_instance('XN_DUM_P')
            else:
                self.remove_instance('XNN')
                self.remove_instance('XNP')
                self.remove_instance('XN_DUM_N')
                self.remove_instance('XN_DUM_P')
                self.remove_pin('sam')
                self.remove_pin('VSS')


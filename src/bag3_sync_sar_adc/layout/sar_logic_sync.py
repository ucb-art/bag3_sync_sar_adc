from typing import Any, Dict, Type, Optional, List, Mapping, Union, Tuple
from itertools import chain
import copy

from pybag.enum import MinLenMode, RoundMode, PinMode

from bag.util.math import HalfInt
from bag.util.immutable import Param, ImmutableSortedDict, ImmutableList
from bag.layout.template import TemplateDB
from bag.layout.routing.base import WireArray, TrackID
from bag.design.database import ModuleDB, Module

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from .digital import NAND2Core, InvChainCore, NOR3Core, InvCore, PassGateCore
from .digital import LatchCore, FlopCore
from .digital import get_adj_tid_list


class OAICore(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'oai')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='segments of transistors',
            seg_pstack0='segments of stack input <3:2>',
            seg_pstack1='segments of stack input <1:0>',
            seg_n0='segments of nmos input <3:2>',
            seg_n1='segments of nmos input <1:0>',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row',
            ridx_p='index for pmos row',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            vertical_sup='True to enable vertical supply (mos_conn layer)',
            vertical_out='True to enable vertical output',
            sig_locs='Optional dictionary of user defined signal locations',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1,
            stack_p=1,
            stack_n=1,
            seg=-1,
            seg_pstack0=-1,
            seg_pstack1=-1,
            seg_n0=-1,
            seg_n1=-1,
            vertical_sup=False,
            vertical_out=True,
            sig_locs={},
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                in2=MinLenMode.NONE,
                in3=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
        )

    def draw_layout(self) -> None:
        # setup floorplan
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        w_n: int = self.params['w_n']
        w_p: int = self.params['w_p']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        stack_n: int = self.params['stack_n']
        stack_p: int = self.params['stack_p']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        seg: int = self.params['seg']
        seg_pstack0: int = self.params['seg_pstack0']
        seg_pstack1: int = self.params['seg_pstack1']
        seg_n0: int = self.params['seg_n0']
        seg_n1: int = self.params['seg_n1']

        if seg_pstack0 <= 0:
            seg_pstack0 = seg
        if seg_pstack1 <= 0:
            seg_pstack1 = seg
        if seg_n0 <= 0:
            seg_n0 = seg
        if seg_n1 <= 0:
            seg_n1 = seg
        min_sep = self.min_sep_col
        min_sep += min_sep & 1

        pports0 = self.add_nand2(ridx_p, 0, seg_pstack0, w=w_p, stack=stack_p)
        nports0 = self.add_nand2(ridx_n, 0, seg_n0, w=w_n, stack=stack_n, other=True)
        port1_col = max(min_sep + 2 * seg_pstack0 * stack_p, min_sep + 2 * seg_n0 * stack_n)
        pports1 = self.add_nand2(ridx_p, port1_col, seg_pstack1, w=w_p, stack=stack_p)
        nports1 = self.add_nand2(ridx_n, port1_col, seg_n1, w=w_n, stack=stack_n, other=True)
        self.set_mos_size()
        xr = self.bound_box.xh

        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')

        nin_tid = get_adj_tid_list(self, ridx_n, sig_locs, MOSWireType.G, 'nin', True, tr_w_h)
        pin_tid = get_adj_tid_list(self, ridx_p, sig_locs, MOSWireType.G, 'pin', False, tr_w_h)

        in0 = self.connect_to_tracks(list(chain(nports0.g0, pports0.g0)), nin_tid[0], min_len_mode=mlm.get('in0', None))
        in1 = self.connect_to_tracks(list(chain(nports0.g1, pports0.g1)), nin_tid[1], min_len_mode=mlm.get('in1', None))
        in2 = self.connect_to_tracks(list(chain(nports1.g0, pports1.g0)), pin_tid[0], min_len_mode=mlm.get('in2', None))
        in3 = self.connect_to_tracks(list(chain(nports1.g1, pports1.g1)), pin_tid[1], min_len_mode=mlm.get('in3', None))

        for p, pname in zip([in0, in1, in2, in3], ['in<0>', 'in<1>', 'in<2>', 'in<3>']):
            self.add_pin(pname, p)

        pd_tidx = sig_locs.get('pout', self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig'))
        nd_tidx0 = sig_locs.get('nout', self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=-1))
        nd_tidx1 = sig_locs.get('nout', self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0))
        nd_tid0 = TrackID(hm_layer, nd_tidx0, width=tr_w_h)
        nd_tid1 = TrackID(hm_layer, nd_tidx1, width=tr_w_h)
        pd_tid = TrackID(hm_layer, pd_tidx, width=tr_w_h)

        pout = self.connect_to_tracks([pports0.d, pports1.d], pd_tid, min_len_mode=mlm.get('out', None))
        nout = self.connect_to_tracks([nports1.d], nd_tid0, min_len_mode=mlm.get('out', None))
        self.connect_to_tracks([nports1.s, nports0.d], nd_tid1, min_len_mode=mlm.get('out', None))

        vm_tidx = sig_locs.get('out', self.grid.coord_to_track(vm_layer, pout.middle, mode=RoundMode.GREATER_EQ))

        if vertical_out:
            out = self.connect_to_tracks([pout, nout], TrackID(vm_layer, vm_tidx))
            self.add_pin('pout', pout, hide=True)
            self.add_pin('nout', nout, hide=True)
            self.add_pin('out', out)
        else:
            self.add_pin('pout', pout, label='out:')
            self.add_pin('nout', nout, label='out:')

        if vertical_sup:
            self.add_pin('VDD', list(chain(pports0.s, pports1.s)), connect=True)
            self.add_pin('VSS', list(chain(nports0.s, nports1.s)), connect=True)
        else:
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
            vdd = self.connect_to_tracks([pports0.s, pports1.s], ps_tid, track_lower=0, track_upper=xr)
            vss = self.connect_to_tracks([nports0.s], ns_tid, track_lower=0, track_upper=xr)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        self.sch_params = dict(
            seg_pstack0=seg_pstack0,
            seg_pstack1=seg_pstack1,
            seg_n0=seg_n0,
            seg_n1=seg_n1,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            stack_p=stack_p,
            stack_n=stack_n,
        )


class SARLogicUnit(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_logic_unit_bot_sync')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Number of segments.',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row',
            ridx_p='index for pmos row',
            substrate_row='True to add substrate row',
            has_pmos_sw='True to add differential signal to drive pmos switch in CDAC',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1,
            substrate_row=False,
            has_pmos_sw=False,
        )

    def draw_layout(self) -> None:
        # setup floorplan
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        w_n: int = self.params['w_n']
        w_p: int = self.params['w_p']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        seg_dict: Dict[str, Any] = self.params['seg_dict']
        substrate_row: bool = self.params['substrate_row']
        has_pmos_sw: bool = self.params['has_pmos_sw']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        seg_oai: Dict[str, int] = seg_dict['oai']
        seg_flop: Dict[str, int] = seg_dict['flop']
        seg_buf: ImmutableList = seg_dict['buf']
        seg_inv_fb = seg_dict['oai_fb']
        seg_nor = seg_dict['nor']
        seg_nand_done = seg_dict['nand_done']
        seg_nand_state = seg_dict['nand_state']
        seg_inv_done = seg_dict['inv_done']
        seg_buf_state = seg_dict['buf_state']
        seg_inv_clk = seg_dict['inv_clk']
        min_sep = self.min_sep_col
        min_sep += min_sep & 1

        # compute track locations
        tr_manager = self.tr_manager
        tr_w_vm = tr_manager.get_width(vm_layer, 'sig')
        tr_sp_vm = tr_manager.get_sep(vm_layer, ('sig', 'sig'))
        tr_w_hm = tr_manager.get_width(hm_layer, 'sig')

        ng0_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        ng2_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=0)
        nd1_tidx = self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=1)
        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=1)

        _, d_fb_tidx = tr_manager.place_wires(vm_layer, ['sig'] * 3,
                                              self.arr_info.col_to_track(vm_layer, 1, mode=RoundMode.NEAREST),
                                              align_idx=0)

        oai_params = dict(pinfo=pinfo, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                          vertical_sup=substrate_row, sig_locs={})
        oai_params.update(**seg_oai)
        flop_rst_params = dict(pinfo=pinfo, seg=seg_flop, seg_ck=4, resetable=True, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                            vertical_sup=substrate_row, sig_locs={ 'clk': ng1_tidx, 'nin': ng2_tidx})
        flop_params = dict(pinfo=pinfo, seg=seg_flop, seg_ck=4, resetable=False, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                            vertical_sup=substrate_row, sig_locs={'clk': ng1_tidx, 'nin': pg0_tidx})

        inv_fb_n_params = dict(pinfo=pinfo, seg=seg_inv_fb, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                               vertical_sup=substrate_row, sig_locs={'nin': ng2_tidx, 'out': d_fb_tidx[0]})
        inv_fb_p_params = dict(pinfo=pinfo, seg=seg_inv_fb, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                               vertical_sup=substrate_row, sig_locs={'nin': ng2_tidx, 'out': d_fb_tidx[1]})
        if has_pmos_sw:
            seg_buf = seg_buf if len(seg_buf) & 1 else seg_buf.to_list() + [seg_buf[-1]]
            buf_params = dict(pinfo=pinfo, seg_list=seg_buf[:-1], w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                              vertical_sup=substrate_row, sig_locs={'nin0': ng2_tidx, 'nin1': ng1_tidx},
                              vertical_out=False)
            buf_np_params = dict(pinfo=pinfo, seg_list=seg_buf, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                                 vertical_sup=substrate_row,
                                 sig_locs={'nin0': ng2_tidx, 'nin1': ng1_tidx, 'nout0': nd0_tidx, 'pout0': pd1_tidx,
                                           'nout1': nd1_tidx, 'pout1': pd0_tidx},
                                 vertical_out=False, dual_output=True)
            pg_params = dict(pinfo=pinfo, seg=max(seg_buf[-1] - self.min_sep_col+2, 2), #hacky
                             w_p=w_p, wn=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                             vertical_sup=substrate_row, vertical_in=False, vertical_out=False,
                             sig_locs={'nd': nd0_tidx, 'pd': pd1_tidx}, is_guarded=True)
            pg_master = self.new_template(PassGateCore, params=pg_params)
        else:
            # check buf seg length and make it even stage
            seg_buf = seg_buf[:-1] if len(seg_buf) & 1 else seg_buf
            buf_params = dict(pinfo=pinfo, seg_list=seg_buf, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                              vertical_sup=substrate_row, sig_locs={'nin0': ng2_tidx, 'nin1': ng1_tidx},
                              vertical_out=False)
            buf_np_params = buf_params
            pg_master = None

        nand_done_params = dict(pinfo=pinfo, seg=seg_nand_done, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                                vertical_sup=substrate_row, sig_locs={'out': d_fb_tidx[2]})
        nand_state_params = dict(pinfo=pinfo, seg=seg_nand_state, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                                 vertical_sup=substrate_row, vertical_out=False,
                                 sig_locs={'nin1': pg1_tidx, 'nin0': ng1_tidx})
        inv_done_params = dict(pinfo=pinfo, seg=seg_inv_done, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                               vertical_sup=substrate_row, sig_locs={'nin': pg0_tidx})
        inv_clk_params = dict(pinfo=pinfo, seg=seg_inv_clk, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                                vertical_sup=substrate_row, sig_locs={'nin': pg1_tidx})
        oai_master = self.new_template(OAICore, params=oai_params)
        inv_fb_n_master = self.new_template(InvCore, params=inv_fb_n_params)
        inv_fb_p_master = self.new_template(InvCore, params=inv_fb_p_params)
        inv_done_master = self.new_template(InvCore, params=inv_done_params)
        nand_done_master = self.new_template(NAND2Core, params=nand_done_params)
        nand_state_master = self.new_template(NAND2Core, params=nand_state_params)
        flop_rst_master = self.new_template(FlopCore, params=flop_rst_params)
        flop_master = self.new_template(FlopCore, params=flop_params)
        inv_clk_master = self.new_template(InvCore, params=inv_clk_params)

        buf_master = self.new_template(InvChainCore, params=buf_params)
        buf_np_master = self.new_template(InvChainCore, params=buf_np_params)

        # Row 0 - bit and retimer
        cur_col = 0
        nand_done = self.add_tile(nand_done_master, 0, cur_col)
        cur_col += nand_done_master.num_cols + min_sep
        cur_col += cur_col & 1
        inv_done = self.add_tile(inv_done_master, 0, cur_col)
        cur_col += inv_done_master.num_cols + min_sep
        cur_col += cur_col & 1
        nand_state = self.add_tile(nand_state_master, 0, cur_col)
        cur_col += nand_state_master.num_cols + min_sep
        cur_col += cur_col & 1
        row0_ncol = cur_col

        # Row 2,3 - dn/p signal
        cur_col = 0
        oai_inv_n = self.add_tile(inv_fb_n_master, 1, cur_col)
        oai_inv_p = self.add_tile(inv_fb_p_master, 2, cur_col)
        cur_col += inv_fb_n_master.num_cols + min_sep
        cur_col += cur_col & 1
        oai_n = self.add_tile(oai_master, 1, cur_col)
        oai_p = self.add_tile(oai_master, 2, cur_col)
        cur_col += oai_master.num_cols + min_sep
        cur_col += cur_col & 1
        buf_n = self.add_tile(buf_np_master, 1, cur_col)
        buf_p = self.add_tile(buf_np_master, 2, cur_col)
        buf_m = self.add_tile(buf_master, 3, cur_col)
        row23_ncol = cur_col + buf_np_master.num_cols
        row23_ncol += row23_ncol & 1
        if has_pmos_sw:
            pg_n = self.add_tile(pg_master, 1, row23_ncol + min_sep)
            pg_p = self.add_tile(pg_master, 2, row23_ncol + min_sep)
            row23_ncol += pg_master.num_cols + 2*min_sep
            row23_ncol += row23_ncol &1
        else:
            pg_n, pg_p = None, None
            if (row23_ncol & 1):
                row23_ncol += 1
            else: 
                row23_ncol += 2
        flop_rst = self.add_tile(flop_rst_master, 1, row23_ncol)
        flop = self.add_tile(flop_master, 2, row23_ncol)
        clk_col = row23_ncol + min_sep +1
        inv_clk = self.add_tile(inv_clk_master, 0, clk_col)
        row23_ncol += max(flop_rst_master.num_cols, flop_master.num_cols)+min_sep
        oai_out_tidx = oai_n.get_pin('out').track_id.base_index

        # Row 1 - dm signal
        cur_col = 1
        nor_params = dict(pinfo=pinfo, seg=seg_nor,
                          w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p, vertical_sup=substrate_row,
                          sig_locs={'nin0': ng0_tidx, 'nin1': ng1_tidx, 'out': oai_out_tidx})
        nor3_master = self.new_template(NOR3Core, params=nor_params)
        nor = self.add_tile(nor3_master, 3, cur_col)
        tot_seg = max(row0_ncol, row23_ncol)
        self.set_mos_size(tot_seg)

        # Route transistor parts


        # Connection for bit logic
        self.connect_to_track_wires(oai_inv_p.get_pin('out'), nand_done.get_pin('nin<1>'))
        self.connect_to_track_wires(oai_inv_n.get_pin('out'), nand_done.get_pin('nin<0>'))
        self.connect_to_track_wires(nand_done.get_pin('out'), inv_done.get_pin('nin'))

        # done wire
        self.connect_to_track_wires(nand_state.get_pin('nin<0>'), inv_done.get_pin('out'))


        # connect bit input
        tidx_lobit = self.grid.coord_to_track(vm_layer, flop_rst.bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hibit = self.grid.coord_to_track(vm_layer, flop_rst.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_listbit = self.get_available_tracks(vm_layer, tidx_lobit, tidx_hibit,
                                                0, flop.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        nin = self.connect_to_tracks([flop_rst.get_pin('nin')], TrackID(vm_layer,  tidx_listbit[0], tr_w_vm))
        bit = self.connect_to_track_wires(nand_state.get_pin('nin<1>'), nin)
        _wrte_ret_tidx = self.grid.coord_to_track(vm_layer, nand_state.get_pin('nout').upper, mode=RoundMode.NEAREST)
        _wrte_ret = self.connect_to_tracks([nand_state.get_pin('nout'), nand_state.get_pin('pout')],
                                           TrackID(vm_layer, _wrte_ret_tidx, tr_w_vm))

        # Connection for clock 
        _clk_wire_idx = self.grid.coord_to_track(vm_layer, flop.get_pin('clk').middle, mode=RoundMode.NEAREST)
        self.extend_wires(inv_clk.get_pin('out'), upper=flop.get_pin('clk').upper)
        
        # rst connections 
        # for vertical oai connections
        tidx_lo = self.grid.coord_to_track(vm_layer, oai_p.bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hi = self.grid.coord_to_track(vm_layer, oai_p.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_list = self.get_available_tracks(vm_layer, tidx_lo, tidx_hi,
                                              0, self.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)

        rst_vm = self.connect_to_tracks([oai_n.get_pin('in<2>'), oai_p.get_pin('in<2>'), nor.get_pin('nin<2>')],
                                        TrackID(vm_layer, tidx_list[-1], tr_w_vm)) #, flop_rst.get_pin('rst')
        _write_vm = self.connect_to_track_wires([oai_n.get_pin('in<0>'), oai_p.get_pin('in<0>')], _wrte_ret)
        # had to connect over to the rst (make another vm connection, connect with above on hm layer)
        tidx_lorst = self.grid.coord_to_track(vm_layer, flop_rst.bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hirst = self.grid.coord_to_track(vm_layer, flop_rst.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_listrst = self.get_available_tracks(vm_layer, tidx_lorst, tidx_hirst,
                                                0, flop.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        rst_vm2 = self.connect_to_tracks([flop_rst.get_pin('rst')],
                                         TrackID(vm_layer, tidx_listrst[len(tidx_listrst)//2], tr_w_vm))
        rst_hm = self.connect_to_track_wires(rst_vm2, nor.get_pin('nin<2>'))

        # Connection between dn/dp
        # -- feedback inv --
        self.connect_to_track_wires(oai_n.get_pin('in<3>'), oai_inv_n.get_pin('out'))
        self.connect_to_track_wires(oai_p.get_pin('in<3>'), oai_inv_p.get_pin('out'))
        # -- oai to buf --
        self.connect_to_track_wires([buf_n.get_pin('in'), oai_inv_n.get_pin('in')], oai_n.get_pin('out'))
        self.connect_to_track_wires([buf_p.get_pin('in'), oai_inv_p.get_pin('in')], oai_p.get_pin('out'))
        # Connection for dm
        self.connect_to_track_wires(nor.get_pin('out'), buf_m.get_pin('in'))
        _, nor_in_tidx = tr_manager.place_wires(vm_layer, ['sig'] * 5,
                                                align_track=oai_inv_p.get_pin('out').track_id.base_index, align_idx=0)
        _dn_fb_vm = self.connect_to_tracks([oai_inv_n.get_pin('in'), nor.get_pin('nin<1>')],
                                           TrackID(vm_layer, nor_in_tidx[1], tr_w_vm))
        _dp_fb_vm = self.connect_to_tracks([oai_inv_p.get_pin('in'), nor.get_pin('nin<0>')],
                                           TrackID(vm_layer, nor_in_tidx[2], tr_w_vm))
        comp_tidx_upper = self.grid.coord_to_track(vm_layer, buf_n.bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_list = self.get_available_tracks(vm_layer, oai_p.get_pin('out').track_id.base_index,
                                              comp_tidx_upper, 0, self.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        
        comp_n, comp_p = self.connect_differential_tracks(oai_n.get_pin('in<1>'), oai_p.get_pin('in<1>'), vm_layer,
                                                          tidx_list[0], tidx_list[1], width=tr_w_vm)

        # Connect retimer's input
        ret_in_tidx = self.grid.coord_to_track(vm_layer, flop.get_pin('nin').upper, mode=RoundMode.LESS_EQ)
        dp_out_tidx = tr_manager.get_next_track(vm_layer, buf_p.get_pin('out').track_id.base_index,
                                                'sig', 'sig', up=False)
        ret_in_tidx = self.get_available_tracks(vm_layer, self.grid.coord_to_track(vm_layer, flop.bound_box.xl),
                                                self.grid.coord_to_track(vm_layer, flop.bound_box.xh),
                                                0, self.get_tile_info(2)[1], width=tr_w_vm, sep=tr_sp_vm)[0]

        # Connect output dn/dp/dm
        if has_pmos_sw:
            out_vm_tidx = self.grid.coord_to_track(vm_layer, buf_n.bound_box.xh, mode=RoundMode.LESS_EQ)
            _, out_vm_locs = tr_manager.place_wires(vm_layer, ['sig'] * 4, align_track=out_vm_tidx, align_idx=-1)
            out_vm_list = [self.connect_to_tracks([buf_m.get_pin('nout'), buf_m.get_pin('pout')],
                                                  TrackID(vm_layer, out_vm_locs[0], tr_w_vm)),
                           self.connect_to_tracks([buf_n.get_pin('noutb'), buf_n.get_pin('poutb')],
                                                  TrackID(vm_layer, out_vm_locs[1], tr_w_vm)),
                           self.connect_to_tracks([buf_p.get_pin('noutb'), buf_p.get_pin('poutb')],
                                                  TrackID(vm_layer, out_vm_locs[2], tr_w_vm))]
            self.connect_wires([buf_n.get_pin('nout'), pg_n.get_pin('ns'), buf_n.get_pin('pout'), pg_n.get_pin('ps')])
            dp_mn = self.connect_wires([buf_p.get_pin('nout'), pg_p.get_pin('ps')])
            dp_mp = self.connect_wires([buf_p.get_pin('pout'), pg_p.get_pin('ns')])
            self.add_pin('dm', out_vm_list[0])
            self.add_pin('dn_b', out_vm_list[1])
            self.add_pin('dp_b', out_vm_list[2])
            _, out_vm_locs = tr_manager.place_wires(vm_layer, ['sig'] * 3 + ['sig'] * 2, align_track=out_vm_tidx,
                                                    align_idx=0)
            out_vm_list.extend([self.connect_to_tracks([pg_n.get_pin('nd'), pg_n.get_pin('pd')],
                                                       TrackID(vm_layer, out_vm_locs[1], tr_w_vm)),
                                self.connect_to_tracks([pg_p.get_pin('nd'), pg_p.get_pin('pd')],
                                                       TrackID(vm_layer, out_vm_locs[2], tr_w_vm)), ])
            en_vm = self.connect_to_tracks([pg_p.get_pin('en'), pg_n.get_pin('en')],
                                           TrackID(vm_layer, out_vm_locs[-1], tr_w_vm))
            enb_vm = self.connect_to_tracks([pg_p.get_pin('enb'), pg_n.get_pin('enb')],
                                            TrackID(vm_layer, out_vm_locs[-2], tr_w_vm))
            self.connect_to_track_wires(en_vm, [pg_n.get_pin('VDD'), pg_p.get_pin('VDD')])
            self.connect_to_track_wires(enb_vm, [pg_n.get_pin('VSS'), pg_p.get_pin('VSS')])
            self.add_pin('dn', out_vm_list[3])
            self.add_pin('dp', out_vm_list[4])
        else:
            out_vm_tidx = self.grid.coord_to_track(vm_layer, self.bound_box.xh, mode=RoundMode.LESS_EQ)
            _, out_vm_locs = tr_manager.place_wires(vm_layer, ['sig'] * 4, align_idx=-1, align_track=out_vm_tidx)
            out_vm_list = []
            for inst, tidx in zip([buf_m, buf_n, buf_p], out_vm_locs[:-1]):
                out_vm_list.append(self.connect_to_tracks([inst.get_pin('nout'), inst.get_pin('pout')],
                                                          TrackID(vm_layer, tidx, tr_w_vm)))
            self.add_pin('dm', out_vm_list[0])
            self.add_pin('dn', out_vm_list[1])
            self.add_pin('dp', out_vm_list[2])

        # Connection for VDD/VSS
        vss_list, vdd_list = [], []
        inst_list = [nand_done, nand_state, inv_done, oai_p, oai_n, oai_inv_p, oai_inv_n, inv_clk, flop_rst, flop,
                     nor, buf_m, buf_n, buf_p] 
        if has_pmos_sw:
            inst_list.extend([pg_n, pg_p])
        for inst in inst_list:
            vdd_list.append(inst.get_pin('VDD'))
            vss_list.append(inst.get_pin('VSS'))
        vdd_hm = self.connect_wires(vdd_list)
        vss_hm = self.connect_wires(vss_list)
        
        # find tracks for passgate dp_m to connect on vm lyer
        tidx_hidm_vm = self.grid.coord_to_track(vm_layer, flop.bound_box.xh, mode=RoundMode.LESS_EQ)

        # connect flop input with pass gate for dp_m
        tidx_lodmflop_vm = self.grid.coord_to_track(vm_layer, flop.bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_listflopdm_vm = self.get_available_tracks(vm_layer, tidx_lodmflop_vm, tidx_hidm_vm,
                                              flop.bound_box.xl, flop.bound_box.xh, width=tr_w_vm, sep=tr_sp_vm)

        # connect up all the dp_m wires
        dp_m = self.connect_to_tracks(dp_mn, TrackID(vm_layer, tidx_listflopdm_vm[0], tr_w_vm))
        dp_m = self.connect_to_track_wires(dp_mp, dp_m)

        dm_vm = self.connect_to_tracks([flop.get_pin('nin')], TrackID(vm_layer, tidx_listflopdm_vm[0], tr_w_vm))

        #extend the out_ret pin so extension does not conflict
        tidx_lo_hm = self.grid.coord_to_track(hm_layer, flop.bound_box.yl, mode=RoundMode.GREATER_EQ)
        tidx_hi_hm = self.grid.coord_to_track(hm_layer, flop.bound_box.yh, mode=RoundMode.LESS_EQ)
        tidx_list_hm = self.get_available_tracks(hm_layer, tidx_lo_hm, tidx_hi_hm,
                                              flop.bound_box.xh, self.bound_box.xh, width=tr_w_vm, sep=tr_sp_vm)
        tidx_lo_vm = self.grid.coord_to_track(vm_layer, flop.bound_box.xh, mode=RoundMode.GREATER_EQ)
        tidx_hi_vm = self.grid.coord_to_track(vm_layer, self.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_list_vm = self.get_available_tracks(vm_layer, tidx_lo_vm, tidx_hi_vm,
                                              flop.bound_box.yl, flop.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        hm = self.connect_to_tracks([flop.get_pin('out')], TrackID(hm_layer, tidx_list_hm[0], tr_w_vm)) 
        out_ret = self.connect_to_tracks([hm], TrackID(vm_layer, tidx_list_vm[0], tr_w_vm))

        # extend the inv_clk layer to vm layer
        tidx_loclk = self.grid.coord_to_track(vm_layer, flop_rst.bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hiclk = self.grid.coord_to_track(vm_layer, flop_rst.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_listclk = self.get_available_tracks(vm_layer, tidx_loclk, tidx_hiclk,
                                                0, flop_rst.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        nin = self.connect_to_tracks([inv_clk.get_pin('pin')], TrackID(vm_layer,  tidx_listclk[len(tidx_listclk)//4], tr_w_vm))

        self.add_pin('comp_p', comp_p)
        self.add_pin('comp_n', comp_n)
        self.add_pin('out_ret', out_ret)
        self.add_pin('bit', bit)
        self.add_pin('rst', rst_vm2) #rst_vm)
        self.add_pin('bit_nxt', flop_rst.get_pin('out'))
        self.add_pin('comp_clk', nin)
        self.add_pin('VDD', vdd_hm, show=self.show_pins, connect=True)
        self.add_pin('VSS', vss_hm, show=self.show_pins, connect=True)

        sch_params_dict = dict(
            inv_done=inv_done_master.sch_params,
            inv_clk=inv_clk_master.sch_params,
            nand_done=nand_done_master.sch_params,
            nand_state=nand_state_master.sch_params,
            rflop=flop_rst_master.sch_params,
            flop=flop_master.sch_params,
            oai=oai_master.sch_params,
            oai_fb=inv_fb_n_master.sch_params,
            nor=nor3_master.sch_params,
            buf=buf_master.sch_params,
            buf_np=buf_np_master.sch_params,
            has_pmos_sw=has_pmos_sw,
        )
        if has_pmos_sw:
            sch_params_dict.update(pg=pg_master.sch_params)
        self.sch_params = sch_params_dict


class SARRetUnit(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_ret_unit')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Number of segments.',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row',
            ridx_p='index for pmos row',
            substrate_row='True to add substrate row',
            sig_locs='Signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1,
            substrate_row=False,
            sig_locs={},
        )

    def draw_layout(self) -> None:
        # setup floorplan
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, Any] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        w_p: int = self.params['w_p']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        substrate_row: bool = self.params['substrate_row']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        seg_buf: List = seg_dict['buf']
        seg_flop: int = seg_dict['flop']
        seg_inv: int = seg_dict['inv']
        min_sep = self.min_sep_col
        min_sep += min_sep & 1

        # compute track locations
        tr_manager = self.tr_manager
        tr_w_vm_sig = tr_manager.get_width(vm_layer, 'sig')
        tr_w_vm_clk = tr_manager.get_width(vm_layer, 'clk')

        ng0_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        ng2_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=0)
        nd1_tidx = self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=1)

        flop_params = dict(pinfo=pinfo, seg=seg_flop, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                            vertical_sup=substrate_row, sig_locs={'nclkb': ng1_tidx, 'nclk': ng0_tidx, 'pin': pg0_tidx})

        inv_params = dict(pinfo=pinfo, seg=seg_inv, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                          vertical_sup=substrate_row, sig_locs={'nin': ng0_tidx})
        buf_params = dict(pinfo=pinfo, seg_list=seg_buf, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                          vertical_sup=substrate_row, sig_locs={'nin0': ng2_tidx, 'nin1': ng1_tidx, 'nout0': nd0_tidx, 'nout1': nd1_tidx},
                          vertical_out=False)

        inv_master = self.new_template(InvCore, params=inv_params)
        buf_master = self.new_template(InvChainCore, params=buf_params)
        latch_master = self.new_template(LatchCore, params=latch_params)

        cur_col = 0
        inv = self.add_tile(inv_master, 0, cur_col)
        cur_col += inv_master.num_cols + min_sep
        latch = self.add_tile(latch_master, 0, cur_col)
        cur_col += latch_master.num_cols
        buf = self.add_tile(buf_master, 0, cur_col)
        tot_seg = cur_col + buf_master.num_cols
        self.set_mos_size(tot_seg)

        self.connect_to_track_wires(inv.get_pin('out'), latch.get_pin('pclkb'))
        self.connect_wires([inv.get_pin('nin'), latch.get_pin('nclk')])
        self.connect_to_track_wires(latch.get_pin('out'), buf.get_pin('nin'))
        vdd_gate_list, vss_gate_list = [], []
        for inst in [inv, latch, buf]:
            vdd_gate_list.append(inst.get_pin('VDD'))
            vss_gate_list.append(inst.get_pin('VSS'))
        vdd_hm = self.connect_wires(vdd_gate_list)
        vss_hm = self.connect_wires(vss_gate_list)
        out_tidx = sig_locs.get('out', 0)

        out_vm_tidx = self.grid.coord_to_track(vm_layer, buf.get_pin('nout').middle, mode=RoundMode.NEAREST)
        out_vm = self.connect_to_tracks([buf.get_pin('nout'), buf.get_pin('pout')],
                                        TrackID(vm_layer, out_vm_tidx + out_tidx, tr_w_vm_sig))
        clk_in_vm_tidx = tr_manager.get_next_track(vm_layer, inv.get_pin('out').track_id.base_index, 'clk', 'clk')
        in_vm_tidx = tr_manager.get_next_track(vm_layer, clk_in_vm_tidx, 'clk', 'sig')
        in_vm_tidx += sig_locs.get('in', 0)

        clk_in_vm = self.connect_to_tracks(inv.get_pin('nin'), TrackID(vm_layer, clk_in_vm_tidx, tr_w_vm_clk))
        in_vm = self.connect_to_tracks(latch.get_pin('nin'), TrackID(vm_layer, in_vm_tidx, tr_w_vm_sig))

        self.add_pin('in', in_vm)
        self.add_pin('out', out_vm)
        self.add_pin('clk', clk_in_vm)
        self.add_pin('VDD', vdd_hm, show=self.show_pins, connect=True)
        self.add_pin('VSS', vss_hm, show=self.show_pins, connect=True)

        self.sch_params = dict(
            latch=latch_master.sch_params,
            inv=inv_master.sch_params,
            buf=buf_master.sch_params,
        )


class SARLogicArray(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._lower_layer_routing = False

    @property
    def lower_layer_routing(self):
        return self._lower_layer_routing

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_logic_array_sync')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Number of segments.',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row',
            ridx_p='index for pmos row',
            logic_unit_row_arr='Array of unit cells',
            flop_out_unit_row_arr='Array of retimer cells',
            substrate_row='True to add substrate row',
            has_pmos_sw='True to add differential signal to drive pmos switch in CDAC',
            lower_layer_routing='Avoid use metal layer above xm for routing'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1,
            logic_unit_row_arr=[],
            flop_out_unit_row_arr=[],
            substrate_row=False,
            has_pmos_sw=False,
            lower_layer_routing=False,
        )

    def draw_layout(self) -> None:
        # setup floorplan
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, Any] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        w_p: int = self.params['w_p']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        substrate_row: bool = self.params['substrate_row']
        has_pmos_sw: bool = self.params['has_pmos_sw']
        lower_layer_routing: bool=self.params['lower_layer_routing']
        logic_unit_row_arr: List[int] = self.params['logic_unit_row_arr']
        flop_out_unit_row_arr: List[int] = self.params['flop_out_unit_row_arr']
        self._lower_layer_routing = lower_layer_routing

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        #seg_ret: Dict[str, int] = seg_dict['retimer']
        seg_flop_out: List[int] = seg_dict['flop_out']
        seg_buf_int: List[int] = seg_dict['buf_int']
        seg_buf_out: List[int] = seg_dict['buf_out']
        seg_logic: ImmutableSortedDict[str, Any] = seg_dict['logic']
        logic_scale_list: List[int] = seg_dict['logic_scale_list']
        min_sep = self.min_sep_col
        min_sep += min_sep & 1

        # Check logic unit arrangement
        num_bits = len(logic_scale_list)
        if not logic_unit_row_arr:
            logic_unit_row_arr = [num_bits]
        if not flop_out_unit_row_arr:
            flop_out_unit_row_arr = [num_bits]
        if num_bits != sum(logic_unit_row_arr):
            raise ValueError("Logic unit array arrangement doesn't match number of units")
        if num_bits != sum(flop_out_unit_row_arr):
            raise ValueError("Logic unit array arrangement doesn't match number of units")

        # compute track locations
        tr_manager = self.tr_manager
        tr_w_vm_clk = tr_manager.get_width(vm_layer, 'clk')
        tr_w_vm_sig = tr_manager.get_width(vm_layer, 'sig')
        tr_w_xm_sig = tr_manager.get_width(xm_layer, 'sig')
        tr_w_xm_clk = tr_manager.get_width(xm_layer, 'clk')
        tr_sp_xm_sig = tr_manager.get_sep(xm_layer, ('sig', 'sig'))
        tr_sp_vm_clk = tr_manager.get_sep(vm_layer, ('clk', 'clk'))
        tr_sp_vm_sig = tr_manager.get_sep(vm_layer, ('sig', 'sig'))
        tr_sp_vm_sig_clk = tr_manager.get_sep(vm_layer, ('sig', 'clk'))
        rt_layer = vm_layer if lower_layer_routing else ym_layer
        tr_w_rt_clk = tr_manager.get_width(rt_layer, 'clk')
        tr_w_rt_sig = tr_manager.get_width(rt_layer, 'sig')
        tr_sp_rt_clk = tr_manager.get_sep(rt_layer, ('clk', 'clk'))
        tr_sp_rt_sig = tr_manager.get_sep(rt_layer, ('sig', 'sig'))

        buf_int_params = dict(pinfo=pinfo, seg_list=seg_buf_int, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                              vertical_sup=substrate_row, sig_locs={}, dual_output=True, vertical_out=False)
        buf_out_params = dict(pinfo=pinfo, seg_list=seg_buf_out, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                              vertical_sup=substrate_row, sig_locs={}, dual_output=True, vertical_out=False)
        flop_params = dict(pinfo=pinfo, seg=seg_flop_out, seg_ck=4, resetable=False, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                            vertical_sup=substrate_row)
        logic_unit_params_list = []
        for idx, scale in enumerate(logic_scale_list):
            _seg_dict = copy.deepcopy(seg_logic.to_dict())
            for key, val in _seg_dict.items():
                if key == 'nor' or key == 'oai_fb':
                    _seg_dict[key] = val * scale
                elif key == 'buf':
                    _seg_dict[key] = [_seg * scale for _seg in val]
                elif key == 'oai':
                    for _key, _val in val.items():
                        _seg_dict[key] = dict(_seg_dict[key])
                        _seg_dict[key][_key] = _seg_dict[key][_key] * scale
            logic_unit_params_list.append(dict(pinfo=pinfo, seg_dict=_seg_dict, substrate_row=substrate_row,
                                               has_pmos_sw=has_pmos_sw, w_n=w_n, w_p=w_p))

        logic_unit_master_list = [self.new_template(SARLogicUnit, params=_params) for _params in logic_unit_params_list]
        buf_int_master = self.new_template(InvChainCore, params=buf_int_params)
        buf_out_master = self.new_template(InvChainCore, params=buf_out_params)

        if lower_layer_routing:
            tot_wires = 2*num_bits if has_pmos_sw else num_bits
            wire_type = 'clk' if tr_w_vm_clk+tr_sp_vm_clk>tr_sp_vm_sig + tr_w_vm_sig else 'sig'
            ntr, _ = tr_manager.place_wires(vm_layer, [wire_type]*tot_wires)
            ncol_rt = self.arr_info.get_column_span(vm_layer, ntr)

        origin_col = ncol_rt//2 if lower_layer_routing else 0

        # Add clock buffer
        buf_int = self.add_tile(buf_int_master, 1, origin_col)
        buf_out = self.add_tile(buf_out_master, 0, origin_col)

        flop_out_ncol, flop_out_row_list = 0, []
        
        # get row indices for the flop rows
        flop_master = self.new_template(FlopCore, params=flop_params)
        nd1_tidx = self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=1)
        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        for idx, row_num in enumerate(flop_out_unit_row_arr):
            flop_params = dict(pinfo=pinfo, seg=seg_flop_out, seg_ck=4, resetable=False, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                            vertical_sup=substrate_row, sig_locs={'clk': nd1_tidx, 'nin': pg0_tidx, 'out':idx})
            flop_shift_master = self.new_template(FlopCore, params=flop_params)
            _flop_list = []
            if idx == 0:
                cur_col_flop = buf_out_master.num_cols + min_sep + origin_col
            elif idx == 1:
                cur_col_flop = buf_int_master.num_cols + 2*min_sep + origin_col
            else:
                cur_col_flop = origin_col
            for jdx in range(row_num):
                _flop_list.append(self.add_tile(flop_shift_master, idx, cur_col_flop))
                cur_col_flop += flop_shift_master.num_cols + 2*min_sep
                cur_col_flop += cur_col_flop & 1 + 2
            flop_out_row_list.append(_flop_list)
            flop_out_ncol = max(cur_col_flop, flop_out_ncol)

        logic_row = max(1, len(flop_out_unit_row_arr)) + 1

        # TODO: reverse size of odd rows
        # Add logic cells
        logic_ncol, logic_row_list = 0, []
        num_logic_rows = logic_unit_master_list[0].num_tile_rows
        for idx, row_num in enumerate(logic_unit_row_arr):
            _logic_list = []
            cur_col, cur_row = origin_col, logic_row + idx * num_logic_rows
            cur_row += (cur_row & 1) * 4
            for jdx in range(row_num):
                _master = logic_unit_master_list[sum(logic_unit_row_arr[:idx + 1]) - jdx - 1]
                if idx & 1:
                    _logic_list.append(self.add_tile(_master, cur_row, cur_col + _master.num_cols, flip_lr=True))
                else:
                    _logic_list.append(self.add_tile(_master, cur_row, cur_col))
                cur_col += _master.num_cols + min_sep
            _logic_list = _logic_list[::-1] if idx & 1 else _logic_list
            logic_row_list.append(_logic_list)
            logic_ncol = max(cur_col, logic_ncol)
        tot_ncol = ncol_rt//2+self.num_cols if lower_layer_routing else self.num_cols
        self.set_mos_size(tot_ncol)

        # Connect clk signal together
        flop_out_flatten_list = [item for sublist in flop_out_row_list for item in sublist]
        logic_flatten_list = [item for sublist in logic_row_list for item in sublist]
        flop_out_clk_list = [inst.get_pin('clk') for inst in flop_out_flatten_list]
        flop_out_clk_vm = self.connect_wires(flop_out_clk_list)
        clk_int_vm_tidx = self.grid.coord_to_track(vm_layer, buf_int.get_pin('nout').upper, mode=RoundMode.NEAREST)
        clk_out_vm_tidx = tr_manager.get_next_track(vm_layer, clk_int_vm_tidx, 'clk', 'clk', up=False)
        buf_int_out_vm = self.connect_to_tracks([buf_int.get_pin('nout'), buf_int.get_pin('pout')],
                                                TrackID(vm_layer, clk_int_vm_tidx, tr_w_vm_clk))
        buf_out_out_vm = self.connect_to_tracks([buf_out.get_pin('nout'), buf_out.get_pin('pout')],
                                                TrackID(vm_layer, clk_out_vm_tidx, tr_w_vm_clk))

        flop_out_clk_xm_tidx = self.grid.coord_to_track(xm_layer, flop_out_clk_vm[0].upper, mode=RoundMode.NEAREST)
        flop_out_clk_xm = self.connect_to_tracks(flop_out_clk_vm, TrackID(xm_layer, flop_out_clk_xm_tidx, tr_w_xm_clk))
        self.connect_to_track_wires(buf_int_out_vm, flop_out_clk_xm)
        self.connect_to_track_wires(buf_int_out_vm, buf_out.get_pin('nin'))

        flop_out_list = [inst.get_pin('out') for inst in flop_out_flatten_list]
        flop_out_in_list = [inst.get_pin('in') for inst in flop_out_flatten_list]
        flop_out_list.sort(key=lambda x: x.track_id.base_index)
        flop_out_in_list.sort(key=lambda x: x.track_id.base_index)
        flop_out_flatten_list.sort(key=lambda x: x.get_pin('in').track_id.base_index)
        #connect up to vm layer because the flop pin is on met 1 and shorting randomly
        for i in range(0, len(flop_out_in_list)):
            if i%2:
                tidx_lo_hm = self.grid.coord_to_track(hm_layer, (flop_out_flatten_list[i].bound_box.yl+flop_out_flatten_list[i].bound_box.yh)//2,
                                                  mode=RoundMode.GREATER_EQ)
            else:
                tidx_lo_hm = self.grid.coord_to_track(hm_layer, (flop_out_flatten_list[i].bound_box.yl+flop_out_flatten_list[i].bound_box.yh)//3,
                                                  mode=RoundMode.GREATER_EQ)
            tidx_hi_hm = self.grid.coord_to_track(hm_layer, flop_out_flatten_list[i].bound_box.yh, mode=RoundMode.LESS_EQ)
            tidx_list_hm = self.get_available_tracks(hm_layer, tidx_lo_hm, tidx_hi_hm,
                                              flop_out_flatten_list[i].bound_box.xl, flop_out_flatten_list[i].bound_box.xh, width=tr_w_vm_clk, sep=tr_sp_vm_clk)
        
            tidx_lo_vm = self.grid.coord_to_track(vm_layer, flop_out_flatten_list[i].bound_box.xl, mode=RoundMode.GREATER_EQ)
            tidx_hi_vm = self.grid.coord_to_track(vm_layer, flop_out_flatten_list[i].bound_box.xh, mode=RoundMode.LESS_EQ)
            tidx_list_vm = self.get_available_tracks(vm_layer, tidx_lo_vm, tidx_hi_vm,
                                              flop_out_flatten_list[i].bound_box.yl, flop_out_flatten_list[i].bound_box.yh, width=tr_w_vm_clk, sep=tr_sp_vm_clk)
            hm = self.connect_to_tracks([flop_out_flatten_list[i].get_pin('in')], TrackID(hm_layer, tidx_list_hm[0], tr_w_vm_clk)) 
            flop_out_in_list[i] = self.connect_to_tracks([hm], TrackID(vm_layer, tidx_list_vm[0], tr_w_vm_clk))
            

        flop_out_list = self.extend_wires(flop_out_list, lower=self.get_tile_info(0)[1])  
         # not shorting seems to just be part of SARRetUnit layout 
        flop_out_in_list = self.extend_wires(flop_out_in_list, upper=self.get_tile_info(len(logic_unit_row_arr))[1])

        # Connection for logic cells
        # -- Rst signal --
        logic_rst_list = []
        for inst_row in logic_row_list:
            rst_vm_list = [inst.get_pin('rst') for inst in inst_row]
            rst_xm_tidx = self.grid.coord_to_track(xm_layer, rst_vm_list[0].lower, mode=RoundMode.NEAREST)
            logic_rst_list.append(self.connect_to_tracks(rst_vm_list, TrackID(xm_layer, rst_xm_tidx, tr_w_xm_clk)))

        # rst_buf_vm_tidx = self.grid.coord_to_track(vm_layer, buf_int.get_pin('nin').lower, mode=RoundMode.LESS_EQ)
        rst_buf_vm_tidx = tr_manager.get_next_track(vm_layer, buf_int.get_pin('outb').track_id.base_index, 'sig', 'sig',
                                                    up=False)
        rst_buf_vm = self.connect_to_tracks(buf_int.get_pin('nin'), TrackID(vm_layer, rst_buf_vm_tidx, tr_w_vm_clk))
        rst_buf_xm_tidx = self.grid.coord_to_track(xm_layer, rst_buf_vm.middle, mode=RoundMode.NEAREST)
        clk_midb_xm_tidx = tr_manager.get_next_track(xm_layer, rst_buf_xm_tidx, 'clk', 'clk')
        rst_buf_xm = self.connect_to_tracks(rst_buf_vm, TrackID(xm_layer, rst_buf_xm_tidx, tr_w_xm_clk))
        clk_midb_xm = self.connect_to_tracks(buf_int.get_pin('outb'), TrackID(xm_layer, clk_midb_xm_tidx, tr_w_xm_clk))

        # rt layer connection
        rt_tidx_start = self.arr_info.col_to_track(rt_layer, 0)
        rt_tidx_stop = self.arr_info.col_to_track(rt_layer, self.num_cols)
        if lower_layer_routing:
            rt_tidx_list = self.get_available_tracks(rt_layer, rt_tidx_start, rt_tidx_stop, self.bound_box.yl,
                                                     self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
            rt_tidx_coord_list = [self.grid.track_to_coord(rt_layer, x) for x in rt_tidx_list]
            rst_rt_tidx = self.get_nearest_tidx(rst_buf_xm, rt_tidx_list, rt_tidx_coord_list)
            clk_midb_rt_tidx = self.get_nearest_tidx(rst_buf_xm, rt_tidx_list, rt_tidx_coord_list)
        else:
            rst_rt_tidx = self.grid.coord_to_track(rt_layer, rst_buf_xm.lower, mode=RoundMode.LESS_EQ)
            clk_midb_rt_tidx = tr_manager.get_next_track(rt_layer, rst_rt_tidx, 'clk', 'clk')
            ym_tidx_start = tr_manager.get_next_track(rt_layer, clk_midb_rt_tidx, 'clk', 'clk')

        rst_rt = self.connect_to_tracks([rst_buf_xm] + logic_rst_list, TrackID(rt_layer, rst_rt_tidx, tr_w_rt_clk))

        # -- Compn/Compp signal --
        logic_compn_list, logic_compp_list = [], []
        for inst_row in logic_row_list:
            rst_vm_list = [inst.get_pin('rst') for inst in inst_row]
            compn_list = [inst.get_pin('comp_n') for inst in inst_row]
            compp_list = [inst.get_pin('comp_p') for inst in inst_row]
            rst_xm_tidx = self.grid.coord_to_track(xm_layer, rst_vm_list[0].lower, mode=RoundMode.NEAREST)
            _, comp_xm_locs = tr_manager.place_wires(xm_layer, ['clk', 'sig', 'sig'], rst_xm_tidx, 0)
            logic_compn_list.append(self.connect_to_tracks(compn_list, TrackID(xm_layer, comp_xm_locs[1], tr_w_xm_sig)))
            logic_compp_list.append(self.connect_to_tracks(compp_list, TrackID(xm_layer, comp_xm_locs[2], tr_w_xm_sig)))

        if lower_layer_routing:
            comp_p_ym = logic_compp_list
            comp_n_ym = logic_compn_list
        else:
            # ---- Connect comp/n to ym ----
            middle_coord = self.arr_info.col_to_coord(self.num_cols // 2)
            _, comp_ym_locs = tr_manager.place_wires(ym_layer, ['sig'] * 2, center_coord=middle_coord)
            comp_p_ym = self.connect_to_tracks(logic_compp_list, TrackID(rt_layer, comp_ym_locs[0], tr_w_rt_sig),
                                               track_upper=self.bound_box.yh)
            comp_n_ym = self.connect_to_tracks(logic_compn_list, TrackID(rt_layer, comp_ym_locs[1], tr_w_rt_sig),
                                               track_upper=self.bound_box.yh)

        # -- bit/bit_nxt --
        out_flop_list = []
        last_bit = None
        bit_nxt_list = []
        ym_tidx_stop = self.grid.coord_to_track(ym_layer, self.bound_box.xh, mode=RoundMode.LESS)
        # Also take care of comp_clk xm in this section
        comp_clk_xm_list = []
        for idx, inst_row in enumerate(logic_row_list):
            _row = logic_row + (idx + 1) * num_logic_rows
            _coord = self.get_tile_info(_row)[1]
            _tidx_start = self.grid.coord_to_track(xm_layer, _coord, mode=RoundMode.GREATER_EQ)
            num_units = len(inst_row)
            _, xm_locs = tr_manager.place_wires(xm_layer, ['sup']+ ['sig'] * (num_units + 4), _tidx_start, 0)
            xm_locs = xm_locs[1:] # Avoid put on top of sup
            bit_sig_list = []
            bit_nxt_sig_list = []
            # inst_row = inst_row[::-1] if idx & 1 else inst_row
            for jdx, inst in enumerate(inst_row):
                if jdx & 1:
                    _tidx, _tidx_nxt = xm_locs[1], xm_locs[2]
                else:
                    _tidx, _tidx_nxt = xm_locs[2], xm_locs[1]
                if idx == len(logic_row_list)-1 and jdx == len(inst_row)-1:
                    _tidx = xm_locs[0]
                bit_sig_list.append(self.connect_to_tracks(inst.get_pin('bit'),
                                                           TrackID(xm_layer, _tidx, tr_w_xm_sig)))
                if (idx>0 or jdx>0):
                    bit_nxt_sig_list.append(self.connect_to_tracks(inst.get_pin('bit_nxt'),
                                                                TrackID(xm_layer, _tidx_nxt, tr_w_xm_sig)))
                else:
                    self.add_pin(f'state<0>', inst.get_pin('bit_nxt'))
                comp_clk_xm_list.append(self.connect_to_tracks(inst.get_pin('comp_clk'),
                                                 TrackID(xm_layer, xm_locs[3 + num_units], tr_w_xm_sig)))
            bit_nxt_list.extend(bit_nxt_sig_list)
            
            # connect bit and bit_nxt
            conn_bit_nxt_sig_list = bit_nxt_sig_list[1:] if idx>0 else bit_nxt_sig_list
            for _bit, _bitnxt in zip(conn_bit_nxt_sig_list, bit_sig_list[:-1]):
                self.connect_wires([_bit, _bitnxt])
            out_flop_list.extend([self.connect_to_tracks(inst_row[idx].get_pin('out_ret'),
                                                        TrackID(xm_layer, xm_locs[3 + idx], tr_w_xm_sig))
                                 for idx in range(num_units)])

            # connect bit and bit_nxt across rows
            if idx > 0:
                _coord = max(bit_nxt_sig_list[0].middle, last_bit.middle) if idx & 1 else \
                    min(bit_nxt_sig_list[0].middle, last_bit.middle)
                if lower_layer_routing:
                    rt_tidx_list = self.get_available_tracks(rt_layer, rt_tidx_start, rt_tidx_stop,
                                                             last_bit.bound_box.yh,
                                                             self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                    rt_tidx_coord_list = [self.grid.track_to_coord(rt_layer, x) for x in rt_tidx_list]
                    _y_wire = self.connect_to_tracks([last_bit, bit_nxt_sig_list[0]],
                                                     TrackID(rt_layer, self.get_nearest_tidx(last_bit, rt_tidx_list,
                                                                                             rt_tidx_coord_list, _coord),
                                                             tr_w_rt_sig))
                else:
                    ym_tidx_list = self.get_available_tracks(ym_layer, ym_tidx_start, ym_tidx_stop, last_bit.bound_box.yh,
                                                             self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                    ym_tidx_coord_list = [self.grid.track_to_coord(ym_layer, x) for x in ym_tidx_list]
                    _y_wire = self.connect_to_tracks([last_bit, bit_nxt_sig_list[0]],
                                                     TrackID(ym_layer, self.get_nearest_tidx(last_bit, ym_tidx_list,
                                                                                             ym_tidx_coord_list, _coord),
                                                             tr_w_rt_sig))
            last_bit = bit_sig_list[-1]

        # Connect the comp_clk wires together
        self.extend_wires(comp_clk_xm_list, lower=self.bound_box.xl)
        tidx_comp_clk_vm = self.grid.coord_to_track(vm_layer, self.bound_box.xl, mode=RoundMode.NEAREST)
        comp_clk_vm = self.connect_to_tracks(comp_clk_xm_list, TrackID(vm_layer, tidx_comp_clk_vm, tr_w_rt_sig*4))

        # Connect clk midb
        self.connect_to_tracks([clk_midb_xm, last_bit], TrackID(rt_layer, clk_midb_rt_tidx, tr_w_rt_clk))

        # -- flop input --
        flop_out_in_tidx_start = self.grid.coord_to_track(xm_layer, self.get_tile_info(len(flop_out_unit_row_arr))[1],
                                                         mode=RoundMode.NEAREST)
        _, flop_out_in_xm_locs = tr_manager.place_wires(xm_layer, ['sig'] * (num_bits + 1), flop_out_in_tidx_start, 0)
        flop_out_in_xm_list = []
        for tidx, flop_in in zip(flop_out_in_xm_locs[1:], flop_out_in_list):
            flop_out_in_xm_list.append(self.connect_to_tracks(flop_in, TrackID(xm_layer, tidx, tr_w_xm_sig))) #FIXME

        # Connect from logic to output flop
        if lower_layer_routing:
            rt_tidx_core_l = self.arr_info.col_to_track(rt_layer, ncol_rt//2)
            rt_tidx_core_h = self.arr_info.col_to_track(rt_layer, self.num_cols-ncol_rt//2)
            rt_tidx_list_lower = self.get_available_tracks(rt_layer, rt_tidx_start, rt_tidx_core_l, self.bound_box.yl,
                                                             self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
            rt_tidx_list_upper = self.get_available_tracks(rt_layer, rt_tidx_core_h, rt_tidx_stop, self.bound_box.yl,
                                                           self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
            middle = -(-num_bits//2)
            flop_out_in_tidx = rt_tidx_list_lower[-1 - middle:] + rt_tidx_list_upper[:middle]
            for idx, (flop_in, logic_out) in enumerate(zip(flop_out_in_xm_list, out_flop_list)):
                self.connect_to_tracks([flop_in, logic_out], TrackID(rt_layer, flop_out_in_tidx[idx], tr_w_rt_sig))
        else:
            stop_ym_tidx = self.arr_info.col_to_track(ym_layer, self.num_cols, mode=RoundMode.NEAREST)
            tidx_list = self.get_available_tracks(ym_layer, ym_tidx_start, stop_ym_tidx,
                                                  0, self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
            flop_out_in_tidx = tidx_list[:num_bits // 2] + tidx_list[-1 - num_bits // 2:]
            for idx, (ret_in, logic_out) in enumerate(zip(flop_out_in_xm_list, out_flop_list)):
                self.connect_to_tracks([ret_in, logic_out], TrackID(ym_layer, flop_out_in_tidx[idx], tr_w_rt_sig))

        # Connect dm/dn/dp dp_b/dn_b to xm_layer
        dm_xm_list, dn_xm_list, dp_xm_list, dpb_xm_list, dnb_xm_list = [], [], [], [], []
 
        for idx, inst_row in enumerate(logic_row_list):
            _row_start = logic_row + idx * num_logic_rows + 1
            _row_stop = logic_row + (idx + 1) * num_logic_rows + 1
            _tidx_start = self.grid.coord_to_track(xm_layer, self.get_tile_info(_row_start)[1],
                                                   mode=RoundMode.GREATER_EQ)
            _tidx_stop = self.grid.coord_to_track(xm_layer, self.get_tile_info(_row_stop)[1], mode=RoundMode.LESS_EQ)
            tidx_list = self.get_available_tracks(xm_layer, _tidx_start, _tidx_stop, self.bound_box.xl,
                                                  self.bound_box.xh, width=tr_w_xm_sig, sep=tr_sp_xm_sig)
            # inst_row = inst_row[::-1] if idx & 1 else inst_row
            if has_pmos_sw:
                for jdx, inst in enumerate(inst_row):
                    dm_xm_list.append(self.connect_to_tracks(inst.get_pin('dm'),
                                                             TrackID(xm_layer, tidx_list[5 * jdx], tr_w_xm_sig)))
                    dn_xm_list.append(self.connect_to_tracks(inst.get_pin('dn'),
                                                             TrackID(xm_layer, tidx_list[5 * jdx + 1], tr_w_xm_sig)))
                    dp_xm_list.append(self.connect_to_tracks(inst.get_pin('dp'),
                                                             TrackID(xm_layer, tidx_list[5 * jdx + 2], tr_w_xm_sig)))
                    dnb_xm_list.append(self.connect_to_tracks(inst.get_pin('dn_b'),
                                                              TrackID(xm_layer, tidx_list[5 * jdx + 3], tr_w_xm_sig)))
                    dpb_xm_list.append(self.connect_to_tracks(inst.get_pin('dp_b'),
                                                              TrackID(xm_layer, tidx_list[5 * jdx + 4], tr_w_xm_sig)))
            else:
                for jdx, inst in enumerate(inst_row):
                    dm_xm_list.append(self.connect_to_tracks(inst.get_pin('dm'),
                                                             TrackID(xm_layer, tidx_list[3 * jdx], tr_w_xm_sig)))
                    dn_xm_list.append(self.connect_to_tracks(inst.get_pin('dn'),
                                                             TrackID(xm_layer, tidx_list[3 * jdx + 1], tr_w_xm_sig)))
                    dp_xm_list.append(self.connect_to_tracks(inst.get_pin('dp'),
                                                             TrackID(xm_layer, tidx_list[3 * jdx + 2], tr_w_xm_sig)))

        # Connect dm/dn/dp to ym_layer and extend to top
        _tidx_start = self.grid.coord_to_track(ym_layer, self.bound_box.xl, mode=RoundMode.GREATER_EQ)
        _tidx_stop = self.grid.coord_to_track(ym_layer, self.bound_box.xh, mode=RoundMode.LESS)
        if has_pmos_sw:
            for idx, pin in enumerate(dm_xm_list):
                if lower_layer_routing:
                    _y_wire = pin
                else:
                    ym_tidx_list = self.get_available_tracks(ym_layer, _tidx_start, _tidx_stop, pin.bound_box.yh,
                                                             self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                    ym_tidx_coord_list = [self.grid.track_to_coord(ym_layer, x) for x in ym_tidx_list]
                    _y_wire = self.connect_to_tracks(pin, TrackID(ym_layer,
                                                                  self.get_nearest_tidx(pin, ym_tidx_list, ym_tidx_coord_list),
                                                                  tr_w_rt_sig), track_upper=self.bound_box.yh)
                self.add_pin(f'dm<{idx}>', _y_wire, mode=PinMode.UPPER)
            for d_list, pname in zip([dpb_xm_list, dn_xm_list], ['dp_b', 'dn']):
                for idx, pin in enumerate(d_list):
                    if lower_layer_routing:
                        _y_wire = pin
                    else:
                        ym_tidx_list = self.get_available_tracks(ym_layer, _tidx_start, _tidx_stop, pin.bound_box.yh,
                                                                 self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                        ym_tidx_coord_list = [self.grid.track_to_coord(ym_layer, x) for x in ym_tidx_list]
                        _y_wire = self.connect_to_tracks(pin, TrackID(ym_layer, self.get_nearest_tidx(pin, ym_tidx_list,
                                                                                                 ym_tidx_coord_list),
                                                                      tr_w_rt_sig), track_upper=self.bound_box.yh)
                    self.add_pin(f'{pname}<{idx}>', _y_wire, mode=PinMode.UPPER)
            for d_list, pname in zip([dnb_xm_list, dp_xm_list], ['dn_b', 'dp']):
                for idx, pin in enumerate(d_list):
                    if lower_layer_routing:
                        _y_wire = pin
                    else:
                        ym_tidx_list = self.get_available_tracks(ym_layer, _tidx_start, _tidx_stop, pin.bound_box.yh,
                                                                 self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                        ym_tidx_coord_list = [self.grid.track_to_coord(ym_layer, x) for x in ym_tidx_list]
                        _y_wire = self.connect_to_tracks(pin, TrackID(ym_layer, self.get_nearest_tidx(pin, ym_tidx_list,
                                                                                                 ym_tidx_coord_list),
                                                                      tr_w_rt_sig), track_upper=self.bound_box.yh)
                    self.add_pin(f'{pname}<{idx}>', _y_wire, mode=PinMode.UPPER)

        else:
            for d_list, pname in zip([dm_xm_list, dn_xm_list, dp_xm_list], ['dm', 'dn', 'dp']):
                for idx, pin in enumerate(d_list):
                    if lower_layer_routing:
                        _y_wire = pin
                    else:
                        ym_tidx_list = self.get_available_tracks(ym_layer, _tidx_start, _tidx_stop, pin.bound_box.yh,
                                                                 self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                        ym_tidx_coord_list = [self.grid.track_to_coord(ym_layer, x) for x in ym_tidx_list]
                        _y_wire = self.connect_to_tracks(pin,
                                                         TrackID(ym_layer,
                                                                 self.get_nearest_tidx(pin, ym_tidx_list, ym_tidx_coord_list),
                                                                 tr_w_rt_sig), track_upper=self.bound_box.yh)
                    self.add_pin(f'{pname}<{idx}>', _y_wire, mode=PinMode.UPPER)
        # Connection for VDD/VSS
        vss_list, vdd_list = [], []
        inst_list = [buf_int, buf_out] + flop_out_flatten_list + logic_flatten_list
        for inst in inst_list:
            vdd_list.append(inst.get_all_port_pins('VDD'))
            vss_list.append(inst.get_all_port_pins('VSS'))
        vdd_list = [vdd for sublist in vdd_list for vdd in sublist]
        vss_list = [vss for sublist in vss_list for vss in sublist]
        vdd_hm = self.connect_wires(vdd_list)
        vss_hm = self.connect_wires(vss_list)

        for idx, pin in enumerate(bit_nxt_list):
            self.add_pin(f'state<{idx+1}>', pin)
        self.add_pin('VDD', vdd_hm, show=self.show_pins, connect=True)
        self.add_pin('VSS', vss_hm, show=self.show_pins, connect=True)
        self.add_pin('comp_clk', comp_clk_vm)
        self.add_pin('rst', rst_rt)
        self.add_pin('comp_p', comp_p_ym)
        self.add_pin('comp_n', comp_n_ym)
        self.add_pin('clk_out', buf_out_out_vm)
        [self.add_pin(f'data_out<{idx}>', pin) for idx, pin in enumerate(flop_out_list)]
        self.sch_params = dict(
            flop=flop_master.sch_params,
            buf_clk=buf_int_master.sch_params,
            buf_out=buf_out_master.sch_params,
            logic_list=[master.sch_params for master in logic_unit_master_list][::-1],
            nbits=num_bits,
            has_pmos_sw=has_pmos_sw,
        )

    @classmethod
    def get_nearest_tidx(cls, wire: WireArray, aval_tidx_list: List[HalfInt], aval_tidx_coord_list: List[float],
                         coord: Tuple[int, None] = None):
        wire_coord = wire.middle if coord is None else coord
        nearest_track_coord = min(aval_tidx_coord_list, key=lambda x: abs(x - wire_coord))
        nearest_idx = aval_tidx_coord_list.index(nearest_track_coord)
        nearest_track = aval_tidx_list[nearest_idx]
        aval_tidx_list.pop(nearest_idx)
        aval_tidx_coord_list.pop(nearest_idx)
        return nearest_track


class SARLogic(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_logic_array_sync')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            w_n='nmos width',
            w_p='pmos width',
            pinfo='The MOSBasePlaceInfo object.',
            logic_array='Parameter for logic array',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
        )

    def draw_layout(self) -> None:
        # setup floorplan
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        w_p = self.params['w_p']
        w_n = self.params['w_n']
        self.draw_base(pinfo)

        logic_array_param: ImmutableSortedDict[str, Any] = self.params['logic_array']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        # compute track locations
        tr_manager = self.tr_manager
        tr_w_vm_sig = tr_manager.get_width(vm_layer, 'sig')
        tr_w_xm_sig = tr_manager.get_width(xm_layer, 'sig')
        tr_w_ym_sig = tr_manager.get_width(ym_layer, 'sig')
        tr_sp_vm_sig = tr_manager.get_sep(vm_layer, ('sig', 'sig'))

        logic_array_param = logic_array_param.copy(append={'pinfo': pinfo, 'w_n': w_n, 'w_p': w_p})
        logic_array_master = self.new_template(SARLogicArray, params=logic_array_param)

        # Add clock buffer
        logic_ntile = logic_array_master.num_tile_rows
        logic_array = self.add_tile(logic_array_master, 0, 0)
        self.set_mos_size()

        nbits = len(logic_array_param['seg_dict']['logic_scale_list']) 
        lower_layer_routing = logic_array_master.lower_layer_routing
        rt_layer = vm_layer
        rt_tidx_start = self.arr_info.col_to_track(rt_layer, 0)
        rt_tidx_stop = self.arr_info.col_to_track(rt_layer, self.num_cols)
        tr_w_rt_clk = tr_manager.get_width(rt_layer, 'clk')
        tr_w_rt_sig = tr_manager.get_width(rt_layer, 'sig')
        tr_sp_rt_clk = tr_manager.get_sep(rt_layer, ('clk', 'clk'))
        tr_sp_rt_sig = tr_manager.get_sep(rt_layer, ('sig', 'sig'))

        # vertical comp will not cross output flops, lower coord uses supply to find lowest array height that matters
        lower_coord = self.grid.track_to_coord(hm_layer, logic_array.get_all_port_pins('VSS')[3].track_id.base_htr-logic_array.get_all_port_pins('VSS')[0].track_id.base_htr)
        if lower_layer_routing:
            middle_coord = self.arr_info.col_to_coord(self.num_cols // 2)
            rt_tidx_lim1 = self.arr_info.col_to_track(rt_layer, self.num_cols //2 -12)
            rt_tidx_lim2 = self.arr_info.col_to_track(rt_layer, self.num_cols //2 +12)
            rt_tidx_list1 = self.get_available_tracks(rt_layer, rt_tidx_start, rt_tidx_lim1, lower_coord,
                                                     self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig) 
            rt_tidx_list2 = self.get_available_tracks(rt_layer, rt_tidx_lim2, rt_tidx_stop, lower_coord,
                                                     self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)

            rt_tidx_coord_list1 = [self.grid.track_to_coord(rt_layer, x) for x in rt_tidx_list1]
            rt_tidx_coord_list2 = [self.grid.track_to_coord(rt_layer, x) for x in rt_tidx_list2]
            compp_rt_tidx = SARLogicArray.get_nearest_tidx(logic_array.get_pin('comp_n'), rt_tidx_list2, rt_tidx_coord_list2, middle_coord)
            compn_rt_tidx = SARLogicArray.get_nearest_tidx(logic_array.get_pin('comp_p'), rt_tidx_list1, rt_tidx_coord_list1, middle_coord)            

            comp_p_ym = self.connect_to_tracks(logic_array.get_all_port_pins('comp_p'), TrackID(rt_layer, compp_rt_tidx, tr_w_rt_sig),
                                               track_upper=self.bound_box.yh)
            comp_n_ym = self.connect_to_tracks(logic_array.get_all_port_pins('comp_n'), TrackID(rt_layer, compn_rt_tidx, tr_w_rt_sig),
                                               track_upper=self.bound_box.yh)
            self.add_pin('comp_p', comp_p_ym)
            self.add_pin('comp_n', comp_n_ym)
            for pname in ['rst', 'clk_out','comp_clk']:
                self.reexport(logic_array.get_port(pname))
            for pname in ['state']:
                [self.reexport(logic_array.get_port(f'{pname}<{idx}>')) for idx in range(nbits)]
            for idx in range(nbits):    
                for pname in ['dn', 'dp', 'dm']:
                    pin = logic_array.get_pin(f'{pname}<{idx}>')
                    rt_tidx_list = self.get_available_tracks(rt_layer, rt_tidx_start, rt_tidx_stop, pin.bound_box.yh,
                                                             self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                    rt_tidx_coord_list = [self.grid.track_to_coord(rt_layer, x) for x in rt_tidx_list]
                    _y_wire = self.connect_to_tracks(pin, TrackID(rt_layer,
                                                                  SARLogicArray.get_nearest_tidx(pin, rt_tidx_list,
                                                                                        rt_tidx_coord_list),
                                                                  tr_w_rt_sig), track_upper=self.bound_box.yh)
                    self.add_pin(f'{pname}<{idx}>', _y_wire, mode=PinMode.UPPER)
            if logic_array_param['has_pmos_sw']: #logic_array_master.sch_params['has_pmos_sw']:FIXME
                for pname in ['dn_b', 'dp_b']:
                    for idx in range(nbits):
                        pin = logic_array.get_pin(f'{pname}<{idx}>')
                        rt_tidx_list = self.get_available_tracks(rt_layer, rt_tidx_start, rt_tidx_stop,
                                                                 pin.bound_box.yh,
                                                                 self.bound_box.yh, width=tr_w_rt_sig, sep=tr_sp_rt_sig)
                        rt_tidx_coord_list = [self.grid.track_to_coord(rt_layer, x) for x in rt_tidx_list]
                        _y_wire = self.connect_to_tracks(pin, TrackID(rt_layer,
                                                                      SARLogicArray.get_nearest_tidx(pin, rt_tidx_list,
                                                                                            rt_tidx_coord_list),
                                                                      tr_w_rt_sig), track_upper=self.bound_box.yh)
                        self.add_pin(f'{pname}<{idx}>', _y_wire, mode=PinMode.UPPER)

        else:
            for pname in ['rst', 'comp_p', 'comp_n', 'clk_out']:
                self.reexport(logic_array.get_port(pname))
            for pname in ['dn', 'dp', 'dm', 'state']:
                [self.reexport(logic_array.get_port(f'{pname}<{idx}>')) for idx in range(nbits)]
            if logic_array_master.sch_params['has_pmos_sw']:
                for pname in ['dn_b', 'dp_b']:
                    [self.reexport(logic_array.get_port(f'{pname}<{idx}>')) for idx in range(nbits)]
        [self.reexport(logic_array.get_port(f'data_out<{idx}>')) for idx in range(nbits)]
        self.reexport(logic_array.get_port('VDD'))
        self.reexport(logic_array.get_port('VSS'))
        self.sch_params = logic_array_master.sch_params

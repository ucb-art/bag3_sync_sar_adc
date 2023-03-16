from typing import Any, Dict, Type, Optional, Mapping, Union

from bag.design.database import ModuleDB, Module
# from bag.layout.routing.base import TrackID
from bag.layout.routing.base import TrackManager
from bag.layout.template import TemplateDB
from bag.util.immutable import Param
from bag.util.math import HalfInt
from pybag.enum import MinLenMode, RoundMode
from xbase.layout.enum import MOSWireType, SubPortMode
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from .util.template import TrackIDZL as TrackID, TemplateBaseZL
from .digital import NAND2Core, InvChainCore, NOR2Core, InvCore
from .util.util import fill_tap_intv, fill_tap


class NORNANDDynCore(MOSBase):
    """A single inverter.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    def get_schematic_class(self) -> Optional[Type[Module]]:
        if self.params['nand']:
            # noinspection PyTypeChecker
            return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'nand_dyn')
        else:
            # noinspection PyTypeChecker
            return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'nor_dyn')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='segments of transistors',
            w_p='pmos width, can be list or integer if all widths are the same.',
            w_n='pmos width, can be list or integer if all widths are the same.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            vertical_out='True to draw output on vertical metal layer.',
            vertical_sup='True to have supply unconnected on conn_layer.',
            vertical_in='False to not draw the vertical input wire when is_guarded = True.',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            tr_manager='override track manager',
            nand='False to generate nor gate',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            is_guarded=False,
            sig_locs={},
            vertical_out=True,
            vertical_sup=False,
            vertical_in=True,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            tr_manager=None,
            nand=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        grid = self.grid

        seg_dict: Dict = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        vertical_in: bool = self.params['vertical_in']

        is_nand = self.params['nand']
        if is_nand:
            raise ValueError("Not implemented yet")

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # Placement
        seg_pu, seg_pd, seg_in = seg_dict['pu'], seg_dict['pd'], seg_dict['in']

        # Total number of cols
        tot_cols = max(seg_pu, seg_pd + 2 * seg_in)

        col_pu, col_pd = (tot_cols - seg_pu) // 2, seg_in
        pu = self.add_mos(ridx_p, col_pu, seg_pu, w=w_p, g_on_s=bool(col_pu & 1))
        pd = self.add_mos(ridx_n, col_pd, seg_pd, w=w_n, g_on_s=bool(col_pd & 1))
        ina = self.add_mos(ridx_n, 0, seg_in, w=w_n)
        inb = self.add_mos(ridx_n, col_pd + seg_pd, seg_in, w=w_p)

        self.set_mos_size()

        # get wire_indices from sig_locs
        tr_manager = self.params['tr_manager'] if self.params['tr_manager'] else self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        nout_tidx = sig_locs.get('nout', self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=0))
        pout_tidx = sig_locs.get('pout', self.get_track_index(ridx_p, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=-1))
        nout_tid = TrackID(hm_layer, nout_tidx, tr_w_h)
        pout_tid = TrackID(hm_layer, pout_tidx, tr_w_h)

        pout = self.connect_to_tracks(pu.d, pout_tid, min_len_mode=mlm.get('pout', MinLenMode.NONE))
        nout = self.connect_to_tracks([ina.d, inb.d], nout_tid, min_len_mode=mlm.get('nout', MinLenMode.NONE))

        ns_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=1)
        _ns = self.connect_to_tracks([ina.s, inb.s, pd.s], ns_tid)

        if vertical_out:
            vm_tidx = sig_locs.get('out', grid.coord_to_track(vm_layer, max(pout.middle, nout.middle),
                                                              mode=RoundMode.NEAREST))
            vm_tid = TrackID(vm_layer, vm_tidx, width=tr_w_v)
            self.add_pin('out', self.connect_to_tracks([pout, nout], vm_tid))
        else:
            self.add_pin('out', [pout, nout], connect=True)

        default_clk_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        clk_tidx = sig_locs.get('clk', default_clk_tidx)
        clk_warr = self.connect_to_tracks([pd.g, pu.g], TrackID(hm_layer, clk_tidx, width=tr_w_h),
                                          min_len_mode=mlm.get('nin', MinLenMode.NONE))

        ina_tidx = sig_locs.get('ina', self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1))
        inb_tidx = sig_locs.get('inb', self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0))
        ina, inb = self.connect_differential_tracks(ina.g, inb.g, hm_layer, ina_tidx, inb_tidx, width=tr_w_h)

        self.add_pin('clk', clk_warr)
        self.add_pin('in<0>', ina)
        self.add_pin('in<1>', inb)

        if vertical_sup:
            self.add_pin('VDD', pu.s, connect=True)
            self.add_pin('VSS', pd.d, connect=True)
        else:
            xr = self.bound_box.xh
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sup')
            vdd = self.connect_to_tracks(pu.d, ps_tid, track_lower=0, track_upper=xr)
            vss = self.connect_to_tracks(pd.d, ns_tid, track_lower=0, track_upper=xr)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        default_wp = self.place_info.get_row_place_info(ridx_p).row_info.width
        default_wn = self.place_info.get_row_place_info(ridx_n).row_info.width
        thp = self.place_info.get_row_place_info(ridx_p).row_info.threshold
        thn = self.place_info.get_row_place_info(ridx_n).row_info.threshold
        lch = self.place_info.lch
        self.sch_params = dict(
            seg_pd=seg_pd,
            seg_pu=seg_pu,
            seg_in=seg_in,
            lch=lch,
            w_p=default_wp if w_p == 0 else w_p,
            w_n=default_wn if w_n == 0 else w_n,
            th_n=thn,
            th_p=thp,
        )
        #


class MuxPGwBuf(MOSBase):
    """A single inverter.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    def get_schematic_class(self) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'mux_pg')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='segments of transistors',
            w_p='pmos width, can be list or integer if all widths are the same.',
            w_n='pmos width, can be list or integer if all widths are the same.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            vertical_out='True to draw output on vertical metal layer.',
            vertical_sup='True to have supply unconnected on conn_layer.',
            vertical_in='False to not draw the vertical input wire when is_guarded = True.',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            tr_manager='override track manager',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            is_guarded=False,
            sig_locs={},
            vertical_out=True,
            vertical_sup=False,
            vertical_in=True,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            tr_manager=None,
            nand=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        grid = self.grid

        seg_dict: Dict = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        vertical_sup: bool = self.params['vertical_sup']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # Placement
        seg_inv, seg_p, seg_n = seg_dict.get('inv', 0), seg_dict['p'], seg_dict['n']
        cur_col = 0

        en_tidx = sig_locs.get('en', self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1))
        enb_tidx = sig_locs.get('enb', self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0))
        if seg_inv:
            inv_sig_locs = dict(nin=en_tidx)
            inv_template = self.new_template(InvCore, params=dict(pinfo=self.params['pinfo'], w_n=w_n, w_p=w_p,
                                                                  ridx_n=ridx_n, ridx_p=ridx_p, seg=seg_inv,
                                                                  vertical_sup=vertical_sup, vertical_out=True,
                                                                  sig_locs=inv_sig_locs))
            inv = self.add_tile(inv_template, 0, 0)
            cur_col += inv_template.num_cols + self.min_sep_col
            self.reexport(inv.get_port('VDD'))
            self.reexport(inv.get_port('VSS'))
        else:
            inv, inv_template = None, None

        cur_col += cur_col & 1

        # setup tr manager
        tr_manager = self.params['tr_manager'] if self.params['tr_manager'] else self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')

        # Total number of cols
        seg_pg = max(seg_n, seg_p)
        pa = self.add_mos(ridx_p, cur_col, seg_p, w=w_p, g_on_s=True)
        na = self.add_mos(ridx_n, cur_col, seg_n, w=w_p)
        cur_col += seg_pg + self.min_sep_col
        out_col = cur_col - self.min_sep_col // 2
        ina_col = out_col - seg_pg // 2 - self.min_sep_col // 2
        inb_col = out_col + seg_pg // 2 + self.min_sep_col // 2

        pb = self.add_mos(ridx_p, cur_col, seg_p, w=w_p, g_on_s=True)
        nb = self.add_mos(ridx_n, cur_col, seg_n, w=w_p)
        sig_vm_sp_col = self.arr_info.track_to_col(vm_layer, self.grid.get_sep_tracks(vm_layer, tr_w_v, tr_w_v))
        if abs(ina_col - cur_col) < sig_vm_sp_col or abs(inb_col - cur_col) < sig_vm_sp_col:
            raise ValueError("No enough space for in, out vm routing")

        self.set_mos_size()

        # get wire_indices from sig_locs
        nout_tidx = sig_locs.get('nout', self.get_track_index(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=0))
        pout_tidx = sig_locs.get('pout', self.get_track_index(ridx_p, MOSWireType.DS,
                                                              wire_name='sig', wire_idx=-1))
        nin_tidx = sig_locs.get('nin', self.get_track_index(ridx_n, MOSWireType.DS,
                                                            wire_name='sig', wire_idx=1))
        pin_tidx = sig_locs.get('pin', self.get_track_index(ridx_p, MOSWireType.DS,
                                                            wire_name='sig', wire_idx=-2))

        nout = self.connect_to_tracks([pa.s, pb.s], TrackID(hm_layer, pout_tidx, tr_w_h))
        pout = self.connect_to_tracks([na.s, nb.s], TrackID(hm_layer, nout_tidx, tr_w_h))
        nin_a = self.connect_to_tracks(na.d, TrackID(hm_layer, nin_tidx, tr_w_h))
        nin_b = self.connect_to_tracks(nb.d, TrackID(hm_layer, nin_tidx, tr_w_h))
        pin_a = self.connect_to_tracks(pa.d, TrackID(hm_layer, pin_tidx, tr_w_h))
        pin_b = self.connect_to_tracks(pb.d, TrackID(hm_layer, pin_tidx, tr_w_h))
        self.add_pin('nd<0>', nin_a, show=False)
        self.add_pin('pd<0>', pin_a, show=False)
        self.add_pin('nd<1>', nin_b, show=False)
        self.add_pin('pd<1>', nin_b, show=False)
        if seg_inv:
            en_hm = self.connect_to_track_wires([na.g, pb.g], inv.get_pin('nin'))
            enb_hm = self.connect_to_tracks([nb.g, pa.g], TrackID(hm_layer, enb_tidx, tr_w_h))
            self.connect_to_track_wires(enb_hm, inv.get_pin('out'))
            self.add_pin('en', en_hm)
        else:
            en_hm = self.connect_to_tracks([na.g, nb.g], TrackID(hm_layer, en_tidx, tr_w_h))
            enb_hm = self.connect_to_tracks([nb.g, pa.g], TrackID(hm_layer, enb_tidx, tr_w_h))
            self.add_pin('en', en_hm)
            self.add_pin('enb', enb_hm)

        ina_vm_tidx = self.arr_info.col_to_track(vm_layer, ina_col, RoundMode.NEAREST)
        inb_vm_tidx = self.arr_info.col_to_track(vm_layer, inb_col, RoundMode.NEAREST)
        out_vm_tidx = self.arr_info.col_to_track(vm_layer, out_col, RoundMode.NEAREST)

        ina_vm = self.connect_to_tracks([nin_a, pin_a], TrackID(vm_layer, ina_vm_tidx, tr_w_v))
        inb_vm = self.connect_to_tracks([nin_b, pin_b], TrackID(vm_layer, inb_vm_tidx, tr_w_v))
        out_vm = self.connect_to_tracks([nout, pout], TrackID(vm_layer, out_vm_tidx, tr_w_v))

        self.add_pin('out', out_vm)
        self.add_pin('d<0>', ina_vm)
        self.add_pin('d<1>', inb_vm)

        default_wp = self.place_info.get_row_place_info(ridx_p).row_info.width
        default_wn = self.place_info.get_row_place_info(ridx_n).row_info.width
        thp = self.place_info.get_row_place_info(ridx_p).row_info.threshold
        thn = self.place_info.get_row_place_info(ridx_n).row_info.threshold
        lch = self.place_info.lch
        pg_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            w_p=default_wp if w_p == 0 else w_p,
            w_n=default_wn if w_n == 0 else w_n,
            th_n=thn,
            th_p=thp,
            lch=lch,
        )
        self.sch_params = dict(
            inv_params=inv_template.sch_params if seg_inv else None,
            pg_params=pg_params
        )
        

class SARAsyncClkSimple(MOSBase, TemplateBaseZL):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._middle_col = 0

    @property
    def middle_col(self):
        return self._middle_col

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_async_clk_core_simple')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            tr_manager='',
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Number of segments.',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row',
            ridx_p='index for pmos row',
            substrate_row='True to add substrate row'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1,
            substrate_row=True,
            tr_manager=None,
        )

    def draw_layout(self) -> None:
        # setup floorplan
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        gate_tr_manager = self.params['tr_manager']
        tr_manager = TrackManager(self.grid, tr_widths=gate_tr_manager['tr_widths'],
                                  tr_spaces=gate_tr_manager['tr_spaces']) if gate_tr_manager else self.tr_manager

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        w_p: int = self.params['w_p']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        substrate_row: bool = self.params['substrate_row']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        seg_nand_en = seg_dict['nand_en']
        seg_nor_en = seg_dict['nor_en']
        seg_comp_buf = seg_dict['comp_buf']
        seg_clk_buf = seg_dict['clk_buf']
        seg_start_buf = seg_dict['start_buf']
        seg_delay_buf = seg_dict['delay_buf']
        seg_nor_dyn = seg_dict['nor_dyn']
        seg_mux = seg_dict['mux']
        min_sep = self.min_sep_col
        min_sep += min_sep & 1

        # compute track locations
        ng0_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        ng2_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)

        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)

        gate_params_temp_dict = dict(pinfo=pinfo, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p, tr_manager=tr_manager)
        nand_en_params = dict(seg=seg_nand_en, vertical_sup=substrate_row, **gate_params_temp_dict, vertical_out=False,
                              sig_locs={'nin0': ng0_tidx, 'nin1': ng1_tidx, 'nout': nd0_tidx, 'pout': pd1_tidx})
        nor_en_params = dict(seg=seg_nor_en, **gate_params_temp_dict, vertical_sup=substrate_row,
                             sig_locs={'nin0': pg0_tidx, 'nin1': pg1_tidx, 'nout': nd0_tidx, 'pout': pd1_tidx})
        comp_buf_params = dict(seg_list=seg_comp_buf, vertical_sup=substrate_row, **gate_params_temp_dict,
                               sig_locs={'nin0': pg1_tidx, 'nin1': pg0_tidx}, dual_output=True)
        clk_buf_params = dict(seg_list=seg_clk_buf, vertical_sup=substrate_row, **gate_params_temp_dict,
                              sig_locs={'nin0': ng2_tidx, 'nin1': ng1_tidx}, dual_output=False)
        start_buf_params = dict(seg_list=seg_start_buf, vertical_sup=substrate_row, **gate_params_temp_dict,
                                sig_locs={'nin0': ng2_tidx, 'nin1': ng1_tidx, 'nout': nd0_tidx, 'pout': pd1_tidx},
                                dual_output=False)
        delay_buf_params = dict(seg_list=seg_delay_buf, vertical_sup=substrate_row, **gate_params_temp_dict,
                                sig_locs={'nin0': ng2_tidx, 'nin1': ng1_tidx, 'nout': nd0_tidx, 'pout': pd1_tidx},
                                dual_output=False)
        nor_dyn_params = dict(seg_dict=seg_nor_dyn, **gate_params_temp_dict, vertical_sup=substrate_row)
        mux_params = dict(seg_dict=seg_mux, **gate_params_temp_dict, vertical_sup=substrate_row)

        nand_en_master = self.new_template(NAND2Core, params=nand_en_params)
        nor_en_master = self.new_template(NOR2Core, params=nor_en_params)
        comp_buf_master = self.new_template(InvChainCore, params=comp_buf_params)
        start_buf_master = self.new_template(InvChainCore, params=start_buf_params)
        delay_buf_master = self.new_template(InvChainCore, params=delay_buf_params)
        clk_buf_master = self.new_template(InvChainCore, params=clk_buf_params)

        nor_dyn_master = self.new_template(NORNANDDynCore, params=nor_dyn_params)
        mux_master = self.new_template(MuxPGwBuf, params=mux_params)

        # Place gates
        cur_col = 0
        second_row_idx = 2 if substrate_row else 1
        start_buf = self.add_tile(start_buf_master, second_row_idx, cur_col)
        cur_col += min_sep + start_buf_master.num_cols
        cur_col += cur_col & 1
        nor_en = self.add_tile(nor_en_master, second_row_idx, cur_col)
        col_logic_buf = max(clk_buf_master.num_cols, cur_col + nor_en_master.num_cols)
        cur_col += min_sep + nor_en_master.num_cols
        cur_col += cur_col & 1

        clk_buf = self.add_tile(clk_buf_master, 0, col_logic_buf, flip_lr=True)
        col_nor_dyn = cur_col
        max_nor_nand = max(nor_dyn_master.num_cols, nand_en_master.num_cols)
        # middle col needs to be odd number
        self._middle_col = col_nor_dyn + max_nor_nand//2
        self._middle_col += 0 if self._middle_col & 1 else 1

        nor_dyn = self.add_tile(nor_dyn_master, second_row_idx,
                                col_nor_dyn + (max_nor_nand - nor_dyn_master.num_cols) // 2 + (self._middle_col & 1))
        nand_en = self.add_tile(nand_en_master, 0, col_nor_dyn + (max_nor_nand - nand_en_master.num_cols) // 2)

        cur_col = col_nor_dyn + max(nor_dyn_master.num_cols, nand_en_master.num_cols) + min_sep
        cur_col += cur_col & 1

        comp_buf = self.add_tile(comp_buf_master, second_row_idx, cur_col)
        mux = self.add_tile(mux_master, 0, cur_col+mux_master.num_cols, flip_lr=True)
        cur_col += mux_master.num_cols + min_sep
        cur_col += cur_col & 1
        delay_buf = self.add_tile(delay_buf_master, 0, cur_col + delay_buf_master.num_cols, flip_lr=True)

        self.set_mos_size()

        # Route between gates:
        tr_w_vm = tr_manager.get_width(vm_layer, 'sig')
        tr_w_vm_clk = tr_manager.get_width(vm_layer, 'clk')
        # -- NOR enable gate
        start_vm_tidx = self.grid.coord_to_track(vm_layer, start_buf.bound_box.xl, RoundMode.LESS)
        start_vm = self.connect_to_tracks(start_buf.get_pin('nin'),
                                          TrackID(vm_layer, start_vm_tidx, tr_w_vm_clk, grid=self.grid))
        self.connect_to_track_wires(start_buf.get_pin('out'), nor_en.get_pin('nin<0>'))
        nor_stop_vm_tidx = self.grid.coord_to_track(vm_layer, nor_en.bound_box.xl, RoundMode.NEAREST)
        nor_stop_vm = self.connect_to_tracks(nor_en.get_pin('nin<1>'), TrackID(vm_layer, nor_stop_vm_tidx, tr_w_vm_clk, grid=self.grid))

        # Connect to logic clock
        logic_clk_in_vm_tidx = self.grid.coord_to_track(vm_layer, nand_en.bound_box.xl, RoundMode.NEAREST)
        self.connect_to_tracks([clk_buf.get_pin('in'), nand_en.get_pin('nout'), nand_en.get_pin('pout')],
                               TrackID(vm_layer, logic_clk_in_vm_tidx, tr_w_vm_clk, grid=self.grid))

        # -- NAND enable gate
        self.connect_to_track_wires(nor_dyn.get_pin('out'), nand_en.get_pin('nin<1>'))
        self.connect_to_track_wires(nor_en.get_pin('out'), nand_en.get_pin('nin<0>'))

        # Connect to mux
        delay_buf_in_vm_tidx = self.grid.coord_to_track(vm_layer, delay_buf.bound_box.xh, RoundMode.NEAREST)
        delay_buf_in_vm = self.connect_to_tracks(delay_buf.get_pin('nin'),
                                                 TrackID(vm_layer, delay_buf_in_vm_tidx, tr_w_vm))
        mux_clk_in_vm_tidx = self.grid.coord_to_track(vm_layer, nand_en.bound_box.xh, RoundMode.NEAREST)
        mux_clk_in_vm = self.connect_to_tracks([nand_en.get_pin('nout'), nand_en.get_pin('pout')],
                                               TrackID(vm_layer, mux_clk_in_vm_tidx, tr_w_vm))
        # self.connect_to_track_wires([mux.get_pin('nd<1>'), mux.get_pin('pd<1>')], delay_buf.get_pin('out'))
        xm_layer = vm_layer + 1
        tr_w_xm = tr_manager.get_width(xm_layer, 'sig')
        tr_w_xm_clk = tr_manager.get_width(xm_layer, 'clk')
        tr_w_xm_ctrl = tr_manager.get_width(xm_layer, 'ctrl')
        mux_d_xm_tidx = self.grid.coord_to_track(xm_layer, mux_clk_in_vm.middle, RoundMode.NEAREST)
        mux_xm_tidx = tr_manager.get_next_track(xm_layer, mux_d_xm_tidx, 'sig', 'sig')
        self.connect_to_tracks([mux.get_pin('d<1>'), mux_clk_in_vm, delay_buf_in_vm],
                               TrackID(xm_layer, mux_xm_tidx, tr_w_xm))
        self.connect_to_tracks([mux.get_pin('d<0>'), delay_buf.get_pin('out')],
                               TrackID(xm_layer, mux_d_xm_tidx, tr_w_xm))
        mux_enable_vm_tidx = self.grid.coord_to_track(vm_layer, mux.bound_box.xl, RoundMode.NEAREST)
        mux_enable_vm = self.connect_to_tracks(mux.get_pin('en'), TrackID(vm_layer, mux_enable_vm_tidx, tr_w_vm))

        ctrl_ext_clk_xm_tidx = mux_xm_tidx - self.get_track_sep(xm_layer, tr_w_xm_clk, tr_w_xm)
        ctrl_ext_clk_xm = self.connect_to_tracks(mux_enable_vm,
                                                 TrackID(xm_layer, ctrl_ext_clk_xm_tidx, tr_w_xm_clk, grid=self.grid),
                                                 min_len_mode=MinLenMode.MIDDLE)

        # conenct to comp clk
        self.connect_to_track_wires(mux.get_pin('out'), comp_buf.get_pin('nin'))

        # === Some interfaces to other blocks
        # Start/stop
        _, start_stop_xm_locs = tr_manager.place_wires(xm_layer, ['ctrl', 'ctrl'], center_coord=start_vm.middle)
        start_xm = self.connect_to_tracks(start_vm, TrackID(xm_layer, start_stop_xm_locs[0], tr_w_xm_ctrl),
                                          track_lower=self.bound_box.xl)
        stop_xm = self.connect_to_tracks(nor_stop_vm, TrackID(xm_layer, start_stop_xm_locs[1], tr_w_xm_ctrl),
                                         track_lower=self.bound_box.xl)

        # Clock
        self.connect_to_track_wires(nor_dyn.get_pin('clk'), comp_buf.get_pin('outb'))
        # logic_clk_xm_tidx = self.grid.coord_to_track(xm_layer, clk_buf.get_pin('outb').middle, RoundMode.NEAREST)
        comp_clk_xm_tidx = self.grid.coord_to_track(xm_layer, comp_buf.get_pin('outb').middle, RoundMode.NEAREST)
        logic_clk_xm_tidx = self.get_track_sep(xm_layer, tr_w_xm_clk, tr_w_xm_clk)
        logic_clk_xm = self.connect_to_tracks(clk_buf.get_pin('outb'),
                                              TrackID(xm_layer, logic_clk_xm_tidx+ctrl_ext_clk_xm_tidx, tr_w_xm_clk,
                                                      grid=self.grid),
                                              min_len_mode=MinLenMode.MIDDLE)
        comp_clk_xm = self.connect_to_tracks(comp_buf.get_pin('outb'),
                                             TrackID(xm_layer, comp_clk_xm_tidx, tr_w_xm_clk, grid=self.grid),
                                             min_len_mode=MinLenMode.MIDDLE)

        ym_layer = xm_layer + 1
        tr_w_ym_clk = tr_manager.get_width(ym_layer, 'clk')
        logic_clk_ym_tidx = self.grid.coord_to_track(ym_layer, logic_clk_xm.lower, RoundMode.LESS)
        logic_clk_ym = self.connect_to_tracks(logic_clk_xm, TrackID(ym_layer, logic_clk_ym_tidx, tr_w_ym_clk,
                                                                    grid=self.grid), min_len_mode=MinLenMode.MIDDLE)
        vdd_gate_list, vss_gate_list = [], []

        # Fill
        row0, row1 = fill_tap(self, 0, port_mode=SubPortMode.EVEN, extra_margin=True)
        row3, row2 = fill_tap(self, 2, port_mode=SubPortMode.EVEN, extra_margin=True)

        for inst in [nand_en, nor_en, mux, nor_dyn, clk_buf, comp_buf, start_buf, delay_buf]:
            vdd_gate_list.extend(inst.get_all_port_pins('VDD'))
            vss_gate_list.extend(inst.get_all_port_pins('VSS'))
        flip_np = False # True to fliip np rows, vdd outward
        if substrate_row:
            tap = self.add_substrate_contact(0, 0, seg=self.num_cols,  tile_idx=1)
            sup_tap = self.connect_to_tracks(tap, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=1))
            self.connect_to_track_wires(vdd_gate_list + row0 + row3 if flip_np else vss_gate_list + row0 + row3,
                                        sup_tap)
            sup_conn = vss_gate_list if flip_np else vdd_gate_list
            sup_conn += row0+row3 if flip_np else row1+row2
            self.add_pin('VSS_bot' if flip_np else 'VDD_bot',
                         [w for w in sup_conn if w.middle < self.bound_box.h//2],
                         show=self.show_pins, label='VSS' if flip_np else 'VDD', connect=True)
            self.add_pin('VSS_top' if flip_np else 'VDD_top',
                         [w for w in sup_conn if w.middle > self.bound_box.h//2],
                         show=self.show_pins, label='VSS' if flip_np else 'VDD', connect=True)
            self.add_pin('VDD' if flip_np else 'VSS', sup_tap,
                         show=self.show_pins, connect=True)
        else:
            vdd_hm = self.connect_wires(vdd_gate_list, lower=self.bound_box.xl, upper=self.bound_box.xh)
            vss_hm = self.connect_wires(vss_gate_list, lower=self.bound_box.xl, upper=self.bound_box.xh)
            self.add_pin('VSS', vss_hm, show=self.show_pins, connect=True)
            self.add_pin('VDD', vdd_hm, show=self.show_pins, connect=True)

        # Add pins
        self.add_pin('start', start_xm)
        self.add_pin('stop', stop_xm)
        self.add_pin('ctrl_ext_clk', ctrl_ext_clk_xm)
        self.add_pin('comp_p', nor_dyn.get_pin('in<0>'))
        self.add_pin('comp_n', nor_dyn.get_pin('in<1>'))
        self.add_pin('clk_out', comp_clk_xm)
        self.add_pin('logic_clk', logic_clk_ym)

        #
        self.sch_params = dict(
            nor_en=nor_en_master.sch_params,
            nand_en=nand_en_master.sch_params,
            compbuf=comp_buf_master.sch_params,
            clkbuf=clk_buf_master.sch_params,
            startbuf=start_buf_master.sch_params,
            delaybuf=delay_buf_master.sch_params,
            mux=mux_master.sch_params,
            nor_dyn=nor_dyn_master.sch_params
        )

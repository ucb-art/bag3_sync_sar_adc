from queue import Empty
from typing import Any, Dict, Optional, Type

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.routing import TrackID
from bag.layout.template import TemplateDB
from bag.util.immutable import Param
from pybag.enum import RoundMode
from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase


class SamplerUnit(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg='segments dictionary.',
            seg_p='segments dictionary.',
            seg_n='segments dictionary.',
            ridx_n='',
            ridx_p='',
            sampler_type='',
            xcp_dummy='',
            xcp_metal='',
            w_n='',
            w_p='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg_n=0,
            seg_p=0,
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1,
            sampler_type='nmos',
            xcp_dummy=False,
            xcp_metal=True,
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_n = self.params['seg_n']
        seg_p = self.params['seg_p']
        seg = self.params['seg']
        ridx_n = self.params['ridx_n']
        ridx_p = self.params['ridx_p']
        w_n = self.params['w_n']
        w_p = self.params['w_p']
        sampler_type: str = self.params['sampler_type']

        # check xcp dummy setting
        xcp_metal = self.params['xcp_metal']
        xcp_dummy = self.params['xcp_dummy']
        if xcp_metal and xcp_dummy:
            raise ValueError('sar_samp/SamplerUnit: choose only metal or device dummy, cant enable both')

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        has_nmos = sampler_type == 'cmos' or sampler_type == 'nmos'
        has_pmos = sampler_type == 'cmos' or sampler_type == 'pmos'
        if has_pmos:
            raise NotImplementedError

        nsam = self.add_mos(ridx_n, 0, seg_n, w=w_n, tile_idx=0) if has_nmos else None
        psam = self.add_mos(ridx_p, 0, seg_p, w=w_p, tile_idx=0) if has_pmos else None
        max_sam_col = max(seg_n, seg_p)

        nd_conn, ns_conn = ([nsam.d], [nsam.s]) if has_nmos else ([], [])
        pd_conn, ps_conn = ([psam.d], [psam.s]) if has_pmos else ([], [])

        if xcp_dummy:
            ndum = self.add_mos(ridx_n - 1, max_sam_col, seg_n, w=w_n) if has_nmos else None
            pdum = self.add_mos(ridx_p + 1, max_sam_col, seg_p, w=w_p) if has_pmos else None
            nd_conn.append(ndum.d)
            ns_conn.append(ndum.s)
            pd_conn.append(pdum.d)
            ps_conn.append(pdum.s)

        if xcp_metal:
            pinfo, tile_yb, flip_tile = self.used_array.get_tile_info(0)
            if has_nmos:
                row_info, y0, orient = MOSBase.get_mos_row_info(pinfo, tile_yb, flip_tile, ridx_n - 1)
                if row_info.flip:
                    xcp_metal_upper, xcp_metal_lower = (y0 - row_info.ds_conn_y[0],
                                                        y0 - row_info.ds_conn_y[1])
                else:
                    xcp_metal_lower, xcp_metal_upper = (y0 + row_info.ds_conn_y[0], y0 + row_info.ds_conn_y[1])
                d_start_tidx = self.arr_info.col_to_track(self.conn_layer, 1)
                s_start_tidx = self.arr_info.col_to_track(self.conn_layer, 0)
                xcp_metal_d = self.add_wires(self.conn_layer, d_start_tidx, lower=xcp_metal_lower,
                                             upper=xcp_metal_upper, num=seg_n // 2, pitch=2)
                xcp_metal_s = self.add_wires(self.conn_layer, s_start_tidx, lower=xcp_metal_lower,
                                             upper=xcp_metal_upper, num=seg_n // 2 + 1, pitch=2)
                nd_conn.append(xcp_metal_d)
                d_num_tid_n = self.get_track_id(ridx_n - 1, MOSWireType.DS, 'sig', 1, tile_idx=0)
                dum_s_n = self.connect_to_tracks(xcp_metal_s, d_num_tid_n)
                self.add_pin('in_c', dum_s_n, connect=True)
            if has_pmos:
                row_info, y0, orient = MOSBase.get_mos_row_info(pinfo, tile_yb, flip_tile, ridx_p + 1)
                if row_info.flip:
                    xcp_metal_upper, xcp_metal_lower = (y0 - row_info.ds_conn_y[0],
                                                        y0 - row_info.ds_conn_y[1])
                else:
                    xcp_metal_lower, xcp_metal_upper = (y0 + row_info.ds_conn_y[0], y0 + row_info.ds_conn_y[1])
                xcp_metal_lower, xcp_metal_upper = psam.s.lower, psam.d.upper
                d_start_tidx = self.arr_info.col_to_track(self.conn_layer, max_sam_col + 1)
                s_start_tidx = self.arr_info.col_to_track(self.conn_layer, max_sam_col + 2)
                xcp_metal_d = self.add_wires(self.conn_layer, d_start_tidx, lower=xcp_metal_lower,
                                             upper=xcp_metal_upper, num=seg_p // 2 + 1)
                xcp_metal_s = self.add_wires(self.conn_layer, s_start_tidx, lower=xcp_metal_lower,
                                             upper=xcp_metal_upper, num=seg_p // 2)
                pd_conn.append(xcp_metal_d)
                d_num_tid_p = self.get_track_id(ridx_p + 1, MOSWireType.DS, 'sig', 1, tile_idx=0)
                dum_s_p = self.connect_to_tracks(xcp_metal_s, d_num_tid_p)
                self.add_pin('in_c', dum_s_p, connect=True)

        if has_nmos:
            en_tid_n = self.get_track_id(ridx_n, MOSWireType.G_MATCH, 'clk', 0, tile_idx=0)
            d_tid_n = self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 0, tile_idx=0)
            s_tid_n = self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 1, tile_idx=0)
            clk_n = self.connect_to_tracks(nsam.g, en_tid_n)
            d_n = self.connect_to_tracks(nd_conn, d_tid_n)
            s_n = self.connect_to_tracks(ns_conn, s_tid_n)
            self.add_pin('sam', clk_n)
            self.add_pin('in', s_n)
            self.add_pin('out', d_n)
        if has_pmos:
            en_tid_p = self.get_track_id(ridx_p, MOSWireType.G_MATCH, 'clk', 0)
            d_tid_p = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', 0)
            s_tid_p = self.get_track_id(ridx_p, MOSWireType.DS, 'sig', 1)
            clk_p = self.connect_to_tracks(psam.g, en_tid_p)
            d_p = self.connect_to_tracks(pd_conn, d_tid_p)
            s_p = self.connect_to_tracks(ps_conn, s_tid_p)
            self.add_pin('sam_b', clk_p)
            self.add_pin('in', s_p)
            self.add_pin('out', d_p)

        self.set_mos_size()
        sampler_unit_params = dict(
            lch=self.arr_info.lch,
        )
        if has_pmos:
            sampler_unit_params['seg_p'] = seg_p
            sampler_unit_params['w_p'] = w_p
            sampler_unit_params['th_p'] = self.place_info.get_row_place_info(ridx_p).row_info.threshold
            if xcp_dummy:
                sampler_unit_params['seg_p_dum'] = seg_p
        if has_nmos:
            sampler_unit_params['seg_n'] = seg_n
            sampler_unit_params['w_n'] = w_n
            sampler_unit_params['th_n'] = self.place_info.get_row_place_info(ridx_n).row_info.threshold
            if xcp_dummy:
                sampler_unit_params['seg_n_dum'] = seg_n

        self._sch_params = sampler_unit_params


class Sampler(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            m_list='',
            nbits='',
            sampler_unit_params='',
            unit_pinfo=True,
            pinfo='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            m_list=[],
            nbits=1,
            unit_pinfo=True,
            pinfo='',
        )

    def get_schematic_class(self) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'nmos_sampler_diff')

    def draw_layout(self):
        # Parse parameters
        if (self.params['unit_pinfo']):
            place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['sampler_unit_params']['pinfo'])
            sampler_master = self.new_template(SamplerUnit, params=self.params['sampler_unit_params'])
        else:
            place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
            sampler_master = self.new_template(SamplerUnit, params=self.params['sampler_unit_params'].copy(
                                                            append=dict(pinfo=self.params['pinfo'])))
        self.draw_base(place_info)

        # by default use binary segments, otherwise used passed m_list parameter
        sampler_params_list = []
        nbits = self.params['nbits']
        if self.params['m_list']:
            m_list = self.params['m_list']
        else:
            m_list = [2 ** idx for idx in range(nbits)]

        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col
        tap_vdd_list, tap_vss_list = [], []
        #self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=0)
        cur_col = tap_n_cols + tap_sep_col
        vm_col_l = cur_col
        sampler_list, sampler_list_list = [], []
        for s in m_list:
            sampler_sub_list = []
            for idx in range(s):
                sampler_sub_list.append(self.add_tile(sampler_master, 0, cur_col))
                cur_col += sampler_master.num_cols
            sampler_list.extend(sampler_sub_list)
            sampler_list_list.append(sampler_sub_list)

        vm_col_r = self.num_cols
        self.set_mos_size()

        in_warr = self.connect_wires([s.get_pin('in') for s in sampler_list])
        self.add_pin('in', in_warr)

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        tr_w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')

        # export output
        for idx, s_list in enumerate(sampler_list_list):
            unit_out_pins = []
            for s in s_list:
                out_vm_tid = self.grid.coord_to_track(vm_layer, s.get_pin('out').middle, mode=RoundMode.NEAREST)
                self.connect_to_tracks(s.get_pin('out'), TrackID(vm_layer, out_vm_tid, tr_w_sig_vm))
                unit_out_pins.append(s.get_pin('out'))
            self.connect_wires(unit_out_pins)
        # Collect vm tid list to export input
        vm_tid_list = []
        for idx in range(vm_col_l, vm_col_r + 1, 2):
            vm_tid_list.append(self.arr_info.col_to_track(vm_layer, idx, mode=RoundMode.NEAREST))
        tr_sp_sig_vm = tr_manager.get_sep(vm_layer, ('sig', 'sig'))
        vm_tid_list = self.get_available_tracks(vm_layer, self.arr_info.col_to_track(vm_layer, vm_col_l),
                                                self.arr_info.col_to_track(vm_layer, vm_col_r), self.bound_box.yl,
                                                self.bound_box.yh, sep=tr_sp_sig_vm, sep_margin=tr_sp_sig_vm,
                                                include_last=True)

        in_vm_list = [self.connect_to_tracks(in_warr, TrackID(vm_layer, _tid, tr_w_sig_vm))
                      for _tid in vm_tid_list]

        # Uncomment to connect dummies up to xm_layer
        # in_xm_tid = self.grid.coord_to_track(xm_layer, in_vm_list[0].middle, mode=RoundMode.NEAREST)
        # in_vm_list_ret = []
        # in_xm = self.connect_to_tracks(in_vm_list, TrackID(xm_layer, in_xm_tid, tr_w_sig_xm),
        #                                ret_wire_list=in_vm_list_ret)
        # out_vm_upper = in_vm_list_ret[0].upper

        for idx, s_list in enumerate(sampler_list_list):
            _out_vm_list = []
            for s in s_list:
                out_vm_tid = self.grid.coord_to_track(vm_layer, s.get_pin('out').middle, mode=RoundMode.NEAREST)
                # _out_vm_list.append(s.get_pin('out')) 
                # Uncomment (and replace prev line) to connect dummies up to xm layer
                _out_vm_list.append(self.connect_to_tracks(s.get_pin('out'),
                                                       TrackID(vm_layer, out_vm_tid, tr_w_sig_vm)))
            self.add_pin(f'out<{idx}>', _out_vm_list)

        if sampler_list[0].has_port('in_c'):
            inc_warr = self.connect_wires([s.get_pin('in_c') for s in sampler_list])
            self.add_pin('in_c', inc_warr)
            inc_vm_list = [self.connect_to_tracks(inc_warr, TrackID(vm_layer, _tid, tr_w_sig_vm))
                           for _tid in vm_tid_list]
            self.add_pin('in_c', inc_warr)

            # Uncomment (and replace pin assignment) to connect dummies up to xm_layer
            # inc_xm_tid = self.grid.coord_to_track(xm_layer, inc_vm_list[0].middle, mode=RoundMode.NEAREST)
            # inc_xm = self.connect_to_tracks(inc_vm_list, TrackID(xm_layer, inc_xm_tid, tr_w_sig_xm))
            # self.add_pin('in_c', inc_xm)

        if sampler_list[0].has_port('sam'):
            clk = self.connect_wires([s.get_pin('sam') for s in sampler_list])
            self.add_pin('sam', clk)

        if sampler_list[0].has_port('sam_b'):
            clkb = self.connect_wires([s.get_pin('sam_b') for s in sampler_list])
            self.add_pin('sam_b', clkb)
            clkb_vm_list = [self.connect_to_tracks(clkb, TrackID(vm_layer, _tid, tr_w_sig_vm))
                            for _tid in vm_tid_list]
            clkb_xm_tid = self.grid.coord_to_track(xm_layer, clkb_vm_list[0].middle, mode=RoundMode.NEAREST)
            clkb_xm = self.connect_to_tracks(clkb_vm_list, TrackID(xm_layer, clkb_xm_tid, tr_w_sig_xm))
            self.add_pin('sam_b', clkb_xm)

        self._sch_params = dict(
            m_list=m_list,
            **sampler_master.sch_params
        )

class Sampler_orig(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            m_list='',
            nbits='',
            sampler_unit_params='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            m_list=[],
            nbits=1,
        )

    def get_schematic_class(self) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'nmos_sampler_diff')

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['sampler_unit_params']['pinfo'])
        self.draw_base(place_info)

        sampler_master = self.new_template(SamplerUnit, params=self.params['sampler_unit_params'])

        # by default use binary segments
        sampler_params_list = []
        nbits = self.params['nbits']
        if self.params['m_list']:
            m_list = self.params['m_list']
        else:
            m_list = [2 ** idx for idx in range(nbits)]

        tap_n_cols = self.get_tap_ncol()
        tap_sep_col = self.sub_sep_col
        tap_vdd_list, tap_vss_list = [], []
        self.add_tap(0, tap_vdd_list, tap_vss_list, tile_idx=0)
        cur_col = tap_n_cols + tap_sep_col
        vm_col_l = cur_col
        sampler_list, sampler_list_list = [], []
        for s in m_list:
            sampler_sub_list = []
            for idx in range(s):
                sampler_sub_list.append(self.add_tile(sampler_master, 0, cur_col))
                cur_col += sampler_master.num_cols
            sampler_list.extend(sampler_sub_list)
            sampler_list_list.append(sampler_sub_list)

        vm_col_r = self.num_cols
        self.add_tap(self.num_cols + tap_n_cols + tap_sep_col, tap_vdd_list, tap_vss_list, tile_idx=0, flip_lr=True)

        self.set_mos_size()

        in_warr = self.connect_wires([s.get_pin('in') for s in sampler_list])
        self.add_pin('in', in_warr)

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        tr_w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')
        # export output

        for idx, s_list in enumerate(sampler_list_list):
            for s in s_list:
                out_vm_tid = self.grid.coord_to_track(vm_layer, s.get_pin('out').middle, mode=RoundMode.NEAREST)
                self.connect_to_tracks(s.get_pin('out'), TrackID(vm_layer, out_vm_tid, tr_w_sig_vm))

        # Collect vm tid list to export input
        vm_tid_list = []
        for idx in range(vm_col_l, vm_col_r + 1, 2):
            vm_tid_list.append(self.arr_info.col_to_track(vm_layer, idx, mode=RoundMode.NEAREST))
        tr_sp_sig_vm = tr_manager.get_sep(vm_layer, ('sig', 'sig'))
        vm_tid_list = self.get_available_tracks(vm_layer, self.arr_info.col_to_track(vm_layer, vm_col_l),
                                                self.arr_info.col_to_track(vm_layer, vm_col_r), self.bound_box.yl,
                                                self.bound_box.yh, sep=tr_sp_sig_vm, sep_margin=tr_sp_sig_vm,
                                                include_last=True)

        in_vm_list = [self.connect_to_tracks(in_warr, TrackID(vm_layer, _tid, tr_w_sig_vm))
                      for _tid in vm_tid_list]

        in_xm_tid = self.grid.coord_to_track(xm_layer, in_vm_list[0].middle, mode=RoundMode.NEAREST)
        in_vm_list_ret = []
        in_xm = self.connect_to_tracks(in_vm_list, TrackID(xm_layer, in_xm_tid, tr_w_sig_xm),
                                       ret_wire_list=in_vm_list_ret)

        out_vm_upper = in_vm_list_ret[0].upper
        for idx, s_list in enumerate(sampler_list_list):
            _out_vm_list = []
            for s in s_list:
                out_vm_tid = self.grid.coord_to_track(vm_layer, s.get_pin('out').middle, mode=RoundMode.NEAREST)
                _out_vm_list.append(self.connect_to_tracks(s.get_pin('out'),
                                                           TrackID(vm_layer, out_vm_tid, tr_w_sig_vm),
                                                           track_upper=out_vm_upper))
            self.add_pin(f'out<{idx}>', _out_vm_list)
        #self.add_pin('in', in_xm)

        if sampler_list[0].has_port('in_c'):
            inc_warr = self.connect_wires([s.get_pin('in_c') for s in sampler_list])
            self.add_pin('in_c', inc_warr)
            inc_vm_list = [self.connect_to_tracks(inc_warr, TrackID(vm_layer, _tid, tr_w_sig_vm))
                           for _tid in vm_tid_list]
            inc_xm_tid = self.grid.coord_to_track(xm_layer, inc_vm_list[0].middle, mode=RoundMode.NEAREST)
            inc_xm = self.connect_to_tracks(inc_vm_list, TrackID(xm_layer, inc_xm_tid, tr_w_sig_xm))
            self.add_pin('in_c', inc_xm)

        if sampler_list[0].has_port('sam'):
            clk = self.connect_wires([s.get_pin('sam') for s in sampler_list])
            self.add_pin('sam_b', clk)
            clk_vm_list = [self.connect_to_tracks(clk, TrackID(vm_layer, _tid, tr_w_sig_vm))
                           for _tid in vm_tid_list]
            clk_xm_tid = self.grid.coord_to_track(xm_layer, clk_vm_list[0].middle, mode=RoundMode.NEAREST)
            clk_xm = self.connect_to_tracks(clk_vm_list, TrackID(xm_layer, clk_xm_tid, tr_w_sig_xm))
            self.add_pin('sam', clk_xm)
        if sampler_list[0].has_port('sam_b'):
            clkb = self.connect_wires([s.get_pin('sam_b') for s in sampler_list])
            self.add_pin('sam_b', clkb)
            clkb_vm_list = [self.connect_to_tracks(clkb, TrackID(vm_layer, _tid, tr_w_sig_vm))
                            for _tid in vm_tid_list]
            clkb_xm_tid = self.grid.coord_to_track(xm_layer, clkb_vm_list[0].middle, mode=RoundMode.NEAREST)
            clkb_xm = self.connect_to_tracks(clkb_vm_list, TrackID(xm_layer, clkb_xm_tid, tr_w_sig_xm))
            self.add_pin('sam_b', clkb_xm)

        self._sch_params = dict(
            m_list=m_list,
            **sampler_master.sch_params
        )

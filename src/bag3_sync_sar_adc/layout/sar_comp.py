from itertools import chain
from typing import Mapping, Union, Any, Dict, Type, Optional, Tuple, cast

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.io import read_yaml
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB
from bag.util.immutable import Param, ImmutableSortedDict
from bag.util.importlib import import_class
from bag.util.math import HalfInt
from pybag.core import BBox, BBoxArray
from pybag.enum import RoundMode
from xbase.layout.enum import MOSWireType, SubPortMode, MOSType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from .sar_async_clkgen import SARAsyncClkSimple
from .digital import InvChainCore
from .util.template import TemplateBaseZL
from .util.util import connect_conn_dummy_rows, fill_conn_layer_intv, fill_tap_intv


class PreAmpHalf(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._has_ofst = False

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ridx_n='bottom nmos row index.',
            ridx_p='pmos row index.',
            has_ofst='True to add bridge switch.',
            vertical_out='True to connect outputs to vm_layer.',
            vertical_sup='True to connect outputs to vm_layer.',
            sig_locs='Optional dictionary of user defined signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            sig_locs={},
            ridx_n=0,
            ridx_p=-1,
            has_ofst=False,
            vertical_out=True,
            vertical_sup=True,
        )

    @property
    def has_ofst(self) -> bool:
        return self._has_ofst

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        vertical_sup: bool = self.params['vertical_sup']

        w_dict, th_dict = self._get_w_th_dict(ridx_n, ridx_p)

        seg_in = seg_dict['in']
        seg_tail = seg_dict['tail']
        seg_cas = seg_dict['cas']
        seg_load = seg_dict['load']
        seg_os = seg_dict.get('os', 0)
        self._has_ofst = bool(seg_os)

        w_in = w_dict['in']
        w_tail = w_dict['tail']
        w_cas = w_dict['cas']
        w_load = w_dict['load']
        w_os = w_dict.get('os', w_dict['in'])

        if seg_in & 1 or (seg_tail % 4 != 0) or seg_cas & 1 or seg_load & 1:
            raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        seg_tail = seg_tail // 2

        # placement
        ridx_in = ridx_n + 1
        ridx_cas = ridx_in + 1
        m_tail = self.add_mos(ridx_n, 0, seg_tail, w=w_tail)
        m_cas = self.add_mos(ridx_cas, 0, seg_cas, w=w_cas)
        m_load = self.add_mos(ridx_p, 0, seg_load, w=w_load)
        if seg_os:
            m_in = self.add_mos(ridx_in, 0, seg_in, w=w_in)
            m_os = self.add_mos(ridx_in, seg_in, seg_os, w=w_os)
            self.add_pin('os', m_os.g)
            tail_conn = [m_tail.d, m_in.d, m_os.d]
            mid_conn = [m_in.s, m_cas.s, m_os.s]
        else:
            m_in = self.add_mos(ridx_in, 0, seg_in, w=w_in)
            tail_conn = [m_tail.d, m_in.d]
            mid_conn = [m_in.s, m_cas.s]

        ng_tid = self.get_track_id(ridx_cas, MOSWireType.G, wire_name='sig', wire_idx=-1)
        mid_tid = self.get_track_id(ridx_cas, MOSWireType.DS, wire_name='sig')
        pg_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')
        nclk_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)
        tail_tid = self.get_track_id(ridx_in, MOSWireType.DS, wire_name='sig')

        vss_conn = m_tail.s
        clk_conn = m_tail.g

        # NOTE: force even number of columns to make sure VDD conn_layer wires are on even columns.
        ncol_tot = self.num_cols
        self.set_mos_size(num_cols=ncol_tot + (ncol_tot & 1))

        # routing
        conn_layer = self.conn_layer
        vm_layer = conn_layer + 2
        vm_w = self.tr_manager.get_width(vm_layer, 'sig')
        grid = self.grid

        tail = self.connect_to_tracks(tail_conn, tail_tid)
        out = self.connect_wires([m_cas.d, m_load.d])
        if vertical_sup:
            vdd = m_load.s
        else:
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sup')
            vdd = self.connect_to_tracks([m_load.s], vdd_tid)

        # -- Connect middle node --
        self.connect_to_tracks(mid_conn, mid_tid)

        if vertical_sup:
            vss = vss_conn
        else:
            vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            vss = self.connect_to_tracks(vss_conn, vss_tid)

        nclk = self.connect_to_tracks(clk_conn, nclk_tid)
        nvdd = self.connect_to_tracks(m_cas.g, ng_tid)
        pclk = self.connect_to_tracks(m_load.g, pg_tid)

        xclk = grid.track_to_coord(conn_layer, m_tail.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xclk, mode=RoundMode.GREATER_EQ)
        vm_tidx = sig_locs.get('clk', vm_tidx)
        clk_vm = self.connect_to_tracks([nclk, pclk], TrackID(vm_layer, vm_tidx, width=vm_w))

        self.add_pin('clk_vm', clk_vm)
        self.add_pin('tail', tail)
        self.add_pin('clk', nclk)
        self.add_pin('in', m_in.g)
        self.add_pin('out', out)

        self.add_pin('VSS', vss)
        self.add_pin('VDD', vdd, connect=True)
        self.add_pin('VDD', nvdd, connect=True)

        self.sch_params = dict(
            lch=self.arr_info.lch,
            seg_dict=seg_dict,
            w_dict=w_dict,
            th_dict=th_dict,
            has_ofst=self.has_ofst,
        )

    def _get_w_th_dict(self, ridx_n: int, ridx_p: int, ) \
            -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        th_ans = {}
        for name, row_idx in [('tail', ridx_n), ('in', ridx_n + 1), ('cas', ridx_n + 2),
                              ('load', ridx_p), ('os', ridx_n + 1)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class PreAmp(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._has_ofst = False

    @property
    def has_ofst(self) -> bool:
        return self._has_ofst

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'comp_preamp')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = PreAmpHalf.get_params_info()
        ans['even_center'] = 'True to force center column to be even.'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = PreAmpHalf.get_default_param_values()
        ans['even_center'] = False
        return ans

    def draw_layout(self):
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        even_center: bool = self.params['even_center']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']

        master: PreAmpHalf = self.new_template(PreAmpHalf, params=self.params)
        self.draw_base(master.draw_base_info)
        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        tr_w_vm = tr_manager.get_width(vm_layer, 'ana_sig')

        # placement
        nsep = self.min_sep_col
        nsep += (nsep & 1)
        if even_center and nsep % 4 == 2:
            nsep += 2

        nhalf = master.num_cols
        corel = self.add_tile(master, 0, nhalf, flip_lr=True)
        corer = self.add_tile(master, 0, nhalf + nsep)
        self.set_mos_size(num_cols=nsep + 2 * nhalf)

        # Routing
        # -- Get track index --
        ridx_in = ridx_n + 1
        ridx_cas = ridx_in + 1
        inn_tidx, hm_w = self.get_track_info(ridx_in, MOSWireType.G_MATCH, wire_name='sig', wire_idx=1)
        inp_tidx = self.get_track_index(ridx_in, MOSWireType.G_MATCH, wire_name='sig', wire_idx=2)
        outn_tidx = self.get_track_index(ridx_cas, MOSWireType.DS_MATCH, wire_name='sig', wire_idx=-1)
        outp_tidx = self.get_track_index(ridx_cas, MOSWireType.DS_MATCH, wire_name='sig', wire_idx=-2)

        inp, inn = self.connect_differential_tracks(corel.get_pin('in'), corer.get_pin('in'),
                                                    hm_layer, inp_tidx, inn_tidx, width=hm_w)
        if master.has_ofst:
            os_tidx = self.get_track_index(ridx_in, MOSWireType.G_MATCH, wire_name='sig', wire_idx=0)
            osp = self.connect_to_tracks(corel.get_pin('os'), TrackID(hm_layer, os_tidx, hm_w))
            osn = self.connect_to_tracks(corer.get_pin('os'), TrackID(hm_layer, os_tidx, hm_w))

            osp_vm_tidx = self.grid.coord_to_track(vm_layer, osp.middle, mode=RoundMode.NEAREST)
            osn_vm_tidx = self.grid.coord_to_track(vm_layer, osn.middle, mode=RoundMode.NEAREST)
            osp_vm = self.connect_to_tracks(osp, TrackID(vm_layer, osp_vm_tidx, tr_w_vm))
            osn_vm = self.connect_to_tracks(osn, TrackID(vm_layer, osn_vm_tidx, tr_w_vm))
            self.add_pin('osp', osp_vm)
            self.add_pin('osn', osn_vm)
        # outp, outn = self.connect_differential_tracks(corer.get_all_port_pins('out'),
        #                                               corel.get_all_port_pins('out'),
        #                                               hm_layer, outp_tidx, outn_tidx, width=hm_w)
        outp = self.connect_to_tracks(corer.get_all_port_pins('out'), TrackID(hm_layer, outp_tidx, hm_w))
        outn = self.connect_to_tracks(corel.get_all_port_pins('out'), TrackID(hm_layer, outn_tidx, hm_w))
        outp_vm_tidx = tr_manager.get_next_track(vm_layer, corer.get_pin('clk_vm').track_id.base_index, 'clk', 'ana_sig')
        outp_vm_tidx = max(outp_vm_tidx, corer.get_pin('clk_vm').track_id.base_index + sig_locs.get('out', 0))
        outn_vm_tidx = tr_manager.get_next_track(vm_layer, corel.get_pin('clk_vm').track_id.base_index, 'clk', 'ana_sig',
                                                 up=False)
        outn_vm_tidx = min(outn_vm_tidx, corel.get_pin('clk_vm').track_id.base_index - sig_locs.get('out', 0))
        if vertical_out:
            outp_vm, outn_vm = self.connect_differential_tracks(outp, outn, vm_layer, outp_vm_tidx, outn_vm_tidx,
                                                                width=tr_w_vm)
            self.add_pin('outp', outp_vm)
            self.add_pin('outn', outn_vm)
        else:
            self.add_pin('outp', outp, connect=True)
            self.add_pin('outn', outn, connect=True)

        clk = self.connect_wires([corel.get_pin('clk'), corer.get_pin('clk')])
        if vertical_sup:
            vss = [corel.get_pin('VSS'), corer.get_pin('VSS')]
            vdd = list(chain(corel.get_all_port_pins('VDD', layer=self.conn_layer),
                             corer.get_all_port_pins('VDD', layer=self.conn_layer)))
        else:
            vss = self.connect_wires([corel.get_pin('VSS'), corer.get_pin('VSS')])
            vdd = self.connect_wires(list(chain(corel.get_all_port_pins('VDD'), corer.get_all_port_pins('VDD'))))

        vdd_tx = self.connect_wires(list(chain(corel.get_all_port_pins('VDD', layer=self.conn_layer + 1),
                                               corer.get_all_port_pins('VDD', layer=self.conn_layer + 1))))

        self.add_pin('inp', inp)
        self.add_pin('inn', inn)

        self.add_pin('outp_hm', outp, hide=True)
        self.add_pin('outn_hm', outn, hide=True)

        self.connect_wires([corel.get_pin('tail'), corer.get_pin('tail')])
        self.add_pin('clk', clk)
        self.add_pin('VDD', vdd, connect=True)
        self.add_pin('VDD', vdd_tx, connect=True)
        self.add_pin('VSS', vss)
        self.reexport(corel.get_port('clk_vm'), net_name='clkl', hide=True)
        self.reexport(corer.get_port('clk_vm'), net_name='clkr', hide=True)

        self._has_ofst = master.has_ofst
        self.sch_params = master.sch_params


class HalfLatchHalf(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ridx_n='bottom nmos row index.',
            ridx_p='pmos row index.',
            vertical_out='True to connect outputs to vm_layer.',
            vertical_sup='True to connect outputs to vm_layer.',
            sig_locs='Optional dictionary of user defined signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_n=0,
            ridx_p=-1,
            vertical_out=True,
            vertical_sup=True,
            sig_locs={}
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']

        w_dict, th_dict = self._get_w_th_dict(ridx_n, ridx_p)

        seg_in = seg_dict['in']
        seg_cp = seg_dict['cp']
        seg_tail = seg_dict['tail']
        w_in = w_dict['in']
        w_cp = w_dict['cp']
        w_tail = w_dict['tail']

        if seg_in & 1 or (seg_tail % 4 != 0) or seg_cp & 1:
            raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        seg_tail = seg_tail // 2

        # placement
        ridx_cp = ridx_n + 1
        m_tail = self.add_mos(ridx_p, 0, seg_tail, w=w_tail)
        m_cp = self.add_mos(ridx_cp, 0, seg_cp, w=w_cp)
        m_in = self.add_mos(ridx_n, 0, seg_in, w=w_in, g_on_s=True)

        nd_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        pd_tid = self.get_track_id(ridx_cp, MOSWireType.DS, wire_name='sig', wire_idx=-2)
        tail_tid = self.get_track_id(ridx_cp, MOSWireType.DS, wire_name='sig', wire_idx=-1)

        pclk_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1)

        vss_conn = m_in.d
        tail_conn = m_tail.d
        clk_conn = m_tail.g

        ncol_tot = self.num_cols
        self.set_mos_size(num_cols=ncol_tot + (ncol_tot & 1))

        # routing
        conn_layer = self.conn_layer
        vm_layer = conn_layer + 2
        vm_w = self.tr_manager.get_width(vm_layer, 'ana_sig')
        grid = self.grid

        tail = self.connect_to_tracks([tail_conn, m_cp.d], tail_tid)
        if vertical_sup:
            vdd = m_tail.s
        else:
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sup')
            vdd = self.connect_to_tracks([m_tail.s], vdd_tid)

        if vertical_sup:
            vss = vss_conn
        else:
            vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            vss = self.connect_to_tracks(vss_conn, vss_tid)
        pclk = self.connect_to_tracks(clk_conn, pclk_tid)
        nout = self.connect_to_tracks(m_in.s, nd_tid)
        pout = self.connect_to_tracks(m_cp.s, pd_tid)

        xclk = grid.track_to_coord(conn_layer, m_tail.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xclk, mode=RoundMode.GREATER_EQ)
        vm_tidx = sig_locs.get('clk', vm_tidx)
        clk_vm = self.connect_to_tracks([pclk], TrackID(vm_layer, vm_tidx, width=vm_w))

        self.add_pin('clk_vm', clk_vm)

        xout = grid.track_to_coord(conn_layer, m_cp.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xout, mode=RoundMode.GREATER_EQ)
        vm_tidx = sig_locs.get('out', vm_tidx)

        if vertical_out:
            out_vm = self.connect_to_tracks([nout, pout], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('out_vm', out_vm)
        else:
            self.add_pin('nout', nout)
            self.add_pin('pout', pout)

        self.add_pin('VSS', vss)
        self.add_pin('VDD', vdd)
        self.add_pin('clk', pclk)
        self.add_pin('tail', tail)
        self.add_pin('in', m_in.g)
        self.add_pin('out', m_cp.g)

        self.sch_params = dict(
            lch=self.arr_info.lch,
            seg_dict=seg_dict,
            w_dict=w_dict,
            th_dict=th_dict,
        )

    def _get_w_th_dict(self, ridx_n: int, ridx_p: int, ) \
            -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        th_ans = {}
        for name, row_idx in [('in', ridx_n), ('cp', ridx_n + 1), ('tail', ridx_p)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        # w_ans['swm'] = w_ans['swo'] = w_ans['pfb']
        # th_ans['swm'] = th_ans['swo'] = th_ans['pfb']
        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class HalfLatch(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'comp_half_latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = HalfLatchHalf.get_params_info()
        ans['even_center'] = 'True to force center column to be even.'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = HalfLatchHalf.get_default_param_values()
        ans['even_center'] = False
        return ans

    def draw_layout(self):
        master: HalfLatchHalf = self.new_template(HalfLatchHalf, params=self.params)
        self.draw_base(master.draw_base_info)

        ridx_n: int = self.params['ridx_n']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        even_center: bool = self.params['even_center']

        # placement
        nsep = self.min_sep_col
        nsep += (nsep & 1)
        if even_center and nsep % 4 == 2:
            nsep += 2

        nhalf = master.num_cols
        corel = self.add_tile(master, 0, nhalf, flip_lr=True)
        corer = self.add_tile(master, 0, nhalf + nsep)
        self.set_mos_size(num_cols=nsep + 2 * nhalf)

        # routing
        ridx_cp = ridx_n + 1
        inn_tidx, hm_w = self.get_track_info(ridx_n, MOSWireType.G_MATCH, wire_name='sig', wire_idx=0)
        inp_tidx = self.get_track_index(ridx_n, MOSWireType.G_MATCH, wire_name='sig', wire_idx=-1)
        outn_tidx = self.get_track_index(ridx_cp, MOSWireType.G_MATCH, wire_name='sig', wire_idx=-1)
        outp_tidx = self.get_track_index(ridx_cp, MOSWireType.G_MATCH, wire_name='sig', wire_idx=0)

        hm_layer = self.conn_layer + 1
        inp, inn = self.connect_differential_tracks(corel.get_pin('in'), corer.get_pin('in'),
                                                    hm_layer, inp_tidx, inn_tidx, width=hm_w)
        self.add_pin('inp', inp)
        self.add_pin('inn', inn)

        outp, outn = self.connect_differential_tracks(corer.get_all_port_pins('out'),
                                                      corel.get_all_port_pins('out'),
                                                      hm_layer, outp_tidx, outn_tidx, width=hm_w)
        if vertical_out:
            outp_vm = corel.get_pin('out_vm')
            outn_vm = corer.get_pin('out_vm')
            self.connect_to_track_wires(outp, outp_vm)
            self.connect_to_track_wires(outn, outn_vm)
            self.add_pin('outp', outn_vm)
            self.add_pin('outn', outp_vm)
        else:
            self.add_pin('outp', outp, connect=True)
            self.add_pin('outn', outn, connect=True)
            self.add_pin('outp', corel.get_pin('pout'), connect=True)
            self.add_pin('outn', corer.get_pin('pout'), connect=True)

        self.add_pin('outp_hm', outp, hide=True)
        self.add_pin('outn_hm', outn, hide=True)

        clk = self.connect_wires([corel.get_pin('clk'), corer.get_pin('clk')])
        if vertical_sup:
            vss = [corel.get_pin('VSS'), corer.get_pin('VSS')]
            vdd = list(chain(corel.get_all_port_pins('VDD', layer=self.conn_layer),
                             corer.get_all_port_pins('VDD', layer=self.conn_layer)))
        else:
            vss = self.connect_wires([corel.get_pin('VSS'), corer.get_pin('VSS')])
            vdd = self.connect_wires(list(chain(corel.get_all_port_pins('VDD'), corer.get_all_port_pins('VDD'))))
        self.connect_wires([corel.get_pin('tail'), corer.get_pin('tail')])

        self.add_pin('clkb', clk)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)
        self.reexport(corel.get_port('clk_vm'), net_name='clkl', hide=True)
        self.reexport(corer.get_port('clk_vm'), net_name='clkr', hide=True)

        self.sch_params = master.sch_params


class DynLatchHalf(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ridx_n='bottom nmos row index.',
            ridx_p='pmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            flip_np='True to flip nmos and pmos',
            has_rst='True to add reset devices and connect tail to output of previous stage',
            vertical_sup='True to connect outputs to vm_layer.',
            vertical_out='True to connect outputs to vm_layer.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_n=0,
            ridx_p=-1,
            sig_locs={},
            has_rst=False,
            flip_np=False,
            vertical_out=True,
            vertical_sup=True,
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        has_rst: bool = self.params['has_rst']
        flip_np: bool = self.params['flip_np']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']

        ridx_tail = ridx_p if flip_np else ridx_n
        ridx_nfb = ridx_n if flip_np else ridx_n + 1
        ridx_pfb = ridx_p - 1 if flip_np else ridx_p
        ridx_in = ridx_nfb if flip_np else ridx_pfb

        w_dict, th_dict = self._get_w_th_dict(ridx_tail, ridx_nfb, ridx_pfb, ridx_in)
        seg_in = seg_dict['in']
        seg_nfb = seg_dict['nfb']
        seg_pfb = seg_dict['pfb']
        seg_tail = seg_dict['tail']
        w_in = w_dict['in']
        w_tail = w_dict['tail']
        w_nfb = w_dict['nfb']
        w_pfb = w_dict['pfb']

        if seg_in & 1 or (seg_tail % 4 != 0) or seg_nfb & 1 or seg_pfb & 1:
            raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        seg_tail = seg_tail // 2

        # placement
        m_tail = self.add_mos(ridx_tail, 0, seg_tail, w=w_tail, g_on_s=True)
        m_nfb = self.add_mos(ridx_nfb, 0, seg_nfb, w=w_nfb)
        m_pfb = self.add_mos(ridx_pfb, 0, seg_pfb, w=w_pfb)
        m_in = self.add_mos(ridx_in, seg_nfb if flip_np else seg_pfb, seg_in, w=w_in)
        if has_rst:
            ridx_rst = ridx_in
            col_rst = seg_nfb if flip_np else seg_pfb
            seg_rst = seg_dict.get('rst', 2)
            m_rst = self.add_mos(ridx_rst, col_rst + seg_in, seg_rst, w=w_pfb if flip_np else w_nfb)
        else:
            m_rst = None

        ng_tid = self.get_track_id(ridx_nfb, MOSWireType.G, wire_name='sig')
        pg_tid = self.get_track_id(ridx_pfb, MOSWireType.G, wire_name='sig')

        nclk_tid = self.get_track_id(ridx_tail, MOSWireType.G, wire_name='sig', wire_idx=-1)
        tail_tid = self.get_track_id(ridx_pfb if flip_np else ridx_nfb, MOSWireType.DS, wire_name='sig', wire_idx=-1)

        tail_conn = [m_tail.s, m_pfb.s] if flip_np else [m_tail.s, m_nfb.s]
        clk_conn = m_tail.g

        # NOTE: force even number of columns to make sure VDD conn_layer wires are on even columns.
        ncol_tot = self.num_cols
        self.set_mos_size(num_cols=ncol_tot + (ncol_tot & 1))

        # routing
        conn_layer = self.conn_layer
        vm_layer = conn_layer + 2
        vm_w = self.tr_manager.get_width(vm_layer, 'ana_sig')
        grid = self.grid

        if has_rst:
            tail_conn.append(m_rst.d)
        tail = self.connect_to_tracks(tail_conn, tail_tid)
        out = self.connect_wires([m_nfb.d, m_pfb.d, m_in.d])

        nclk = self.connect_to_tracks(clk_conn, nclk_tid)
        nout = self.connect_to_tracks(m_nfb.g, ng_tid)
        pout = self.connect_to_tracks(m_pfb.g, pg_tid)
        if flip_np:
            vdd_conn = m_tail.d
            vss_conn = [m_nfb.s, m_in.s, m_rst.s] if has_rst else [m_nfb.s, m_in.s]
        else:
            vdd_conn = [m_pfb.s, m_in.s, m_rst.s] if has_rst else [m_pfb.s, m_in.s]
            vss_conn = m_tail.d
        if vertical_sup:
            vdd = vdd_conn
            vss = vss_conn
        else:
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sup')
            vdd = self.connect_to_tracks(vdd_conn, vdd_tid)
            vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            vss = self.connect_to_tracks(vss_conn, vss_tid)

        xout = grid.track_to_coord(conn_layer, m_pfb.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xout, mode=RoundMode.GREATER_EQ)
        vm_tidx = sig_locs.get('out', vm_tidx)

        if vertical_out:
            out_vm = self.connect_to_tracks([nout, pout], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('out_vm', out_vm)
        else:
            self.add_pin('pout', pout)
            self.add_pin('nout', nout)
        self.add_pin('VSS', vss)
        self.add_pin('VDD', vdd)
        self.add_pin('tail', tail)
        if has_rst:
            rst_in_m = self.connect_to_tracks(m_rst.g, ng_tid if flip_np else pg_tid)
            vm_tidx = grid.coord_to_track(vm_layer, rst_in_m.middle, mode=RoundMode.GREATER_EQ)
            vm_tidx = sig_locs.get('in_m', vm_tidx)
            in_m_vm = self.connect_to_tracks([nclk, rst_in_m], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('in_m', in_m_vm)
        else:
            xclk = grid.track_to_coord(conn_layer, m_tail.g.track_id.base_index)
            vm_tidx = grid.coord_to_track(vm_layer, xclk, mode=RoundMode.GREATER_EQ)
            vm_tidx = sig_locs.get('clk', vm_tidx)
            clk_vm = self.connect_to_tracks([nclk], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('clk_vm', clk_vm)
            self.add_pin('clk', nclk)

        self.add_pin('in', m_in.g)
        self.add_pin('out', out)

        self.sch_params = dict(
            lch=self.arr_info.lch,
            seg_dict=seg_dict,
            w_dict=w_dict,
            th_dict=th_dict,
            has_rst=has_rst,
            flip_np=flip_np,
        )

    def _get_w_th_dict(self, ridx_tail: int, ridx_nfb: int, ridx_pfb: int, ridx_in: int) \
            -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']
        has_rst: bool = self.params['has_rst']

        w_ans = {}
        th_ans = {}
        for name, row_idx in [('nfb', ridx_nfb), ('in', ridx_in), ('pfb', ridx_pfb), ('tail', ridx_tail)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        if has_rst:
            rinfo = self.get_row_info(ridx_in, 0)
            w = w_dict.get('rst', 0)
            if w == 0:
                w = rinfo.width
            w_ans['rst'] = w
            th_ans['rst'] = rinfo.threshold

        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class DynLatch(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'comp_dyn_latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = DynLatchHalf.get_params_info()
        ans['even_center'] = 'True to force center column to be even.'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = DynLatchHalf.get_default_param_values()
        ans['even_center'] = False
        return ans

    def draw_layout(self):
        master: DynLatchHalf = self.new_template(DynLatchHalf, params=self.params)
        self.draw_base(master.draw_base_info)

        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        even_center: bool = self.params['even_center']
        has_rst: bool = self.params['has_rst']
        flip_np: bool = self.params['flip_np']

        # placement
        nsep = self.min_sep_col
        nsep += (nsep & 1)
        if even_center and nsep % 4 == 2:
            nsep += 2

        nhalf = master.num_cols
        corel = self.add_tile(master, 0, nhalf, flip_lr=True)
        corer = self.add_tile(master, 0, nhalf + nsep)
        self.set_mos_size(num_cols=nsep + 2 * nhalf)

        # routing
        ridx_nfb = ridx_n if flip_np else ridx_n + 1
        ridx_pfb = ridx_p - 1 if flip_np else ridx_p
        inn_tidx, hm_w = \
            self.get_track_info(ridx_nfb if flip_np else ridx_pfb, MOSWireType.G_MATCH, wire_name='sig', wire_idx=1)
        inp_tidx = self.get_track_index(ridx_nfb if flip_np else ridx_pfb, MOSWireType.G_MATCH, wire_name='sig',
                                        wire_idx=-1)
        # outn_tidx = self.get_track_index(ridx_pfb if flip_np else ridx_nfb,
        #                                  MOSWireType.DS_MATCH, wire_name='sig', wire_idx=1)
        # outp_tidx = self.get_track_index(ridx_pfb if flip_np else ridx_nfb,
        #                                  MOSWireType.DS_MATCH, wire_name='sig', wire_idx=0)
        outn_tidx = self.get_track_index(ridx_nfb if flip_np else ridx_pfb,
                                         MOSWireType.DS_MATCH, wire_name='sig', wire_idx=1)
        outp_tidx = self.get_track_index(ridx_nfb if flip_np else ridx_pfb,
                                         MOSWireType.DS_MATCH, wire_name='sig', wire_idx=0)
        hm_layer = self.conn_layer + 1
        inp, inn = self.connect_differential_tracks(corel.get_pin('in'), corer.get_pin('in'),
                                                    hm_layer, inp_tidx, inn_tidx, width=hm_w)
        self.add_pin('inp', inp)
        self.add_pin('inn', inn)

        # outp, outn = self.connect_differential_tracks(corer.get_all_port_pins('out'),
        #                                               corel.get_all_port_pins('out'),
        #                                               hm_layer, outp_tidx, outn_tidx, width=hm_w)
        outp = self.connect_to_tracks(corer.get_all_port_pins('out'), TrackID(hm_layer, outp_tidx, hm_w))
        outn = self.connect_to_tracks(corel.get_all_port_pins('out'), TrackID(hm_layer, outn_tidx, hm_w))
        if vertical_out:
            outp_vm = corel.get_pin('out_vm')
            outn_vm = corer.get_pin('out_vm')
            self.connect_to_track_wires(outp, outp_vm)
            self.connect_to_track_wires(outn, outn_vm)
            self.add_pin('outp', outp_vm)
            self.add_pin('outn', outn_vm)
        else:
            self.add_pin('outp', [corel.get_pin('pout'), corel.get_pin('nout'), outp], connect=True)
            self.add_pin('outn', [corer.get_pin('pout'), corer.get_pin('nout'), outn], connect=True)

        if vertical_sup:
            vss = list(chain(corel.get_all_port_pins('VSS', layer=self.conn_layer),
                             corer.get_all_port_pins('VSS', layer=self.conn_layer)))
            vdd = list(chain(corel.get_all_port_pins('VDD', layer=self.conn_layer),
                             corer.get_all_port_pins('VDD', layer=self.conn_layer)))
        else:
            vss = self.connect_wires(list(chain(corel.get_all_port_pins('VSS'), corer.get_all_port_pins('VSS'))))
            vdd = self.connect_wires(list(chain(corel.get_all_port_pins('VDD'), corer.get_all_port_pins('VDD'))))

        if has_rst:
            self.reexport(corel.get_port('in_m'), net_name='inp_m')
            self.reexport(corer.get_port('in_m'), net_name='inn_m')
        else:
            clk = self.connect_wires([corel.get_pin('clk'), corer.get_pin('clk')])
            self.connect_wires([corel.get_pin('tail'), corer.get_pin('tail')])
            self.add_pin('clk', clk)
            self.reexport(corel.get_port('clk_vm'), net_name='clkl', hide=True)
            self.reexport(corer.get_port('clk_vm'), net_name='clkr', hide=True)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        self.sch_params = master.sch_params


class SAHalf(MOSBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ridx_n='bottom nmos row index.',
            ridx_p='pmos row index.',
            has_bridge='True to add bridge switch.',
            vertical_out='True to connect outputs to vm_layer.',
            vertical_sup='True to left vdd on conn_layer.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ridx_n=0,
            ridx_p=-1,
            has_bridge=False,
            vertical_out=True,
            vertical_sup=True,
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        has_bridge: bool = self.params['has_bridge']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']

        w_dict, th_dict = self._get_w_th_dict(ridx_n, ridx_p, has_bridge)

        seg_in = seg_dict['in']
        seg_tail = seg_dict['tail']
        seg_nfb = seg_dict['nfb']
        seg_pfb = seg_dict['pfb']
        seg_swm = seg_dict['sw']
        w_in = w_dict['in']
        w_tail = w_dict['tail']
        w_nfb = w_dict['nfb']
        w_pfb = w_dict['pfb']

        if seg_in & 1 or (seg_tail % 4 != 0) or seg_nfb & 1 or seg_pfb & 1:
            raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        # NOTE: make seg_swo even so we can abut transistors
        seg_swo = seg_swm + (seg_swm & 1)
        seg_tail = seg_tail // 2

        # placement
        ridx_in = ridx_n + 1
        ridx_nfb = ridx_in + 1
        m_in = self.add_mos(ridx_n, 0, seg_in, w=w_in)
        m_nfb = self.add_mos(ridx_nfb, 0, seg_nfb, w=w_nfb)
        m_pfb = self.add_mos(ridx_p, 0, seg_pfb, w=w_pfb)

        m_tail = self.add_mos(ridx_in, 0, seg_tail, w=w_tail)
        m_swo = self.add_mos(ridx_p, seg_pfb, seg_swo, w=w_pfb)
        m_swm = self.add_mos(ridx_p, seg_pfb + seg_swo, seg_swm, w=w_pfb)

        ng_tid = self.get_track_id(ridx_nfb, MOSWireType.G, wire_name='sig', wire_idx=-1)
        mid_tid = self.get_track_id(ridx_nfb, MOSWireType.DS, wire_name='sig')
        pg_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')
        pclk_tid = pg_tid
        nclk_tid = self.get_track_id(ridx_in, MOSWireType.G, wire_name='sig')
        nclk_vss = self.get_track_id(ridx_in, MOSWireType.G, wire_name='sig', wire_idx=-1)
        in_mid_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=1)

        vss_hm = self.connect_to_tracks(m_tail.s, nclk_vss)
        vdd_conn = [m_pfb.s, m_swo.s, m_swm.s]
        tail_conn = m_tail.d
        clk_conn = m_tail.g
        mid_in_conn = self.connect_to_tracks(m_in.s, in_mid_tid)

        # NOTE: force even number of columns to make sure VDD conn_layer wires are on even columns.
        ncol_tot = self.num_cols
        self.set_mos_size(num_cols=ncol_tot + (ncol_tot & 1))

        # routing
        conn_layer = self.conn_layer
        vm_layer = conn_layer + 2
        vm_w = self.tr_manager.get_width(vm_layer, 'ana_sig')
        grid = self.grid

        # tail_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig')
        tail_in_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=2)
        tail = self.connect_to_tracks(tail_conn, tail_in_tid)
        _tail_in = self.connect_to_tracks(m_in.d, tail_in_tid)
        out = self.connect_wires([m_nfb.d, m_pfb.d, m_swo.d])
        #mid_tidx_vm = grid.coord_to_track(vm_layer, m_in.d.track_id.base_index)
        mid_hm = self.connect_to_tracks([m_nfb.s, m_swm.d], mid_tid)
        mid = self.connect_to_tracks([mid_hm, mid_in_conn], TrackID(vm_layer, m_in.d.track_id.base_index+2, width=vm_w))

        nclk = self.connect_to_tracks(clk_conn, nclk_tid)
        nout = self.connect_to_tracks(m_nfb.g, ng_tid)
        pout = self.connect_to_tracks(m_pfb.g, pg_tid)
        pclk = self.connect_to_tracks([m_swo.g, m_swm.g], pclk_tid)

        xclk = grid.track_to_coord(conn_layer, m_swo.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xclk, mode=RoundMode.GREATER_EQ)
        clk_vm = self.connect_to_tracks([nclk, pclk], TrackID(vm_layer, vm_tidx, width=vm_w))
        self.add_pin('clk_vm', clk_vm)

        xout = grid.track_to_coord(conn_layer, m_pfb.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xout, mode=RoundMode.GREATER_EQ)
        
        if vertical_out:
            out_vm = self.connect_to_tracks([nout, pout], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('out_vm', out_vm)
        else:
            self.add_pin('pout', pout)
            self.add_pin('nout', nout)

        if vertical_sup:
            #vss_conn = m_tail.s
            vdd = vdd_conn
            #vss = vss_conn
            w_conn, h_conn = grid.get_block_size(conn_layer, half_blk_x=True, half_blk_y=True)
            vss_tid_conn = grid.coord_to_track(conn_layer, mid_hm.upper//w_conn*w_conn)
            vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            vss_conn = self.connect_to_tracks(vss_hm, TrackID(conn_layer, vss_tid_conn, width=vm_w))
            vss = vss_conn #self.connect_to_tracks(vss_conn, vss_tid)

        else:
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sup')
            vdd = self.connect_to_tracks(vdd_conn, vdd_tid)
            vss_conn = m_tail.s
            vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            vss = self.connect_to_tracks(vss_conn, vss_tid)
            # vss_tid_conn = grid.coord_to_track(conn_layer, m_in.d.track_id.base_index+10)
            # vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            # vss_conn = self.connect_to_tracks(vss_hm, TrackID(conn_layer, vss_tid_conn, width=vm_w))
            # vss = self.connect_to_tracks(vss_conn, vss_tid)
            
        self.add_pin('VSS', vss)
        self.add_pin('VDD', vdd)
        self.add_pin('tail', tail)
        self.add_pin('clk', nclk)
        self.add_pin('in', m_in.g)
        self.add_pin('out', out)
        self.add_pin('mid', mid)

        append_dict = dict(swo=seg_swo, swm=seg_swm)
        if has_bridge:
            append_dict['br'] = 1
        sch_seg_dict = seg_dict.copy(append=append_dict, remove=['sw'])
        self.sch_params = dict(
            lch=self.arr_info.lch,
            seg_dict=sch_seg_dict,
            w_dict=w_dict,
            th_dict=th_dict,
            has_bridge=has_bridge,
        )

    def _get_w_th_dict(self, ridx_n: int, ridx_p: int, has_bridge: bool
                       ) -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        th_ans = {}
        for name, row_idx in [('tail', ridx_n), ('in', ridx_n + 1), ('nfb', ridx_n + 2),
                              ('pfb', ridx_p)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        w_ans['swm'] = w_ans['swo'] = w_ans['pfb']
        th_ans['swm'] = th_ans['swo'] = th_ans['pfb']
        if has_bridge:
            w_ans['br'] = w_ans['in']
            th_ans['br'] = th_ans['in']
        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class SA(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'comp_strongarm_core')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = SAHalf.get_params_info()
        ans['even_center'] = 'True to force center column to be even.',
        ans['add_tap'] = 'Add tap rows to have complete  layout',
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = SAHalf.get_default_param_values()
        ans['even_center'] = False
        ans['add_tap'] = False
        return ans

    def draw_layout(self):
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        has_bridge: bool = self.params['has_bridge']
        add_tap: bool = self.params['add_tap']
        vertical_out: bool = self.params['vertical_out']
        even_center: bool = self.params['even_center']
        master_params = self.params.copy().to_dict()
        if add_tap:
            pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
            self.draw_base(pinfo)
            master_pinfo = self.get_draw_base_sub_pattern(1, 2)
            master_params['pinfo'] = master_pinfo
            master: SAHalf = self.new_template(SAHalf, params=master_params)
        else:
            master: SAHalf = self.new_template(SAHalf, params=master_params)
            self.draw_base(master.draw_base_info)

        nsep = self.min_sep_col
        nsep += (nsep & 1)
        if even_center and nsep % 4 == 2:
            nsep += 2

        # placement
        inst_tile_idx = 1 if add_tap else 0
        nhalf = master.num_cols
        corel = self.add_tile(master, inst_tile_idx, nhalf, flip_lr=True)
        corer = self.add_tile(master, inst_tile_idx, nhalf + nsep)
        self.set_mos_size(num_cols=nsep + 2 * nhalf)

        # routing
        ridx_in = ridx_n + 1
        ridx_nfb = ridx_in + 1
        inn_tidx, hm_w = self.get_track_info(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0,
                                             tile_idx=inst_tile_idx)
        inp_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1, tile_idx=inst_tile_idx)
        outn_tidx = self.get_track_index(ridx_nfb, MOSWireType.DS, wire_name='sig', wire_idx=-1,
                                         tile_idx=inst_tile_idx)
        outp_tidx = self.get_track_index(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=0, tile_idx=inst_tile_idx)


        hm_layer = self.conn_layer + 1
        inp, inn = self.connect_differential_tracks(corel.get_pin('in'), corer.get_pin('in'),
                                                    hm_layer, inp_tidx, inn_tidx, width=hm_w)
        self.add_pin('inp', inp)
        self.add_pin('inn', inn)
        self.reexport(corer.get_port('mid'), net_name='midp')
        self.reexport(corel.get_port('mid'), net_name='midn')

        outp, outn = self.connect_differential_tracks(corer.get_all_port_pins('out'),
                                                      corel.get_all_port_pins('out'),
                                                      hm_layer, outp_tidx, outn_tidx, width=hm_w)
        if vertical_out:
            outp_vm = corel.get_pin('out_vm')
            outn_vm = corer.get_pin('out_vm')
            self.connect_to_track_wires(outp, outp_vm)
            self.connect_to_track_wires(outn, outn_vm)
            self.add_pin('outp', outp_vm)
            self.add_pin('outn', outn_vm)
        else:
            self.add_pin('outp', outp, connect=True)
            self.add_pin('outn', outn, connect=True)
            self.add_pin('outp', corel.get_pin('pout'), connect=True)
            self.add_pin('outp', corel.get_pin('nout'), connect=True)
            self.add_pin('outn', corer.get_pin('pout'), connect=True)
            self.add_pin('outn', corer.get_pin('nout'), connect=True)

        self.add_pin('outp_hm', outp, hide=True)
        self.add_pin('outn_hm', outn, hide=True)

        clk = self.connect_wires([corel.get_pin('clk'), corer.get_pin('clk')])
        _tail = self.connect_wires([corel.get_pin('tail'), corer.get_pin('tail')])
        vss = self.connect_wires(corel.get_all_port_pins('VSS') + corer.get_all_port_pins('VSS'))
        #connect horizontal section of vss
        w_hm, h_hm = self.grid.get_block_size(hm_layer, half_blk_x=True, half_blk_y=True)
        vss_tidx_hm = self.grid.coord_to_track(hm_layer, corel.get_all_port_pins('VSS')[0].upper//h_hm * h_hm)
        self.connect_to_tracks([corel.get_all_port_pins('VSS')[0], corer.get_all_port_pins('VSS')[0]], 
                                        TrackID(hm_layer, vss_tidx_hm, hm_w))
        
        vdd = self.connect_wires(corel.get_all_port_pins('VDD') + corer.get_all_port_pins('VDD'))
        if add_tap:
            ncols, nrows = master.num_cols, master.num_rows
            ntap = self.add_substrate_contact(0, 0, seg=2 * ncols + nsep, tile_idx=2, port_mode=SubPortMode.EVEN)
            ptap = self.add_substrate_contact(0, 0, seg=2 * ncols + nsep, tile_idx=0, port_mode=SubPortMode.EVEN)
            vss = self.connect_to_tracks(vss + [ptap], self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=0))
            vdd = self.connect_to_tracks(vdd + [ntap], self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=2))
        self.add_pin('clk', clk)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)
        self.reexport(corel.get_port('clk_vm'), net_name='clkl', hide=True)
        self.reexport(corer.get_port('clk_vm'), net_name='clkr', hide=True)

        # bridge_switch
        if has_bridge:
            m_br0 = self.add_mos(ridx_n + 1, nhalf, 1, w=master.sch_params['w_dict']['br'],
                                 stack=nsep)
            self.connect_to_track_wires(m_br0.g, clk)

        if corel.has_port('nrstb'):
            self.connect_wires([corel.get_pin('tail_in'), corer.get_pin('tail_in')])
            rstb = self.connect_wires([corel.get_pin('nrstb'), corer.get_pin('nrstb')])
            # if vertical_rstb:
            #     self.add_pin('rstb', rstb)
            # else:
            rstl = corel.get_pin('prstb')
            rstr = corer.get_pin('prstb')
            self.add_pin('rstb', rstb, connect=True)
            self.add_pin('rstb', rstl, connect=True)
            self.add_pin('rstb', rstr, connect=True)
            self.add_pin('nrstb', rstb, hide=True)
            self.add_pin('prstbl', rstl, hide=True)
            self.add_pin('prstbr', rstr, hide=True)

        if has_bridge:
            self.sch_params = master.sch_params.copy(append=dict(stack_br=nsep))
        else:
            self.sch_params = master.sch_params


class DualSACore(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'comp_dual_strongarm_core')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            vertical_out='True to connect outputs to vm_layer.',
            even_center='True to force center column to be even.',
            signal_locs='Signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            vertical_out=True,
            even_center=False,
            signal_locs={},
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        vertical_out: bool = self.params['vertical_out']
        even_center: bool = self.params['even_center']
        seg_dict: Dict[str, Dict] = self.params['seg_dict']
        w_dict: Dict[str, Dict] = self.params['w_dict']
        signal_locs: Dict[str, int] = self.params['signal_locs']

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        vm_w = tr_manager.get_width(vm_layer, 'ana_sig')

        # Make templates
        # -- Get tidx for clock --
        _, clk_tidx_locs = tr_manager.place_wires(vm_layer, ['clk', 'clk', 'sig', 'sig'],
                                                  self.arr_info.col_to_track(vm_layer, 0), 0)
        sa_params = dict(pinfo=self.get_tile_pinfo(1), even_center=even_center,
                         vertical_sup=True, seg_dict=seg_dict['sa'], w_dict=w_dict['sa'])
        dyn_latch_params = dict(pinfo=self.get_tile_pinfo(3), vertical_out=vertical_out, even_center=even_center,
                                vertical_sup=True, seg_dict=seg_dict['dyn_latch'], has_rst=True, flip_np=True,
                                w_dict=w_dict['dyn_latch'], )

        sa_master = self.new_template(SA, params=sa_params)
        dyn_latch_master = self.new_template(DynLatch, params=dyn_latch_params)

        # floorplanning

        sa_ncol = sa_master.num_cols
        dyn_latch_ncol = dyn_latch_master.num_cols

        tot_ncol = max(sa_ncol, dyn_latch_ncol)

        # placement
        sa = self.add_tile(sa_master, 1, (tot_ncol - sa_ncol) // 2)
        dyn_latch = self.add_tile(dyn_latch_master, 3, (tot_ncol - dyn_latch_ncol) // 2)

        # Add substrate connection
        ptap0 = self.add_substrate_contact(0, 0, seg=tot_ncol, tile_idx=0, port_mode=SubPortMode.EVEN)
        ntap0 = self.add_substrate_contact(0, 0, seg=tot_ncol, tile_idx=2, port_mode=SubPortMode.EVEN)
        ptap1 = self.add_substrate_contact(0, 0, seg=tot_ncol, tile_idx=4, port_mode=SubPortMode.EVEN)
        self.set_mos_size()

        # Connect supplies
        vss0 = self.connect_to_tracks(ptap0, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=0))
        vss1 = self.connect_to_tracks(ptap1, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=4))
        vdd0 = self.connect_to_tracks(ntap0, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=2))
        self.connect_to_track_wires(vss0, sa.get_all_port_pins('VSS'))
        self.connect_to_track_wires(vdd0, sa.get_all_port_pins('VDD', layer=self.conn_layer) +
                                    dyn_latch.get_all_port_pins('VDD'))
        self.connect_to_track_wires(vss1, dyn_latch.get_all_port_pins('VSS'))

        # -- Inter connection --
        # ---- in/out signals ----
        self.connect_differential_wires(sa.get_pin('outp'), sa.get_pin('outn'),
                                        dyn_latch.get_pin('inn'), dyn_latch.get_pin('inp'))
        inn_vm_tidx = tr_manager.get_next_track(vm_layer, sa.get_pin('outn').track_id.base_index,
                                                'sig', 'sig')
        inp_vm_tidx = tr_manager.get_next_track(vm_layer, sa.get_pin('outp').track_id.base_index,
                                                'sig', 'sig', up=False)
        inn, inp = self.connect_differential_tracks(sa.get_pin('inn'), sa.get_pin('inp'), vm_layer, inn_vm_tidx,
                                                    inp_vm_tidx, width=tr_manager.get_width(vm_layer, 'sig'),
                                                    track_lower=self.bound_box.yl)
        self.connect_to_track_wires(sa.get_pin('midp'), dyn_latch.get_pin('inn_m'))
        self.connect_to_track_wires(sa.get_pin('midn'), dyn_latch.get_pin('inp_m'))

        if vertical_out:
            outp_tidx, outn_tidx = signal_locs.get('outp', 0), signal_locs.get('outn', 0)
            outn_hm, outp_hm = dyn_latch.get_all_port_pins('outp_hm'), dyn_latch.get_all_port_pins('outn_hm')

            outn = self.connect_to_tracks(outn_hm, TrackID(vm_layer, outn_tidx, vm_w))
            outp = self.connect_to_tracks(outp_hm, TrackID(vm_layer, outp_tidx, vm_w))

            self.add_pin('outp', outp)
            self.add_pin('outn', outn)
        else:
            self.reexport(dyn_latch.get_port('outp'), net_name='outn')
            self.reexport(dyn_latch.get_port('outn'), net_name='outp')

        self.add_pin('inn', inn)
        self.add_pin('inp', inp)
        self.add_pin('VSS', [vss0, vss1], connect=True)
        self.add_pin('VDD', [vdd0], connect=True)
        self.reexport(sa.get_port('clk'))

        self.sch_params = dict(
            sa0=sa_master.sch_params,
            sa1=dyn_latch_master.sch_params,
        )


class SARCompCore(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'comp_tri_tail_core')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            vertical_out='True to connect outputs to vm_layer.',
            vm_sup='True to connect supply to vm_layer.',
            even_center='True to force center column to be even.',
            signal_locs='Signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            vertical_out=True,
            vm_sup=False,
            even_center=False,
            signal_locs={},
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        vertical_out: bool = self.params['vertical_out']
        even_center: bool = self.params['even_center']
        seg_dict: Dict[str, Dict] = self.params['seg_dict']
        w_dict: Dict[str, Dict] = self.params['w_dict']
        signal_locs: Dict[str, int] = self.params['signal_locs']

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        vm_w = tr_manager.get_width(vm_layer, 'ana_sig')

        # Make templates
        # -- Get tidx for clock --
        _, clk_tidx_locs = tr_manager.place_wires(vm_layer, ['clk', 'clk', 'sig', 'sig'],
                                                  self.arr_info.col_to_track(vm_layer, 0), 0)
        preamp_params = dict(pinfo=self.get_tile_pinfo(1), even_center=even_center,
                             vertical_sup=True, seg_dict=seg_dict['preamp'], w_dict=w_dict['preamp'],
                             sig_locs={'clk': clk_tidx_locs[0], 'out': clk_tidx_locs[3] - clk_tidx_locs[0]})
        half_latch_params = dict(pinfo=self.get_tile_pinfo(3), even_center=even_center,
                                 vertical_sup=True, seg_dict=seg_dict['half_latch'],
                                 w_dict=w_dict['half_latch'],
                                 sig_locs={'clk': clk_tidx_locs[1], 'out': clk_tidx_locs[2]})
        dyn_latch_params = dict(pinfo=self.get_tile_pinfo(5), vertical_out=vertical_out, even_center=even_center,
                                vertical_sup=True, seg_dict=seg_dict['dyn_latch'],
                                w_dict=w_dict['dyn_latch'],
                                sig_locs={'clk': clk_tidx_locs[0], 'out': clk_tidx_locs[3]})

        preamp_master = self.new_template(PreAmp, params=preamp_params)
        half_latch_master = self.new_template(HalfLatch, params=half_latch_params)
        dyn_latch_master = self.new_template(DynLatch, params=dyn_latch_params)

        # floorplanning

        preamp_ncol = preamp_master.num_cols
        half_latch_ncol = half_latch_master.num_cols
        dyn_latch_ncol = dyn_latch_master.num_cols

        tot_ncol = max(preamp_ncol, half_latch_ncol, dyn_latch_ncol)

        # placement
        preamp = self.add_tile(preamp_master, 1, (tot_ncol - preamp_ncol) // 2)
        half_latch = self.add_tile(half_latch_master, 3, (tot_ncol - half_latch_ncol) // 2)
        dyn_latch = self.add_tile(dyn_latch_master, 5, (tot_ncol - dyn_latch_ncol) // 2)

        # Add substrate connection
        ptap0 = self.add_substrate_contact(0, 0, seg=tot_ncol, tile_idx=0, port_mode=SubPortMode.EVEN)
        ntap0 = self.add_substrate_contact(0, 0, seg=tot_ncol, tile_idx=2, port_mode=SubPortMode.EVEN)
        ptap1 = self.add_substrate_contact(0, 0, seg=tot_ncol, tile_idx=4, port_mode=SubPortMode.EVEN)
        ntap1 = self.add_substrate_contact(0, 0, seg=tot_ncol, tile_idx=6, port_mode=SubPortMode.EVEN)
        self.set_mos_size()

        # Fill preamp
        pre_amp_nrows = self.get_tile_pinfo(1).num_rows
        preamp_p_dum, preamp_n_dum = [], []
        for idx in range(pre_amp_nrows):
            s, d, dev_type = fill_conn_layer_intv(self, 1, idx)
            if dev_type == MOSType.nch:
                preamp_n_dum.append(s + d)
            elif dev_type == MOSType.pch:
                preamp_p_dum.append(s + d)

        # Fill half latch
        half_latch_nrows = self.get_tile_pinfo(3).num_rows
        half_latch_p_dum, half_latch_n_dum = [], []
        for idx in range(half_latch_nrows):
            s, d, dev_type = fill_conn_layer_intv(self, 3, idx)
            if dev_type == MOSType.nch:
                half_latch_n_dum.append(s + d)
            elif dev_type == MOSType.pch:
                half_latch_p_dum.append(s + d)

        half_latch_p_dum = half_latch_p_dum[::-1]
        half_latch_n_dum = half_latch_n_dum[::-1]

        # Fill dyn latch
        dyn_latch_nrows = self.get_tile_pinfo(5).num_rows
        dyn_latch_p_dum, dyn_latch_n_dum = [], []
        for idx in range(dyn_latch_nrows):
            s, d, dev_type = fill_conn_layer_intv(self, 5, idx)
            if dev_type == MOSType.nch:
                dyn_latch_n_dum.append(s + d)
            elif dev_type == MOSType.pch:
                dyn_latch_p_dum.append(s + d)

        connect_conn_dummy_rows(self, preamp_p_dum, connect_to_sup=True, sup_dum_idx=-1, sup_coord=ntap0[0].middle)
        connect_conn_dummy_rows(self, preamp_n_dum, connect_to_sup=True, sup_dum_idx=0, sup_coord=ptap0[0].middle)
        connect_conn_dummy_rows(self, half_latch_p_dum, connect_to_sup=True, sup_dum_idx=0, sup_coord=ntap0[0].middle)
        connect_conn_dummy_rows(self, half_latch_n_dum, connect_to_sup=True, sup_dum_idx=-1, sup_coord=ptap1[0].middle)
        connect_conn_dummy_rows(self, dyn_latch_p_dum, connect_to_sup=True, sup_dum_idx=-1, sup_coord=ntap1[0].middle)
        connect_conn_dummy_rows(self, dyn_latch_n_dum, connect_to_sup=True, sup_dum_idx=0, sup_coord=ptap1[0].middle)

        # Connect supplies
        vss0 = self.connect_to_tracks(ptap0, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=0))
        vss1 = self.connect_to_tracks(ptap1, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=4))
        vdd0 = self.connect_to_tracks(ntap0, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=2))
        vdd1 = self.connect_to_tracks(ntap1, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=6))
        vdd2 = self.extend_wires(preamp.get_pin('VDD', layer=self.conn_layer + 1), lower=self.bound_box.xl,
                                 upper=self.bound_box.xh)[0]
        self.connect_to_track_wires(vss0, preamp.get_all_port_pins('VSS'))
        self.connect_to_track_wires(vdd0, preamp.get_all_port_pins('VDD', layer=self.conn_layer) +
                                    half_latch.get_all_port_pins('VDD'))
        self.connect_to_track_wires(vss1, dyn_latch.get_all_port_pins('VSS') +
                                    half_latch.get_all_port_pins('VSS'))
        self.connect_to_track_wires(vdd1, dyn_latch.get_all_port_pins('VDD'))

        # -- Inter connection --
        # ---- clock signals ----
        clk_vm_list = [preamp.get_pin('clkr'), preamp.get_pin('clkl'),
                       dyn_latch.get_pin('clkr'), dyn_latch.get_pin('clkl')]
        nclk_vm_list = [half_latch.get_pin('clkr'), half_latch.get_pin('clkl')]
        clk_vm = self.connect_wires(clk_vm_list, lower=self.bound_box.yl)
        nclk_vm = self.connect_wires(nclk_vm_list, lower=self.bound_box.yl)

        # ---- in/out signals ----
        self.connect_differential_wires(preamp.get_pin('outp'), preamp.get_pin('outn'),
                                        half_latch.get_pin('inp'), half_latch.get_pin('inn'))
        self.connect_differential_wires(half_latch.get_pin('outp'), half_latch.get_pin('outn'),
                                        dyn_latch.get_pin('inp'), dyn_latch.get_pin('inn'))
        inn_vm_tidx = tr_manager.get_next_track(vm_layer, preamp.get_pin('outn').track_id.base_index,
                                                'sig', 'sig')
        inp_vm_tidx = tr_manager.get_next_track(vm_layer, preamp.get_pin('outp').track_id.base_index,
                                                'sig', 'sig', up=False)
        inn, inp = self.connect_differential_tracks(preamp.get_pin('inn'), preamp.get_pin('inp'), vm_layer, inn_vm_tidx,
                                                    inp_vm_tidx, width=tr_manager.get_width(vm_layer, 'sig'),
                                                    track_lower=self.bound_box.yl)

        self.add_pin('clk', clk_vm)
        self.add_pin('clkb', nclk_vm)

        if vertical_out:
            outp_tidx, outn_tidx = signal_locs.get('outp', 0), signal_locs.get('outn', 0)
            outn, outp = dyn_latch.get_all_port_pins('outn'), dyn_latch.get_all_port_pins('outp')
            if outn[0].layer_id == hm_layer:
                outn = self.connect_to_tracks(outn, TrackID(vm_layer, outn_tidx, vm_w))
                outp = self.connect_to_tracks(outp, TrackID(vm_layer, outp_tidx, vm_w))
            self.add_pin('outp', outp)
            self.add_pin('outn', outn)
        else:
            self.reexport(dyn_latch.get_port('outp'))
            self.reexport(dyn_latch.get_port('outn'))

        if self.params['vm_sup']:
            sup_vm_locs_l = self.arr_info.col_to_track(vm_layer, 0, mode=RoundMode.GREATER_EQ)
            _, sup_vm_locs_l = tr_manager.place_wires(vm_layer, ['sig', 'sup', 'sup'], align_idx=0,
                                                      align_track=sup_vm_locs_l)
            sup_vm_locs_r = self.arr_info.col_to_track(vm_layer, self.num_cols, mode=RoundMode.LESS_EQ)
            _, sup_vm_locs_r = tr_manager.place_wires(vm_layer, ['sig', 'sup', 'sup'], align_idx=-1,
                                                      align_track=sup_vm_locs_r)
            # sup_vm_locs = sup_vm_locs_l + sup_vm_locs_r
            tr_w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
            vssr, vddr = self.connect_matching_tracks([[vss0, vss1], [vdd0, vdd1, vdd2]], vm_layer,
                                                      [sup_vm_locs_r[0], sup_vm_locs_r[1]], width=tr_w_sup_vm)
            vssl, vddl = self.connect_matching_tracks([[vss0, vss1], [vdd0, vdd1, vdd2]], vm_layer,
                                                      [sup_vm_locs_l[0], sup_vm_locs_l[1]], width=tr_w_sup_vm)

            self.add_pin('VSS', [vssr, vssl], connect=True)
            self.add_pin('VDD', [vddr, vddl], connect=True)
            # self.conn
        else:
            self.add_pin('VSS', [vss0, vss1], connect=True)
            self.add_pin('VDD', [vdd0, vdd1, vdd2], connect=True)

        self.add_pin('inn', inn)
        self.add_pin('inp', inp)

        if preamp_master.has_ofst:
            self.reexport(preamp.get_port('osp'))
            self.reexport(preamp.get_port('osn'))

        self.sch_params = dict(
            preamp=preamp_master.sch_params,
            half_latch=half_latch_master.sch_params,
            dyn_latch=dyn_latch_master.sch_params,
        )


class SARComp(MOSBase, TemplateBaseZL):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    def get_schematic_class(self) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_comp_clk') if self.params['clk_params'] else\
            ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_comp')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='comparator core',
            sch_cls='comparator core sch name',
            sup_top_layer='Top supply layer',
            pinfo='placement information object.',
            ncols_tot='Total number of columns',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            shield='Add ground shield if True',
            ngroups_dict='',
            ridx_dict='',
            clk_params='',
            num_comp_tiles='',
            add_tap='',
            substrate_row='',
            vertical_out=''
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ncols_tot=0,
            clk_params=None,
            num_comp_tiles=1 ,
            ngroups_dict='',
            ridx_dict='',
            add_tap=False,
            substrate_row=False,
            vertical_out=True,
            shield=False
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, Dict] = self.params['seg_dict']
        w_dict: Dict[str, Dict] = self.params['w_dict']
        shield = self.params['shield']

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xm1_layer = ym_layer + 1
        # Make templates
        clk_params = self.params['clk_params']
        total_tiles = self.params['num_comp_tiles']
        comp_params = dict(pinfo=(self.get_tile_subpattern(0, total_tiles), self.tile_table),
                           seg_dict=seg_dict, w_dict=w_dict, add_tap=self.params['add_tap'],
                           substrate_row=self.params['substrate_row'], vertical_out=self.params['vertical_out'],
                           ngroups_dict=self.params['ngroups_dict'], ridx_dict=self.params['ridx_dict'])
        buf_params = dict(pinfo=self.get_tile_pinfo(total_tiles), segn_list=seg_dict['buf']['n'],
                          segp_list=seg_dict['buf']['p'], vertical_sup=True, export_pins=True,
                          w_p=w_dict['buf']['p'], w_n=w_dict['buf']['n'], dual_output=False,
                          vertical_out=False, mid_int_vm=True)

        gen_cls = cast(Type[MOSBase], import_class(self.params['cls_name']))
        comp_master: MOSBase = self.new_template(gen_cls, params=comp_params)
        buf_master: MOSBase = self.new_template(InvChainCore, params=buf_params)

        # floorplanning
        if isinstance(clk_params, str):
            clk_yaml = read_yaml(clk_params)
            clk_params = clk_yaml['params']['params']
            logic_pinfo = self.get_tile_pinfo(total_tiles-1)
            clk_params['pinfo'] = (self.get_tile_subpattern(total_tiles+2, total_tiles+5),self.tile_table)
            clk_master = self.new_template(SARAsyncClkSimple, params=clk_params)
            clk_ncol = clk_master.num_cols
        else:
            clk_ncol = 0
            clk_master = None
        clk_ncol += clk_ncol & 1
        min_sep = self.min_sep_col
        min_sep += min_sep & 1
        comp_ncol = comp_master.num_cols
        buf_ncol = buf_master.num_cols

        out_layer = comp_master.get_port('outp').get_single_layer()
        buf_mid_sep = abs(int(comp_master.get_port('outp').get_pins(out_layer)[0]._tid.base_htr
                        - comp_master.get_port('outn').get_pins(out_layer)[0]._tid.base_htr))//2
                        #comp_master.out_col[0] - comp_master.out_col[1]
        buf_mid_sep += buf_mid_sep & 1
        buf_mid_sep += 2 * int(not bool((buf_mid_sep // 2) & 1))
        buf_ncol_tot = 2 * buf_ncol + buf_mid_sep
        clk_ncol += buf_mid_sep
        core_ncol = max(buf_ncol_tot, comp_ncol, clk_ncol)
        side_sup = self.params['ncols_tot'] > core_ncol
        tot_ncol = self.params['ncols_tot'] if side_sup else core_ncol

        #FIXME making out of range errors
        # if buf_ncol_tot > comp_master.num_cols:
        #     comp_params = dict(pinfo=(self.get_tile_subpattern(0, total_tiles), self.tile_table),
        #                        seg_dict=seg_dict, w_dict=w_dict, ncols_tot=buf_ncol_tot,
        #                        ngroups_dict=self.params['ngroups_dict'], ridx_dict=self.params['ridx_dict'])
        #     comp_master: MOSBase = self.new_template(gen_cls, params=comp_params)
        #     comp_ncol = comp_master.num_cols

        # placement
        comp = self.add_tile(comp_master, 0, (tot_ncol - comp_ncol) // 2, commit=False)
        bufn = self.add_tile(buf_master, total_tiles, (tot_ncol - buf_mid_sep) // 2, flip_lr=True)
        bufp = self.add_tile(buf_master, total_tiles, (tot_ncol + buf_mid_sep) // 2)
        comp.commit()

        if clk_master:
            clkgen = self.add_tile(clk_master, total_tiles + 2, tot_ncol//2-clk_master.middle_col)
            self.connect_differential_wires(clkgen.get_pin('comp_n'), clkgen.get_pin('comp_p'),
                                            comp.get_pin('outn'), comp.get_pin('outp'))
        else:
            clkgen = None

        # fill dummy and put tap for output buffers
        port_mode = SubPortMode.ODD if tot_ncol % 4 != 0 else SubPortMode.EVEN
        ndum_side = (tot_ncol - core_ncol) // 2 - min_sep
        # Connect supplies
        if not (total_tiles // 2) & 1:
            ntap_top = self.add_substrate_contact(0, ndum_side + min_sep, seg=core_ncol, tile_idx=total_tiles+1,
                                                  port_mode=port_mode)
            vdd_top = self.connect_to_tracks(ntap_top,
                                             self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=total_tiles+1))
            vss_top = max(comp.get_all_port_pins('VSS'), key=lambda x: x.track_id.base_index)
        else:
            ptap_top = self.add_substrate_contact(0, ndum_side + min_sep, seg=core_ncol, tile_idx=total_tiles+1,
                                                  port_mode=port_mode)
            vss_top = self.connect_to_tracks(ptap_top,
                                             self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=total_tiles+1))
            vdd_top = max(comp.get_all_port_pins('VDD'), key=lambda x: x.track_id.base_index)

        if clk_params:
            self.connect_to_tracks(clkgen.get_all_port_pins('VDD_bot'),
                                   self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=total_tiles+1))
            vdd_top_clk = self.add_substrate_contact(0, ndum_side + min_sep, seg=core_ncol, tile_idx=total_tiles+5,
                                                  port_mode=port_mode)
            vdd_top_clk = self.connect_to_tracks(clkgen.get_all_port_pins('VDD_top') + [vdd_top_clk],
                                   self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=total_tiles+5))

        buffer_p_dum, buffer_n_dum = [], []
        # fill_and_collect(self, 7, buffer_p_dum, buffer_n_dum, start_col=ndum_side - 1, stop_col=tot_ncol - ndum_side)
        #
        # connect_conn_dummy_rows(self, buffer_n_dum, connect_to_sup=True, sup_dum_idx=-1, sup_coord=ptap_top[0].middle)
        # connect_conn_dummy_rows(self, buffer_p_dum, connect_to_sup=True, sup_dum_idx=0,
        #                         sup_coord=self.grid.track_to_coord(hm_layer, vdd_top.track_id.base_index))

        self.set_mos_size(tot_ncol)

        # Connect comp to buf
        # self.extend_wires(comp.get_pin('outp'), upper=self.bound_box.yh)
        # self.extend_wires(comp.get_pin('outn'), upper=self.bound_box.yh)
        comp_p_int = self.connect_to_track_wires(bufn.get_pin('nin'), comp.get_pin('outp')) #had to flip bufn and bufp
        comp_n_int = self.connect_to_track_wires(bufp.get_pin('nin'), comp.get_pin('outn'))
        # self.add_pin('comp_p', bufp.get_pin('nin'))
        # self.add_pin('comp_n', bufn .get_pin('nin'))
        self.match_warr_length([comp_n_int, comp_p_int])

        # Connect supply
        vss_hm_top = [self.connect_to_track_wires(bufp.get_all_port_pins('VSS') + bufn.get_all_port_pins('VSS'),
                                                  vss_top)]
        vdd_hm_top = [self.connect_to_track_wires(bufp.get_all_port_pins('VDD') + bufn.get_all_port_pins('VDD'),
                                                  vdd_top)]
        # self.add_pin('VSS',vss_hm_top)
        # self.add_pin('VDD', vdd_hm_top)
        self.extend_wires(vss_hm_top, lower=self.bound_box.xl, upper=self.bound_box.xh)
        # self.reexport(bufn.get_port('outb'), net_name='outn')
        # self.reexport(bufp.get_port('outb'), net_name='outp')
        outn_hm = bufn.get_all_port_pins('out', layer=hm_layer)
        # outn_bbox = BBox(outn_hm[0].bound_box.xl + outn_hm[0].bound_box.w//3, outn_hm[0].bound_box.yl, #swapped 0 and 1 index
        #                 outn_hm[0].bound_box.xh - outn_hm[0].bound_box.w//3, outn_hm[1].bound_box.yh)
        # outn_viaup = self.via_stack_up(tr_manager, outn_hm, hm_layer, vm_layer, 'ana_sig', RoundMode.GREATER,
        #                   bbox=outn_bbox)  ##FIXME?? changed ym_layer+2 to vm_layer
        outp_hm = bufp.get_all_port_pins('out', layer=hm_layer)
        # outp_bbox = BBox(outp_hm[0].bound_box.xl + outp_hm[0].bound_box.w//3, outp_hm[0].bound_box.yl, #swapped 0 and 1 index
        #                 outp_hm[0].bound_box.xh - outp_hm[0].bound_box.w//3, outp_hm[1].bound_box.yh)
        # outp_viaup = self.via_stack_up(tr_manager, outp_hm, hm_layer, vm_layer, 'ana_sig', RoundMode.GREATER,
        #                   bbox=outp_bbox, align_higher_x=True)

        for pinname in ['inn', 'inp']:
            pin = comp.get_all_port_pins(pinname)
            # pin = self.extend_wires(pin, lower=self.bound_box.yl)
            self.add_pin(pinname, pin)

        # self.reexport(bufn.get_port('mid<0>'), net_name='outn_m')
        # self.reexport(bufp.get_port('mid<0>'), net_name='outp_m')

        # FIXME?? no osn for SA
        # osn = comp.get_all_port_pins('osn')
        # osp = comp.get_all_port_pins('osp')
        # tr_w_sig_xm = tr_manager.get_width(xm_layer, 'ana_sig')
        # tr_w_sig_ym = tr_manager.get_width(ym_layer, 'ana_sig')
        # os_xm_tidx = self.grid.coord_to_track(xm_layer, osn[0].middle, RoundMode.NEAREST)
        # osn_xm = self.connect_to_tracks(osn, TrackID(xm_layer, os_xm_tidx, tr_w_sig_xm), min_len_mode=MinLenMode.MIDDLE)
        # osp_xm = self.connect_to_tracks(osp, TrackID(xm_layer, os_xm_tidx, tr_w_sig_xm), min_len_mode=MinLenMode.MIDDLE)
        # osn_ym_tidx = self.grid.coord_to_track(ym_layer, osn_xm.middle, RoundMode.NEAREST)
        # osp_ym_tidx = self.grid.coord_to_track(ym_layer, osp_xm.middle, RoundMode.NEAREST)
        # osn_ym = self.connect_to_tracks(osn_xm, TrackID(ym_layer, osn_ym_tidx, tr_w_sig_ym),
        #                                 track_lower=self.bound_box.xl)
        # osp_ym = self.connect_to_tracks(osp_xm, TrackID(ym_layer, osp_ym_tidx, tr_w_sig_ym),
        #                                 track_lower=self.bound_box.yl)

        # tr_w_sig_xm1 = tr_manager.get_width(xm1_layer, 'ana_sig')
        # _, os_xm1_locs = tr_manager.place_wires(xm1_layer, ['ana_sig']*2, center_coord=osp_ym.middle)
        # osn_xm1, osp_xm1 = self.connect_matching_tracks([osn_ym, osp_ym], xm1_layer, os_xm1_locs,
        #                                                 width=tr_w_sig_xm1)
        # self.add_pin('osn', osn_xm1)
        # self.add_pin('osp', osp_xm1)
        # connect clk to ym
        clk_vm = comp.get_pin('clk')
        # self.add_pin('clk_vm', clk_vm)
        # clk_bbox = BBox(clk_vm.bound_box.xl, clk_vm.bound_box.yl+clk_vm.bound_box.h//4,
        #                 clk_vm.bound_box.xh, clk_vm.bound_box.yh-clk_vm.bound_box.h//4)
        # clk_viaup = self.via_stack_up(tr_manager, clk_vm, hm_layer, vm_layer, 'clk', RoundMode.LESS,
        #                   bbox=clk_bbox) #changed from xm and ym_layer
        # clk_ym_tid = self.grid.coord_to_track(ym_layer, clk_xm[0].middle, RoundMode.NEAREST)
        # clk_ym = self.connect_to_tracks(clk_xm, TrackID(ym_layer, clk_ym_tid,
        #                                                 tr_manager.get_width(ym_layer, 'clk')))
        # self.add_pin('clk_ym', clk_ym, label='clk')

        # fill_conn_layer_intv(self, self.num_tile_rows-2, 0, False)
        # fill_conn_layer_intv(self, self.num_tile_rows-2, 1, False)

        dum_info = []
        vss_hm_top.extend(comp.get_all_port_pins('VSS'))
        vdd_hm_top.extend(comp.get_all_port_pins('VDD'))

        if clk_params:
            vss_hm_top.append(clkgen.get_pin('VSS'))
            vdd_hm_top.append(vdd_top_clk)
        vss_hm_top = self.extend_wires(vss_hm_top, lower=comp.bound_box.xl, upper=comp.bound_box.xh)
        vdd_hm_top = self.extend_wires(vdd_hm_top, lower=comp.bound_box.xl, upper=comp.bound_box.xh)

        if clk_params:
            # self.connect_to_track_wires(clk_viaup[vm_layer], clkgen.get_pin('clk_out')) #change from ym_layer
            self.reexport(clkgen.get_port('logic_clk'))
            self.reexport(clkgen.get_port('ctrl_ext_clk'))
            self.reexport(clkgen.get_port('start'))
            self.reexport(clkgen.get_port('stop'))
        else:
            self.add_pin('clk', clk_vm) #clk_viaup[vm_layer])
        self.add_pin('outn', outn_hm) #outn_viaup[vm_layer]) #change from vm_layer + 2
        self.add_pin('outp', outp_hm)

        if shield:
            coord_x = self.grid.track_to_coord(vm_layer, 1) #FIXME Not sure why bound box is not quite covering whole thing
            coord_y = self.grid.track_to_coord(hm_layer, 1)
            shield_box = BBox(min(bufn.bound_box.xl, comp.bound_box.xl)-coord_x,
                              comp.bound_box.yl, 
                              max(bufp.bound_box.xh, comp.bound_box.xh)+coord_x, 
                              ptap_top.bound_box.yh+coord_y)
            vss_shield = self.add_bbox_array((f'met{xm_layer}', 'drawing'), BBoxArray(shield_box))
            self.add_pin_primitive('VSS_shield', xm_layer, shield_box)
        if side_sup:
            # port_mode = SubPortMode.EVEN if tot_ncol % 4 != 0 else SubPortMode.ODD
            core_ntile = comp_master.num_tile_rows
            core_ntile += 4 if clk_params else 0
            sup_top_layer = self.params['sup_top_layer']
            # Fill side tap and dummy
            for idx in range(core_ntile + 2):
                _nrows = self.get_tile_pinfo(idx).num_rows
                _ptap_list, _ntap_list, _pch_list, _nch_list = [], [], [], []
                tap_side_ncol = ndum_side if idx >= comp_master.num_tile_rows else (tot_ncol - comp_ncol) // 2 - min_sep
                for jdx in range(_nrows):
                    row_info = self.get_row_info(jdx, idx)
                    if row_info.row_type == MOSType.ptap or row_info.row_type == MOSType.ntap:
                        fill_tap_intv(self, idx, 0, tap_side_ncol)
                        fill_tap_intv(self, idx, tot_ncol-tap_side_ncol, tot_ncol,
                                      port_mode=SubPortMode.EVEN if tot_ncol & 1 else SubPortMode.ODD)
                    else:
                        fill_conn_layer_intv(self, idx, jdx, False)
            # Fill supply at side
            bbox_l = BBox(self.arr_info.col_to_coord(0), 0, self.arr_info.col_to_coord(ndum_side), self.bound_box.yh)
            bbox_r = BBox(self.arr_info.col_to_coord(tot_ncol - ndum_side), 0,
                          self.arr_info.col_to_coord(tot_ncol), self.bound_box.yh)

            # vss_xm_core, vdd_xm_core = [], []
            # for vss_hm in vss_hm_top:
            #     print(vss_hm_top)
            #     vss_xm_core.append(self.export_tap_hm(tr_manager, vss_hm, hm_layer, xm_layer,
            #                                           bbox=[vss_hm.lower, vss_hm.middle])[0])
            #     vss_xm_core.append(self.export_tap_hm(tr_manager, vss_hm, hm_layer, xm_layer,
            #                                           bbox=[vss_hm.middle, vss_hm.upper], align_upper=True)[0])
            # for vdd_hm in [vdd_hm_top[0]]:
            #     print(vdd_hm_top)
            #     vdd_xm_core.append(self.export_tap_hm(tr_manager, vdd_hm, hm_layer, xm_layer,
            #                                           bbox=[vdd_hm.lower, vdd_hm.middle])[0])
            #     vdd_xm_core.append(self.export_tap_hm(tr_manager, vdd_hm, hm_layer, xm_layer,
            #                                           bbox=[vdd_hm.middle, vdd_hm.upper], align_upper=True)[0])

            # vdd_xm_core = self.connect_wires(vdd_xm_core, lower=self.bound_box.xl, upper=self.bound_box.xh)
            # vss_xm_core = self.connect_wires(vss_xm_core, lower=self.bound_box.xl, upper=self.bound_box.xh)
            # # for vdd_hm in vdd_hm_top:
            # #     vdd_xm_core.append(self.export_tap_hm(tr_manager, vdd_hm, hm_layer, xm_layer,
            # #                                           bbox=[vss_hm.middle, vss_hm.upper]))
            # print(ndum_side)
           
            #sup_stack_dict_l = self.connect_supply_stack_warr(tr_manager, [vdd_hm_top, vss_hm_top], hm_layer,
            #                                                  vm_layer, bbox_l, side_sup=True)
            #sup_stack_dict_r = self.connect_supply_stack_warr(tr_manager, [vdd_hm_top, vss_hm_top], hm_layer,
            #                                                  vm_layer, bbox_r, side_sup=True, align_upper=True)
            # self.connect_to_track_wires(vdd_xm_core, sup_stack_dict_r[0][vm_layer]+sup_stack_dict_l[0][vm_layer])
            # self.connect_to_track_wires(vss_xm_core, sup_stack_dict_r[1][vm_layer]+sup_stack_dict_l[1][vm_layer])
            vss_hm_top = self.extend_wires(vss_hm_top, lower=bbox_l.xl, upper=bbox_r.xh)
            vdd_hm_top = self.extend_wires(vdd_hm_top, lower=bbox_l.xl, upper=bbox_r.xh)
            sup_vdd_vm_l, sup_vss_vm_l = self.do_power_fill(vm_layer, tr_manager, vdd_hm_top, vss_hm_top, bbox_l)
            sup_vdd_vm_r, sup_vss_vm_r = self.do_power_fill(vm_layer, tr_manager, vdd_hm_top, vss_hm_top, bbox_r)
            # bbox_l.extend(x=(self.bound_box.xl+self.bound_box.xh)//2-1000)
            # bbox_r.extend(x=(self.bound_box.xl+self.bound_box.xh)//2+1000)
            # sup_stack_dict_l = self.connect_supply_stack_warr(tr_manager, [vdd_xm_core, vss_xm_core], xm_layer,
            #                                                   sup_top_layer, bbox_l, side_sup=False)
            # sup_stack_dict_r = self.connect_supply_stack_warr(tr_manager, [vdd_xm_core, vss_xm_core], xm_layer,
            #                                                   sup_top_layer, bbox_r, side_sup=False, align_upper=True)
            # vdd_comp_l, vss_comp_l = self.connect_supply_warr(tr_manager, [comp.get_all_port_pins('VDD'), comp.get_all_port_pins('VSS')], hm_layer,
            #                          BBox(bbox_l.xl, bbox_l.yl, comp.bound_box.xl, comp.bound_box.yh), side_sup=False)
            # vdd_comp_r, vss_comp_r = self.connect_supply_warr(tr_manager, [comp.get_all_port_pins('VDD'), comp.get_all_port_pins('VSS')], hm_layer,
            #                          BBox(comp.bound_box.xh, bbox_l.yl, bbox_r.xh, comp.bound_box.yh), side_sup=False,
            #                          align_upper=True)
            

            # print("VDD PINS: ", sup_vss_vm_l+sup_vss_vm_r)
            # print(self.top_layer)
            # self.add_pin('VSS', sup_vss_vm_l+sup_vss_vm_r)
            # self.add_pin('VDD', sup_vdd_vm_l+sup_vdd_vm_r)
    
            self.add_pin('VSS', sup_vss_vm_l+sup_vss_vm_r, connect=True)
            self.add_pin('VDD', sup_vdd_vm_l+sup_vdd_vm_r, connect=True)

            # for xm in vdd_xm_core:
            #     if xm.bound_box.yh < comp.bound_box.yh:
            #         self.connect_to_track_wires(vdd_comp_l+vdd_comp_r, xm)
            # for xm in vss_xm_core:
            #     if xm.bound_box.yh < comp.bound_box.yh:
            #         self.connect_to_track_wires(vss_comp_l+vss_comp_r, xm)

            # for idx in range(xm1_layer, sup_top_layer+1):
            #     self.connect_wires(sup_stack_dict_l[0][idx]+sup_stack_dict_r[0][idx])
            #     self.connect_wires(sup_stack_dict_l[1][idx]+sup_stack_dict_r[1][idx])

            # # self.add_pin('VDD', sup_stack_dict_l[0][sup_top_layer-1]+sup_stack_dict_r[0][sup_top_layer-1])
            # # self.add_pin('VSS', sup_stack_dict_l[1][sup_top_layer-1]+sup_stack_dict_r[1][sup_top_layer-1])
            #     self.add_pin('VDD', sup_stack_dict_l[0][idx]+sup_stack_dict_r[0][idx])
            #     self.add_pin('VSS', sup_stack_dict_l[1][idx]+sup_stack_dict_r[1][idx])
        else:
            self.add_pin('VSS', vss_top)
            self.add_pin('VSS', comp.get_all_port_pins('VSS'))
            self.add_pin('VDD', comp.get_all_port_pins('VDD'))

    
        self.sch_params = dict(
            buf_outb=False,
            sch_cls=self.params['sch_cls'],
            sa_params=comp_master.sch_params,
            buf_params=buf_master.sch_params,
            dum_info=dum_info,
            clk_params=clk_master.sch_params if clk_params else None
        )

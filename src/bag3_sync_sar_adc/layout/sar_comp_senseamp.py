from itertools import chain
from typing import Mapping, Union, Any, Dict, Type, Optional, Tuple

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB
from bag.util.immutable import Param, ImmutableSortedDict
from bag.util.math import HalfInt
from pybag.enum import RoundMode
from xbase.layout.enum import MOSWireType, SubPortMode, MOSType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from .digital import InvChainCore
from .sar_comp_match import TriTailWrap
from .util.util import connect_conn_dummy_rows, fill_conn_layer_intv, fill_and_collect


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

        tr_w_vm = tr_manager.get_width(vm_layer, 'sig')

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
        outp_vm_tidx = tr_manager.get_next_track(vm_layer, corer.get_pin('clk_vm').track_id.base_index, 'clk', 'sig')
        outp_vm_tidx = max(outp_vm_tidx, corer.get_pin('clk_vm').track_id.base_index + sig_locs.get('out', 0))
        outn_vm_tidx = tr_manager.get_next_track(vm_layer, corel.get_pin('clk_vm').track_id.base_index, 'clk', 'sig',
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
        vm_w = self.tr_manager.get_width(vm_layer, 'sig')
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
        vm_w = self.tr_manager.get_width(vm_layer, 'sig')
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
            ridx_p=-3,
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

        ridx_in = ridx_n + 1
        ridx_nfb = ridx_in + 1
        ridx_swm = ridx_p + 2 
        ridx_swo = ridx_p + 1 
        #ridx_p = ridx_swo -1
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
        w_sw= w_dict['swo']

        if seg_in & 1 or (seg_tail % 4 != 0) or seg_nfb & 1 or seg_pfb & 1:
            raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        # NOTE: make seg_swo even so we can abut transistors
        seg_swo = seg_swm + (seg_swm & 1)
        seg_tail = seg_tail // 2

        # MOS placement
        m_in = self.add_mos(ridx_in, 0, seg_in, w=w_in)
        m_nfb = self.add_mos(ridx_nfb, 0, seg_nfb, w=w_nfb)
        m_pfb = self.add_mos(ridx_p, 0, seg_pfb, w=w_pfb)
        m_tail = self.add_mos(ridx_n, 0, seg_tail, w=w_tail)
        m_swo = self.add_mos(ridx_swo, 0, seg_swo, w=w_sw)
        m_swm = self.add_mos(ridx_swm, 0, seg_swm, w=w_sw)

        # get the track index 
        ng_tid = self.get_track_id(ridx_nfb, MOSWireType.G, wire_name='sig', wire_idx=-1)
        mid_tid = self.get_track_id(ridx_nfb, MOSWireType.DS, wire_name='sig')
        midp_tid = self.get_track_id(ridx_swm, MOSWireType.DS, wire_name='sig')
        pg_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig')
        pclk_tid = self.get_track_id(ridx_swo, MOSWireType.G, wire_name='sig')
        pclk_swm_tid = self.get_track_id(ridx_swm, MOSWireType.G, wire_name='sig')
        nclk_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-1)
        swo_tid = self.get_track_id(ridx_swo, MOSWireType.DS, wire_name='sig')

        vss_conn = m_tail.s
        vdd_conn = [m_pfb.s, m_swo.s, m_swm.s]
        tail_conn = m_tail.d
        clk_conn = m_tail.g

        # NOTE: force even number of columns to make sure VDD conn_layer wires are on even columns.
        # also increasing columns by 2 so there is extra well area
        ncol_tot = self.num_cols + 2
        self.set_mos_size(num_cols=ncol_tot + (ncol_tot & 1)+2)

        # routing
        conn_layer = self.conn_layer
        vm_layer = conn_layer + 2
        vm_w = self.tr_manager.get_width(vm_layer, 'sig')
        grid = self.grid

        # connect tracks/ mos connections that are right next to each other
        tail_in_tid = self.get_track_id(ridx_in, MOSWireType.DS, wire_name='sig')
        tail = self.connect_to_tracks(tail_conn, tail_in_tid)
        out = self.connect_wires([m_nfb.d, m_pfb.d]) #m_swo.d
        mid = self.connect_to_tracks([m_in.s, m_nfb.s], mid_tid ) #m_swm.d
        midp = self.connect_to_tracks([m_swm.d], midp_tid)
        nclk = self.connect_to_tracks(clk_conn, nclk_tid)
        nout = self.connect_to_tracks(m_nfb.g, ng_tid)
        pout = self.connect_to_tracks(m_pfb.g, pg_tid)
        pclk = self.connect_to_tracks(m_swo.g, pclk_tid) #switch output to vdd
        pclk_swm = self.connect_to_tracks(m_swm.g, pclk_swm_tid) #switch mid to vdd
        
        # route clk up one side
        xclk = grid.track_to_coord(conn_layer, m_swm.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xclk, mode=RoundMode.GREATER_EQ)
        clk_vm = self.connect_to_tracks([nclk, pclk,pclk_swm], TrackID(vm_layer, vm_tidx+1.5, width=vm_w))
        self.add_pin('clk_vm', clk_vm)

        # connect mid with the pmos mid transistor
        mid = self.connect_to_tracks([mid, midp],  TrackID(vm_layer, vm_tidx-2, width=vm_w))

        # connect out 
        xout = grid.track_to_coord(conn_layer, m_pfb.g.track_id.base_index)
        vm_tidx = grid.coord_to_track(vm_layer, xout, mode=RoundMode.GREATER_EQ)

        if vertical_out:
            out_sw = self.connect_to_tracks(m_swo.d, swo_tid)
            out_vm = self.connect_to_tracks([nout, pout, out_sw], TrackID(vm_layer, vm_tidx, width=vm_w))
            self.add_pin('out_vm', out_vm)
        else:
            self.add_pin('pout', pout)
            self.add_pin('nout', nout)

        if vertical_sup:
            vdd = vdd_conn
            vss = vss_conn
        else:
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sup')
            vdd = self.connect_to_tracks(vdd_conn, vdd_tid)
            vss_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sup')
            vss = self.connect_to_tracks(vss_conn, vss_tid)

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
                              ('pfb', ridx_p), ('swm', ridx_p+1), ('swo', ridx_p+1)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        w_ans['swo'] = w_ans['swm'] 
        th_ans['swo'] = th_ans['swm'] 
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
        inn_tidx, hm_w = self.get_track_info(ridx_in, MOSWireType.G, wire_name='sig', wire_idx=0,
                                             tile_idx=inst_tile_idx)
        inp_tidx = self.get_track_index(ridx_in, MOSWireType.G, wire_name='sig', wire_idx=-1, tile_idx=inst_tile_idx)
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
        vdd = self.connect_wires(corel.get_all_port_pins('VDD') + corer.get_all_port_pins('VDD'))
        if add_tap:
            ncols, nrows = master.num_cols, master.num_rows

            # placing contacts at non-0 column because columns has been expanded to allow for bigger well
            ntap = self.add_substrate_contact(0, 4, seg=2 * (ncols-4)+ nsep, tile_idx=2, port_mode=SubPortMode.BOTH)
            ptap = self.add_substrate_contact(0, 4, seg=2 * (ncols-4)+ nsep, tile_idx=0, port_mode=SubPortMode.BOTH)
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
        vm_w = tr_manager.get_width(vm_layer, 'sig')

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
        vm_w = tr_manager.get_width(vm_layer, 'sig')

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


class SARComp(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'strongarm_tri')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            ncols_tot='Total number of columns',
            seg_dict='segments dictionary.',
            w_dict='widths dictionary.',
            ngroups_dict='',
            ridx_dict='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_dict={},
            ncols_tot=0,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, Dict] = self.params['seg_dict']
        w_dict: Dict[str, Dict] = self.params['w_dict']

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # Make templates
        # -- Get tidx for clock --
        comp_params = dict(pinfo=(self.get_tile_subpattern(0, 7), self.tile_table), seg_dict=seg_dict, w_dict=w_dict,
                           ngroups_dict=self.params['ngroups_dict'], ridx_dict=self.params['ridx_dict'])
        buf_params = dict(pinfo=self.get_tile_pinfo(7), seg_list=seg_dict['buf'], vertical_sup=True,
                          w_p=w_dict['buf']['p'], w_n=w_dict['buf']['n'], dual_output=True)

        comp_master: MOSBase = self.new_template(TriTailWrap, params=comp_params)
        buf_master: MOSBase = self.new_template(InvChainCore, params=buf_params)

        # floorplanning
        min_sep = self.min_sep_col
        min_sep += min_sep & 1
        comp_ncol = comp_master.num_cols
        buf_ncol = buf_master.num_cols
        buf_ncol_tot = 2 * buf_ncol + min_sep

        core_ncol = max(buf_ncol_tot, comp_ncol)
        side_sup = self.params['ncols_tot'] > core_ncol
        tot_ncol = self.params['ncols_tot'] if side_sup else core_ncol

        # placement
        comp = self.add_tile(comp_master, 0, (tot_ncol - comp_ncol) // 2, commit=False)
        bufp = self.add_tile(buf_master, 7, (tot_ncol - min_sep) // 2, flip_lr=True)
        bufn = self.add_tile(buf_master, 7, (tot_ncol + min_sep) // 2)

        # FIXME: potentially short with comparator internal signals
        outp_tidx = tr_manager.get_next_track(vm_layer, bufp.get_pin('outb').track_id.base_index,
                                              'sig', 'sig', up=False)
        outn_tidx = tr_manager.get_next_track(vm_layer, bufn.get_pin('outb').track_id.base_index, 'sig', 'sig')
        comp.new_master_with(signal_locs={'outn': outn_tidx, 'outp': outp_tidx}, vertical_out=True)
        comp.commit()

        # fill dummy and put tap for output buffers
        port_mode = SubPortMode.ODD if tot_ncol % 4 != 0 else SubPortMode.EVEN
        ndum_side = (tot_ncol - core_ncol) // 2
        ptap_top = self.add_substrate_contact(0, ndum_side, seg=core_ncol, tile_idx=8, port_mode=port_mode)
        # Connect supplies
        vss_top = self.connect_to_tracks(ptap_top, self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=8))
        vdd_top = max(comp.get_all_port_pins('VDD'), key=lambda x: x.track_id.base_index)

        buffer_p_dum, buffer_n_dum = [], []
        fill_and_collect(self, 7, buffer_p_dum, buffer_n_dum, start_col=ndum_side - 1, stop_col=tot_ncol - ndum_side)

        connect_conn_dummy_rows(self, buffer_n_dum, connect_to_sup=True, sup_dum_idx=-1, sup_coord=ptap_top[0].middle)
        connect_conn_dummy_rows(self, buffer_p_dum, connect_to_sup=True, sup_dum_idx=0,
                                sup_coord=self.grid.track_to_coord(hm_layer, vdd_top.track_id.base_index))

        self.set_mos_size(tot_ncol)

        # Connect comp to buf
        self.connect_to_track_wires(comp.get_pin('outp'), bufp.get_pin('nin'))
        self.connect_to_track_wires(comp.get_pin('outn'), bufn.get_pin('nin'))

        # Connect supply
        self.connect_to_track_wires(vss_top, bufp.get_all_port_pins('VSS') + bufn.get_all_port_pins('VSS'))
        self.connect_to_track_wires(vdd_top, bufp.get_all_port_pins('VDD') + bufn.get_all_port_pins('VDD'))

        self.reexport(bufn.get_port('out'), net_name='outp')
        self.reexport(bufp.get_port('out'), net_name='outn')
        [self.reexport(comp.get_port(pname)) for pname in ['osn', 'osp', 'clkb', 'clk', 'inn', 'inp']]
        self.reexport(comp.get_port('outp'), net_name='outp_m')
        self.reexport(comp.get_port('outn'), net_name='outn_m')

        if side_sup:
            port_mode = SubPortMode.EVEN if tot_ncol % 4 != 0 else SubPortMode.ODD
            core_ntile = comp_master.num_tile_rows
            vdd_tap_list, vss_tap_list = [], []
            for idx in range(core_ntile + 2):
                _nrows = self.get_tile_pinfo(idx).num_rows
                for jdx in range(_nrows):
                    row_info = self.get_row_info(jdx, idx)
                    if row_info.row_type == MOSType.ptap:
                        _ptap_l = self.add_substrate_contact(0, 0, seg=ndum_side, tile_idx=idx, port_mode=port_mode)
                        _ptap_r = self.add_substrate_contact(0, tot_ncol, seg=ndum_side, tile_idx=idx, flip_lr=True,
                                                             port_mode=port_mode)
                        _tid = self.get_track_id(jdx, MOSWireType.DS, 'sup', 0, tile_idx=idx)
                        vss_tap_list.append(self.connect_to_tracks([_ptap_l, _ptap_r], _tid))
                    if row_info.row_type == MOSType.ntap:
                        _ntap_l = self.add_substrate_contact(0, 0, seg=ndum_side, tile_idx=idx, port_mode=port_mode)
                        _ntap_r = self.add_substrate_contact(0, tot_ncol, seg=ndum_side, tile_idx=idx, flip_lr=True,
                                                             port_mode=port_mode)
                        _tid = self.get_track_id(jdx, MOSWireType.DS, 'sup', 0, tile_idx=idx)
                        vdd_tap_list.append(self.connect_to_tracks([_ntap_l, _ntap_r], _tid))

            vdd_vm_list, vss_vm_list = [], []
            vdd_hm_list, vss_hm_list = [], []
            vdd_hm_list.extend(vdd_tap_list)
            vss_hm_list.extend(vss_tap_list)
            for idx in range(1, core_ntile + 2, 2):
                dum_n, dum_p = [], []
                fill_and_collect(self, idx, dum_p, dum_n, True, True, 0, ndum_side, True)
                dum = self.connect_wires(dum_n[0] + dum_p[0])[0].to_warr_list()

                _flag = (idx // 2) & 1
                _top_sup = vss_tap_list[idx // 4 + 1] if _flag else vdd_tap_list[idx // 4]
                _bot_sup = vdd_tap_list[idx // 4] if _flag else vss_tap_list[idx // 4]
                sup_locs = self.get_available_tracks(hm_layer, _bot_sup.track_id.base_index,
                                                     _top_sup.track_id.base_index,
                                                     self.arr_info.col_to_coord(0),
                                                     self.arr_info.col_to_coord(ndum_side - 1),
                                                     tr_manager.get_width(hm_layer, 'sup'),
                                                     tr_manager.get_sep(hm_layer, ('sup', 'sup')), include_last=True)
                # FIXME: place sup tracks center span
                if not len(sup_locs) & 1:
                    sup_locs = sup_locs[1:]
                for locs in sup_locs[::2]:
                    _e = self.connect_to_tracks(dum[1::2],
                                                TrackID(hm_layer, locs, tr_manager.get_width(hm_layer, 'sup')))
                    if _flag:
                        vdd_hm_list.append(_e)
                    else:
                        vss_hm_list.append(_e)
                for locs in sup_locs[1::2]:
                    _o = self.connect_to_tracks(dum[::2],
                                                TrackID(hm_layer, locs, tr_manager.get_width(hm_layer, 'sup')))
                    if _flag:
                        vss_hm_list.append(_o)
                    else:
                        vdd_hm_list.append(_o)

                connect_conn_dummy_rows(self, [dum[1::2]], connect_to_sup=True, sup_dum_idx=0,
                                        sup_coord=self.grid.track_to_coord(hm_layer, _bot_sup.track_id.base_index))
                connect_conn_dummy_rows(self, [dum[::2]], connect_to_sup=True, sup_dum_idx=-1,
                                        sup_coord=self.grid.track_to_coord(hm_layer, _top_sup.track_id.base_index))
            sup_vm_locs = self.get_available_tracks(vm_layer, self.arr_info.col_to_track(vm_layer, 0),
                                                    self.arr_info.col_to_track(vm_layer, ndum_side),
                                                    self.bound_box.yl, self.bound_box.yh,
                                                    tr_manager.get_width(vm_layer, 'sup'),
                                                    tr_manager.get_sep(vm_layer, ('sup', 'sup')),
                                                    include_last=True)
            for locs in sup_vm_locs[::2]:
                vss_vm_list.append(self.connect_to_tracks(vss_hm_list,
                                                          TrackID(vm_layer, locs,
                                                                  tr_manager.get_width(vm_layer, 'sup'))))
            for locs in sup_vm_locs[1::2]:
                vdd_vm_list.append(self.connect_to_tracks(vdd_hm_list,
                                                          TrackID(vm_layer, locs,
                                                                  tr_manager.get_width(vm_layer, 'sup'))))

            vdd_hm_list, vss_hm_list = [], []
            vdd_hm_list.extend(vdd_tap_list)
            vss_hm_list.extend(vss_tap_list)
            for idx in range(1, core_ntile + 2, 2):
                dum_n, dum_p = [], []
                fill_and_collect(self, idx, dum_p, dum_n, True, True, tot_ncol - ndum_side, tot_ncol, True)
                dum = self.connect_wires(dum_n[0] + dum_p[0])[0].to_warr_list()

                _flag = (idx // 2) & 1
                _top_sup = vss_tap_list[idx // 4 + 1] if _flag else vdd_tap_list[idx // 4]
                _bot_sup = vdd_tap_list[idx // 4] if _flag else vss_tap_list[idx // 4]
                sup_locs = self.get_available_tracks(hm_layer, _bot_sup.track_id.base_index,
                                                     _top_sup.track_id.base_index,
                                                     self.arr_info.col_to_coord(tot_ncol - ndum_side + 1),
                                                     self.arr_info.col_to_coord(tot_ncol),
                                                     tr_manager.get_width(hm_layer, 'sup'),
                                                     tr_manager.get_sep(hm_layer, ('sup', 'sup')), include_last=True)
                if not len(sup_locs) & 1:
                    sup_locs = sup_locs[1:]
                vdd_hm_list.extend(vdd_tap_list)
                vss_hm_list.extend(vss_tap_list)
                for locs in sup_locs[::2]:
                    _e = self.connect_to_tracks(dum[1::2],
                                                TrackID(hm_layer, locs, tr_manager.get_width(hm_layer, 'sup')))
                    if _flag:
                        vdd_hm_list.append(_e)
                    else:
                        vss_hm_list.append(_e)
                for locs in sup_locs[1::2]:
                    _o = self.connect_to_tracks(dum[::2],
                                                TrackID(hm_layer, locs, tr_manager.get_width(hm_layer, 'sup')))
                    if _flag:
                        vss_hm_list.append(_o)
                    else:
                        vdd_hm_list.append(_o)

                connect_conn_dummy_rows(self, [dum[1::2]], connect_to_sup=True, sup_dum_idx=0,
                                        sup_coord=self.grid.track_to_coord(hm_layer, _bot_sup.track_id.base_index))
                connect_conn_dummy_rows(self, [dum[::2]], connect_to_sup=True, sup_dum_idx=-1,
                                        sup_coord=self.grid.track_to_coord(hm_layer, _top_sup.track_id.base_index))
            sup_vm_locs = self.get_available_tracks(vm_layer,
                                                    self.arr_info.col_to_track(vm_layer, tot_ncol - ndum_side + 1),
                                                    self.arr_info.col_to_track(vm_layer, tot_ncol),
                                                    self.bound_box.yl, self.bound_box.yh,
                                                    tr_manager.get_width(vm_layer, 'sup'),
                                                    tr_manager.get_sep(vm_layer, ('sup', 'sup')),
                                                    include_last=True)
            sup_vm_locs = sup_vm_locs[::-1]
            for locs in sup_vm_locs[::2]:
                vss_vm_list.append(self.connect_to_tracks(vss_hm_list,
                                                          TrackID(vm_layer, locs,
                                                                  tr_manager.get_width(vm_layer, 'sup'))))
            for locs in sup_vm_locs[1::2]:
                vdd_vm_list.append(self.connect_to_tracks(vdd_hm_list,
                                                          TrackID(vm_layer, locs,
                                                                  tr_manager.get_width(vm_layer, 'sup'))))

            self.add_pin('VDD', vdd_vm_list, show=self.show_pins)
            self.add_pin('VSS', vss_vm_list, show=self.show_pins)
        else:
            self.add_pin('VSS', vss_top, connect=True)
            self.add_pin('VSS', comp.get_all_port_pins('VSS'), connect=True)
            self.add_pin('VDD', comp.get_all_port_pins('VDD'), connect=True)

        self.sch_params = dict(
            sa_params=comp_master.sch_params,
            buf_params=buf_master.sch_params,
        )

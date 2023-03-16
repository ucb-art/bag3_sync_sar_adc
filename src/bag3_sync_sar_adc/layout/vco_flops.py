from itertools import chain
from typing import Any, Dict, Type, Optional, Tuple
from typing import Mapping, Union

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.routing.base import TrackID
from bag.layout.template import TemplateDB
from bag.util.immutable import ImmutableSortedDict
from bag.util.immutable import Param
from bag.util.math import HalfInt
from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from pybag.enum import RoundMode

class CnterLatchHalf(MOSBase):
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
            vertical_out=True,
            vertical_sup=False,
        )

    def draw_layout(self):
        place_info = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(place_info)

        seg_dict: ImmutableSortedDict[str, int] = self.params['seg_dict']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']

        tr_manager = self.tr_manager

        w_dict, th_dict = self._get_w_th_dict(ridx_n, ridx_p)
        seg_nin = seg_dict['nin']
        seg_pin = seg_dict['pin']
        seg_nfb = seg_dict['nfb']
        seg_pfb = seg_dict['pfb']
        seg_ptail = seg_dict['ptail']
        seg_ntail = seg_dict['ntail']
        w_nin = w_dict['nin']
        w_pin = w_dict['pin']
        w_ntail = w_dict['ntail']
        w_ptail = w_dict['ptail']
        w_nfb = w_dict['nfb']
        w_pfb = w_dict['pfb']

        if seg_nin & 1 or seg_pin & 1:
        # if seg_nin & 1 or seg_pin & 1 or (seg_ntail % 4 != 0) or (seg_ptail % 4 != 0):
                raise ValueError('in, tail, nfb, or pfb must have even number of segments')
        seg_ptail = seg_ptail // 2
        seg_ntail = seg_ntail // 2

        # placement
        min_sep = self.min_sep_col
        m_ntail = self.add_mos(ridx_n, 0, seg_ntail, w=w_ntail, g_on_s=bool(seg_ntail & 1))
        m_ptail = self.add_mos(ridx_p, 0, seg_ptail, w=w_ptail, g_on_s=bool(seg_ptail & 1))
        m_nin = self.add_mos(ridx_n, seg_ntail, seg_nin, w=w_nin)
        m_pin = self.add_mos(ridx_p, seg_ptail, seg_pin, w=w_pin)

        m_nfb = self.add_mos(ridx_n, seg_nin+seg_ntail+min_sep, seg_nfb, w=w_nfb, g_on_s=bool(seg_nfb & 1))
        m_pfb = self.add_mos(ridx_p, seg_pin+seg_ptail+min_sep, seg_pfb, w=w_pfb, g_on_s=bool(seg_pfb & 1))

        nout_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=0)
        pout_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        nclk_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0+sig_locs.get('clk', 0))
        pclk_tid = self.get_track_id(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0+sig_locs.get('clk', 0))
        ntail_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=1)
        ptail_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=-2)
        in_tid = self.get_track_id(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=3)

        # # NOTE: force even number of columns to make sure VDD conn_layer wires are on even columns.
        ncol_tot = self.num_cols
        ncol_tot += 1  # left some space for clock signal routing
        # self.set_mos_size(num_cols=ncol_tot + (ncol_tot & 1))
        self.set_mos_size(ncol_tot)
        # routing
        conn_layer = self.conn_layer
        vm_layer = conn_layer + 2
        sig_vm_w = self.tr_manager.get_width(vm_layer, 'sig')

        nclk = self.connect_to_tracks([m_ntail.g], nclk_tid)
        pclk = self.connect_to_tracks([m_ptail.g], pclk_tid)
        out = self.connect_wires([m_nfb.g, m_pfb.g])
        nout = self.connect_to_tracks([m_nin.d, m_nfb.s], nout_tid)
        pout = self.connect_to_tracks([m_pin.d, m_pfb.s], pout_tid)
        ntail = self.connect_to_tracks([m_nin.s, m_ntail.d] if seg_ntail & 1 else [m_nin.s, m_ntail.s], ntail_tid)
        ptail = self.connect_to_tracks([m_pin.s, m_ptail.d] if seg_ptail & 1 else [m_pin.s, m_ptail.s], ptail_tid)
        fb_g = self.connect_wires([m_nfb.g, m_pfb.g])
        in_g = self.connect_wires([m_nin.g, m_pin.g])
        in_hm = self.connect_to_tracks(in_g, in_tid)

        vdd_conn = [m_pfb.d, m_ptail.s] if seg_ptail & 1 else [m_pfb.d, m_ptail.d]
        vss_conn = [m_nfb.d, m_ntail.s] if seg_ntail & 1 else [m_nfb.d, m_ntail.d]
        if vertical_sup:
            tr_w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sup')
            vdd_hm = self.connect_to_tracks(vdd_conn, vdd_tid)
            vss_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sup')
            vss_hm = self.connect_to_tracks(vss_conn, vss_tid)
            # export to vm
            sup_vm_tidx = self.arr_info.col_to_track(vm_layer, self.num_cols, mode=RoundMode.NEAREST)
            sup_vm_tidx = tr_manager.get_next_track(vm_layer, sup_vm_tidx, 'clk', 'clk', up=False)
            sup_vm_tidx = tr_manager.get_next_track(vm_layer, sup_vm_tidx, 'clk', 'sup', up=False)
            vdd = [self.connect_to_tracks(vdd_hm, TrackID(vm_layer, sup_vm_tidx, tr_w_sup_vm))]
            vss = [self.connect_to_tracks(vss_hm, TrackID(vm_layer, sup_vm_tidx, tr_w_sup_vm))]
            # sup_vm_tidx = self.arr_info.col_to_track(vm_layer, 0, mode=RoundMode.NEAREST)
            # vdd.append(self.connect_to_tracks(vdd_hm, TrackID(vm_layer, sup_vm_tidx, tr_w_sup_vm)))
            # vss.append(self.connect_to_tracks(vss_hm, TrackID(vm_layer, sup_vm_tidx, tr_w_sup_vm)))

        else:
            vdd_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sup')
            vdd = self.connect_to_tracks(vdd_conn, vdd_tid)
            vss_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sup')
            vss = self.connect_to_tracks(vss_conn, vss_tid)

        if vertical_out:
            vm_tidx = self.arr_info.col_to_track(vm_layer, 0, mode=RoundMode.NEAREST)
            vm_tidx = sig_locs.get('out', vm_tidx)
            out_vm = self.connect_to_tracks([nout, pout], TrackID(vm_layer, vm_tidx, width=sig_vm_w))
            vm_tidx = self.arr_info.col_to_track(vm_layer, 1, mode=RoundMode.NEAREST)
            vm_tidx = sig_locs.get('in', vm_tidx)
            in_vm = self.connect_to_tracks(in_hm, TrackID(vm_layer, vm_tidx, width=sig_vm_w))
            self.add_pin('out_vm', out_vm)
            self.add_pin('in_vm', in_vm)
        else:
            self.add_pin('in', in_hm)
            self.add_pin('pout', pout)
            self.add_pin('nout', nout)

        self.add_pin('VSS', vss)
        self.add_pin('VDD', vdd)
        self.add_pin('ntail', ntail)
        self.add_pin('ptail', ptail)

        #extend clk
        clk_ext_x = self.arr_info.col_to_coord(max(seg_ptail, seg_ntail)+max(seg_nin, seg_pin)//2)
        nclk = self.extend_wires(nclk, upper=clk_ext_x)
        pclk = self.extend_wires(pclk, upper=clk_ext_x)

        self.add_pin('nclk', nclk)
        self.add_pin('pclk', pclk)
        self.add_pin('out', out)
        self.add_pin('fb_in', fb_g)

        self.sch_params = dict(
            lch=self.arr_info.lch,
            seg_dict=seg_dict,
            w_dict=w_dict,
            th_dict=th_dict,
        )

    def _get_w_th_dict(self, ridx_n: int, ridx_p: int)\
            -> Tuple[ImmutableSortedDict[str, int], ImmutableSortedDict[str, str]]:
        w_dict: Mapping[str, int] = self.params['w_dict']

        w_ans = {}
        th_ans = {}
        for name, row_idx in [('nfb', ridx_n), ('nin', ridx_n), ('pfb', ridx_p), ('pin', ridx_p), ('ntail', ridx_n),
                              ('ptail', ridx_p)]:
            rinfo = self.get_row_info(row_idx, 0)
            w = w_dict.get(name, 0)
            if w == 0:
                w = rinfo.width
            w_ans[name] = w
            th_ans[name] = rinfo.threshold

        return ImmutableSortedDict(w_ans), ImmutableSortedDict(th_ans)


class CnterLatch(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'vco_cnter_latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = CnterLatchHalf.get_params_info()
        ans['even_center'] = 'True to force center column to be even.'
        ans['flip_io'] = 'True to flip input/output, easier for inter-connection'
        ans['vertical_clk'] = 'True to add vertical clock signals'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = CnterLatchHalf.get_default_param_values()
        ans['even_center'] = False
        ans['flip_io'] = False
        ans['vertical_clk'] = False
        return ans

    def draw_layout(self):
        master: CnterLatchHalf = self.new_template(CnterLatchHalf, params=self.params)
        self.draw_base(master.draw_base_info)

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        flip_io: bool = self.params['flip_io']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        vertical_clk: bool = self.params['vertical_clk']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        # placement
        nsep = self.min_sep_col if vertical_clk else 0
        nsep += nsep & 1
        nhalf = master.num_cols
        out_vm_tidx = self.arr_info.col_to_track(vm_layer, 1)
        in_vm_tidx = self.arr_info.col_to_track(vm_layer, 2)
        sig_locs_new = {'out': sig_locs.get('out', out_vm_tidx), 'in': sig_locs.get('in', in_vm_tidx)}

        # shift hm clk idx
        if flip_io:
            sig_locs_new['clk'] = 1
        master_params = self.params.copy(append=dict(sig_locs=sig_locs_new))
        master: CnterLatchHalf = self.new_template(CnterLatchHalf, params=master_params)
        corel = self.add_tile(master, 0, nhalf, flip_lr=True)
        corer = self.add_tile(master, 0, nhalf + nsep)
        self.set_mos_size(num_cols=nsep + 2 * nhalf)

        # routing
        # in_tidx, hm_w = self.get_track_info(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=3)
        hm_w = tr_manager.get_width(hm_layer, 'sig')
        # inp_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=-2)
        outn_tidx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2)
        outp_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=2)
        # if flip_io:
        #     inn_tidx, inp_tidx, outn_tidx, outp_tidx = outp_tidx, outn_tidx, inp_tidx, inn_tidx
        #
        # inp = self.connect_to_tracks(corel.get_pin('in'), TrackID(hm_layer, in_tidx, hm_w))
        # inn = self.connect_to_tracks(corer.get_pin('in'), TrackID(hm_layer, in_tidx, hm_w))
        outp, outn = self.connect_differential_tracks(corer.get_all_port_pins('out'),
                                                      corel.get_all_port_pins('out'),
                                                      hm_layer, outp_tidx, outn_tidx, width=hm_w)
        if vertical_out:
            outp_vm = corel.get_pin('out_vm')
            outn_vm = corer.get_pin('out_vm')
            inp_vm = corel.get_pin('in_vm')
            inn_vm = corer.get_pin('in_vm')
            outp, outn = self.connect_differential_wires(outp_vm, outn_vm, outp, outn)

            inp_vm = self.extend_wires(inp_vm, upper=outp_vm.upper, lower=outp_vm.lower)
            inn_vm = self.extend_wires(inn_vm, upper=outp_vm.upper, lower=outp_vm.lower)

            self.add_pin('d', inp_vm)
            self.add_pin('dn', inn_vm)

            self.add_pin('outn', outp)
            self.add_pin('outp', outn)
            self.add_pin('outn', outp_vm)
            self.add_pin('outp', outn_vm)
        else:
            self.reexport(corel.get_port('in'), net_name='d')
            self.reexport(corer.get_port('in'), net_name='dn')
            self.add_pin('outp', [corel.get_pin('pout'), corel.get_pin('nout'), outp], connect=True)
            self.add_pin('outn', [corer.get_pin('pout'), corer.get_pin('nout'), outn], connect=True)

        if vertical_sup:
            vss = list(chain(corel.get_all_port_pins('VSS'), corer.get_all_port_pins('VSS')))
            vdd = list(chain(corel.get_all_port_pins('VDD'), corer.get_all_port_pins('VDD')))
            vdd_hm_tid = self.grid.coord_to_track(hm_layer, vdd[0].middle, mode=RoundMode.NEAREST)
            vss_hm_tid = self.grid.coord_to_track(hm_layer, vss[0].middle, mode=RoundMode.NEAREST)
            tr_w_sup_hm = tr_manager.get_width(hm_layer, 'sup')
            vdd_hm = self.connect_to_tracks(vdd, TrackID(hm_layer, vdd_hm_tid, tr_w_sup_hm))
            vss_hm = self.connect_to_tracks(vss, TrackID(hm_layer, vss_hm_tid, tr_w_sup_hm))
            self.add_pin('VDD', vdd_hm)
            self.add_pin('VSS', vss_hm)
        else:
            vss = self.connect_wires(list(chain(corel.get_all_port_pins('VSS'), corer.get_all_port_pins('VSS'))))
            vdd = self.connect_wires(list(chain(corel.get_all_port_pins('VDD'), corer.get_all_port_pins('VDD'))))
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        if vertical_clk:
            _, clk_tidxs = self.tr_manager.place_wires(vm_layer, ['clk', 'clk'],
                                                       center_coord=self.arr_info.col_to_coord(self.num_cols//2))
            clk_vm = self.connect_to_tracks([corel.get_pin('pclk'), corer.get_pin('pclk')],
                                            TrackID(vm_layer, clk_tidxs[0], tr_manager.get_width(vm_layer, 'clk')))
            nclk_vm = self.connect_to_tracks([corel.get_pin('nclk'), corer.get_pin('nclk')],
                                             TrackID(vm_layer, clk_tidxs[1], tr_manager.get_width(vm_layer, 'clk')))
            self.add_pin('clkn', clk_vm)
            self.add_pin('clkp', nclk_vm)
        else:
            self.add_pin('clkn', self.connect_wires([corel.get_pin('pclk'), corer.get_pin('pclk')]))
            self.add_pin('clkp', self.connect_wires([corel.get_pin('nclk'), corer.get_pin('nclk')]))

        self.connect_wires([corel.get_pin('ntail'), corer.get_pin('ntail')])
        self.connect_wires([corel.get_pin('ptail'), corer.get_pin('ptail')])

        self.sch_params = master.sch_params
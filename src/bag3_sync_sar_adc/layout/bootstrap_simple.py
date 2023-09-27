from typing import Any, Dict, Type, Optional, List, Mapping, Union
import copy

from pybag.enum import MinLenMode, RoundMode, Orientation, Direction
from pybag.core import Transform, BBox, BBoxArray, COORD_MAX, COORD_MIN

from bag.util.immutable import Param, ImmutableSortedDict
from bag.layout.template import TemplateDB, TemplateBase
from bag.design.database import ModuleDB, Module
from bag.layout.routing.base import TrackManager, TrackID, WireArray
from bag.layout.routing.base import WDictType, SpDictType
from bag3_sync_sar_adc.layout.sar_samp import Sampler
from bag3_sync_sar_adc.layout.util.util import fill_conn_layer_intv

from xbase.layout.enum import MOSWireType, SubPortMode
from xbase.layout.mos.placement.data import TilePatternElement, TilePattern
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase, MOSArrayPlaceInfo
from xbase.layout.mos.top import GenericWrapper
from xbase.layout.cap.mim import MIMCap


class BootstrapNMOS_simple(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._dum_sampler = False

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            sampler_params='Unit sampler parameters',
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Number of segments.',
            w_n='nmos width',
            ridx_samp='index for nmos row with sampler',
            ridx_off0='index for nmos row with turn-off transistor 0',
            ridx_off1='index for nmos row with turn-off transistor 1',
            dum_sampler='True to enable dummy sampler',
            fast_on='True to fast turn-on XON_N',
            break_outputs='True to break output to connect to different caps',
            nside='Negative side of sampler, to xcp input',
            swap_inout='Change input and output for samplers',
            no_sampler='True to exclude sampler, export vg and vin signals',
            dummy_off='For signal sampler, need to match gate resistance of dummy',

        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=84,
            ridx_samp=0,
            ridx_off0=0,
            ridx_off1=1,
            dum_sampler=False,
            fast_on=False,
            break_outputs=False,
            nside=True,
            swap_inout=False,
            no_sampler=False,
            dummy_off=False,

        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        dum_sampler: bool = self.params['dum_sampler']
        fast_on: bool = self.params['fast_on']
        break_outputs: bool = self.params['break_outputs']
        no_sampler: bool = self.params['no_sampler']
        dummy_off: bool = self.params['dummy_off']
        swap_inout: bool = self.params['swap_inout']
        
        if dum_sampler:
            pinfo0 = [TilePatternElement(pinfo[1]['samp_tile'], flip=True)] + \
                     [TilePatternElement(pinfo[1][tile['name']]) for tile in self.params['pinfo']['tiles']]
            self.draw_base((TilePattern(pinfo0), pinfo[1]))
            tidx_samp, tidx_off = 2, 3
        else:
            self.draw_base(pinfo)
            tidx_samp, tidx_off = 1, 2

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        ridx_samp: int = self.params['ridx_samp']
        ridx_off0: int = self.params['ridx_off0']
        ridx_off1: int = self.params['ridx_off1']

        seg_samp = seg_dict['sampler']
        seg_off0 = seg_dict['off0']
        seg_off1 = seg_dict['off1']
        seg_on_n = seg_dict['on_n']
        seg_capn = seg_dict['cap_n']
        min_sep = self.min_sep_col

        # Calculate segments
        seg_max = max(seg_off0 + seg_on_n + min_sep, seg_off1 + seg_capn + min_sep)
        if (no_sampler or dummy_off):
            sampler_master = None
        else: 
            sampler_params = self.params['sampler_params'].copy(append=dict(unit_pinfo = False, pinfo=self.get_draw_base_sub_pattern(1,2))) 
            sampler_master = self.new_template(Sampler, params=sampler_params)
            sampler_ncols = sampler_master.num_cols
            seg_max = max(seg_max, sampler_ncols )
            if ((seg_max - sampler_ncols) // 2) & 1:
                seg_max += 2
        seg_tot = seg_max
        seg_tot2 = seg_tot // 2
        

        # Place other transistors
        # Align cap_n and on_n
        cur_col_off0 = seg_tot2 - (seg_on_n + seg_off0 + min_sep) // 2
        cur_col_off1 = seg_tot2 - (seg_off1 + seg_capn + min_sep) // 2
        cur_col0 = min(cur_col_off0, cur_col_off1)
        cur_col0 += (cur_col0 & 1)

        # Middle row: on_n and off0
        on_n = self.add_mos(ridx_off0, cur_col0, seg_on_n, w=w_n, g_on_s=False, tile_idx=tidx_off)
        cur_col = cur_col0 + seg_on_n + min_sep
        cur_col += (cur_col & 1)
        off0 = self.add_mos(ridx_off0, cur_col, seg_off0, w=w_n, g_on_s=True, tile_idx=tidx_off)

        # Top row: cap_n and off1
        capn = self.add_mos(ridx_off1, cur_col0, seg_capn, w=w_n, g_on_s=True, tile_idx=tidx_off)
        cur_col = cur_col0 + seg_capn + min_sep
        cur_col += (cur_col & 1)
        off1 = self.add_mos(ridx_off1, cur_col, seg_off1, g_on_s=False, tile_idx=tidx_off)

        if not no_sampler:
            # Place sampler first, If has dummy sampler, put output btw dum_samp and samp
            sampler_col = (seg_max // 2 - sampler_ncols // 2)
            sampler_col += sampler_col & 1
            sampler = self.add_tile(sampler_master, 1, sampler_col+1) 
        else:
            fill_conn_layer_intv(self, 1, 0, extend_to_gate=False, fill_empty=True)
            if dummy_off:
                fill_conn_layer_intv(self, 1, 1, extend_to_gate=False, fill_empty=True)
            sampler = None

        nouts = len(seg_dict['sampler'])
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        tr_manager = self.tr_manager
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')
        out_xm_tidx = self.grid.coord_to_track(xm_layer, (sampler if not no_sampler else self).bound_box.yl,
                                               RoundMode.NEAREST)

        if sampler:
            if nouts > 1:
                for idx in range(nouts):
                    self.add_pin(f'out<{idx}>', sampler.get_all_port_pins(f'out<{idx}>'))
            
            else:
                if (not swap_inout):
                    self.add_pin('out', sampler.get_all_port_pins('out<0>'))
                else: 
                    self.add_pin('out', sampler.get_all_port_pins('in'))

        # Add sub-contact
        sub = self.add_substrate_contact(0, 0, seg=seg_tot, tile_idx=tidx_off + 1, port_mode=SubPortMode.ODD)
        sub_bot = self.add_substrate_contact(0, 0, seg=seg_tot, tile_idx=0, port_mode=SubPortMode.ODD)
        
        self.set_mos_size()

        # --- Connection
        # - sampler input and vg
        if (not swap_inout):
            vin = sampler.get_pin('in') 
        else:
            vin = sampler.get_pin('out<0>')
       
        # # - sampler -> on_n
        self.connect_to_track_wires(on_n.s, vin)
        
        cap_bot_on = self.connect_to_tracks(on_n.d,
                                            self.get_track_id(ridx_off0, MOSWireType.DS, 'sig', 0, tile_idx=tidx_off))
        self.connect_to_track_wires(sampler.get_pin('sam'), on_n.g)
        # # - sampler -> off0
        self.connect_to_track_wires(sampler.get_pin('sam'), off0.d)
        # - off0
        vdd_off0 = self.connect_to_tracks(off0.g, self.get_track_id(ridx_off0, MOSWireType.G, 'sup', 0,
                                                                    tile_idx=tidx_off))
        _vg_off = self.connect_to_tracks(off0.s, self.get_track_id(ridx_off0, MOSWireType.DS, 'sig', 0,
                                                                   tile_idx=tidx_off))
        # - off0 -> off1
        self.connect_wires([off0.s, off1.s])
        # - off1
        samp_b = self.connect_to_tracks([off1.g, capn.g],
                                        self.get_track_id(ridx_off1, MOSWireType.G, 'clk', 0, tile_idx=tidx_off))
        self.connect_to_tracks(off1.s,
                               self.get_track_id(ridx_off1, MOSWireType.DS, 'sig', 0, tile_idx=tidx_off))

        # - on_n -> cap_n
        # cap_n
        cap_bot = self.connect_to_tracks(capn.s,
                                         self.get_track_id(ridx_off1, MOSWireType.DS, 'sig', 0, tile_idx=tidx_off))
        
        # - tap
        vss = self.connect_to_tracks([sub, off1.d, capn.d],
                                     self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=tidx_off + 1))
        vss2 = self.connect_to_tracks(sub_bot,
                                     self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=0))
        for pname, p in zip(['in', 'vg', 'sample_b', 'cap_bot'],
                            [vin, sampler.get_pin('sam'), samp_b, [cap_bot, cap_bot_on]]):
            self.add_pin(pname, p, show=self.show_pins)

        self.add_pin('VSS', [vss, vss2])
        self.add_pin('VDD', vdd_off0)

        dev_info = dict(
            XSAMPLE={'nf': seg_dict['sampler'], 'seg_n': sampler_params['sampler_unit_params']['seg'],
                      'w_n': self.params['w_n'],
                     'intent': self.get_tile_pinfo(0).get_row_place_info(ridx_samp).row_info.threshold},
            XON_N={'nf': seg_dict['on_n'], 'w_n': self.params['w_n'],
                   'intent': self.get_tile_pinfo(tidx_off).get_row_place_info(ridx_off0).row_info.threshold},
            XOFF_N0={'nf': seg_dict['off0'], 'w_n': self.params['w_n'],
                     'intent': self.get_tile_pinfo(tidx_off).get_row_place_info(ridx_off0).row_info.threshold},
            XOFF_N1={'nf': seg_dict['off1'], 'w_n': self.params['w_n'],
                     'intent': self.get_tile_pinfo(tidx_off).get_row_place_info(ridx_off1).row_info.threshold},
            XCAP_N={'nf': seg_dict['cap_n'],'w_n': self.params['w_n'],
                    'intent': self.get_tile_pinfo(tidx_off).get_row_place_info(ridx_off1).row_info.threshold},
        )
        if dum_sampler:
            dev_info.update({'XSAMPLE_DUM': dev_info['XSAMPLE']})
        if seg_dict.get('pre', 0):
            dev_info.update(
                XPRE={'nf': seg_dict['pre'],
                      'intent': self.get_tile_pinfo(tidx_off).get_row_place_info(ridx_off1).row_info.threshold},
            )
        self.sch_params = dict(
            dev_info=dev_info,
            has_dum_sampler=dum_sampler
        )


class BootstrapNWL_simple(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='Number of segments.',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row with turn-off transistor 0',
            ridx_p='index for pmos row with turn-off transistor 1',
            fast_on='True to enable fast turn on ON_N',
            has_buf='True to add buffer before sample',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=84,
            w_p=110,
            ridx_n=0,
            ridx_p=1,
            fast_on=False,
            has_buf=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        seg_dict: Dict[str, Union[int, List]] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        w_p: int = self.params['w_p']
        ridx_n: int = self.params['ridx_n']
        ridx_p: int = self.params['ridx_p']
        fast_on: int = self.params['fast_on']
        has_buf: int = self.params['has_buf']

        min_sep = self.min_sep_col

        seg_cap_p = seg_dict['cap_p']
        seg_on_p = seg_dict['on_p']
        seg_cap_p_aux = seg_dict.get('cap_p_aux', 0)

        seg_mid = seg_dict['mid']
        seg_invn = seg_dict['inv_n']
        seg_invp = seg_dict['inv_p']

        # buffers layout in a reversed order

        # Assign tiles for each part
        tidx_inv, tidx_nwl = 1, 3
        tidx_ntapb, tidx_ptap, tidx_ntapt = 2, 0, 4

        # Setup track manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        tr_manager = self.tr_manager

        # --- Floorplan ---
        # = calc. total seg.
        seg_tot_nwl = seg_cap_p_aux + min_sep if seg_cap_p_aux else 0
        seg_tot_nwl += seg_cap_p + seg_on_p + min_sep

        if fast_on:
            seg_inv_p = seg_dict['inv_p_vg2']
            seg_inv_n0 = seg_dict['inv_n0_vg2']
            seg_inv_n1 = seg_dict['inv_n1_vg2']
            seg_tot_nwl += min_sep + seg_inv_p
            seg_tot_nwl = max(seg_tot_nwl, seg_inv_n0 + seg_inv_n1 + min_sep)
        else:
            seg_inv_p, seg_inv_n0, seg_inv_n1 = 0, 0, 0

        if has_buf:
            seg_invn_buf = seg_dict.get('inv_n_buf', 2)
            seg_tot0_inv = max(seg_invp, seg_invn + min_sep + seg_mid + min_sep + seg_invn_buf + min_sep)
            seg_buf_list_n = seg_dict['buf_n'][::-1]
            seg_buf_list_p = seg_dict['buf_p'][::-1]
            if len(seg_buf_list_n) != len(seg_buf_list_p):
                raise ValueError("Buffer length n/p doesn't match")
            seg_tot_inv = seg_tot0_inv + sum([max(a, b) for a, b in zip(seg_buf_list_n, seg_buf_list_p)])
        else:
            seg_tot0_inv = max(seg_invp, seg_invn + min_sep + seg_mid + min_sep)
            seg_buf_list_n, seg_buf_list_p = [], []
            seg_tot_inv = seg_tot0_inv
            seg_invn_buf = 0
        seg_tot = max(seg_tot_nwl, seg_tot_inv)

        # Add substrate connection
        ntap_b = self.add_substrate_contact(0, 0, seg=seg_tot, tile_idx=tidx_ntapb)
        ntap_t = self.add_substrate_contact(0, 0, seg=seg_tot, tile_idx=tidx_ntapt)
        ptap = self.add_substrate_contact(0, 0, seg=seg_tot, tile_idx=tidx_ptap)

        # Add inv for vg
        invp = self.add_mos(ridx_p, seg_tot0_inv - seg_invp, seg=seg_invp, tile_idx=tidx_inv, w=w_p)
        if has_buf:
            cur_col = seg_tot0_inv - seg_invn_buf - min_sep
            invn_buf = self.add_mos(ridx_n, cur_col, seg=seg_invn_buf, tile_idx=tidx_inv, w=w_n)
            cur_col -= min_sep + seg_invn
            invn = self.add_mos(ridx_n, cur_col, seg=seg_invn, tile_idx=tidx_inv, w=w_n)
            cur_col -= min_sep + seg_mid
            mid = self.add_mos(ridx_n, cur_col, seg=seg_mid, tile_idx=tidx_inv, w=w_n)
        else:
            cur_col = seg_tot0_inv - seg_invn - min_sep
            invn = self.add_mos(ridx_n, cur_col, seg=seg_invn, tile_idx=tidx_inv, w=w_n)
            cur_col -= min_sep + seg_mid
            mid = self.add_mos(ridx_n, cur_col, seg=seg_mid, tile_idx=tidx_inv, w=w_n)
            invn_buf = None

        # Add buffers for inv_vg
        buf_n_list, buf_p_list = [], []
        cur_loc = seg_tot0_inv
        if has_buf:
            for idx, (segn, segp) in enumerate(zip(seg_buf_list_n, seg_buf_list_p)):
                buf_n_list.append(self.add_mos(ridx_n, cur_loc, seg=segn, tile_idx=tidx_inv, w=w_n))
                buf_p_list.append(self.add_mos(ridx_p, cur_loc, seg=segp, tile_idx=tidx_inv, w=w_p))
                cur_loc += max(segn, segp)

        # Add XCAP_P and inv for vg2
        cur_col = 0
        if seg_cap_p_aux:
            cap_p_aux = self.add_mos(ridx_p, cur_col, seg_cap_p_aux, w=w_p, tile_idx=tidx_nwl)
            cur_col += seg_cap_p_aux + min_sep
        else:
            cap_p_aux = None

        cap_p = self.add_mos(ridx_p, cur_col, seg_cap_p, w=w_p, tile_idx=tidx_nwl)
        cur_col += seg_cap_p + min_sep
        on_p = self.add_mos(ridx_p, cur_col, seg_on_p, w=w_p, tile_idx=tidx_nwl, g_on_s=True)
        cur_col += seg_on_p + min_sep
        cur_col += (cur_col & 1)
        if fast_on:
            inv_p_2 = self.add_mos(ridx_p, cur_col, seg_inv_p, w=w_n, tile_idx=tidx_nwl, g_on_s=True)
            cur_col += seg_inv_p - seg_inv_n0
            inv_n0 = self.add_mos(ridx_n, cur_col, seg_inv_n0, w=w_n, tile_idx=tidx_nwl, g_on_s=True)
            cur_col -= min_sep + seg_inv_n1
            inv_n1 = self.add_mos(ridx_n, cur_col, seg_inv_n1, w=w_n, tile_idx=tidx_nwl)
           
        else:
            inv_p_2, inv_n0, inv_n1 = None, None, None

        seg_pre = seg_dict.get('pre', 0)
        if seg_pre:
            if cur_col < seg_pre + min_sep:
                raise ValueError("pre charge device is too large, reduce the size")
            pre = self.add_mos(ridx_n, 0, seg_pre, w=w_n, tile_idx=tidx_nwl)
            samp_pre = self.connect_to_tracks([pre.g],
                                              self.get_track_id(ridx_n, MOSWireType.G, 'clk', 0, tile_idx=tidx_nwl),
                                              min_len_mode=MinLenMode.MIDDLE)
            vdd_pre = self.connect_to_tracks([pre.s], self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 0,
                                                                        tile_idx=tidx_nwl))
            vg_pre = self.connect_to_tracks([pre.d], self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 1,
                                                                       tile_idx=tidx_nwl))
        else:
            samp_pre, vdd_pre, vg_pre = None, None, None

        self.set_mos_size()

        # --- Connections ---

        cap_vg_conn = [cap_p.g, cap_p_aux.g, on_p.d] if seg_cap_p_aux else [cap_p.g, on_p.d]
        vg_cap = self.connect_to_tracks(cap_vg_conn,
                                        self.get_track_id(ridx_p, MOSWireType.G, 'sig', 0, tile_idx=tidx_nwl))
        _vg_on = self.connect_to_tracks(on_p.d, self.get_track_id(ridx_p, MOSWireType.DS, 'sig', 0, tile_idx=tidx_nwl))

        cap_vdd_conn = [cap_p.d, cap_p_aux.d] if seg_cap_p_aux else [cap_p.d]
        cap_vdd = self.connect_to_tracks(cap_vdd_conn,
                                         self.get_track_id(ridx_p, MOSWireType.DS, 'sup', 0, tile_idx=tidx_nwl))

        if fast_on:
            vmid = self.connect_to_tracks([inv_p_2.g, on_p.g],
                                          self.get_track_id(ridx_p, MOSWireType.G, 'clk', 0, tile_idx=tidx_nwl))
            inv_vdd = self.connect_to_tracks([inv_n0.g],
                                             self.get_track_id(ridx_n, MOSWireType.G, 'clk', 0, tile_idx=tidx_nwl),
                                             min_len_mode=MinLenMode.MIDDLE)
            self.connect_to_tracks([inv_n0.s, inv_n1.d],
                                   self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 0, tile_idx=tidx_nwl))
            _inv1_d_n = self.connect_to_tracks([inv_n0.d],
                                               self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 1, tile_idx=tidx_nwl),
                                               min_len_mode=MinLenMode.MIDDLE)
            _inv1_d_p = self.connect_to_tracks([inv_p_2.d],
                                               self.get_track_id(ridx_p, MOSWireType.DS, 'cap', 0, tile_idx=tidx_nwl),
                                               min_len_mode=MinLenMode.MIDDLE)
            sample_b = self.connect_to_tracks([inv_n1.g],
                                              self.get_track_id(ridx_n, MOSWireType.G, 'clk', 0, tile_idx=tidx_nwl),
                                              min_len_mode=MinLenMode.MIDDLE)
            # -- vg2
            vg2_hm = self.connect_to_tracks([inv_p_2.d, inv_n0.d],
                                            self.get_track_id(ridx_n, MOSWireType.G, 'clk', 1, tile_idx=tidx_nwl),
                                            min_len_mode=MinLenMode.MIDDLE)
            _vss = self.connect_to_tracks([ptap, inv_n1.s],
                                          self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=tidx_ptap))
            self.add_pin('VDD_inv', inv_vdd, label='VDD:', show=self.show_pins)

        else:
            vg2_hm, sample_b, inv_vdd = None, None, None
            vmid = self.connect_to_tracks([on_p.g],
                                          self.get_track_id(ridx_p, MOSWireType.G, 'clk', 0, tile_idx=tidx_nwl))

        if seg_cap_p_aux:
            cap_top_aux_conn = [ntap_t, cap_p_aux.s, inv_p_2.s] if fast_on else [ntap_t, cap_p_aux.s]
            cap_top_aux = self.connect_to_tracks(cap_top_aux_conn,
                                                 self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=tidx_ntapt))
            cap_top = self.connect_to_tracks([on_p.s, cap_p.s],
                                             self.get_track_id(ridx_p, MOSWireType.DS, 'cap', 0, tile_idx=tidx_nwl))
            self.add_pin('cap_top_aux', cap_top_aux, show=self.show_pins)
        else:
            cap_top_conn = [ntap_t, cap_p.s, inv_p_2.s, on_p.s] if fast_on else [ntap_t, cap_p.s, on_p.s]
            cap_top = self.connect_to_tracks(cap_top_conn,
                                             self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=tidx_ntapt))

        # --- Inv's connection:
        vss_inv = self.connect_to_tracks([ptap] + [tx.s for tx in buf_n_list],
                                         self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=tidx_ptap))
        vdd_inv = self.connect_to_tracks([ntap_b, invp.s] + [tx.s for tx in buf_p_list],
                                         self.get_track_id(0, MOSWireType.DS, 'sup', 0, tile_idx=tidx_ntapb))

        if has_buf:
            # - buffer inter-connections
            buf_g_conn, buf_d_conn = [], []
            for idx, (bufn, bufp) in enumerate(zip(buf_n_list, buf_p_list)):
                _ridx = ridx_p if idx ^ 1 else ridx_n
                _in = self.connect_to_tracks([bufn.g, bufp.g],
                                             self.get_track_id(_ridx, MOSWireType.G, 'sig', idx & 1, tile_idx=1))
                _pout = self.connect_to_tracks(bufp.d, self.get_track_id(ridx_p, MOSWireType.DS,
                                                                         'sig', idx ^ 1, tile_idx=1))
                _nout = self.connect_to_tracks(bufn.d, self.get_track_id(ridx_n, MOSWireType.DS,
                                                                         'sig', idx & 1, tile_idx=1))
                buf_d_conn.append([_pout, _nout])
                buf_g_conn.append([_in])
            samp_inv = self.connect_to_tracks([invp.g, invn.g],
                                              self.get_track_id(ridx_p, MOSWireType.G, 'sig', 0, tile_idx=1))
            samp_b_inv = self.connect_to_tracks(invn_buf.s,
                                                self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 2, tile_idx=1))
            cap_bot = self.connect_to_tracks([mid.s, invn.s],
                                             self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 0, tile_idx=1))
            samp_b_delay = [
                self.connect_to_tracks(invn_buf.g, self.get_track_id(ridx_n, MOSWireType.G, 'sig', 1, tile_idx=1),
                                       track_upper=samp_inv.upper)]
            samp_b_buf = buf_g_conn[-1][0]

            buf_mid_conn_vm_list = []
            for _in, _out in zip([samp_b_delay] + buf_g_conn[:-1], buf_d_conn):
                vm_tidx = self.grid.coord_to_track(vm_layer, _in[0].upper, mode=RoundMode.NEAREST)
                vm_tid = TrackID(vm_layer, vm_tidx, tr_manager.get_width(vm_layer, 'clk'))
                buf_mid_conn_vm_list.append(self.connect_to_tracks(_in + _out, vm_tid))
            vmid_tidx = tr_manager.get_next_track(vm_layer, buf_mid_conn_vm_list[0].track_id.base_index,
                                                  'clk', 'clk', up=False)

        else:
            samp_inv = self.connect_to_tracks([invp.g, invn.g],
                                              self.get_track_id(ridx_p, MOSWireType.G, 'sig', 0, tile_idx=1))
            cap_bot = self.connect_to_tracks([mid.s, invn.s],
                                             self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 0, tile_idx=1))
            vmid_tidx = self.arr_info.col_to_track(vm_layer, seg_tot0_inv//2, mode=RoundMode.NEAREST)
            samp_b_buf, samp_b_inv = None, None
            buf_mid_conn_vm_list = []

        vg_inv = self.connect_to_tracks(mid.g, self.get_track_id(ridx_n, MOSWireType.G, 'sig', 0, tile_idx=1))
        vmid_inv_p = self.connect_to_tracks(invp.d, self.get_track_id(ridx_p, MOSWireType.DS, 'sig', 0, tile_idx=1))
        vmid_conn_n = [invn.d, mid.d, invn_buf.d] if has_buf else [invn.d, mid.d]
        vmid_inv_n = self.connect_to_tracks(vmid_conn_n,
                                            self.get_track_id(ridx_n, MOSWireType.DS, 'sig', 1, tile_idx=1))

        # -- pre charge device --

        # --- vm connections ---
        vm_tidx_l = self.arr_info.col_to_track(vm_layer, 0, mode=RoundMode.NEAREST)
        vm_tidx_r = self.arr_info.col_to_track(vm_layer, self.num_cols, mode=RoundMode.NEAREST)
        _, vm_locs_l = tr_manager.place_wires(vm_layer, ['sup', 'sup', 'sig'], align_track=vm_tidx_l, align_idx=0)
        _, vm_locs_r = tr_manager.place_wires(vm_layer, ['sup', 'sup'], align_track=vm_tidx_r, align_idx=-1)
        # Connect vmid:
        vmid_tid = TrackID(vm_layer, vmid_tidx, tr_manager.get_width(vm_layer, 'clk'))
        vmid = self.connect_to_tracks([vmid_inv_n, vmid_inv_p, vmid], vmid_tid)

        # Connect XMID vg:
        vg_tid = TrackID(vm_layer, vm_locs_l[-1], tr_manager.get_width(vm_layer, 'sig'))
        _vg_inv_vm = self.connect_to_tracks([vg_inv, vg_cap], vg_tid)

        # Connect vg2 to vm
        if fast_on:
            vg2_tidx = tr_manager.get_next_track(vm_layer, vmid_tidx, 'clk', 'sig')
            vg2_tid = TrackID(vm_layer, vg2_tidx, tr_manager.get_width(vm_layer, 'sig'))
            vg2 = self.connect_to_tracks(vg2_hm, vg2_tid)
            self.add_pin('vg2', vg2, show=self.show_pins)

        # Connect sample_b
        sample_b_coord = sample_b.upper if fast_on else COORD_MAX
        sample_b_hm = [sample_b] if fast_on else []
        if has_buf:
            sample_b_coord = min(buf_mid_conn_vm_list[0].bound_box.xl, sample_b_coord)
            sample_b_hm.extend([samp_b_buf, samp_b_inv])
        sample_b_tidx = self.grid.coord_to_track(vm_layer, sample_b_coord, mode=RoundMode.NEAREST)
        sample_b_tidx = tr_manager.get_next_track(vm_layer, sample_b_tidx, 'clk', 'clk', up=True)
        sample_b_tid = TrackID(vm_layer, sample_b_tidx, tr_manager.get_width(vm_layer, 'clk'))
        samp_b_vm = self.connect_to_tracks(sample_b_hm, sample_b_tid)

        # Connect Supply to vm
        vm_vss_tidx_list = [vm_locs_l[0], vm_locs_r[-1]]
        vm_vdd_tidx_list = [vm_locs_l[1], vm_locs_r[-2]]
        tr_w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
        vdd_warr_hm, vss_warr_hm = [cap_vdd, vdd_inv], [vss_inv]
        vss_vm_list, vdd_vm_list = [], []
        for vss_tidx, vdd_tidx in zip(vm_vss_tidx_list, vm_vdd_tidx_list):
            _vss, _vdd = self.connect_differential_tracks(vss_warr_hm, vdd_warr_hm, vm_layer,
                                                          vss_tidx, vdd_tidx, width=tr_w_sup_vm)
            vss_vm_list.append(_vss)
            vdd_vm_list.append(_vdd)
        if fast_on:
            self.connect_to_tracks(inv_vdd, TrackID(vm_layer, vm_vdd_tidx_list[-1], tr_w_sup_vm))

        vg_vm_start = tr_manager.get_next_track(vm_layer, vm_locs_l[-1], 'sig', 'sig')
        vg_vm_tidx = self.get_available_tracks(vm_layer, vg_vm_start, vmid_tidx, 0, self.bound_box.yh,
                                               width=tr_manager.get_width(vm_layer, 'sig'),
                                               sep=tr_manager.get_sep(vm_layer, ('sig', 'sig')))
        vg_vm_list = []
        for tidx in vg_vm_tidx[0:2]: #FIXME
            vg_vm_list.append(self.connect_to_tracks(vg_cap, TrackID(vm_layer, tidx,
                                                                     tr_manager.get_width(vm_layer, 'sig'))))
        if seg_pre:
            self.connect_to_tracks(vdd_pre, TrackID(vm_layer, vm_vdd_tidx_list[0], tr_w_sup_vm))
            samp_inv_tidx = self.grid.coord_to_track(vm_layer, samp_inv.lower, mode=RoundMode.NEAREST)
            samp_inv = self.connect_to_tracks([samp_inv, samp_pre], TrackID(vm_layer, samp_inv_tidx,
                                                                            tr_manager.get_width(vm_layer, 'clk')))
            vg_tidx_start = tr_manager.get_next_track(vm_layer, samp_inv_tidx, 'clk', 'sig')
            for vg in vg_vm_list:
                if vg.track_id.base_index > vg_tidx_start:
                    self.connect_to_track_wires(vg_pre, vg)
        
        self.add_pin('VSS', vss_vm_list, show=self.show_pins, connect=True)
        self.add_pin('VDD', vdd_vm_list, show=self.show_pins, connect=True)
        self.add_pin('cap_top', cap_top, show=self.show_pins)
        self.add_pin('cap_bot', cap_bot, show=self.show_pins)
        self.add_pin('vg', vg_vm_list, show=self.show_pins)
        self.add_pin('vmid', vmid, show=self.show_pins)
        self.add_pin('sample', samp_inv, show=self.show_pins)
        if fast_on or has_buf:
            self.add_pin('sample_b', samp_b_vm, show=self.show_pins)

        nrow_info, prow_info = self.get_tile_pinfo(tidx_nwl).get_row_place_info(0), self.get_tile_pinfo(
            tidx_nwl).get_row_place_info(1)
        ninv_info, pinv_info = self.get_tile_pinfo(tidx_inv).get_row_place_info(0), self.get_tile_pinfo(
            tidx_inv).get_row_place_info(1)

        dev_info = dict(
            XON_P={'nf': seg_dict['on_p'], 'w_n': prow_info.row_info.width,
                    'intent': prow_info.row_info.threshold},
            XCAP_P={'nf': seg_dict['cap_p'], 'w_n': prow_info.row_info.width,
                    'intent': prow_info.row_info.threshold},
            XINV_P={'nf': seg_dict['inv_p'], 'w_n': pinv_info.row_info.width,
                    'intent': pinv_info.row_info.threshold},
            XINV_N={'nf': seg_dict['inv_n'], 'w_n': ninv_info.row_info.width,
                    'intent': ninv_info.row_info.threshold},
            XMID={'nf': seg_dict['mid'], 'w_n': ninv_info.row_info.width,
                    'intent': ninv_info.row_info.threshold},
        )
        for idx, (segn, segp) in enumerate(zip(seg_buf_list_n[::-1], seg_buf_list_p[::-1])):
            dev_info.update({f"XSAMPLE_INVN<{idx}>": {'nf': segn, 'intent': ninv_info.row_info.threshold}})
            dev_info.update({f"XSAMPLE_INVP<{idx}>": {'nf': segp, 'intent': pinv_info.row_info.threshold}})
        if seg_invn_buf:
            dev_info.update({'XINV_N_BUF': {'nf': seg_invn_buf, 'intent': pinv_info.row_info.threshold}})

        if seg_cap_p_aux:
            dev_info.update({'XCAP_P_AUX': {'nf': seg_dict['cap_p_aux'], 'intent': prow_info.row_info.threshold}})
        if fast_on:
            dev_info.update(
                dict(
                    XINV_P_VG2={'nf': seg_dict['inv_p_vg2'], 'intent': prow_info.row_info.threshold},
                    XINV_N0_VG2={'nf': seg_dict['inv_n0_vg2'], 'intent': nrow_info.row_info.threshold},
                    XINV_N1_VG2={'nf': seg_dict['inv_n1_vg2'], 'intent': nrow_info.row_info.threshold},

                )
            )
        self.sch_params = dict(
            lch=self.arr_info.lch,
            intent='standard',
            dev_info=dev_info
        )


class Bootstrap_simple(TemplateBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'bootstrap')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            nmos_params='NMOS part parameters.',
            nwl_params='NWL part parameters.',
            cap_params='capacitor parameters',
            top_layer='Top routing layer',
            tr_widths='track widths',
            tr_spaces='track widths',
            fast_on='True to fast turn on XON_N',
            no_sampler="True to remove sampler",
            shield="True to add ground shield over MOS components"
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(fast_on=False, 
                    no_sampler=False,
                    shield=False
                    )

    def draw_layout(self) -> None:
        top_layer = self.params['top_layer']
        nmos_params: Param = self.params['nmos_params']
        nwl_params: Param = self.params['nwl_params']
        cap_params: ImmutableSortedDict[str, Any] = self.params['cap_params']
        tr_widths: WDictType = self.params['tr_widths']
        tr_spaces: SpDictType = self.params['tr_spaces']
        fast_on: bool = self.params['fast_on']
        shield: bool = self.params['shield']

        has_cap_aux = bool(nwl_params['seg_dict'].get('cap_p_aux', 0))

        # get track manager
        tr_mgr = TrackManager(grid=self.grid, tr_widths=tr_widths, tr_spaces=tr_spaces)
        conn_layer = MOSArrayPlaceInfo.get_conn_layer(self.grid.tech_info,
                                                      nmos_params['pinfo']['tile_specs']['arr_info']['lch'])
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        nmos_new_params = nmos_params.to_dict()
        nwl_new_params = nwl_params.to_dict()
        nmos_new_params.update(dict(fast_on=fast_on))
        nwl_new_params.update(dict(fast_on=fast_on))
        btstrp_nmos_params = dict(
            cls_name=BootstrapNMOS_simple.get_qualified_name(),
            params=nmos_new_params
        )
        btstrp_nwl_params = dict(
            cls_name=BootstrapNWL_simple.get_qualified_name(),
            params=nwl_new_params
        )
        nmos_master: TemplateBase = self.new_template(GenericWrapper, params=btstrp_nmos_params)
        nwl_master: TemplateBase = self.new_template(GenericWrapper, params=btstrp_nwl_params)

        w_cap: int = cap_params['height']
        cboot_params = copy.deepcopy(cap_params.to_dict())

        w_blk, h_blk = self.grid.get_block_size(top_layer)
        nmos_box, nwl_box = nmos_master.bound_box, nwl_master.bound_box

        w_nmos, h_nmos = nmos_box.w, nmos_box.h
        w_nwl, h_nwl = nwl_box.w, nwl_box.h

        w_cap = -(-w_cap // w_blk) * w_blk
        h_tot = h_nmos + h_nwl + h_blk
        h_tot = -(-h_tot // h_blk) * h_blk

        if has_cap_aux:
            h_cap_tot = h_tot - 4 * cap_params['margin'] - 4 * h_blk
            cboot_height = (int(0.85 * h_cap_tot) // h_blk) * h_blk
        else:
            h_cap_tot = h_tot - 2 * cap_params['width'] - 2 * h_blk
            cboot_height = (h_cap_tot // h_blk) * h_blk

        cboot_master: TemplateBase = self.new_template(MIMCap, params=cboot_params['cap_config'])

        vmw_blk, vmh_blk = self.grid.get_block_size(vm_layer)
        w_cap = cboot_master.bound_box.h
        w_tot = max(w_nmos, w_nwl+w_cap+(vmw_blk)) #+ w_cap
        w_tot = -(-w_tot // w_blk) * w_blk

        nmos_x = -(-(w_tot - w_nmos)//w_blk)*w_blk
        nmos = self.add_instance(nmos_master, inst_name='XNMOS', xform=Transform(nmos_x, 0)) 

        capb_nmos = nmos.get_pin('cap_bot')
        cap_coord = capb_nmos.upper
        if (cap_coord < w_tot-(w_nwl+w_cap+vmw_blk)):
            nwl = self.add_instance(nwl_master, inst_name='XNWL',
                    xform=Transform(cap_coord+vmw_blk//2+w_cap, h_nmos + h_blk))
            capb_nwl = nwl.get_pin('cap_bot')
            vm_layer = capb_nmos.track_id.layer_id + 1
            cap_bot_coord = max(int(nwl.bound_box.yl) + 2*h_blk, self.grid.track_to_coord(capb_nwl.track_id.layer_id, capb_nwl.track_id.base_index))
            cboot = self.add_instance(cboot_master, inst_name='CBOOT', xform=Transform(-(-(cap_coord+w_cap-vmw_blk//2)//vmw_blk )* vmw_blk, cap_bot_coord, mode=Orientation.R90))
        else:
            nwl = self.add_instance(nwl_master, inst_name='XNWL',
                                xform=Transform(w_tot - w_nwl, h_nmos + h_blk))

            capb_nwl = nwl.get_pin('cap_bot')
            vm_layer = capb_nmos.layer_id + 1
            cap_bot_coord = max(nwl.bound_box.yl + h_blk, self.grid.track_to_coord(capb_nwl.track_id.layer_id, capb_nwl.track_id.base_index))
            cboot = self.add_instance(cboot_master, inst_name='CBOOT', xform=Transform(-(-(nwl.bound_box.xl-vmw_blk//2)//vmw_blk )* vmw_blk, cap_bot_coord, mode=Orientation.R90)) 
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot))

        # --- Connections:
        # - vg2
        if fast_on:
            vg2_nmos = nmos.get_port('vg2')
            vg2_nwl = nwl.get_port('vg2')
            vm_lay_purp = (vg2_nwl.get_single_layer(), 'drawing')
            vg2_nwl_bbox: List[BBox] = vg2_nwl.get_pins()
            vg2_bnds = [COORD_MAX, COORD_MIN]
            self.connect_bbox_to_track_wires(Direction.UPPER, vm_lay_purp, vg2_nwl_bbox[0],
                                             vg2_nmos.get_pins(), ret_bnds=vg2_bnds)
            [bbox.extend(y=vg2_bnds[1]) for bbox in vg2_nwl_bbox]
            vg2_vm_bbox_list = [bbox.extend(y=vg2_bnds[0]) for bbox in vg2_nwl_bbox]
            self.add_pin_primitive('vg2', vm_lay_purp[0], vg2_vm_bbox_list[0], show=self.show_pins)

        # - sample_b
        if nwl.has_port('sample_b'):
            sampb_nmos = nmos.get_port('sample_b')
            sampb_nwl = nwl.get_port('sample_b')
            vm_lay_purp = (sampb_nwl.get_single_layer(), 'drawing')
            sampb_nwl_bbox: List[BBox] = sampb_nwl.get_pins()
            sample_b_bnds = [COORD_MAX, COORD_MIN]
            self.connect_bbox_to_track_wires(Direction.UPPER, vm_lay_purp, sampb_nwl_bbox[0],
                                             sampb_nmos.get_pins(), ret_bnds=sample_b_bnds)
            [bbox.extend(y=sample_b_bnds[1]) for bbox in sampb_nwl_bbox]
            samp_b_vm_bbox_list = [bbox.extend(y=sample_b_bnds[0]) for bbox in sampb_nwl_bbox]
            self.add_pin_primitive('sample_b', vm_lay_purp[0], samp_b_vm_bbox_list[0], show=self.show_pins)
        else:
            self.reexport(nmos.get_port('sample_b'))

        # cap_bot
        capb_nmos = nmos.get_pin('cap_bot') 
        capb_nwl = nwl.get_pin('cap_bot')
        cap_bot_vm_coord = min(nwl.bound_box.xl, capb_nmos.upper, cboot.get_pin('BOT').xm)
        cap_bot_vm_tidx = self.grid.coord_to_track(vm_layer, cap_bot_vm_coord, mode=RoundMode.NEAREST)
        cap_bot_vm_tid = TrackID(vm_layer, cap_bot_vm_tidx, tr_mgr.get_width(vm_layer, 'cap'))
        capb_nmos_vm = self.connect_to_tracks(nmos.get_all_port_pins('cap_bot'), cap_bot_vm_tid)
        self.extend_wires(capb_nmos_vm, upper=cap_bot_coord)

        # vg
        vg_nmos = nmos.get_port('vg')
        vg_nwl = nwl.get_port('vg')
        vg_nwl_bbox: List[BBox] = vg_nwl.get_pins()
        vm_lay_purp = (vg_nwl.get_single_layer(), 'drawing')
        vg_vm_bbox_list = []
        for bbox in vg_nwl_bbox:
            _ret_bnds = [COORD_MAX, COORD_MIN]
            self.connect_bbox_to_track_wires(Direction.UPPER, vm_lay_purp, bbox, vg_nmos.get_pins(), ret_bnds=_ret_bnds)
            bbox.extend(y=_ret_bnds[0])
            vg_vm_bbox_list.append(bbox.extend(y=_ret_bnds[1]))

        # supply vm
        vdd_nmos = nmos.get_pin('VDD')
        vss_nmos = nmos.get_all_port_pins('VSS')
        vdd_nwl= nwl.get_all_port_pins('VDD') 
        vss_nwl= nwl.get_all_port_pins('VSS') 
        
        vdd_list=[]
        for wire in vdd_nwl:
            vdd_list.append(self.connect_bbox_to_track_wires(Direction.UPPER, vm_lay_purp, wire, vdd_nmos))
        self.add_pin('VDD', vdd_list)

        vss_list=[]
        for wire in vss_nwl:
            vss_list.append(self.connect_bbox_to_track_wires(Direction.UPPER, vm_lay_purp, wire, vss_nmos[0]))
        self.add_pin('VSS', nmos.get_all_port_pins('VSS'))
    
        if nmos_master.sch_params['has_dum_sampler']:
            self.add_pin('vss_dum', nmos.get_pin('vss_dum'), label='VSS', connect=True)


        # Connection with mom cap
        # 1. Get available tracks in space
        # 2. Connect pins from transistor to vm_layer
        # 3. Connect to xm_layer then ym_layer
        cap_vm_tidx_start = self.grid.coord_to_track(vm_layer, cap_bot_vm_coord-w_cap, mode=RoundMode.GREATER_EQ)
        cap_vm_tidx_list = self.get_available_tracks(vm_layer, cap_vm_tidx_start, cap_bot_vm_tidx,0, self.bound_box.yh,
                                                     width=tr_mgr.get_width(vm_layer, 'cap'),
                                                     sep=tr_mgr.get_sep(vm_layer, ('cap', 'cap')))[1:]                                            
        cap_bot_hm = nmos.get_all_port_pins('cap_bot')
        cap_top_hm = nwl.get_pin('cap_top')
        cap_bot_vm_list, cap_top_vm_list = [], []
        tr_w_cap_vm = tr_mgr.get_width(vm_layer, 'cap')
        tr_w_cap_ym = tr_mgr.get_width(ym_layer, 'cap')
       
        if has_cap_aux:
            cap_top_aux_vm_list = []
            cap_top_aux_hm = nwl.get_pin('cap_top_aux')
            for idx in cap_vm_tidx_list:
                cap_top_aux_vm_list.append(self.connect_to_tracks(cap_top_aux_hm, TrackID(vm_layer, idx, tr_w_cap_vm)))
            cap_top_aux_xm_tidx = self.grid.coord_to_track(xm_layer, cap_top_aux_vm_list[0].middle,
                                                           mode=RoundMode.NEAREST)
            cap_top_xm_tidx = max(cap_top_xm_tidx, tr_mgr.get_next_track(xm_layer, cap_top_aux_xm_tidx, 'cap', 'cap'))
        else:
            cap_top_aux_vm_list = []
            cap_top_aux_xm_tidx = COORD_MAX

        cap_top_layer = cboot_master.top_layer
        cap_bot_layer = cap_top_layer-1
        mim_cap_top =  cboot.get_pin('TOP')
        mim_cap_bot = cboot.get_pin('BOT')

        # Use a MIM Cap
        if ((cap_top_layer %2) == 0): 
            idx_t = self.grid.find_next_track(cap_top_layer-1, int(mim_cap_top.yh), tr_width=1, half_track=True, mode=RoundMode.GREATER_EQ)
            idx_b = self.grid.find_next_track(cap_bot_layer-1, int(mim_cap_bot.yh), tr_width=1, half_track=True, mode=RoundMode.LESS_EQ)
            cap_top = self.connect_bbox_to_tracks(Direction.UPPER, ('met4','drawing'), mim_cap_top,
                                                      TrackID(cap_top_layer-1, idx_t, 1))
            cap_bot = self.connect_bbox_to_tracks(Direction.UPPER, ('met3','drawing'), mim_cap_bot,
                                                     TrackID(capb_nmos_vm.track_id.layer_id, 
                                                            capb_nmos_vm.track_id.base_index, 1)) #TrackID(cap_bot_layer-1, idx_b, 1))
            self.connect_wires([capb_nmos_vm, cap_bot])
            self.connect_to_track_wires(capb_nwl, capb_nmos_vm) #FIXME
            cap_top_tr_w = round(self.grid.coord_to_track(vm_layer, w_cap, mode=RoundMode.LESS_EQ)*3//2)
            self.extend_wires(cap_top_hm, lower=cap_top.lower)
            idx_vm_top = self.grid.coord_to_track(vm_layer, -(-(cap_top.lower+cap_top.upper)//(h_blk*2))*h_blk, mode=RoundMode.NEAREST)
            cap_top_mid = self.connect_to_tracks(cap_top_hm, TrackID(vm_layer, idx_vm_top, int(cap_top_tr_w) ))
            self.connect_to_track_wires(cap_top_mid, cap_top)


        tr_w_cap_xm = tr_mgr.get_width(xm_layer, 'cap')

        # Reexport pins:
        vin = nmos.get_pin('in')

        # export input signal to higher layer
        vm_tr_lo = self.grid.coord_to_track(vm_layer, vin.lower, mode=RoundMode.GREATER_EQ)
        vm_tr_hi = self.grid.coord_to_track(vm_layer, vin.upper, mode=RoundMode.LESS_EQ)
        sig_tr_list = self.get_available_tracks(vm_layer, vm_tr_lo, vm_tr_hi, lower=vin.bound_box.xl,
                                                upper=vin.bound_box.xh, width=tr_mgr.get_width(vm_layer, 'sig'),
                                                sep=tr_mgr.get_sep(vm_layer, ('sig', 'sig')),
                                                sep_margin=tr_mgr.get_sep(vm_layer, ('sig', 'sup')))
        tr_w_sig_vm = tr_mgr.get_width(vm_layer, 'sig')
        tr_w_sig_xm = tr_mgr.get_width(xm_layer, 'sig')
        self.add_pin('in', vin)

        nout = 0
        for pname in nmos.port_names_iter():
            if 'out' in pname:
                self.reexport(nmos.get_port(pname))
                nout += 1

        # Connect sample to a horizontal layer
        samp_vm_tidx_lower = self.grid.coord_to_track(vm_layer, (nwl.bound_box.xl//h_blk)*h_blk)
        samp_vm_tidx_upper = self.grid.coord_to_track(vm_layer, (-(-nwl.bound_box.xh//h_blk))*h_blk)
        samp_vm_avail_tracks = self.get_available_tracks(vm_layer, samp_vm_tidx_lower, samp_vm_tidx_upper, 
                                                            nmos.bound_box.yh, nwl.bound_box.yh)
        samp_vm = self.connect_to_tracks(nwl.get_pin('sample'), TrackID(vm_layer, samp_vm_avail_tracks[0], tr_w_sig_vm))
        samp_hm_tidx_lower = self.grid.coord_to_track(hm_layer, nmos.bound_box.yh)
        samp_hm_tidx_upper = self.grid.coord_to_track(hm_layer, nwl.bound_box.yl)
        samp_hm_avail_tracks = self.get_available_tracks(hm_layer, samp_hm_tidx_lower, samp_hm_tidx_upper, 
                                                            nmos.bound_box.yl, nmos.bound_box.yh)
        samp_hm = self.connect_to_tracks(samp_vm, TrackID(hm_layer, samp_hm_avail_tracks[0], tr_w_sig_vm))
        self.add_pin("sample", samp_hm)

        # Add pins:
        self.add_pin('cap_bot', nmos.get_pin('cap_bot'), show=self.show_pins)
        self.add_pin('cap_top', nwl.get_pin('cap_top'), show=self.show_pins)
        vm_lay_purp = self.grid.tech_info.get_lay_purp_list(vm_layer)[0]

        for bbox in vg_vm_bbox_list:
            self.add_pin_primitive('vg', vm_lay_purp[0], bbox, show=self.show_pins)

        if shield:
            shield_layer_purp = (f'met{xm_layer}', 'drawing')
            vss_shield = [self.add_bbox_array(shield_layer_purp, BBoxArray(BBox(nwl.bound_box.xl, nmos.bound_box.yh, nwl.bound_box.xh, nwl.bound_box.yh))), 
                      self.add_bbox_array(shield_layer_purp, BBoxArray(nmos.bound_box))]
            self.add_pin_primitive('VSS_shield', shield_layer_purp[0], BBox(nwl.bound_box.xl, nmos.bound_box.yh, nwl.bound_box.xh, nwl.bound_box.yh))
            self.add_pin_primitive('VSS_shield', shield_layer_purp[0], nmos.bound_box)
       
        dev_info = {
            **nmos_master.sch_params['dev_info'],
            **nwl_master.sch_params['dev_info']
        }
        self._sch_params = dict(
            lch=nwl_master.sch_params['lch'],
            intent='standard',
            dev_info=dev_info,
            cap_params=cboot_params['cap_config'],
            fast_on=fast_on,
            break_outputs= True if nout > 1 else False, 
            no_sampler=False
        )

class BootstrapDiff_simple(TemplateBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'bootstrap_diff')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            sampler_params='single end sampler parameters.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict()

    def draw_layout(self) -> None:
        sampler_params: Param = self.params['sampler_params']

        sampler_master: TemplateBase = self.new_template(Bootstrap_simple, params=sampler_params)

        top_layer = sampler_master.top_layer
        w_blk, h_blk = self.grid.get_block_size(top_layer)
        w_samp, h_samp = sampler_master.bound_box.w, sampler_master.bound_box.h

        w_tot = 2 * (w_samp+2*w_blk)
        h_tot = h_samp
        h_tot = -(-h_tot // h_blk) * h_blk
        w_tot = -(-w_tot // w_blk) * w_blk

        samp_n = self.add_instance(sampler_master, inst_name='XN', xform=Transform(0, 0))
        samp_p = self.add_instance(sampler_master, inst_name='XP',
                                   xform=Transform(w_tot, 0, mode=Orientation.MY))

        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot))

        # --- export pins:
        if sampler_params['dum_sampler']:
            self.add_pin('vss_dum', samp_n.get_pin('vss_dum'), label='VSS', connect=True)
            self.add_pin('vss_dum', samp_p.get_pin('vss_dum'), label='VSS', connect=True)

        self.connect_wires([samp_n.get_pin('VDD'), samp_p.get_pin('VDD')])
        self.connect_wires([samp_n.get_pin('VSS'), samp_p.get_pin('VSS')])
        self.connect_wires([samp_n.get_pin('sample'), samp_p.get_pin('sample')])
        self.connect_wires([samp_n.get_pin('sample_b'), samp_p.get_pin('sample_b')])
        self.add_pin('VDD', samp_n.get_pin('VDD'), connect=True)
        self.add_pin('VDD', samp_p.get_pin('VDD'), connect=True)
        self.add_pin('VSS', samp_n.get_pin('VSS'), connect=True)
        self.add_pin('VSS', samp_p.get_pin('VSS'), connect=True)
        self.add_pin('sample', samp_n.get_pin('sample'), connect=True)
        self.add_pin('sample', samp_p.get_pin('sample'), connect=True)
        self.add_pin('sample_b', samp_n.get_pin('sample_b'), connect=True)
        self.add_pin('sample_b', samp_p.get_pin('sample_b'), connect=True)
        # self.reexport(samp_n.get_port('VDD'), connect=True)
        # self.reexport(samp_p.get_port('VDD'), connect=True)
        # self.reexport(samp_n.get_port('VSS'), connect=True)
        # self.reexport(samp_p.get_port('VSS'), connect=True)
        # self.reexport(samp_n.get_port('sample'), connect=True)
        # self.reexport(samp_p.get_port('sample'), connect=True)    
        # self.reexport(samp_n.get_port('sample_b'), connect=True)
        # self.reexport(samp_p.get_port('sample_b'), connect=True)

        pin_list = ['cap_top', 'cap_bot', 'vg']
        if samp_n.has_port('cap_top_aux'):
            pin_list.append('cap_top_aux')

        for pinname in samp_p.port_names_iter():
            if 'out' in pinname:
                ppin = pinname.replace('out', 'out_p')
                npin = pinname.replace('out', 'out_n')
                self.reexport(samp_n.get_port(pinname), net_name=npin)
                self.reexport(samp_p.get_port(pinname), net_name=ppin)

            for pinparttern in pin_list:
                if pinparttern == pinname:
                    self.reexport(samp_n.get_port(pinname), net_name=pinname + '_n')
                    self.reexport(samp_p.get_port(pinname), net_name=pinname + '_p')

        self.add_pin('sig_n', samp_n.get_pin('in'))
        self.add_pin('sig_p', samp_p.get_pin('in'))
        #self.reexport(samp_n.get_port('in'), net_name='sig_n', connect=True)
        #self.reexport(samp_n.get_port('in_c'), net_name='sig_p', connect=True)
        #self.reexport(samp_p.get_port('in'), net_name='sig_p', connect=True)
        #self.reexport(samp_p.get_port('in_c'), net_name='sig_n', connect=True)
        if samp_n.has_port('VSS_shield'):
            self.reexport(samp_n.get_port('VSS_shield'))
        if samp_n.has_port('VSS_shield'):
            self.reexport(samp_p.get_port('VSS_shield'))

        self._sch_params = dict(
            sampler_params=sampler_master.sch_params
        )

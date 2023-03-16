from typing import Any, Dict, Type, Optional
import copy

from bag.design.database import ModuleDB, Module
from bag.io.file import read_yaml
from bag.layout.routing.base import TrackManager
from bag.layout.template import TemplateDB, TemplateBase
from bag.util.immutable import Param
from pybag.core import BBox, Transform
from pybag.enum import Orientation, Orient2D, RoundMode, MinLenMode, Direction

from xbase.layout.enum import MOSWireType, SubPortMode
from xbase.layout.mos.placement.data import TilePatternElement, TilePattern
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase, MOSArrayPlaceInfo
from xbase.layout.mos.top import GenericWrapper
from xbase.layout.cap.mim import MIMCap

from .bootstrap_simple import Bootstrap_simple, BootstrapDiff_simple
from .util.template import TemplateBaseZL
from .util.template import TrackIDZL as TrackID

class CMSwitch(MOSBase, TemplateBaseZL):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            segp='segments.',
            segn='segments.',
            wp='widths.',
            wn='widths.',
            ncols_tot='Total number of fingersa',
            swap_inout=''
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            wp=110,
            wn=84,
            ncols_tot=0,
            swap_inout=False,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        tr_manager = self.tr_manager

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        segn: int = self.params['segn']
        segp: int = self.params['segp']
        wn: int = self.params['wn']
        wp: int = self.params['wp']

        tap_ncol = self.get_tap_ncol(tile_idx=0)
        tap_sep_col = self.sub_sep_col
        tap_sep_col += tap_sep_col & 1
        tap_ncol += tap_sep_col

        tot_cols = max(self.params['ncols_tot'], max(segn, segp) + 2 * tap_ncol + 2) 
        seg_sub_conn = (tot_cols - max(segn, segp) - 2) // 2

        # Placement
        ntap_l = self.add_substrate_contact(0, 0, seg=seg_sub_conn - tap_sep_col, w=wn)
        ptap_l = self.add_substrate_contact(1, 0, seg=seg_sub_conn - tap_sep_col, w=wp)

        sw_n = self.add_mos(0, (tot_cols - segn) // 2, segn, w=wn)
        sw_p = self.add_mos(1, (tot_cols - segp) // 2, segp, w=wp)

        ntap_r = self.add_substrate_contact(0, tot_cols, seg=seg_sub_conn - tap_sep_col, w=wn, flip_lr=True)
        ptap_r = self.add_substrate_contact(1, tot_cols, seg=seg_sub_conn - tap_sep_col, w=wp, flip_lr=True)
        vdd_list, vss_list = [ptap_l, ptap_r], [ntap_l, ntap_r]
        self.set_mos_size()

        ntid_g = self.get_track_id(0, MOSWireType.G, wire_name='clk', wire_idx=0)
        ntid_sig = self.get_track_id(0, MOSWireType.DS, wire_name='sig', wire_idx=0)

        ptid_g = self.get_track_id(1, MOSWireType.G, wire_name='clk', wire_idx=0)
        ptid_sig = self.get_track_id(1, MOSWireType.DS, wire_name='sig', wire_idx=0)
        ntid_ref = ptid_ref = self.grid.get_middle_track(ntid_sig.base_index, ptid_sig.base_index)

        sam_hm = self.connect_to_tracks(sw_n.g, ntid_g)
        sam_b_hm = self.connect_to_tracks(sw_p.g, ptid_g)
        ref_hm = self.connect_to_tracks([sw_n.d, sw_p.d],
                                        TrackID(hm_layer, ntid_ref, tr_manager.get_width(hm_layer, 'sig')))
        n_sig_hm = self.connect_to_tracks(sw_n.s, ntid_sig)
        p_sig_hm = self.connect_to_tracks(sw_p.s, ptid_sig)

        mid_vm_tidx = self.arr_info.col_to_track(vm_layer, tot_cols // 2, RoundMode.NEAREST)
        sam_vm = self.connect_to_tracks(sam_hm, TrackID(vm_layer, mid_vm_tidx, tr_manager.get_width(vm_layer, 'clk')))
        sam_b_vm = self.connect_to_tracks(sam_b_hm,
                                          TrackID(vm_layer, mid_vm_tidx, tr_manager.get_width(vm_layer, 'clk')))
        tid_l = self.arr_info.col_to_track(vm_layer, tap_ncol, mode=RoundMode.NEAREST)
        tid_r = self.arr_info.col_to_track(vm_layer, self.num_cols - tap_ncol, mode=RoundMode.NEAREST)

        tr_w_vref_vm = tr_manager.get_width(vm_layer, 'vref')
        tr_w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        tr_w_clk_vm = tr_manager.get_width(vm_layer, 'clk')
        vref_vm_locs = self.get_tids_between(vm_layer, tid_l,
                                             mid_vm_tidx - self.get_track_sep(vm_layer, tr_w_vref_vm, tr_w_clk_vm),
                                             tr_w_vref_vm, 0, 0, True)
        sig_vm_locs = self.get_tids_between(vm_layer,
                                            mid_vm_tidx + self.get_track_sep(vm_layer, tr_w_clk_vm, tr_w_sig_vm),
                                            tid_r, tr_manager.get_width(vm_layer, 'sig'), 0, 0, True)
        vref_vm = [self.connect_to_tracks(ref_hm, _tid) for _tid in vref_vm_locs]
        nsig_vm = [self.connect_to_tracks(n_sig_hm, _tid) for _tid in sig_vm_locs]
        psig_vm = [self.connect_to_tracks(p_sig_hm, _tid)
                   for _tid in sig_vm_locs]
        vm_warrs = vref_vm + nsig_vm + psig_vm
        vm_warrs_max_coord, vm_warrs_min_coord = max([v.upper for v in vm_warrs]), min([v.lower for v in vm_warrs])
        vref_vm = self.extend_wires(vref_vm, upper=vm_warrs_max_coord, lower=vm_warrs_min_coord)
        sig_vm = self.extend_wires(nsig_vm + psig_vm, upper=vm_warrs_max_coord, lower=vm_warrs_min_coord)

        tr_w_sup_xm = tr_manager.get_width(xm_layer, 'sup')
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')
        tr_w_clk_xm = tr_manager.get_width(xm_layer, 'clk')

        # Connect supplies
        tr_sup_vm_w = tr_manager.get_width(vm_layer, 'sup')
        tr_sup_xm_w = tr_manager.get_width(xm_layer, 'sup')
        vss_hm = self.connect_to_tracks(vss_list, self.get_track_id(0, MOSWireType.G, wire_name='sup'))
        vdd_hm = self.connect_to_tracks(vdd_list, self.get_track_id(1, MOSWireType.G, wire_name='sup'))

        vdd_vm_list, vss_vm_list = [], []
        sup_vm_locs = self.get_tids_between(vm_layer, self.arr_info.col_to_track(vm_layer, 0),
                                            self.arr_info.col_to_track(vm_layer, tap_ncol),
                                            tr_sup_vm_w, 0, 0, True)
        sup_vm_locs += self.get_tids_between(vm_layer, self.arr_info.col_to_track(vm_layer, tot_cols - seg_sub_conn),
                                             self.arr_info.col_to_track(vm_layer, tot_cols),
                                             tr_sup_vm_w, 0, 0, True)
        for tid in sup_vm_locs:
            vss_vm_list.append(self.connect_to_tracks(vss_hm, tid))
            vdd_vm_list.append(self.connect_to_tracks(vdd_hm, tid))

        swap_inout = self.params['swap_inout']
        _, sig_xm_locs = tr_manager.place_wires(xm_layer, ['sig', 'sig'],
                                                center_coord=self.grid.track_to_coord(hm_layer, ptid_ref))
        vdd_xm_tidx = self.grid.coord_to_track(xm_layer, self.grid.track_to_coord(hm_layer, vdd_hm.track_id.base_index),
                                               RoundMode.NEAREST)
        vss_xm_tidx = self.grid.coord_to_track(xm_layer, self.grid.track_to_coord(hm_layer, vss_hm.track_id.base_index),
                                               RoundMode.NEAREST)
        vdd_xm = self.connect_to_tracks(vdd_vm_list, TrackID(xm_layer, vdd_xm_tidx, tr_w_sup_xm))
        vss_xm = self.connect_to_tracks(vss_vm_list, TrackID(xm_layer, vss_xm_tidx, tr_w_sup_xm))

        xm_avail_locs = self.get_available_tracks(xm_layer, vss_xm_tidx, vdd_xm_tidx,
                                                  self.bound_box.xl, self.bound_box.xh,
                                                  tr_manager.get_width(xm_layer, 'clk'),
                                                  tr_manager.get_sep(xm_layer, ('clk', 'clk')),
                                                  sep_margin=tr_manager.get_sep(xm_layer, ('clk', 'sup')),
                                                  include_last=True)

        sam_xm = self.connect_to_tracks(sam_vm, TrackID(xm_layer, xm_avail_locs[0], tr_w_clk_xm))
        sam_b_xm = self.connect_to_tracks(sam_b_vm, TrackID(xm_layer, xm_avail_locs[-1], tr_w_clk_xm))

        self.add_pin('sam', sam_xm)
        self.add_pin('sam_b', sam_b_xm)
        self.add_pin('ref', vref_vm)
        self.add_pin('sig', sig_vm)
        self.add_pin('VSS', vss_xm)
        self.add_pin('VDD', vdd_xm)

        self.sch_params = dict(
            n=dict(
                lch=self.arr_info.lch,
                seg=segn,
                w=wn,
                intent=self.place_info.get_row_place_info(0).row_info.threshold
            ),
            p=dict(
                lch=self.arr_info.lch,
                seg=segn,
                w=wn,
                intent=self.place_info.get_row_place_info(0).row_info.threshold
            )
        )

class NMOSSwitch(MOSBase, TemplateBaseZL):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            segn='segments.',
            wn='widths.',
            ncols_tot='Total number of fingersa',
            swap_inout=''
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            wn=84,
            ncols_tot=0,
            swap_inout=False,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        tr_manager = self.tr_manager

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        segn: int = self.params['segn']
        wn: int = self.params['wn']

        tap_ncol = self.get_tap_ncol(tile_idx=0)
        tap_sep_col = self.sub_sep_col
        tap_sep_col += tap_sep_col & 1
        tap_ncol += tap_sep_col

        tot_cols = max(self.params['ncols_tot'], segn + 2 * tap_ncol + 2)  # +2 for J_ error in intel22
        seg_sub_conn = (tot_cols - segn - 2) // 2
        # Placement
        ntap_l = self.add_substrate_contact(0, 0, seg=seg_sub_conn - tap_sep_col, w=wn)
    
        sw_n = self.add_mos(0, (tot_cols - segn) // 2, segn, w=wn)

        ntap_r = self.add_substrate_contact(0, tot_cols, seg=seg_sub_conn - tap_sep_col, w=wn, flip_lr=True)
        vss_list = [ntap_l, ntap_r]
        self.set_mos_size()

        ntid_g = self.get_track_id(0, MOSWireType.G, wire_name='clk', wire_idx=0)
        ntid_ref = self.get_track_id(0, MOSWireType.DS, wire_name='sig', wire_idx=1)
        ntid_sig = self.get_track_id(0, MOSWireType.DS, wire_name='sig', wire_idx=0)

        sam_hm = self.connect_to_tracks(sw_n.g, ntid_g)
        ref_hm = self.connect_to_tracks([sw_n.d],
                                        TrackID(hm_layer, ntid_ref.base_index, tr_manager.get_width(hm_layer, 'sig')))
        n_sig_hm = self.connect_to_tracks(sw_n.s, ntid_sig)

        # get middle track for sample signal
        mid_vm_tidx = self.arr_info.col_to_track(vm_layer, tot_cols // 2, RoundMode.NEAREST)
        sam_vm = self.connect_to_tracks(sam_hm, TrackID(vm_layer, mid_vm_tidx, tr_manager.get_width(vm_layer, 'clk')))
        tid_l = self.arr_info.col_to_track(vm_layer, tap_ncol, mode=RoundMode.NEAREST)
        tid_r = self.arr_info.col_to_track(vm_layer, self.num_cols - tap_ncol, mode=RoundMode.NEAREST)

        tr_w_vref_vm = tr_manager.get_width(vm_layer, 'vref')
        tr_w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        tr_w_clk_vm = tr_manager.get_width(vm_layer, 'clk')

        tr_w_sup_xm = tr_manager.get_width(xm_layer, 'sup')
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')
        tr_w_clk_xm = tr_manager.get_width(xm_layer, 'clk')

        # Connect supplies
        tr_sup_vm_w = tr_manager.get_width(vm_layer, 'sup')
        tr_sup_xm_w = tr_manager.get_width(xm_layer, 'sup')
        vss_hm = self.connect_to_tracks(vss_list, self.get_track_id(0, MOSWireType.G, wire_name='sup'))

        vdd_vm_list, vss_vm_list = [], []
        sup_vm_locs = self.get_tids_between(vm_layer, self.arr_info.col_to_track(vm_layer, 0),
                                            self.arr_info.col_to_track(vm_layer, tap_ncol),
                                            tr_sup_vm_w, self.get_track_sep(vm_layer, tr_sup_vm_w, tr_sup_vm_w), 0, True)
        sup_vm_locs += self.get_tids_between(vm_layer, self.arr_info.col_to_track(vm_layer, tot_cols - seg_sub_conn),
                                             self.arr_info.col_to_track(vm_layer, tot_cols),
                                             tr_sup_vm_w, self.get_track_sep(vm_layer, tr_sup_vm_w, tr_sup_vm_w), 0, True)
        for tid in sup_vm_locs:
            vss_vm_list.append(self.connect_to_tracks(vss_hm, tid))

        swap_inout = self.params['swap_inout']
        vss_xm_tidx = self.grid.coord_to_track(xm_layer, self.grid.track_to_coord(hm_layer, vss_hm.track_id.base_index),
                                               RoundMode.NEAREST)

        vss_xm = vss_vm_list 

        self.add_pin('sam', sam_vm)
        self.add_pin('ref', ref_hm)
        self.add_pin('sig', n_sig_hm)
        self.add_pin('VSS', vss_xm)

        self.sch_params = dict(
            n=dict(
                lch=self.arr_info.lch,
                seg=segn,
                w=wn,
                intent=self.place_info.get_row_place_info(0).row_info.threshold
            ),
            p=dict(
                lch=self.arr_info.lch,
                seg=segn,
                w=wn,
                intent=self.place_info.get_row_place_info(0).row_info.threshold
            )
        )


class SamplerTop(TemplateBaseZL):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sampler_top')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            vcm_sampler='Path to vcm sampler yaml(diff)',
            sig_sampler='Path to signal sampler yaml',
            vcm_mid_sw='Path to middle switch yaml',
            route_power='True to add top layer supply',
            decap_width='Decoupling cap width',
            cap_config='Settings for decoupling capacitors'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(route_power=True, decap_width=0, cap_config={})

    def draw_layout(self) -> None:
        sig_sampler_params: Param = Param(read_yaml(self.params['sig_sampler'])['params'])
        vcm_sampler_params: Param = read_yaml(self.params['vcm_sampler'])['params']
        mid_sw_params = read_yaml(self.params['vcm_mid_sw'])['params']

        mid_sw_master = self.new_template(GenericWrapper, params=mid_sw_params)
        sig_sampler_n_master = self.new_template(Bootstrap_simple, params=sig_sampler_params)
        sig_sampler_p_master = self.new_template(Bootstrap_simple, params=sig_sampler_params.copy(append=dict(nside=False)))
        vcm_sampler_master = self.new_template(BootstrapDiff_simple, params=vcm_sampler_params)

        conn_layer = mid_sw_master.core.conn_layer
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        tr_manager = TrackManager(grid=self.grid, tr_widths=sig_sampler_params['tr_widths'],
                                  tr_spaces=sig_sampler_params['tr_spaces'], )
        top_layer_btstrp = max(sig_sampler_n_master.top_layer, vcm_sampler_master.top_layer)
        top_layer_midsw = mid_sw_master.top_layer

        # top_layer = sampler_p_master.top_layer
        w_blk_midsw, h_blk_midsw = self.grid.get_block_size(top_layer_midsw)
        w_blk, h_blk = self.grid.get_block_size(top_layer_btstrp)
        w_cmsw, h_cmsw = mid_sw_master.bound_box.w, mid_sw_master.bound_box.h
        w_sig_sampler_se, h_sig_sampler_se = \
            sig_sampler_n_master.bound_box.w, sig_sampler_n_master.bound_box.h
        w_vcm_sampler_diff, h_vcm_sampler_diff = \
            vcm_sampler_master.bound_box.w, vcm_sampler_master.bound_box.h

        h_tot = max(h_sig_sampler_se, h_vcm_sampler_diff + h_cmsw)
        h_tot = -(-h_tot // h_blk) * h_blk

        # Calculate extra decap width for floorplanning (can be 0 for no decaps)
        route_power = self.params['route_power']
        decap_width = self.params['decap_width']
        if route_power:
            if decap_width>0:
                cap_width = -(-decap_width//(w_blk))*w_blk
                decap_vref_params = copy.deepcopy(self.params['cap_config'].to_dict())
                decap_vref_params['num_cols'] = cap_width//decap_vref_params['unit_width']
                decap_vref_params['num_rows'] = (2*h_tot//3)//decap_vref_params['unit_height']
                decap_vref_master = self.new_template(MIMCap,
                                    params=decap_vref_params)
                cap_w = decap_vref_master.bound_box.w
        else:
            cap_w = 0

        w_tot = max(w_vcm_sampler_diff, w_cmsw) + 2 * w_sig_sampler_se + 2 * cap_w
        w_tot = -(-w_tot // w_blk) * w_blk
        w_tot2 = w_tot // 2

        # get xm1layer supply pitch
        top_hor_lay = top_layer_btstrp if self.grid.get_direction(
            top_layer_btstrp) == Orient2D.x else top_layer_btstrp - 1

        top_hor_sup_w = tr_manager.get_width(top_hor_lay, 'sup')
        top_hor_sup_pitch = self.grid.get_track_pitch(top_hor_lay) * \
                            self.get_track_sep(top_hor_lay, top_hor_sup_w, top_hor_sup_w)

        sup_align_ofst=0
        sup_align_ofst += 2 * top_hor_sup_pitch 
        sup_align_ofst += -(-h_cmsw // 2 // top_hor_sup_pitch) * 2 * top_hor_sup_pitch
        y_cm_sampler = -(-sup_align_ofst // h_blk) * h_blk

        cm_sw_mid = self.add_instance(mid_sw_master, xform=Transform(w_tot2+w_cmsw//2, -y_cm_sampler, Orientation.MY))
        mid_sw_params['params']['swap_inout'] = True
        mid_sw_master = self.new_template(GenericWrapper, params=mid_sw_params)

        cm_sampler = self.add_instance(vcm_sampler_master,
                                       xform=Transform(w_tot2 - w_vcm_sampler_diff // 2, 0, Orientation.R0))
        sig_sampler_p = self.add_instance(sig_sampler_n_master, xform=Transform(w_tot-(w_sig_sampler_se+cap_w), 0, Orientation.R0))
        sig_sampler_n = self.add_instance(sig_sampler_p_master, xform=Transform(w_sig_sampler_se + cap_w, 0, Orientation.MY))

        # only add decaps if the decap width is > 0
        if cap_w > 0:
            decap_l = self.add_instance(decap_vref_master, inst_name='XDECAP', xform=Transform(0, 0))
            decap_r = self.add_instance(decap_vref_master, inst_name='XDECAP', xform=Transform(w_tot, 0, mode=Orientation.MY))
        
            self.set_size_from_bound_box(top_layer_btstrp, BBox(0, 0, w_tot, max(h_tot, decap_vref_master.bound_box.h)))
        else:
            self.set_size_from_bound_box(top_layer_btstrp, BBox(0, 0, w_tot, h_tot))

        self.connect_wires(cm_sampler.get_all_port_pins('VDD', layer=top_hor_lay) +
                           sig_sampler_n.get_all_port_pins('VDD', layer=top_hor_lay))
        self.connect_wires(cm_sampler.get_all_port_pins('VDD', layer=top_hor_lay) +
                           sig_sampler_p.get_all_port_pins('VDD', layer=top_hor_lay))
        self.connect_wires(cm_sampler.get_all_port_pins('VSS', layer=top_hor_lay) +
                           sig_sampler_n.get_all_port_pins('VSS', layer=top_hor_lay))
        self.connect_wires(cm_sampler.get_all_port_pins('VSS', layer=top_hor_lay) +
                           sig_sampler_p.get_all_port_pins('VSS', layer=top_hor_lay))

        # Connect between middle switch and cm sampler
        tr_w_sig_hm = tr_manager.get_width(hm_layer, 'sig')

        # connect S and D of middle switch
        mid_sw_p_hm = cm_sw_mid.get_pin('ref')
        mid_sw_n_hm = cm_sw_mid.get_pin('sig')
        cm_sw_n_ym = cm_sampler.get_pin('sig_n')
        cm_sw_p_ym = cm_sampler.get_pin('sig_p')
        
        w_conn_blk, h_conn_blk = self.grid.get_block_size(conn_layer)
        w_hm_blk, h_hm_blk = self.grid.get_block_size(hm_layer)
        vm_sw_n_tidx = int(round( self.grid.coord_to_track(vm_layer, -(-cm_sw_n_ym.upper//h_hm_blk)*h_hm_blk, mode=RoundMode.GREATER)))
        vm_sw_p_tidx = int(round(self.grid.coord_to_track(vm_layer, (cm_sw_p_ym.lower//h_hm_blk)*h_hm_blk, mode=RoundMode.LESS)))
        
        vm_swn = self.connect_to_tracks([mid_sw_n_hm, cm_sw_n_ym], TrackID(vm_layer, vm_sw_n_tidx, width=tr_w_sig_hm))
        vm_swp = self.connect_to_tracks([mid_sw_p_hm, cm_sw_p_ym], TrackID(vm_layer, vm_sw_p_tidx, width=tr_w_sig_hm),
                               track_lower=vm_swn.lower)
        
        # connect vg_cm
        cm_sw_vg_n = cm_sampler.get_pin('vg_n')
        cm_sw_vg_p = cm_sampler.get_pin('vg_p')
        vg_cm_hm_tidx = self.grid.coord_to_track(hm_layer, (cm_sw_vg_n.yl//(h_hm_blk//2))*(h_hm_blk//2))
        vg_cm_hm =self.add_wires(hm_layer, vg_cm_hm_tidx, lower=cm_sw_vg_n.xl, 
                                            upper=cm_sw_vg_p.xh, width=tr_w_sig_hm)
        vg_cm_vm = self.connect_bbox_to_track_wires(Direction.UPPER, (f'met{vm_layer}', 'drawing'), 
                                                        cm_sw_mid.get_pin('sam'), vg_cm_hm)
        self.add_pin('vg_cm', vg_cm_vm)
        self.reexport(sig_sampler_n.get_port('vg'), net_name='vg_n')
        self.reexport(sig_sampler_p.get_port('vg'), net_name='vg_p')

        # Routing clock signals
        sam = self.connect_wires([cm_sampler.get_pin('sample'), sig_sampler_n.get_pin('sample'), sig_sampler_p.get_pin('sample')])
        sam_b = self.connect_wires([cm_sampler.get_pin('sample_b'), sig_sampler_n.get_pin('sample_b'), sig_sampler_p.get_pin('sample_b')])
        self.add_pin('sam', sam)
        self.add_pin('sam_b', sam_b)

        for pinname in sig_sampler_p.port_names_iter():
            if 'out' in pinname:
                ppin = pinname.replace('out', 'out_p')
                npin = pinname.replace('out', 'out_n')
                self.reexport(sig_sampler_n.get_port(pinname), net_name=npin)
                self.reexport(sig_sampler_p.get_port(pinname), net_name=ppin)

        self.add_pin('out_p_bot', vm_swp)
        self.add_pin('out_n_bot', vm_swn)

        self.reexport(sig_sampler_n.get_port('in'), net_name='sig_n')
        self.reexport(sig_sampler_p.get_port('in'), net_name='sig_p')

        # connect power signals
        vdd_topm = [sig_sampler_n.get_pin('VDD'),
                   sig_sampler_p.get_pin('VDD'),
                     cm_sampler.get_pin('VDD')]
        vdd_topm = self.connect_wires(vdd_topm)

        vss_topm = self.connect_wires(sig_sampler_n.get_all_port_pins('VSS') + \
                   sig_sampler_p.get_all_port_pins('VSS') + \
                   cm_sampler.get_all_port_pins('VSS'))
        for cm_vss in cm_sw_mid.get_all_port_pins('VSS'):
            # connect cm sw to lower VSS track of samplers
            self.connect_bbox_to_track_wires(Direction.UPPER, (cm_sw_mid.get_port('VSS').get_single_layer(), 'drawing'), 
                                            cm_vss, sig_sampler_n.get_all_port_pins('VSS')[1])

        vcm_topm2 = [cm_sampler.get_pin('out_n'), cm_sampler.get_pin('out_p')]

        cmsw_params = dict()
        mid_sw_master_params = mid_sw_master.sch_params.to_dict()
        cmsw_params['n'] = mid_sw_master_params['n']
        cmsw_params['p'] = mid_sw_master_params['p']

        # top_layer_supply = self.params['top_layer_supply']
        # decap_width = self.params['decap_width']
        # if top_layer_supply:
        #     top_sup_layer = top_layer_btstrp + 1
        #     tr_w_sup_top = tr_manager.get_width(top_sup_layer, 'sup')
        #     tr_sp_sup_top = tr_manager.get_sep(top_sup_layer, ('sup', 'sup'))
        #     if self.grid.get_direction(top_sup_layer) == Orient2D.y:
        #         top_coord_mid = (self.bound_box.xl + self.bound_box.xh) // 2
        #         top_coord_upper = self.bound_box.xh
        #         top_upper = self.bound_box.yh
        #     else:
        #         vcm_top=[]
        #         for bbox in vcm_topm2:
        #             tidx = self.grid.coord_to_track(vm_layer, bbox.xm//w_conn_blk*w_conn_blk) 
        #             vcm_top.append(self.add_wires(vm_layer, tidx, lower=bbox.yl, upper=bbox.yh, width=tr_w_sig_hm))

        #         top_coord_mid = ((self.bound_box.yl + self.bound_box.yh) // (2*w_blk))*w_blk
        #         top_coord_upper = self.bound_box.yh
        #         top_upper = self.bound_box.xh
        #         top_cm_tid = self.grid.coord_to_track(top_sup_layer, top_coord_mid)
        #         #vcm_top = self.connect_to_tracks(vcm_topm, TrackID(top_layer_btstrp + 1, top_cm_tid, tr_w_sup_top))
        #         top_cm_lower = vcm_topm2.xl if self.grid.get_direction(
        #             top_sup_layer) == Orient2D.y else vcm_top.bound_box.yl
        #         top_cm_upper = vcm_topm2.xh if self.grid.get_direction(
        #             top_sup_layer) == Orient2D.y else vcm_top.bound_box.yh
        #         top_locs = self.get_tids_between(top_sup_layer,
        #                                          self.grid.coord_to_track(top_sup_layer, 0, RoundMode.GREATER),
        #                                          self.grid.coord_to_track(top_sup_layer, top_cm_lower, RoundMode.LESS),
        #                                          tr_w_sup_top, tr_sp_sup_top, 0, False, mod=2)
        #         vss_top_list = [self.connect_to_tracks(sig_sampler_n.get_all_port_pins('VSS', layer=top_layer_btstrp) + \
        #                                                cm_sampler.get_all_port_pins('VSS', layer=top_layer_btstrp),
        #                                                tidx, track_lower=0,
        #                                                track_upper=top_upper) for tidx in top_locs[1::2]]
        #         vdd_top_list = [self.connect_to_tracks(sig_sampler_n.get_all_port_pins('VDD', layer=top_layer_btstrp) + \
        #                                                cm_sampler.get_all_port_pins('VDD', layer=top_layer_btstrp),
        #                                                tidx, track_lower=0,
        #                                                track_upper=top_upper) for tidx in top_locs[0::2]]
        #         top_locs = self.get_tids_between(top_sup_layer,
        #                                          self.grid.coord_to_track(top_sup_layer, top_cm_upper,
        #                                                                   RoundMode.GREATER),
        #                                          self.grid.coord_to_track(top_sup_layer, top_coord_upper,
        #                                                                   RoundMode.LESS),
        #                                          tr_w_sup_top, tr_sp_sup_top, 0, False, True, mod=2)[::-1]
        #         vss_top_list += [self.connect_to_tracks(sig_sampler_p.get_all_port_pins('VSS', layer=top_layer_btstrp) + \
        #                                                 cm_sampler.get_all_port_pins('VSS', layer=top_layer_btstrp),
        #                                                 tidx, track_lower=0,
        #                                                 track_upper=top_upper) for tidx in top_locs[1::2]]
        #         vdd_top_list += [self.connect_to_tracks(sig_sampler_p.get_all_port_pins('VDD', layer=top_layer_btstrp) + \
        #                                                 cm_sampler.get_all_port_pins('VDD', layer=top_layer_btstrp),
        #                                                 tidx, track_lower=0,
        #                                                 track_upper=top_upper) for tidx in top_locs[0::2]]
        #         self.add_pin('VSS', vss_top_list)
        #         self.add_pin('VDD', vdd_top_list)
        #         self.add_pin('vcm', vcm_top)
        
        # else:

        route_power = self.params['route_power']
        decap_width = self.params['decap_width']
        if route_power:
            # TODO: space the grids nicely
            shields = False
            if sig_sampler_n.has_port('VSS_shield'):
                self.reexport(sig_sampler_n.get_port('VSS_shield'))
                shields = True
            if sig_sampler_p.has_port('VSS_shield'):
                self.reexport(sig_sampler_p.get_port('VSS_shield'))
                shields = True
            if cm_sampler.has_port('VSS_shield'):
                self.reexport(cm_sampler.get_port('VSS_shield'))
                shields = True 

            if shields:
                vdd_vm_l, vss_vm_l = self.do_power_fill(vm_layer, tr_manager, vdd_topm, vss_topm, 
                                                            bound_box = BBox(0, 0, cm_sampler.get_all_port_pins('VSS_shield')[0].xl, 
                                                               self.grid.htr_to_coord(hm_layer, vdd_topm[0].track_id.base_htr+1)))
                vdd_vm_r, vss_vm_r = self.do_power_fill(vm_layer, tr_manager, vdd_topm, vss_topm, 
                                                            bound_box = BBox(cm_sampler.get_all_port_pins('VSS_shield')[-1].xh, 0, w_tot,
                                                               self.grid.htr_to_coord(hm_layer, vdd_topm[0].track_id.base_htr+1)))
                xm_boxes = [BBox(0, 0, sig_sampler_n.get_all_port_pins('VSS_shield')[0].xl-w_blk, sig_sampler_n.bound_box.yh),
                            BBox(sig_sampler_n.get_all_port_pins('VSS_shield')[1].xh+w_blk, 0, cm_sampler.get_all_port_pins('VSS_shield')[1].xl-w_blk, sig_sampler_n.bound_box.yh-h_blk),
                            BBox(cm_sampler.get_all_port_pins('VSS_shield')[-1].xh+w_blk, 0, sig_sampler_p.get_all_port_pins('VSS_shield')[1].xl-w_blk, sig_sampler_p.bound_box.yh-h_blk),
                            BBox(sig_sampler_p.get_all_port_pins('VSS_shield')[0].xh+w_blk, 0, w_tot, sig_sampler_p.bound_box.yh)]
                vdd_xm, vss_xm = [], []
                for x in xm_boxes:
                    _vdd_xm, _vss_xm = self.do_power_fill(xm_layer, tr_manager, vdd_vm_l+vdd_vm_r, vss_vm_l+vss_vm_r, 
                                                            bound_box = x)
                    if cap_w > 0:
                        if x.xl > sig_sampler_n.get_all_port_pins('VSS_shield')[1].xh and x.xh < sig_sampler_p.get_all_port_pins('VSS_shield')[1].xl:
                            _vss_xm = self.extend_wires(_vss_xm, lower=cm_sampler.get_all_port_pins('VSS_shield')[0].xl, upper=cm_sampler.get_all_port_pins('VSS_shield')[-1].xh)

                        if x.xh < w_tot//2:
                            _vss_xm = self.connect_bbox_to_track_wires(Direction.UPPER, (f'met{ym_layer}', 'drawing'), decap_l.get_port('BOT').get_bounding_box(), _vss_xm)   
                        if x.xl > w_tot//2:
                            _vss_xm = self.connect_bbox_to_track_wires(Direction.UPPER, (f'met{ym_layer}', 'drawing'), decap_r.get_port('BOT').get_bounding_box(), _vss_xm)
                    vdd_xm = vdd_xm + _vdd_xm
                    vss_xm = vss_xm + _vss_xm

                ym_boxes = [BBox(sig_sampler_n.get_all_port_pins('VSS_shield')[1].xh, 0, cm_sampler.get_all_port_pins('VSS_shield')[0].xl, self.grid.htr_to_coord(vm_layer, vdd_xm[-1].track_id.base_htr+1) ),
                            BBox(cm_sampler.get_all_port_pins('VSS_shield')[2].xh, 0, sig_sampler_p.get_all_port_pins('VSS_shield')[1].xl,self.grid.htr_to_coord(vm_layer, vdd_xm[-1].track_id.base_htr+1) ),
                             self.bound_box,
                             BBox(sig_sampler_n.get_all_port_pins('VSS_shield')[0].xh, 0, sig_sampler_n.get_all_port_pins('VSS_shield')[1].xh, self.grid.htr_to_coord(vm_layer, vdd_xm[-1].track_id.base_htr+1)),
                            BBox(sig_sampler_p.get_all_port_pins('VSS_shield')[1].xl, 0, sig_sampler_p.get_all_port_pins('VSS_shield')[0].xl,self.grid.htr_to_coord(vm_layer, vdd_xm[-1].track_id.base_htr+1) ),]
                vdd_ym, vss_ym = [], []
                for y in ym_boxes:
                    _vdd_ym, _vss_ym = self.do_power_fill(ym_layer, tr_manager, vdd_xm, vss_xm, 
                                                            bound_box = y)
                    vdd_ym = vdd_ym + _vdd_ym
                    vss_ym = vss_ym + _vss_ym
            else:
                vdd_vm_l, vss_vm_l = self.do_power_fill(vm_layer, tr_manager, vdd_topm, vss_topm, 
                                                            bound_box = BBox(0, 0, cm_sampler.bound_box.xl, 
                                                               self.grid.htr_to_coord(hm_layer, vdd_topm[0].track_id.base_htr+1)))
                vdd_vm_r, vss_vm_r = self.do_power_fill(vm_layer, tr_manager, vdd_topm, vss_topm, 
                                                            bound_box = BBox(cm_sampler.bound_box.xh, 0, w_tot,
                                                               self.grid.htr_to_coord(hm_layer, vdd_topm[0].track_id.base_htr+1)))

                vdd_xm, vss_xm = self.do_power_fill(xm_layer, tr_manager, vdd_vm_l+vdd_vm_r, vss_vm_l+vss_vm_r,  
                                                            bound_box = self.bound_box)
                vdd_ym, vss_ym = self.do_power_fill(xm_layer, tr_manager, vdd_xm, vss_xm, 
                                                            bound_box = self.bound_box)

                if cap_w > 0:
                    self.connect_bbox_to_track_wires(Direction.LOWER, (f'met{vm_layer}', 'drawing'), decap_r.get_port('BOT').get_bounding_box(), vss_xm)
                    self.connect_bbox_to_track_wires(Direction.LOWER, (f'met{vm_layer}', 'drawing'), decap_l.get_port('BOT').get_bounding_box(), vss_xm)
            # get spaces from available tracks
            
            vdd_x2m, vss_x2m = self.do_power_fill(ym_layer+1, tr_manager, vdd_ym, vss_ym, 
                                                            bound_box = BBox(sig_sampler_n.bound_box.xl+2*w_blk, 
                                                                            0, sig_sampler_p.bound_box.xh-2*w_blk,
                                                                            sig_sampler_p.bound_box.yh)) 
            if cap_w> 0:
                self.extend_wires(vdd_x2m, upper=decap_r.get_port('TOP').get_bounding_box().xh, lower=decap_l.get_port('TOP').get_bounding_box().xl)
    

            self.add_pin('VSS', vss_ym)
            self.add_pin('VDD', vdd_ym)
            
        else:
            self.add_pin('VSS', vss_topm)
            self.add_pin('VDD', vdd_topm)
            # self.add_pin('vcm', vcm_top, connect=True)
        self.reexport(cm_sampler.get_port('out_n'), net_name='vcm') 
        self.reexport(cm_sampler.get_port('out_p'), net_name='vcm') 
        
        # schematic parameters
        self._sch_params = dict(
            cm_sw_params=cmsw_params,
            sig_sampler_params=sig_sampler_n_master.sch_params,
            vcm_sampler_params=vcm_sampler_master.sch_params,
        )

from typing import Any, Dict, Type, Optional, Mapping, Union, Tuple, List

from xbase.layout.mos.base import MOSBase
from xbase.layout.mos.placement.data import MOSArrayPlaceInfo
from xbase.layout.mos.top import GenericWrapper

from bag.design.database import ModuleDB, Module
from bag.layout.routing.base import TrackManager, TrackID
from bag.layout.template import TemplateDB, TemplateBase
from bag.util.immutable import Param
from bag.util.math import HalfInt

from pybag.core import Transform, BBox
from pybag.enum import RoundMode, Orientation, Direction
from .sar_cdac import CapDacColCore
from .sar_comp import SARComp, SA
from .sar_logic_sync import SARLogic, SARLogicArray
from .clk_sync_sar import SyncClkGen
from .sar_samp import Sampler
from .util.util import MOSBaseTapWrapper


class SARSlice(TemplateBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    def get_schematic_class(self) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_slice_wsamp' if self.params['sampler_params'] else
                                            'sar_slice_bot')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            comp_params='Tri-tail comparator parameters.',
            logic_params='SAR logic parameters',
            cdac_params='Comparator DAC parameters',
            sampler_params='Sampler parameters',
            clkgen_params='Clkgen parameters',
            tr_widths='Track width dictionary',
            tr_spaces='Track space dictionary',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(sampler_params=[])

    def draw_layout(self) -> None:
        comp_params: Param = self.params['comp_params']
        logic_params: Param = self.params['logic_params']
        cdac_params: Param = self.params['cdac_params']
        sampler_params: Param = self.params['sampler_params']
        clkgen_params: Param = self.params['clkgen_params']
        tr_widths: Dict[str, Any] = self.params['tr_widths']
        tr_spaces: Mapping[Tuple[str, str], Mapping[int, Union[float, HalfInt]]] = self.params['tr_spaces']
        tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        nbits = cdac_params['nbits'] + 1

        conn_layer = MOSArrayPlaceInfo.get_conn_layer(self.grid.tech_info,
                                                      comp_params['pinfo']['tile_specs']['arr_info']['lch'])
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        comp_params = comp_params.copy(append=dict(cls_name=SA.get_qualified_name(),
                                            sup_top_layer=3))
        cdac_master: CapDacColCore = self.new_template(CapDacColCore, params=cdac_params)
        comp_master_dummy: MOSBase = self.new_template(SARComp, params=comp_params) 
        clkgen_master_dummy: MOSBase = self.new_template(SyncClkGen, params=clkgen_params)
        logic_master_dummy: MOSBase = self.new_template(SARLogic, params=logic_params)
        logic_ncols_tot = logic_master_dummy.bound_box.w // comp_master_dummy.sd_pitch
         #2 * (cdac_master.bound_box.w // comp_master_dummy.sd_pitch)

        logic_gen_params = dict(
            ncols_tot= logic_ncols_tot + 100,
            cls_name=SARLogic.get_qualified_name(),
            params=logic_params
        )

        logic_master: TemplateBase = self.new_template(MOSBaseTapWrapper, params=logic_gen_params)
        cdac_actual_width = cdac_master.actual_width

        top_layer = max(logic_master.top_layer, comp_master_dummy.top_layer, cdac_master.top_layer)
        w_blk, h_blk = self.grid.get_block_size(top_layer)
        w_blk_in_sd_picth = w_blk//comp_master_dummy.sd_pitch
        ncols_tot = 2 * (cdac_actual_width) // comp_master_dummy.sd_pitch
        ncols_tot = ncols_tot//(2*w_blk_in_sd_picth)*(2*w_blk_in_sd_picth)

        comp_gen_params = dict(
            cls_name=SARComp.get_qualified_name(),
            params=comp_params.copy(append=dict(ncols_tot=ncols_tot-100))
        )
        comp_master: TemplateBase = self.new_template(GenericWrapper, params=comp_gen_params) 

        clkgen_params = dict(
            cls_name=SyncClkGen.get_qualified_name(),
            params=clkgen_params
        )
        clkgen_master: TemplateBase = self.new_template(GenericWrapper, params=clkgen_params)


        # Connect digital signals
        # If CDAC has pmos switch, connect logic signal dn/dp together
        sw_type = cdac_params.get('sw_type', ['n', 'n', 'n'])
        has_pmos_sw = 'p' in sw_type

        # floorplanning
        w_clkgen, h_clkgen = clkgen_master.bound_box.w, clkgen_master.bound_box.h
        w_logic, h_logic = logic_master.bound_box.w, logic_master.bound_box.h
        w_comp, h_comp = comp_master.bound_box.w, comp_master.bound_box.h
        w_dac, h_dac = cdac_master.bound_box.w, cdac_master.bound_box.h

        # Calculate logic signal routing
        lower_layer_routing = logic_params['logic_array']['lower_layer_routing'] 
        type_list = ['dig'] * nbits * 6 if has_pmos_sw else ['dig']*nbits*3
        if lower_layer_routing:
            type_list += type_list
        num_dig_tr, _ = tr_manager.place_wires(xm_layer, type_list, align_idx=0)
        coord_dig_tr = self.grid.track_to_coord(xm_layer, num_dig_tr)

        w_tot = max(w_logic, w_comp, 2 * w_dac)
        # w_tot = -(-w_tot // w_blk//2) * w_blk * 2
        w_tot = -(-w_tot//2)*2
        h_tot = -(-h_logic // h_blk) * h_blk + h_comp + coord_dig_tr

        comp_y = -(-h_tot // h_blk) * h_blk
        h_tot = -(-comp_y // h_blk) * h_blk + h_dac

        # logic_x = -(-(w_tot - w_logic) // 2 // w_blk) * w_blk
        # comp_x = -(-(w_tot - w_comp) // 2 // w_blk) * w_blk
        # dac_x = -(-(w_tot - 2 * w_dac) // 2 // w_blk) * w_blk
        logic_x = -(-(w_tot - w_logic)//2)
        comp_x = -(-(w_tot - w_comp) // 2)
        dac_x = -(-(w_tot - 2 * w_dac) // 2)
        clkgen_x = -(-(w_clkgen) // w_blk) * w_blk
        clkgen_y = -(-(h_clkgen) // h_blk) * h_blk

        logic = self.add_instance(logic_master, inst_name='XLOGIC', xform=Transform(logic_x, 0))
        comp = self.add_instance(comp_master, inst_name='XCOMP', xform=Transform(comp_x, comp_y-2*h_blk, mode=Orientation.MX))
        cdac_n = self.add_instance(cdac_master, inst_name='XDAC_N', xform=Transform(dac_x, comp_y, mode=Orientation.R0))
        cdac_p = self.add_instance(cdac_master, inst_name='XDAC_P',
                                   xform=Transform(w_tot - dac_x, comp_y, mode=Orientation.MY))
        clkgen = self.add_instance(clkgen_master, inst_name='XCLK', xform=Transform(logic_x - clkgen_x - 2*w_blk,
                                                                                         clkgen_y, mode=Orientation.MX))

        tr_w_cap_xm = tr_manager.get_width(xm_layer, 'cap')
        sampler_n_xm_list, sampler_p_xm_list = [], []
        if sampler_params:
            sar_gen_params = dict(
                cls_name=Sampler.get_qualified_name(),
                params=sampler_params
            )
            sampler_master: TemplateBase = self.new_template(GenericWrapper, params=sar_gen_params)
            w_sam, h_sam = sampler_master.bound_box.w, sampler_master.bound_box.h
            
            # calculate space for cap bot routing
            sam_y = -(-h_tot // h_blk) * h_blk
            sam_bot_xm_tidx = self.grid.coord_to_track(xm_layer, sam_y, mode=RoundMode.NEAREST)
            num_sam_ntr, sam_bot_locs = tr_manager.place_wires(xm_layer, ['cap']*nbits, align_idx=0,
                                                       align_track=sam_bot_xm_tidx)
            coord_sam_tr = self.grid.track_to_coord(xm_layer, num_sam_ntr)

            sam_bot_locs.pop(0)
            sam_y = -(-(h_tot + coord_sam_tr)// h_blk) * h_blk
            sampler_n = self.add_instance(sampler_master, inst_name='XSAM_N',
                                          xform=Transform(w_tot//2, sam_y, mode=Orientation.MY))
            sampler_p = self.add_instance(sampler_master, inst_name='XSAM_N',
                                          xform=Transform(w_tot//2, sam_y, mode=Orientation.R0))

            sampler_out_lay_purp = (sampler_n.get_port('out<0>').get_single_layer(), 'drawing')
            for idx in range(nbits-1):
                _n_xm = [self.connect_bbox_to_tracks(Direction.LOWER, sampler_out_lay_purp, w,
                                             TrackID(xm_layer, sam_bot_locs[idx], tr_w_cap_xm))
                 for w in sampler_n.get_all_port_pins(f'out<{idx}>')]
                sampler_n_xm_list.append(self.connect_wires(_n_xm)[0])
                _p_xm = [self.connect_bbox_to_tracks(Direction.LOWER, sampler_out_lay_purp, w,
                                             TrackID(xm_layer, sam_bot_locs[idx], tr_w_cap_xm))
                 for w in sampler_p.get_all_port_pins(f'out<{idx}>')]
                sampler_p_xm_list.append(self.connect_wires(_p_xm)[0])
            sampler_clk = self.connect_wires([sampler_n.get_pin('sam'), sampler_p.get_pin('sam')])
            self.add_pin('clk16', sampler_clk, connect=True)
            self.reexport(sampler_n.get_port('in'), net_name='in_n', connect=True)
            self.reexport(sampler_p.get_port('in'), net_name='in_p', connect=True)
            self.reexport(sampler_n.get_port('in_c'), net_name='in_p', connect=True)
            self.reexport(sampler_p.get_port('in_c'), net_name='in_n', connect=True)
        else:
            sampler_master=None
        
        # Expose clkgen pins on vm that are the length of logics
        self.reexport(clkgen.get_port('clk_in'), net_name='clk')
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')
        tr_w_vm_sig = tr_manager.get_width(vm_layer, 'dig')
        tidx_compclk_xm = self.grid.coord_to_track(xm_layer, clkgen.get_pin('clk_comp').ym, mode=RoundMode.NEAREST)
        tidx_clk16_xm = self.grid.coord_to_track(xm_layer, clkgen.get_pin('clk_out').ym, mode=RoundMode.NEAREST)
        compclk_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (clkgen.get_port('clk_comp').get_single_layer(), 'drawing'),
                                                clkgen.get_pin('clk_comp'), TrackID(xm_layer, tidx_compclk_xm, tr_w_vm_sig))
        clk16_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (clkgen.get_port('clk_out').get_single_layer(), 'drawing'),
                                                clkgen.get_pin('clk_out'),
                                                TrackID(xm_layer, tidx_clk16_xm, tr_w_vm_sig))

        tidx_clk_lo = self.grid.coord_to_track(vm_layer, clkgen.bound_box.xh, mode=RoundMode.GREATER)
        tidx_clk_hi = self.grid.coord_to_track(vm_layer, logic.bound_box.xl, mode=RoundMode.LESS)
        tidx_clk_list = self.get_available_tracks(vm_layer, tidx_clk_lo, tidx_clk_hi, 0, logic.bound_box.yh,
                                                    width=tr_w_vm_sig, sep=tr_w_vm_sig)
        
        comp_clk_gen = self.connect_to_tracks(compclk_xm, TrackID(vm_layer, tidx_clk_list[0], tr_w_vm_sig))
        clk16_gen = self.connect_to_tracks(clk16_xm, TrackID(vm_layer, tidx_clk_list[1], tr_w_vm_sig))
        comp_clk_gen = self.extend_wires(comp_clk_gen, lower=logic.bound_box.yl, upper=logic.bound_box.yh)
        clk16_gen = self.extend_wires(clk16_gen, lower=logic.bound_box.yl, upper=logic.bound_box.yh)

        # connect the comp_clk and clk16 wires to logic
        tidx_clklogic_lo = self.grid.coord_to_track(xm_layer, logic.get_pin('comp_clk').yl, mode=RoundMode.GREATER)
        tidx_clklogic_hi = self.grid.coord_to_track(xm_layer, logic.get_pin('comp_clk').yh, mode=RoundMode.LESS)
        tidx_clklogic_list = self.get_available_tracks(xm_layer, tidx_clklogic_lo, tidx_clklogic_hi, 0, logic.bound_box.yh,
                                                    width=tr_w_vm_sig, sep=tr_w_vm_sig)

        tidx_rstlogic_lo = self.grid.coord_to_track(xm_layer, logic.get_pin('rst').yl, mode=RoundMode.GREATER)
        tidx_rstlogic_hi = self.grid.coord_to_track(xm_layer, logic.get_pin('rst').yh, mode=RoundMode.LESS)
        tidx_rstlogic_list = self.get_available_tracks(xm_layer, tidx_rstlogic_lo, tidx_rstlogic_hi, 0, logic.bound_box.yh,
                                                    width=tr_w_vm_sig, sep=tr_w_vm_sig)
        
        compclk_logic_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (logic.get_port('comp_clk').get_single_layer(), 'drawing'),
                                                logic.get_pin('comp_clk'), 
                                                TrackID(xm_layer, tidx_clklogic_list[len(tidx_clklogic_list)//2], tr_w_vm_sig))
        rst_logic_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (logic.get_port('rst').get_single_layer(), 'drawing'),
                                                logic.get_pin('rst'),
                                                TrackID(xm_layer, tidx_rstlogic_list[len(tidx_rstlogic_list)//2], tr_w_vm_sig))
        comp_clk = self.connect_to_track_wires(comp_clk_gen, compclk_logic_xm)
        clk16 = self.connect_to_track_wires(clk16_gen, rst_logic_xm)

        # Connect differential input to comp
        dac_top_n, dac_top_p = cdac_n.get_pin('top'), cdac_p.get_pin('top')
        coord_p = self.grid.track_to_coord(hm_layer, comp.get_pin('inp')._tid.base_index)
        coord_n = self.grid.track_to_coord(hm_layer, comp.get_pin('inn')._tid.base_index)
        dac_top_n = self.extend_wires(dac_top_n, lower=coord_p, min_len_mode=-1)
        dac_top_p = self.extend_wires(dac_top_p, lower=coord_p, min_len_mode=-1)

        dac_topn_tidx = self.grid.coord_to_track(hm_layer, coord_n, mode=RoundMode.NEAREST) 
        dac_topp_tidx = self.grid.coord_to_track(hm_layer, coord_p, mode=RoundMode.NEAREST) 
        dac_n_xm = self.connect_to_tracks(dac_top_n, TrackID(hm_layer, dac_topn_tidx, tr_w_sig_xm))
        dac_p_xm = self.connect_to_tracks(dac_top_p, TrackID(hm_layer, dac_topp_tidx, tr_w_sig_xm))

        self.connect_wires([comp.get_pin('inp'), dac_p_xm]) 
        self.connect_wires([comp.get_pin('inn'), dac_n_xm])
        # self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('inp').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('inp'), dac_p_xm)
        # self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('inn').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('inn'), dac_n_xm)

        # Connect comp_out_m
        tr_w_dig_vm = tr_manager.get_width(vm_layer, 'dig')
        tr_w_clk_xm = tr_manager.get_width(xm_layer, 'clk')
        # comp_p_m_vm_tidx = self.grid.coord_to_track(vm_layer, logic.get_pin('comp_p_m').upper, mode=RoundMode.LESS_EQ)
        # comp_n_m_vm_tidx = self.grid.coord_to_track(vm_layer, logic.get_pin('comp_n_m').upper, mode=RoundMode.LESS_EQ)
        # comp_p_m_vm = self.connect_to_tracks(logic.get_pin('comp_p_m'), TrackID(vm_layer, comp_p_m_vm_tidx,
        #                                                                         tr_w_dig_vm))
        # comp_n_m_vm = self.connect_to_tracks(logic.get_pin('comp_n_m'), TrackID(vm_layer, comp_n_m_vm_tidx,
        # #                                                                         tr_w_dig_vm))

        # Connect comp to logic
        logic_top_hm_tidx = max(logic.get_all_port_pins('VSS', layer=hm_layer),
                                key=lambda x: x.track_id.base_index).track_id.base_index
        logic_top_coord = self.grid.track_to_coord(hm_layer, logic_top_hm_tidx)
        _, comp_out_locs = tr_manager.place_wires(xm_layer, ['clk'] * 2 + ['dig'] * 4, center_coord=logic_top_coord)
        # comp_p_m_xm = self.connect_to_tracks(comp_p_m_vm, TrackID(xm_layer, comp_out_locs[-1], tr_w_sig_xm))
        # comp_n_m_xm = self.connect_to_tracks(comp_n_m_vm, TrackID(xm_layer, comp_out_locs[-2], tr_w_sig_xm))

        # Connect comp_out
        # comp_p_xm = self.connect_to_tracks(logic.get_pin('comp_p'), TrackID(xm_layer, comp_out_locs[-3], tr_w_sig_xm))
        # comp_n_xm = self.connect_to_tracks(logic.get_pin('comp_n'), TrackID(xm_layer, comp_out_locs[-4], tr_w_sig_xm))
        comp_dir = Direction.LOWER if lower_layer_routing else Direction.UPPER
        logic_dir = Direction.LOWER if lower_layer_routing else Direction.UPPER
        
        tidx_comp_lo = self.grid.coord_to_track(xm_layer, (logic.bound_box.yh + comp.bound_box.yl- h_blk*2)//2 , mode=RoundMode.GREATER)
        tidx_comp_hi = self.grid.coord_to_track(xm_layer, comp.bound_box.yl, mode=RoundMode.LESS)
        tidx_comp_list = self.get_available_tracks(xm_layer, tidx_comp_lo, tidx_comp_hi,  cdac_n.bound_box.xl,  cdac_p.bound_box.xh,
                                    width=tr_w_vm_sig, sep=tr_w_vm_sig)
        comp_p_xm = self.connect_bbox_to_tracks(comp_dir, (logic.get_port('comp_p').get_single_layer(), 'drawing'),
                                                logic.get_pin('comp_p'),
                                                TrackID(xm_layer, tidx_comp_list[0], tr_w_sig_xm))  #comp_out_locs[-3]
        comp_n_xm = self.connect_bbox_to_tracks(comp_dir, (logic.get_port('comp_n').get_single_layer(), 'drawing'),
                                                logic.get_pin('comp_n'),
                                                TrackID(xm_layer, tidx_comp_list[0], tr_w_sig_xm)) #comp_out_locs[-4]
        # comp_p_xm = self.grid.coord_to_track(xm_layer, comp.get_pin('outn').yh, mode=RoundMode.NEAREST)
        # comp_n_xm = self.grid.coord_to_track(xm_layer, comp.get_pin('outp').yh, mode=RoundMode.NEAREST)
        comp_p_tidx_vm = self.grid.coord_to_track(vm_layer, (comp.get_pin('outp').lower + comp.get_pin('outp').upper)//2,
                                mode=RoundMode.NEAREST)
        comp_n_tidx_vm = self.grid.coord_to_track(vm_layer, (comp.get_pin('outn').lower + comp.get_pin('outn').upper)//2,
                                        mode=RoundMode.NEAREST)
        comp_p_vm = self.connect_to_tracks(comp_p_xm, TrackID(vm_layer, comp_p_tidx_vm, tr_w_vm_sig))
        comp_n_vm = self.connect_to_tracks(comp_n_xm, TrackID(vm_layer, comp_n_tidx_vm, tr_w_vm_sig))
        
        for idx in range(0, len(comp.get_all_port_pins('outn'))):
            comp_n = self.connect_to_track_wires(comp.get_all_port_pins('outn')[idx], comp_n_vm) #TrackID(xm_layer, comp_n_xm, tr_w_sig_xm))
            comp_p = self.connect_to_track_wires(comp.get_all_port_pins('outp')[idx], comp_p_vm)
        # comp_n = self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('outn').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('outn'), comp_n_vm) #TrackID(xm_layer, comp_n_xm, tr_w_sig_xm))
        # comp_p = self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('outp').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('outp'), comp_p_vm)  #TrackID(xm_layer, comp_p_xm, tr_w_sig_xm))
        # self.connect_wires([comp_n, logic.get_pin('comp_n')])
        # self.connect_wires([comp_p, logic.get_pin('comp_p')])
        # self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('outn').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('outn_m'), comp_n_m_xm)
        # self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('outp').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('outp_m'), comp_p_m_xm)
        
        # Connect comp_clk
        comp_clk_xm, comp_clkb_xm = [], []
        clk_bbox: List[BBox] = comp.get_all_port_pins('clk')
        #clkb_bbox: List[BBox] = comp.get_all_port_pins('clkb')
        for _bbox in clk_bbox:
            comp_clk_vm_tidx = self.grid.coord_to_track(vm_layer, _bbox.middle, mode=RoundMode.NEAREST)
            comp_clk_vm = self.connect_to_tracks(_bbox, TrackID(vm_layer, comp_clk_vm_tidx, tr_w_cap_xm))
            comp_clk_xm.append(self.connect_to_tracks(comp_clk_vm,  TrackID(xm_layer, comp_out_locs[-1], tr_w_clk_xm)))
            
            # self.connect_bbox_to_tracks(Direction.LOWER, (comp.get_port('clk').get_single_layer(), 'drawing'),
            #                                _bbox, TrackID(xm_layer, comp_out_locs[0], tr_w_clk_xm))
        # for _bbox in clkb_bbox:
        #     comp_clk_xm.append(
        #         self.connect_bbox_to_tracks(Direction.LOWER, (comp.get_port('clkb').get_single_layer(), 'drawing'),
        #                                     _bbox, TrackID(xm_layer, comp_out_locs[1], tr_w_clk_xm)))

        comp_clk_xm.append(
            self.connect_bbox_to_tracks(Direction.LOWER, (logic.get_port('comp_clk').get_single_layer(), 'drawing'),
                                        logic.get_pin('comp_clk'), TrackID(xm_layer, comp_out_locs[-1], tr_w_clk_xm)))
        #comp_clk_xm.append(
        #    self.connect_bbox_to_tracks(Direction.LOWER, (logic.get_port('comp_clkb').get_single_layer(), 'drawing'),
        #                                logic.get_pin('comp_clkb'), TrackID(xm_layer, comp_out_locs[1], tr_w_clk_xm)))
        
        # #self.connect_wires(comp_clkb_xm)
        self.connect_wires(comp_clk_xm)
    
        self.set_size_from_bound_box(top_layer, BBox(0, 0, -(-w_tot // w_blk//2) * w_blk * 2, -(-h_tot // h_blk//2) * h_blk * 2))


        # Route digital logic outputs to CDAC switches
        sig_type_list = ['clk'] + ['dig'] * nbits * 5 if has_pmos_sw else ['clk'] + ['dig'] * nbits * 3

        sig_type_list = sig_type_list * 2 if lower_layer_routing else sig_type_list

        _, dig_tr_locs = tr_manager.place_wires(xm_layer, sig_type_list, align_idx=0, align_track=comp_out_locs[5])
        tr_w_dig_xm = tr_manager.get_width(xm_layer, 'dig')
        dig_tr_locs.pop(0)
        dig_tr_locs_r = dig_tr_locs[::-1]
        if lower_layer_routing:
            rt_tidx_start = self.grid.coord_to_track(vm_layer, cdac_n.bound_box.xl)
            rt_tidx_stop = self.grid.coord_to_track(vm_layer, cdac_p.bound_box.xh)
            rt_tidx_cenl = self.grid.coord_to_track(vm_layer, clk_bbox[0].middle-10*w_blk)
            rt_tidx_cenr = self.grid.coord_to_track(vm_layer, clk_bbox[0].middle+10*w_blk)
            tr_w_vm_sig = tr_manager.get_width(vm_layer, 'dig')
            tr_sp_vm_sig = tr_manager.get_sep(vm_layer, ('dig', 'dig'))
            for idx in range(nbits - 1):
                dacp_n, dacp_p, dacp_m = \
                    cdac_p.get_pin(f'ctrl_n<{idx}>'), cdac_p.get_pin(f'ctrl_p<{idx}>'), cdac_p.get_pin(f'ctrl_m<{idx}>')
                dacn_n, dacn_p, dacn_m = \
                    cdac_n.get_pin(f'ctrl_n<{idx}>'), cdac_n.get_pin(f'ctrl_p<{idx}>'), cdac_n.get_pin(f'ctrl_m<{idx}>')
                if has_pmos_sw:
                    dn, dnb, dp, dpb, dm = logic.get_pin(f'dn<{idx + 1}>'), logic.get_pin(f'dn_b<{idx + 1}>'), \
                                           logic.get_pin(f'dp<{idx + 1}>'), logic.get_pin(f'dp_b<{idx + 1}>'), \
                                           logic.get_pin(f'dm<{idx + 1}>')
                    dac_p = self.connect_to_tracks([dacp_p], TrackID(xm_layer, dig_tr_locs_r[5 * idx], tr_w_dig_xm))
                    dac_pb = self.connect_to_tracks([dacp_n], TrackID(xm_layer, dig_tr_locs_r[5 * idx + 1], tr_w_dig_xm))
                    dac_n = self.connect_to_tracks([dacn_p], TrackID(xm_layer, dig_tr_locs_r[5 * idx + 2], tr_w_dig_xm))
                    dac_nb = self.connect_to_tracks([dacn_n], TrackID(xm_layer, dig_tr_locs_r[5 * idx + 3], tr_w_dig_xm))
                    dac_m = self.connect_to_tracks([dacp_m, dacn_m],
                                                   TrackID(xm_layer, dig_tr_locs_r[5 * idx + 4], tr_w_dig_xm))
                    dp = self.connect_bbox_to_tracks(logic_dir,
                                                (logic.get_port(f'dp<{idx + 1}>').get_single_layer(), 'drawing'),
                                                logic.get_pin(f'dp<{idx + 1}>'),
                                                TrackID(xm_layer, dig_tr_locs[5 * idx], tr_w_dig_xm))
                    dnb = self.connect_bbox_to_tracks(logic_dir,
                                                (logic.get_port(f'dn_b<{idx + 1}>').get_single_layer(), 'drawing'),
                                                logic.get_pin(f'dn_b<{idx + 1}>'),
                                                TrackID(xm_layer, dig_tr_locs[5 * idx + 1], tr_w_dig_xm))
                    
                    if (self.grid.coord_to_track(vm_layer, (dn.xh//w_blk)*w_blk) < rt_tidx_cenr 
                                    and rt_tidx_cenl < self.grid.coord_to_track(vm_layer, (dn.xh//w_blk)*w_blk)):
                        tidx_lo = self.grid.coord_to_track(xm_layer, logic.get_pin(f'dn<{idx + 1}>').yl, mode=RoundMode.GREATER)
                        tidx_hi = self.grid.coord_to_track(xm_layer, comp.bound_box.yl, mode=RoundMode.LESS)
                        tidx_list = self.get_available_tracks(xm_layer, tidx_lo, tidx_hi,  cdac_n.bound_box.xl,  cdac_p.bound_box.xh,
                                                    width=tr_w_vm_sig, sep=tr_w_vm_sig)
                        dn = self.connect_bbox_to_tracks(logic_dir,
                                                    (logic.get_port(f'dn<{idx + 1}>').get_single_layer(), 'drawing'),
                                                    logic.get_pin(f'dn<{idx + 1}>'),
                                                    TrackID(xm_layer, tidx_list[0], tr_w_dig_xm))
                    else:    
                        dn = self.connect_bbox_to_tracks(logic_dir,
                                                    (logic.get_port(f'dn<{idx + 1}>').get_single_layer(), 'drawing'),
                                                    logic.get_pin(f'dn<{idx + 1}>'),
                                                    TrackID(xm_layer, dig_tr_locs[5 * idx + 2], tr_w_dig_xm))
                    dpb = self.connect_bbox_to_tracks(logic_dir,
                                                (logic.get_port(f'dp_b<{idx + 1}>').get_single_layer(), 'drawing'),
                                                logic.get_pin(f'dp_b<{idx + 1}>'),
                                                TrackID(xm_layer, dig_tr_locs[5 * idx + 3], tr_w_dig_xm))
                    dm = self.connect_bbox_to_tracks(logic_dir,
                                                (logic.get_port(f'dm<{idx + 1}>').get_single_layer(), 'drawing'),
                                                logic.get_pin(f'dm<{idx + 1}>'),
                                                TrackID(xm_layer, dig_tr_locs[5 * idx + 4], tr_w_dig_xm))
                    rt_tidx_list = self.get_available_tracks(vm_layer, rt_tidx_start, rt_tidx_cenl,
                                                            lower=logic.bound_box.yh,
                                                            upper=comp.bound_box.yl-h_blk*20, width=tr_w_vm_sig, 
                                                            sep=tr_sp_vm_sig) + self.get_available_tracks(vm_layer, rt_tidx_cenr, rt_tidx_stop,
                                                            lower=logic.bound_box.yh,
                                                            upper=comp.bound_box.yl-h_blk*20, width=tr_w_vm_sig, sep=tr_sp_vm_sig)                        
                                                            
                                                             #FIXME
                    rt_tidx_coord_list = [self.grid.track_to_coord(vm_layer, x) for x in rt_tidx_list]
                    mid_coord = (self.bound_box.xl+self.bound_box.xh)//2
                    for _d, _c in zip([dn, dp, dm, dpb, dnb], [dac_n, dac_p, dac_m, dac_nb, dac_pb]):
                        _y_wire = self.connect_to_tracks([_d, _c], TrackID(vm_layer,
                                                                      SARLogicArray.get_nearest_tidx(_d, rt_tidx_list,
                                                                                                     rt_tidx_coord_list,
                                                                                                     mid_coord),
                                                                      tr_w_vm_sig))
                else:
                    dn, dp, dm = logic.get_pin(f'dn<{idx + 1}>'), logic.get_pin(f'dp<{idx + 1}>'), logic.get_pin(
                        f'dm<{idx + 1}>')
                    dac_p = self.connect_to_tracks([dacp_p, dacn_n],
                                                   TrackID(xm_layer, dig_tr_locs[3 * idx], tr_w_dig_xm))
                    dac_n = self.connect_to_tracks([dacp_n, dacn_p],
                                                   TrackID(xm_layer, dig_tr_locs[3 * idx + 1], tr_w_dig_xm))
                    dac_m = self.connect_to_tracks([dacp_m, dacn_m],
                                                   TrackID(xm_layer, dig_tr_locs[3 * idx + 2], tr_w_dig_xm))
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dp<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dp<{idx + 1}>'), dac_p)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dn<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dn<{idx + 1}>'), dac_n)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dm<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dm<{idx + 1}>'), dac_m)
        else:
            for idx in range(nbits - 1):
                dacp_n, dacp_p, dacp_m = \
                    cdac_p.get_pin(f'ctrl_n<{idx}>'), cdac_p.get_pin(f'ctrl_p<{idx}>'), cdac_p.get_pin(
                        f'ctrl_m<{idx}>')
                dacn_n, dacn_p, dacn_m = \
                    cdac_n.get_pin(f'ctrl_n<{idx}>'), cdac_n.get_pin(f'ctrl_p<{idx}>'), cdac_n.get_pin(
                        f'ctrl_m<{idx}>')
                if has_pmos_sw:
                    dn, dnb, dp, dpb, dm = logic.get_pin(f'dn<{idx + 1}>'), logic.get_pin(f'dn_b<{idx + 1}>'), \
                                           logic.get_pin(f'dp<{idx + 1}>'), logic.get_pin(f'dp_b<{idx + 1}>'), \
                                           logic.get_pin(f'dm<{idx + 1}>')
                    dac_p = self.connect_to_tracks([dacp_p], TrackID(xm_layer, dig_tr_locs[5 * idx], tr_w_dig_xm))
                    dac_pb = self.connect_to_tracks([dacp_n], TrackID(xm_layer, dig_tr_locs[5 * idx + 1], tr_w_dig_xm))
                    dac_n = self.connect_to_tracks([dacn_p], TrackID(xm_layer, dig_tr_locs[5 * idx + 2], tr_w_dig_xm))
                    dac_nb = self.connect_to_tracks([dacn_n], TrackID(xm_layer, dig_tr_locs[5 * idx + 3], tr_w_dig_xm))
                    dac_m = self.connect_to_tracks([dacp_m, dacn_m],
                                                   TrackID(xm_layer, dig_tr_locs[5 * idx + 4], tr_w_dig_xm))
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dp<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dp<{idx + 1}>'), dac_p)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dn_b<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dn_b<{idx + 1}>'), dac_pb)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dn<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dn<{idx + 1}>'), dac_n)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dp_b<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dp_b<{idx + 1}>'), dac_nb)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dm<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dm<{idx + 1}>'), dac_m)
                    # self.connect_to_track_wires(dac_p, dp)
                    # self.connect_to_track_wires(dac_pb, dnb)
                    # self.connect_to_track_wires(dac_n, dn)
                    # self.connect_to_track_wires(dac_nb, dmdpb)
                    # self.connect_to_track_wires(dac_m, dm)
                else:
                    dn, dp, dm = logic.get_pin(f'dn<{idx + 1}>'), logic.get_pin(f'dp<{idx + 1}>'), logic.get_pin(
                        f'dm<{idx + 1}>')
                    dac_p = self.connect_to_tracks([dacp_p, dacn_n],
                                                   TrackID(xm_layer, dig_tr_locs[3 * idx], tr_w_dig_xm))
                    dac_n = self.connect_to_tracks([dacp_n, dacn_p],
                                                   TrackID(xm_layer, dig_tr_locs[3 * idx + 1], tr_w_dig_xm))
                    dac_m = self.connect_to_tracks([dacp_m, dacn_m],
                                                   TrackID(xm_layer, dig_tr_locs[3 * idx + 2], tr_w_dig_xm))
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dp<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dp<{idx + 1}>'), dac_p)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dn<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dn<{idx + 1}>'), dac_n)
                    self.connect_bbox_to_track_wires(logic_dir,
                                                     (logic.get_port(f'dm<{idx + 1}>').get_single_layer(), 'drawing'),
                                                     logic.get_pin(f'dm<{idx + 1}>'), dac_m)


        # --- export pins:
        for idx in range(nbits):
            self.reexport(logic.get_port(f'data_out<{idx}>'))
            self.reexport(logic.get_port(f'dn<{idx}>'))
            self.reexport(logic.get_port(f'dp<{idx}>'))
            self.reexport(logic.get_port(f'dm<{idx}>'))
        for idx in range(nbits-1):
            self.reexport(cdac_p.get_port(f'bot<{idx}>'), net_name=f'bot_p<{idx}>')
            self.reexport(cdac_n.get_port(f'bot<{idx}>'), net_name=f'bot_n<{idx}>')

        sampler_xm_p_ret, sampler_xm_n_ret = [], []
        if sampler_params:
            for idx in range(nbits - 1):
                sampler_xm_p_ret.append(self.connect_to_track_wires(cdac_p.get_pin(f'bot<{idx}>'),
                                                                    sampler_p_xm_list[idx]))
                sampler_xm_n_ret.append(self.connect_to_track_wires(cdac_n.get_pin(f'bot<{idx}>'),
                                                                    sampler_n_xm_list[idx]))

            # extend sampler wires
            sampler_n_lower_coord = min([c.lower for c in sampler_xm_n_ret])
            sampler_n_upper_coord = max([c.upper for c in sampler_xm_n_ret])
            self.extend_wires(sampler_xm_n_ret, lower=sampler_n_lower_coord, upper=sampler_n_upper_coord)
            sampler_p_lower_coord = min([c.lower for c in sampler_xm_p_ret])
            sampler_p_upper_coord = max([c.upper for c in sampler_xm_p_ret])
            self.extend_wires(sampler_xm_p_ret, lower=sampler_p_lower_coord, upper=sampler_p_upper_coord)

        self.reexport(cdac_p.get_port('sam'), net_name='clk16', connect=True)
        self.reexport(cdac_n.get_port('sam'), net_name='clk16', connect=True)
        self.reexport(cdac_p.get_port('top'), net_name='top_p')
        self.reexport(cdac_n.get_port('top'), net_name='top_n')
        self.reexport(cdac_p.get_port(f'vref<0>'), connect=True)
        self.reexport(cdac_n.get_port(f'vref<0>'), connect=True)
        self.reexport(cdac_p.get_port(f'vref<1>'), connect=True)
        self.reexport(cdac_n.get_port(f'vref<1>'), connect=True)
        self.reexport(cdac_p.get_port(f'vref<2>'), connect=True)
        self.reexport(cdac_n.get_port(f'vref<2>'), connect=True)
        self.reexport(logic.get_port('clk_out'))
        #self.reexport(logic.get_port('done'))
        self.reexport(logic.get_port('comp_clk'))
        self.reexport(logic.get_port('rst'), net_name='clk16', connect=True)
        self.reexport(comp.get_port('outp'), net_name='comp_p')
        self.reexport(comp.get_port('outn'), net_name='comp_n')
        #self.reexport(comp.get_port('osp'))
        #self.reexport(comp.get_port('osn'))
        for inst in [cdac_p, cdac_n, comp, logic, clkgen]:
            self.reexport(inst.get_port('VSS'), connect=True)
            if inst.has_port('VDD'):
                self.reexport(inst.get_port('VDD'), connect=True)
        self.reexport(comp.get_port('VDD'), connect=True)
        self.reexport(logic.get_port('VDD'), connect=True)

        sar_params=dict(
            nbits=nbits,
            comp=comp_master.sch_params,
            logic=logic_master.sch_params,
            cdac=cdac_master.sch_params,
            clkgen=clkgen_master.sch_params,
            tri_sa=False,
            has_pmos_sw=has_pmos_sw,
        )

        if sampler_params:
            self._sch_params=dict(
                slice_params=sar_params,
                sampler_params=sampler_master.sch_params,
                sync=True
            )
        else:
            self._sch_params = sar_params

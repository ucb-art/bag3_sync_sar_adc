from typing import Any, Dict, Type, Optional, Mapping, Union, Tuple, List
import copy

from xbase.layout.mos.base import MOSBase
from xbase.layout.mos.placement.data import MOSArrayPlaceInfo
from xbase.layout.mos.top import GenericWrapper

from bag.design.database import ModuleDB, Module
from bag.layout.routing.base import TrackManager, TrackID, WireArray
from bag.layout.template import TemplateDB, TemplateBase
from bag.util.immutable import Param
from bag.util.math import HalfInt
from bag.io.file import read_yaml

from pybag.core import Transform, BBox
from pybag.enum import RoundMode, Orientation, Direction, Orient2D
from .sar_cdac import CapDacColCore, CapMIMCore
from .sar_comp import SARComp, SA
from .sar_logic_sync import SARLogic, SARLogicArray
from .clk_sync_sar import SyncClkGen, SyncDivCounter
from .sampler_top import SamplerTop
# from .sar_samp import Sampler
from .util.util import MOSBaseTapWrapper

from xbase.layout.cap.mim import MIMCap

class SARSliceBootstrap(TemplateBase):
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
            decoupling_cap_params='Decoupling cap parameters.',
            route_power="True if want decoupling and power strapping done",
            shield="True if want shielding",
            tr_widths='Track width dictionary',
            tr_spaces='Track space dictionary',
            directory='If the sub-components have separate yamls, then True'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(sampler_params=[],
                    shield=False,
                    route_power=False,
                    directory=False)

    def draw_layout(self) -> None:
        
        if self.params['directory']:
            comp_params: Param = read_yaml(self.params['comp_params'])['params']
            logic_params: Param = read_yaml(self.params['logic_params'])['params']
            cdac_params: Param = read_yaml(self.params['cdac_params'])['params']
            sampler_params: Param = read_yaml(self.params['sampler_params'])['params']
            clkgen_params: Param = read_yaml(self.params['clkgen_params'])['params']
            divcount: bool = read_yaml(self.params['clkgen_params'])['divcount']
        else:
            comp_params: Param = self.params['comp_params']
            comp_params = copy.deepcopy(comp_params).to_dict()
            logic_params: Param = self.params['logic_params']
            cdac_params: Param = self.params['cdac_params']
            sampler_params: Param = read_yaml(self.params['sampler_params'])['params']
            clkgen_params: Param = self.params['clkgen_params']['params']
            divcount: bool = self.params['clkgen_params']['divcount']

        tr_widths: Dict[str, Any] = self.params['tr_widths']
        tr_spaces: Mapping[Tuple[str, str], Mapping[int, Union[float, HalfInt]]] = self.params['tr_spaces']
        tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        nbits = cdac_params['nbits'] + 1
        route_power: Param = self.params['route_power']
        conn_layer = MOSArrayPlaceInfo.get_conn_layer(self.grid.tech_info,
                                                      comp_params['pinfo']['tile_specs']['arr_info']['lch'])
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        x2_layer = ym_layer + 1

        # comp_params = comp_params.copy(append=dict(cls_name=SA.get_qualified_name(),
        #                                     sup_top_layer=3))
        cdac_master: CapDacColCore = self.new_template(CapDacColCore, params=cdac_params)
        comp_master_dummy: MOSBase = self.new_template(SARComp, params=comp_params) 
        # clkgen_master_dummy: MOSBase = self.new_template(SyncClkGen, params=clkgen_params)
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

        
        comp_params['ncols_tot']=ncols_tot-100
        comp_gen_params = dict(
            cls_name=SARComp.get_qualified_name(),
            params=comp_params #.copy(append=dict(ncols_tot=ncols_tot-100))
        )
        comp_master: TemplateBase = self.new_template(GenericWrapper, params=comp_gen_params) 

        clkgen_clsname = SyncDivCounter.get_qualified_name() if divcount else SyncClkGen.get_qualified_name()
        clkgen_master_dummy: MOSBase = self.new_template(SyncDivCounter, params=clkgen_params) if divcount   \
                                            else self.new_template(SyncClkGen, params=clkgen_params)
        clkgen_params = dict(
            ncols_tot = clkgen_master_dummy.bound_box.w // comp_master_dummy.sd_pitch + 20 ,
            cls_name=clkgen_clsname,
            params=clkgen_params
        )
        clkgen_master: TemplateBase = self.new_template(MOSBaseTapWrapper, params=clkgen_params)


        # Connect digital signals
        # If CDAC has pmos switch, connect logic signal dn/dp together
        sw_type = cdac_params.get('sw_type', ['n', 'n', 'n'])
        has_pmos_sw = 'p' in sw_type

        # floorplanning
        w2_blk, h2_blk = self.grid.get_block_size(x2_layer)
        w_clkgen, h_clkgen = clkgen_master.bound_box.w, clkgen_master.bound_box.h
        w_logic, h_logic = logic_master.bound_box.w, logic_master.bound_box.h
        w_comp, h_comp = comp_master.bound_box.w, comp_master.bound_box.h
        w_dac, h_dac = cdac_master.bound_box.w+2*w2_blk, cdac_master.bound_box.h

        # Calculate logic signal routing
        lower_layer_routing = logic_params['logic_array']['lower_layer_routing'] 
        type_list = ['dig'] * nbits * 6 if has_pmos_sw else ['dig']*nbits*3
        if lower_layer_routing:
            type_list += type_list
        num_dig_tr, _ = tr_manager.place_wires(xm_layer, type_list, align_idx=0)
        coord_dig_tr = self.grid.track_to_coord(xm_layer, num_dig_tr)

        w_conn_blk, h_conn_blk = self.grid.get_block_size(conn_layer)
        w_blk, h_blk = self.grid.get_block_size(top_layer)

        w_tot = max(w_logic+w_clkgen+12*w_blk, w_comp, 2 * w_dac)
        w_tot = -(-w_tot//(2*w_conn_blk))*2*w_conn_blk
        h_tot = -(-h_logic // h_blk) * h_blk + h_comp + coord_dig_tr
        comp_y = -(-h_tot // h_blk) * h_blk
        h_tot = -(-comp_y // h_blk) * h_blk + h_dac
        
        w_centered = max(w_logic, w_comp, 2 * w_dac)
        w_clkgen_ext = w_tot - w_centered
        logic_x = -(-(w_centered - w_logic)//(2*w_conn_blk)-w_clkgen_ext//w_conn_blk)*w_conn_blk 
        comp_x = -(-(w_centered - w_comp) // (2*w_conn_blk)-w_clkgen_ext//w_conn_blk)*w_conn_blk
        dac_x = -(-(w_centered - 2 * w_dac) // (2*w_conn_blk)-w_clkgen_ext//w_conn_blk)*w_conn_blk
        clkgen_x = -(-(w_clkgen) // w_blk) * w_blk
        clkgen_y = -(-(h_clkgen) // h_blk) * h_blk

        logic = self.add_instance(logic_master, inst_name='XLOGIC', xform=Transform(logic_x, 0))
        comp = self.add_instance(comp_master, inst_name='XCOMP', xform=Transform(comp_x, comp_y-2*h_blk, mode=Orientation.MX))
        cdac_n = self.add_instance(cdac_master, inst_name='XDAC_N', xform=Transform(dac_x, comp_y, mode=Orientation.R0))
        cdac_p = self.add_instance(cdac_master, inst_name='XDAC_P',
                                   xform=Transform(w_tot - dac_x-(-w_clkgen_ext//w_blk)*w_blk, comp_y, mode=Orientation.MY))
        
        if divcount:
            clkgen = self.add_instance(clkgen_master, inst_name='XCLK', xform=Transform(logic_x - clkgen_x - 12*w_blk, 0))
        else: 
            clkgen = self.add_instance(clkgen_master, inst_name='XCLK', xform=Transform(logic_x - clkgen_x - 12*w_blk,
                                                                                         clkgen_y, mode=Orientation.MX))
        tr_w_cap_xm = tr_manager.get_width(xm_layer, 'cap')
        sampler_n_xm_list, sampler_p_xm_list = [], []
        if sampler_params:
            # sar_gen_params = dict(
            #     cls_name=SamplerTop.get_qualified_name(),
            #     params=sampler_params
            # )
            
            sampler_master: TemplateBase = self.new_template(SamplerTop, params=sampler_params)
            w_sam, h_sam = sampler_master.bound_box.w, sampler_master.bound_box.h
            if route_power:
                decap_params: Param = self.params['decoupling_cap_params']
                decoupling_margin: int = decap_params['decoupling_margin'] 
                # recalculate with decoupling cap width 
                sampler_params['route_power'] = route_power
                sampler_params['decap_width']=(w_tot-w_sam)//2
                sampler_params['cap_config']=decap_params['cap_config'].to_dict()
                sampler_master: TemplateBase = self.new_template(SamplerTop, params=sampler_params)
                w_sam, h_sam = sampler_master.bound_box.w, sampler_master.bound_box.h
            
            # calculate space for cap bot routing and voltage reference routing
            sam_y = (-(-h_tot // h_blk)) * h_blk
            sam_bot_xm_tidx = self.grid.coord_to_track(xm_layer, sam_y, mode=RoundMode.NEAREST)
            num_ref_ntr, sam_ref_locs = tr_manager.place_wires(xm_layer, ['cap']*(4), align_idx=0,
                                                       align_track=sam_bot_xm_tidx)
            num_sam_ntr, sam_bot_locs = tr_manager.place_wires(xm_layer, ['cap']*(nbits), align_idx=0,
                                                       align_track=sam_ref_locs[-1])
            coord_sam_tr = self.grid.track_to_coord(xm_layer, num_sam_ntr+num_ref_ntr)

            #sam_ref_locs.pop(0)
            #sam_bot_locs.pop(0)
            sam_y = -(-(h_tot + coord_sam_tr)// h_blk) * h_blk
            sampler = self.add_instance(sampler_master, inst_name='XSAM',
                              xform=Transform(-(-(w_centered-w_sam)//(2*w_conn_blk))*w_conn_blk + (-(-w_clkgen_ext//w_conn_blk)*w_conn_blk ), sam_y)) #mode=Orientation.MY))
            # sampler_n = self.add_instance(sampler_master, inst_name='XSAM_N',
            #                               xform=Transform(w_tot//2, sam_y, mode=Orientation.MY))
            # sampler_p = self.add_instance(sampler_master, inst_name='XSAM_N',
            #                               xform=Transform(w_tot//2, sam_y, mode=Orientation.R0))

            sampler_out_lay_purp = (sampler.get_port('out_n<0>').get_single_layer(), 'drawing')
            for idx in range(nbits-1):
                _n_xm = [self.connect_bbox_to_tracks(Direction.LOWER, sampler_out_lay_purp, w,
                                             TrackID(xm_layer, sam_bot_locs[idx], tr_w_cap_xm))
                                            for w in sampler.get_all_port_pins(f'out_n<{idx}>')]
                sampler_n_xm_list.append(self.connect_wires(_n_xm)[0])
                _p_xm = [self.connect_bbox_to_tracks(Direction.LOWER, sampler_out_lay_purp, w,
                                             TrackID(xm_layer, sam_bot_locs[idx], tr_w_cap_xm))
                                         for w in sampler.get_all_port_pins(f'out_p<{idx}>')]
                sampler_p_xm_list.append(self.connect_wires(_p_xm)[0])
            #sampler_clk = self.connect_wires([sampler_n.get_pin('sam'), sampler_p.get_pin('sam')])
            # self.add_pin('clk16', sampler.get_all_port_pins('sam'), connect=True)
            # self.add_pin('clk16_b', sampler.get_all_port_pins('sam_b'), connect=True)
            self.reexport(sampler.get_port('sig_n'), net_name='in_n', connect=True)
            self.reexport(sampler.get_port('sig_p'), net_name='in_p', connect=True)
            self.reexport(sampler.get_port('out_n_bot'), net_name='top_n')
            self.reexport(sampler.get_port('out_p_bot'), net_name='top_p')
            self.reexport(sampler.get_port('vcm'), net_name='vref<1>')
            # self.reexport(sampler_n.get_port('in_c'), net_name='in_p', connect=True)
            # self.reexport(sampler_p.get_port('in_c'), net_name='in_n', connect=True)
        else:
            sampler_master=None
        
        # Expose clkgen pins on vm that are the length of logics
        self.reexport(clkgen.get_port('clk_in'), net_name='clk')
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')
        tr_w_vm_sig = tr_manager.get_width(vm_layer, 'dig')

        tr_w_xm_clk = tr_manager.get_width(xm_layer, 'sig')*4
        tr_w_vm_clk = tr_manager.get_width(vm_layer, 'dig')*4

        clkgen_compclk_pin = clkgen.get_pin('comp_clk') #if divcount else clkgen.get_pin('clk_comp')
        clkgen_compclk_port = clkgen.get_port('comp_clk')# if divcount else clkgen.get_port('clk_comp')
        tidx_compclk_xm = self.grid.coord_to_track(xm_layer, clkgen_compclk_pin.ym, mode=RoundMode.NEAREST)

        clkgen_clkout_pin = clkgen.get_pin('clk_out_b') if divcount else clkgen.get_pin('clk_out')
        clkgen_clkout_port = clkgen.get_port('clk_out_b') if divcount else clkgen.get_port('clk_out')
        clkgen_clkoutb_pin = clkgen.get_pin('clk_out') if divcount else clkgen.get_pin('clk_out_b')
        clkgen_clkoutb_port = clkgen.get_port('clk_out') if divcount else clkgen.get_port('clk_out_b')
        tidx_clk16_xm = self.grid.coord_to_track(xm_layer, clkgen.get_pin('clk_out').yl, mode=RoundMode.NEAREST)
        tidx_clk16_b_xm = tidx_clk16_xm + 2 #self.grid.coord_to_track(xm_layer, clkgen.get_pin('clk_out_b').ym, mode=RoundMode.NEAREST)
        compclk_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (clkgen_compclk_port.get_single_layer(), 'drawing'),
                                                clkgen_compclk_pin, TrackID(xm_layer, tidx_compclk_xm, tr_w_sig_xm))
        clk16_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (clkgen_clkout_port.get_single_layer(), 'drawing'),
                                                clkgen_clkout_pin,
                                                TrackID(xm_layer, tidx_clk16_xm, tr_w_sig_xm))
        clk16_b_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (clkgen_clkoutb_port.get_single_layer(), 'drawing'),
                                                clkgen_clkoutb_pin,
                                                TrackID(xm_layer, tidx_clk16_b_xm, tr_w_sig_xm))

        tidx_clk_lo = self.grid.coord_to_track(vm_layer, clkgen.bound_box.xh, mode=RoundMode.GREATER)
        tidx_clk_hi = self.grid.coord_to_track(vm_layer, logic.bound_box.xl, mode=RoundMode.LESS)
        tidx_clk_list = self.get_available_tracks(vm_layer, tidx_clk_lo, tidx_clk_hi, 0, logic.bound_box.yh,
                                                    width=tr_w_vm_clk, sep=tr_w_vm_clk)
        comp_clk_gen = self.connect_to_tracks(compclk_xm, TrackID(vm_layer, tidx_clk_list[0], tr_w_vm_clk))
        clk16_gen = self.connect_to_tracks(clk16_xm, TrackID(vm_layer, tidx_clk_list[1], tr_w_vm_clk))
        clk16_b_gen = self.connect_to_tracks(clk16_b_xm, TrackID(vm_layer, tidx_clk_list[2], tr_w_vm_clk))
        comp_clk_gen = self.extend_wires(comp_clk_gen, lower=logic.bound_box.yl, upper=logic.bound_box.yh)
        clk16_gen = self.extend_wires(clk16_gen, lower=logic.bound_box.yl, upper=logic.bound_box.yh)
        clk16_b_gen = self.connect_to_tracks(clk16_b_xm, TrackID(vm_layer, tidx_clk_list[2], tr_w_vm_clk))
        clk16_b_gen = self.extend_wires(clk16_b_gen, lower=logic.bound_box.yl, upper=logic.bound_box.yh)
        self.add_pin('clk16_b', clk16_b_gen, connect=True) 

        # connect the comp_clk and clk16 wires to logic
        tidx_clklogic_lo = self.grid.coord_to_track(xm_layer, logic.get_pin('comp_clk').yl, mode=RoundMode.GREATER)
        tidx_clklogic_hi = self.grid.coord_to_track(xm_layer, logic.get_pin('comp_clk').yh, mode=RoundMode.LESS)
        tidx_clklogic_list = self.get_available_tracks(xm_layer, tidx_clklogic_lo, tidx_clklogic_hi, 0, logic.bound_box.yh,
                                                    width=tr_w_vm_clk, sep=tr_w_vm_clk)

        tidx_rstlogic_lo = self.grid.coord_to_track(xm_layer, logic.get_pin('rst').yl, mode=RoundMode.GREATER)
        tidx_rstlogic_hi = self.grid.coord_to_track(xm_layer, logic.get_pin('rst').yh, mode=RoundMode.LESS)
        tidx_rstlogic_list = self.get_available_tracks(xm_layer, tidx_rstlogic_lo, tidx_rstlogic_hi, 0, logic.bound_box.yh,
                                                    width=tr_w_vm_clk, sep=tr_w_vm_clk)
        
        compclk_logic_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (logic.get_port('comp_clk').get_single_layer(), 'drawing'),
                                                logic.get_pin('comp_clk'), 
                                                TrackID(xm_layer, tidx_clklogic_list[len(tidx_clklogic_list)//2], tr_w_vm_clk))
        rst_logic_xm = self.connect_bbox_to_tracks(Direction.LOWER,
                                                (logic.get_port('rst').get_single_layer(), 'drawing'),
                                                logic.get_pin('rst'),
                                                TrackID(xm_layer, tidx_rstlogic_list[len(tidx_rstlogic_list)//2], tr_w_vm_clk))
        comp_clk = self.connect_to_track_wires(comp_clk_gen, compclk_logic_xm)
        clk16 = self.connect_to_track_wires(clk16_gen, rst_logic_xm)

        # Connect differential input to comp
        dac_top_n, dac_top_p = cdac_n.get_pin('top'), cdac_p.get_pin('top')
        coord_p = self.grid.track_to_coord(hm_layer, comp.get_pin('inp')._tid.base_index)
        coord_n = self.grid.track_to_coord(hm_layer, comp.get_pin('inn')._tid.base_index)
        dac_top_n = self.extend_wires(dac_top_n, lower=coord_p, min_len_mode=-1)
        dac_top_p = self.extend_wires(dac_top_p, lower=coord_p, min_len_mode=-1)

        dac_topn_tidx = self.grid.coord_to_track(xm_layer, coord_n, mode=RoundMode.NEAREST) 
        dac_topp_tidx = self.grid.coord_to_track(xm_layer, coord_p, mode=RoundMode.NEAREST) 
        dac_n_xm = self.connect_to_tracks(dac_top_n, TrackID(xm_layer, dac_topn_tidx, tr_w_sig_xm))
        dac_p_xm = self.connect_to_tracks(dac_top_p, TrackID(xm_layer, dac_topp_tidx, tr_w_sig_xm))
        
        cdac_n_coord = self.grid.track_to_coord(ym_layer, cdac_n.get_pin('top').track_id.base_index)
        cdac_p_coord = self.grid.track_to_coord(ym_layer, cdac_p.get_pin('top').track_id.base_index)
        dac_topn_vm_tidx = self.grid.coord_to_track(vm_layer, cdac_n_coord)
        dac_topp_vm_tidx = self.grid.coord_to_track(vm_layer,cdac_p_coord)
        dac_n_vm = self.connect_to_tracks(dac_p_xm, TrackID(vm_layer, dac_topp_vm_tidx, tr_w_sig_xm))
        dac_p_vm = self.connect_to_tracks(dac_n_xm, TrackID(vm_layer, dac_topn_vm_tidx, tr_w_sig_xm))

        self.connect_to_track_wires(comp.get_pin('inp'), dac_n_vm) 
        self.connect_to_track_wires(comp.get_pin('inn'), dac_p_vm) 
        # self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('inp').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('inp'), dac_p_xm)
        # self.connect_bbox_to_track_wires(Direction.LOWER, (comp.get_port('inn').get_single_layer(), 'drawing'),
        #                                  comp.get_pin('inn'), dac_n_xm)

        tr_w_dig_vm = tr_manager.get_width(vm_layer, 'dig')
        tr_w_clk_xm = tr_manager.get_width(xm_layer, 'clk')*2

        # Connect comp to logic
        logic_top_hm_tidx = max(logic.get_all_port_pins('VSS', layer=hm_layer),
                                key=lambda x: x.track_id.base_index).track_id.base_index
        logic_top_coord = self.grid.track_to_coord(hm_layer, logic_top_hm_tidx)
        _, comp_out_locs = tr_manager.place_wires(xm_layer, ['clk'] * 2 + ['dig'] * 4, center_coord=logic_top_coord)
        # comp_p_m_xm = self.connect_to_tracks(comp_p_m_vm, TrackID(xm_layer, comp_out_locs[-1], tr_w_sig_xm))
        # comp_n_m_xm = self.connect_to_tracks(comp_n_m_vm, TrackID(xm_layer, comp_out_locs[-2], tr_w_sig_xm))

        # Connect comp_out
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
        
        # Connect comparator clk to the same fast clock used in logic
        comp_clk_xm, comp_clkb_xm = [], []
        clk_bbox: List[BBox] = comp.get_all_port_pins('clk')
        for _bbox in clk_bbox:
            comp_clk_vm_tidx = self.grid.coord_to_track(vm_layer, _bbox.middle, mode=RoundMode.NEAREST)
            comp_clk_vm = self.connect_to_tracks(_bbox, TrackID(vm_layer, comp_clk_vm_tidx, tr_w_cap_xm))
            comp_clk_xm.append(self.connect_to_tracks(comp_clk_vm,  TrackID(xm_layer, comp_out_locs[-1], tr_w_clk_xm)))
            center_coord = _bbox.middle # save to use to calculate differential sampling clock tracks
        comp_clk_xm.append(
            self.connect_bbox_to_tracks(Direction.LOWER, (logic.get_port('comp_clk').get_single_layer(), 'drawing'),
                                        logic.get_pin('comp_clk'), TrackID(xm_layer, comp_out_locs[-1], tr_w_clk_xm)))
        self.connect_wires(comp_clk_xm)
    
        # set design size
        self.set_size_from_bound_box(top_layer, BBox(0, 0, -(-w_tot // w_blk//2) * w_blk * 2, -(-h_tot // h_blk//2) * h_blk * 2))

        # Route digital logic outputs to CDAC switches
        sig_type_list = ['clk'] + ['dig'] * nbits * 5 if has_pmos_sw else ['clk'] + ['dig'] * nbits * 3

        sig_type_list = sig_type_list * 2 if lower_layer_routing else sig_type_list

        _, dig_tr_locs = tr_manager.place_wires(xm_layer, sig_type_list, align_idx=0, align_track=comp_out_locs[5])
        tr_w_dig_xm = tr_manager.get_width(xm_layer, 'dig')
        dig_tr_locs.pop(0)
        dig_tr_locs_r = dig_tr_locs[::-1]
        if lower_layer_routing:
            rt_tidx_start = self.grid.coord_to_track(vm_layer, cdac_n.bound_box.xl-10*w_blk)
            rt_tidx_stop = self.grid.coord_to_track(vm_layer, cdac_p.bound_box.xh+10*w_blk)
            rt_tidx_cenl = comp_n_tidx_vm-1  #self.grid.coord_to_track(vm_layer, -(-clk_bbox[0].middle//w_blk)*w_blk-10*w_blk)
            rt_tidx_cenr = comp_p_tidx_vm+1 #self.grid.coord_to_track(vm_layer, -(-clk_bbox[0].middle//w_blk)*w_blk+10*w_blk)
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
                    if (self.grid.coord_to_track(vm_layer, (dp.xh//w_blk)*w_blk) < rt_tidx_cenr 
                                    and rt_tidx_cenl < self.grid.coord_to_track(vm_layer, (dp.xh//w_blk)*w_blk)):
                        tidx_lo = self.grid.coord_to_track(xm_layer, logic.get_pin(f'dp<{idx + 1}>').yl, mode=RoundMode.GREATER)
                        tidx_hi = self.grid.coord_to_track(xm_layer, comp.bound_box.yl, mode=RoundMode.LESS)
                        tidx_list = self.get_available_tracks(xm_layer, tidx_lo, tidx_hi,  cdac_n.bound_box.xl,  cdac_p.bound_box.xh,
                                                    width=tr_w_vm_sig, sep=tr_w_vm_sig)
                        dp = self.connect_bbox_to_tracks(logic_dir,
                                                    (logic.get_port(f'dp<{idx + 1}>').get_single_layer(), 'drawing'),
                                                    logic.get_pin(f'dp<{idx + 1}>'),
                                                    TrackID(xm_layer, tidx_list[0], tr_w_dig_xm))            
                    else:                
                        dp = self.connect_bbox_to_tracks(logic_dir,
                                                    (logic.get_port(f'dp<{idx + 1}>').get_single_layer(), 'drawing'),
                                                    logic.get_pin(f'dp<{idx + 1}>'),
                                                    TrackID(xm_layer, dig_tr_locs[5 * idx], tr_w_dig_xm))
                    dnb = self.connect_bbox_to_tracks(logic_dir,
                                                (logic.get_port(f'dn_b<{idx + 1}>').get_single_layer(), 'drawing'),
                                                logic.get_pin(f'dn_b<{idx + 1}>'),
                                                TrackID(xm_layer, dig_tr_locs[5 * idx + 1], tr_w_dig_xm))
                    
                    # if (self.grid.coord_to_track(vm_layer, (dn.xh//w_blk)*w_blk) < rt_tidx_cenr 
                    #                 and rt_tidx_cenl < self.grid.coord_to_track(vm_layer, (dn.xh//w_blk)*w_blk)):
                    #     tidx_lo = self.grid.coord_to_track(xm_layer, logic.get_pin(f'dn<{idx + 1}>').yl, mode=RoundMode.GREATER)
                    #     tidx_hi = self.grid.coord_to_track(xm_layer, comp.bound_box.yl, mode=RoundMode.LESS)
                    #     tidx_list = self.get_available_tracks(xm_layer, tidx_lo, tidx_hi,  cdac_n.bound_box.xl,  cdac_p.bound_box.xh,
                    #                                 width=tr_w_vm_sig, sep=tr_w_vm_sig)
                    #     dn = self.connect_bbox_to_tracks(logic_dir,
                    #                                 (logic.get_port(f'dn<{idx + 1}>').get_single_layer(), 'drawing'),
                    #                                 logic.get_pin(f'dn<{idx + 1}>'),
                    #                                 TrackID(xm_layer, tidx_list[0], tr_w_dig_xm))
                    # else:    
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
                                                            upper=comp.bound_box.yl-h_blk, width=tr_w_vm_sig, 
                                                            sep=tr_sp_vm_sig) + self.get_available_tracks(vm_layer, rt_tidx_cenr, rt_tidx_stop,
                                                            lower=logic.bound_box.yh,
                                                            upper=comp.bound_box.yl-h_blk, width=tr_w_vm_sig, sep=tr_sp_vm_sig)                        
                                                                                             
                    rt_tidx_coord_list = [self.grid.track_to_coord(vm_layer, x) for x in rt_tidx_list]
                    mid_coord = -(-clk_bbox[0].middle//w_blk)*w_blk
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

        self.reexport(cdac_p.get_port('top'), net_name='top_p')
        self.reexport(cdac_n.get_port('top'), net_name='top_n') #TODO: connect this to the sampler topn
        # self.reexport(cdac_p.get_port(f'vref<0>'), connect=True)
        # self.reexport(cdac_n.get_port(f'vref<0>'), connect=True)
        # self.reexport(cdac_p.get_port(f'vref<1>'), connect=True)
        # self.reexport(cdac_n.get_port(f'vref<1>'), connect=True)
        # self.reexport(cdac_p.get_port(f'vref<2>'), connect=True)
        # self.reexport(cdac_n.get_port(f'vref<2>'), connect=True)
        
        # Route the voltage references
        vref_xm = []
        for idx in range(3):
            vref = []
            cdac_ref_layer_purp = (cdac_p.get_port(f'vref<{idx}>').get_single_layer(), 'drawing')
            vref.append(self.connect_bbox_to_tracks(Direction.LOWER, cdac_ref_layer_purp, 
                                                                cdac_p.get_all_port_pins(f'vref<{idx}>')[0],  TrackID(xm_layer, sam_ref_locs[idx], tr_w_cap_xm)))
            vref.append(self.connect_bbox_to_tracks(Direction.LOWER, cdac_ref_layer_purp, 
                                                                    cdac_n.get_all_port_pins(f'vref<{idx}>')[0],  TrackID(xm_layer, sam_ref_locs[idx], tr_w_cap_xm)))
            _vref_xm = self.extend_wires(self.connect_wires(vref), lower=sampler.bound_box.xl, upper=sampler.bound_box.xh)
            vref_xm = vref_xm + _vref_xm
            self.add_pin(f"vref<{idx}>", _vref_xm)
        samp_vref = sampler.get_all_port_pins('vcm')
        samp_vref_wires = []
        for ref in samp_vref:
            samp_ref_layer_purp = (sampler.get_port('vcm').get_single_layer(), 'drawing')
            samp_vref_wires.append(self.connect_bbox_to_track_wires(Direction.LOWER, samp_ref_layer_purp, 
                                                                    ref, vref_xm[1]))

        self.reexport(logic.get_port('clk_out'))
        if divcount:    
            self.reexport(clkgen.get_port('comp_clk'), net_name='comp_clk')
        else:
            self.reexport(logic.get_port('comp_clk'))
        self.reexport(logic.get_port('rst'), net_name='clk16', connect=True)
        self.reexport(comp.get_port('outp'), net_name='comp_p')
        self.reexport(comp.get_port('outn'), net_name='comp_n')

        for inst in [cdac_p, cdac_n, comp, logic, clkgen]:
            self.reexport(inst.get_port('VSS'), connect=True)
            if inst.has_port('VDD'):
                self.reexport(inst.get_port('VDD'), connect=True)
        self.reexport(comp.get_port('VDD'), connect=True)
        self.reexport(logic.get_port('VDD'), connect=True)
        if sampler:
            self.reexport(sampler.get_port('VSS'), connect=True)
            self.reexport(sampler.get_port('VDD'), connect=True)
            self.reexport(sampler.get_port('vg_n'))
            self.reexport(sampler.get_port('vg_p'))
            self.reexport(sampler.get_port('vg_cm'))
        
            # Route Sampling Clock
            clk_sam_xm = self.connect_differential_tracks(clk16_b_gen, clk16_gen, xm_layer, tidx_comp_list[1], tidx_comp_list[2])
            comp_clk_ym_tidx = self.grid.coord_to_track(ym_layer, center_coord, mode=RoundMode.NEAREST)
            clk_sam_ym = self.connect_differential_tracks(clk_sam_xm[0], clk_sam_xm[1], ym_layer, tr_manager.get_next_track(vm_layer, comp_clk_ym_tidx, 'sig', 'sig',
                                                up=True), tr_manager.get_next_track(vm_layer, comp_clk_ym_tidx, 'sig', 'sig',
                                                up=False))
            bit_loc = (nbits//2)
            # cannot use connect to differential wires if the tidx is the same for both wires 
            clk_xm1 = self.connect_to_tracks(clk_sam_ym[0], TrackID(xm_layer, sam_bot_locs[bit_loc], 1))
            clk_xm2 = self.connect_to_tracks(clk_sam_ym[1], TrackID(xm_layer, sam_bot_locs[bit_loc], 1))
            samp_vcm_n_vm_tidx =  self.grid.coord_to_track(vm_layer, clk_xm1.middle, mode=RoundMode.GREATER)
            samp_vcm_p_vm_tidx =  self.grid.coord_to_track(vm_layer, clk_xm2.middle, mode=RoundMode.LESS)

            clk_sam_vm1 = self.connect_to_tracks(clk_xm1, TrackID(vm_layer, samp_vcm_n_vm_tidx, 1))
            clk_sam_vm2 = self.connect_to_tracks(clk_xm2, TrackID(vm_layer, samp_vcm_p_vm_tidx, 1))

            self.connect_differential_wires(clk_sam_vm1, clk_sam_vm2, sampler.get_pin('sam_b'), sampler.get_pin('sam'))
        
            # Route top_n and top_p to the sampler
            sam_top_xm = self.connect_differential_tracks(dac_top_p, dac_top_n, xm_layer, 
                                                          sam_bot_locs[bit_loc-1], sam_bot_locs[(bit_loc -2)])
            self.connect_differential_wires(sampler.get_all_port_pins('out_n_bot'), sampler.get_all_port_pins('out_p_bot'), 
                                            sam_top_xm[1], sam_top_xm[0])
        
        shield: Param = self.params['shield']
        if shield:
            VSS_shields = sampler.get_all_port_pins('VSS_shield') + comp.get_all_port_pins('VSS_shield')
            # if not route_power: FIXME
            #     self.reexport(comp.get_port('VSS_shield'), net_name='VSS')
        
        if route_power:
            # Decoupling and power grid 
            decap_params: Param = self.params['decoupling_cap_params']
            decoupling_margin: int = decap_params['decoupling_margin'] 
            decap_vref_params = copy.deepcopy(decap_params['cap_config'].to_dict())

            # Connect Sampler grid to the 'main' grid
            # just make a FAT M2 routing for inputs that go beyond the power grids
            # do multi-power grid over sampler (m4 m5), M3 ground plane coat
            # decoupling cap for references on the top
            # you should be doing the multi power domain over the cap, that's what uses it 

            # samp_power_vm = self.do_power_multi_power_fill(vm_layer, tr_manager, [sampler.get_pin('VSS'), sampler.get_pin('VDD')], 
            #                                                 bound_box = BBox(sampler.bound_box.xl, sampler.bound_box.yl, sampler.bound_box.xh, sampler.bound_box.h//2+sampler.bound_box.yl))
            # # get spaces from available tracks
            # samp_power_xm = self.do_power_multi_power_fill(xm_layer, tr_manager, samp_power_vm, 
            #                                                 bound_box = sampler.bound_box)
            # vref_ym_strap = self.do_power_multi_power_fill(ym_layer, tr_manager, vref_xm + samp_power_xm,
            #                                     bound_box=sampler.bound_box)
            # vref_x2m_strap = self.do_power_multi_power_fill(ym_layer+1, tr_manager, vref_ym_strap,
            #                                     bound_box=sampler.bound_box) 
            # #  Decoupling and Power grid routing
            # can I give blockages, 
            # place caps with some margin
            

            # CDAC N decoupling caps 
            #  VDD
            #  Vref<2>
            #  vref<0>
            #  vref<1>
            #  --------- Mirror over on other side for 8 total
            cdac_cap_width = ((cdac_n.bound_box.xl-clkgen.bound_box.xl)*2//(3*w_blk))*w_blk
            cdac_cap_height = ((cdac_n.bound_box.yh-cdac_n.bound_box.yl-h_blk)*2//(8*3*h_blk))*h_blk+h_blk
            decap_vref_params['num_cols'] = cdac_cap_width//decap_vref_params['unit_width']
            decap_vref_params['num_rows'] = cdac_cap_height//decap_vref_params['unit_height']
            decap_vref_master = self.new_template(MIMCap,
                                params=decap_vref_params)
            cap_w = decap_vref_master.bound_box.w
            decap_y = cdac_n.bound_box.yl-h_blk
            cdac_decaps_n = []
            for i in range(8):
                cdac_decaps_n.append(self.add_instance(decap_vref_master, inst_name=f'XDECAP_CDAC{i}', 
                                                     xform=Transform(cdac_n.bound_box.xl-2* w_blk, decap_y, mode=Orientation.MY)))
                decap_y = decap_y + decap_vref_master.bound_box.h
            decap_bot_wire_n = self.add_wires(ym_layer, self.grid.coord_to_track(ym_layer, cdac_decaps_n[0].get_port('BOT').get_bounding_box().xm//w_blk*w_blk),
                                            int(cdac_decaps_n[0].get_port('BOT').get_bounding_box().yl//h_blk*h_blk), int(decap_y))

            logic_cap_width = ((clkgen.bound_box.xh-clkgen.bound_box.xl)//(2*w_blk))*w_blk
            logic_cap_height = ((comp.bound_box.yl-clkgen.bound_box.yh)*2//(3*h_blk))*h_blk
            decap_logic_n_params = copy.deepcopy(decap_params['cap_config'].to_dict())
            decap_logic_n_params['num_cols'] = logic_cap_width//decap_logic_n_params['unit_width']
            decap_logic_n_params['num_rows'] = logic_cap_height//decap_logic_n_params['unit_height']
            decap_logic_n_master = self.new_template(MIMCap,
                                params=decap_logic_n_params)
            cap_w = decap_logic_n_master.bound_box.w
            decap_logic_n = self.add_instance(decap_logic_n_master, inst_name='XDECAP_0', xform=Transform(cdac_decaps_n[0].bound_box.xl, 
                                                                                                        clkgen.bound_box.yh))

            # other side
            #  CDAC P
            decap_y = max(clkgen.bound_box.yh + decap_vref_master.bound_box.h, cdac_n.bound_box.yl-h_blk)//h_blk*h_blk
            cdac_cap_width = ((cdac_n.bound_box.xl-clkgen.bound_box.xl)*2//(3*w_blk))*w_blk
            cdac_cap_height = ((cdac_n.bound_box.yh-decap_y)*2//(8*3*h_blk))*h_blk+h_blk
            decap_vref_params['num_cols'] = cdac_cap_width//decap_vref_params['unit_width']
            decap_vref_params['num_rows'] = cdac_cap_height//decap_vref_params['unit_height']
            decap_vref_master = self.new_template(MIMCap,
                                params=decap_vref_params)
            cap_w = decap_vref_master.bound_box.w
            cdac_decaps_p = []
            for i in range(8):
                cdac_decaps_p.append(self.add_instance(decap_vref_master, inst_name=f'XDECAP_CDAC{i}', 
                                                     xform=Transform(cdac_p.bound_box.xh+w_blk, decap_y)))
                decap_y = decap_y + decap_vref_master.bound_box.h

            decap_bot_wire_p = self.add_wires(ym_layer, self.grid.coord_to_track(ym_layer, -(-cdac_decaps_p[0].get_port('BOT').get_bounding_box().xm//w_blk)*w_blk),
                                            int(cdac_decaps_p[0].get_port('BOT').get_bounding_box().yl//h_blk*h_blk), int(decap_y))
            
            logic_cap_width = (2*(cdac_decaps_p[0].bound_box.xh-logic.bound_box.xh)//(3*w_blk))*w_blk - w_blk
            logic_cap_height = (2*(comp.bound_box.yl)//(3*h_blk))*h_blk
            decap_logic_p_params = copy.deepcopy(decap_params['cap_config'].to_dict())
            decap_logic_p_params['num_cols'] = logic_cap_width//decap_logic_p_params['unit_width']
            decap_logic_p_params['num_rows'] = logic_cap_height//decap_logic_p_params['unit_height']
            decap_logic_p_master = self.new_template(MIMCap,
                                params=decap_logic_p_params)
            cap_w = decap_logic_p_master.bound_box.w
            decap_logic_p = self.add_instance(decap_logic_p_master, inst_name='XDECAP_3', xform=Transform(logic.bound_box.xh+cap_w+2*w_blk, 0, mode = Orientation.MY))
            decap_y = max(decap_logic_p_master.bound_box.h, comp.bound_box.yl)

            # Grid routing for CDACs (4 voltages here + VSS)
            cdac_coord_xn = sorted(cdac_n.get_all_port_pins('VSS'), key=lambda x: x.xh)[-1].xh 
            cdac_coord_xp = sorted(cdac_p.get_all_port_pins('VSS'), key=lambda x: x.xl)[0].xl 
            cdac_p_ports = [cdac_p.get_all_port_pins(f'VSS'), cdac_p.get_all_port_pins(f'VDD'),
                            cdac_p.get_all_port_pins(f'vref<2>'), cdac_p.get_all_port_pins(f'vref<0>'), cdac_p.get_all_port_pins(f'vref<1>')]
            cdac_n_ports = [cdac_n.get_all_port_pins(f'VSS'),cdac_n.get_all_port_pins(f'VDD'),
                            cdac_n.get_all_port_pins(f'vref<2>'), cdac_n.get_all_port_pins(f'vref<0>'), cdac_n.get_all_port_pins(f'vref<1>')]
            
            cdac_xm_p = self.do_power_multi_power_fill_box(xm_layer, tr_manager, cdac_p_ports, 
                                                bound_box=BBox(cdac_coord_xp, cdac_p.bound_box.yl, cdac_p.bound_box.xh, cdac_p.bound_box.yh))
            cdac_xm_n = self.do_power_multi_power_fill_box(xm_layer, tr_manager, cdac_n_ports, 
                                                bound_box=BBox(cdac_n.bound_box.xl, cdac_n.bound_box.yl, cdac_coord_xn, cdac_n.bound_box.yh))
            cdac_ym_p =self.do_power_multi_power_fill(ym_layer, tr_manager, cdac_xm_p, 
                                                bound_box=BBox(cdac_coord_xp, cdac_p.bound_box.yl, cdac_p.bound_box.xh, cdac_p.bound_box.yh))
            cdac_ym_n =self.do_power_multi_power_fill(ym_layer, tr_manager, cdac_xm_n, 
                                                bound_box=BBox(cdac_n.bound_box.xl, cdac_n.bound_box.yl, cdac_coord_xn, cdac_n.bound_box.yh))
            cdac_ym =[cdac_ym_p[i] + cdac_ym_n[i] for i in range(len(cdac_ym_n))]

            cdac_x2m_box = BBox(cdac_n.bound_box.xl, cdac_n.bound_box.yl, cdac_p.bound_box.xh, cdac_p.bound_box.yh)
            
            cdac_x2m = self.do_power_multi_power_fill(ym_layer+1, tr_manager, cdac_ym, 
                                                bound_box=cdac_x2m_box)
            
            # Connect VSS to capacitors
            self.connect_to_track_wires(cdac_xm_n[4], decap_bot_wire_n)
            self.connect_to_track_wires(cdac_xm_p[4], decap_bot_wire_p)
            
            # Connect the voltage CDAC wires
            num_c = len(cdac_decaps_n)-1
            for idx, x2 in enumerate(cdac_x2m[1:]):
                for wire in x2:
                    wire_coord = self.grid.htr_to_coord(ym_layer+1, wire.track_id.base_htr)
                    if (wire_coord<=cdac_decaps_n[idx].bound_box.yh and wire_coord>=cdac_decaps_n[idx].bound_box.yl) or \
                        (wire_coord<=cdac_decaps_n[num_c-idx].bound_box.yh and wire_coord>=cdac_decaps_n[num_c-idx].bound_box.yl):
                        self.extend_wires(wire, lower=int(cdac_decaps_n[0].get_port('TOP').get_bounding_box().xh//w_blk*w_blk-w_blk),
                                            upper=int(cdac_decaps_p[0].get_port('TOP').get_bounding_box().xl//w_blk*w_blk-w_blk))

            # Just VDD and VSS
            xm_boxes = [BBox(clkgen.bound_box.xl, 0, w_tot, clkgen.bound_box.yh), 
                        BBox(0, clkgen.bound_box.yh, w_tot, logic.bound_box.yh),
                        BBox(0, comp.bound_box.yl, cdac_coord_xn, comp.bound_box.yh),
                        BBox(cdac_coord_xp, comp.bound_box.yl, w_tot, comp.bound_box.yh)]
            ym_boxes = [BBox(0, comp.bound_box.yl, cdac_coord_xn, comp.bound_box.yh),
                        BBox(cdac_coord_xp, comp.bound_box.yl, w_tot, comp.bound_box.yh),
                        clkgen.bound_box, logic.bound_box]
            x2m_boxes = [BBox(0, 0, logic.bound_box.xh, clkgen.bound_box.yh), 
                         BBox(logic.bound_box.xl, clkgen.bound_box.yh, logic.bound_box.xh, comp.bound_box.yh)]

            vm_vdd = comp.get_all_port_pins('VDD', layer=f'met{vm_layer}') + \
                            logic.get_all_port_pins('VDD',layer=f'met{vm_layer}') +\
                            clkgen.get_all_port_pins('VDD',layer=f'met{vm_layer}')
            vm_vss = comp.get_all_port_pins('VSS', layer=f'met{vm_layer}') + \
                            logic.get_all_port_pins('VSS',layer=f'met{vm_layer}') + \
                            clkgen.get_all_port_pins('VSS',layer=f'met{vm_layer}')
                            #if your layer is on a vertical layer (or a box probably) then have to specify the string name
            xm_power = [[],[]]
            for box in xm_boxes:
                _xm_power = self.do_power_multi_power_fill_box(xm_layer, tr_manager, [vm_vdd, vm_vss], 
                                                bound_box=box)
                xm_power = [x + _x for x, _x in zip(xm_power, _xm_power)]
            ym_power =[[],[]]
            for box in ym_boxes:
                _ym_power = self.do_power_multi_power_fill(ym_layer, tr_manager, xm_power,
                                                bound_box=box)
                ym_power = [y + _y for y, _y in zip(ym_power, _ym_power)]
            x2m_power = [[],[]]
            for box in x2m_boxes:
                _x2m_power = self.do_power_multi_power_fill(ym_layer+1, tr_manager, ym_power,
                                                bound_box=box)
                x2m_power = [x2 + _x2 for x2, _x2 in zip(x2m_power, _x2m_power)]
            #Connect VSS
            self.connect_bbox_to_track_wires(Direction.UPPER, (decap_logic_n.get_port('BOT').get_single_layer(), 'drawing'), 
                                                decap_logic_n.get_all_port_pins('BOT')[0], xm_power[1])
            self.connect_bbox_to_track_wires(Direction.UPPER, (decap_logic_p.get_port('BOT').get_single_layer(), 'drawing'), 
                                                decap_logic_p.get_all_port_pins('BOT')[0], xm_power[1])
            #Connect VDD
            for wire in x2m_power[0]:
                wire_coord = self.grid.htr_to_coord(ym_layer+1, wire.track_id.base_htr)
                if (wire_coord<=decap_logic_n.bound_box.yh and wire_coord>=decap_logic_n.bound_box.yl):
                    self.extend_wires(wire, lower=int(decap_logic_n.get_port('TOP').get_bounding_box().xl//w_blk*w_blk-w_blk))
                if (wire_coord<=decap_logic_p.bound_box.yh and wire_coord>=decap_logic_p.bound_box.yl):
                    self.extend_wires(wire, upper=int(decap_logic_p.get_port('TOP').get_bounding_box().xh//w_blk*w_blk+w_blk))

        # Schematic parameters
        sar_params=dict(
            nbits=nbits,
            comp=comp_master.sch_params,
            logic=logic_master.sch_params,
            cdac=cdac_master.sch_params,
            clkgen=clkgen_master.sch_params,
            tri_sa=False,
            has_pmos_sw=has_pmos_sw,
            divcount=divcount
        )

        if sampler_params:
            self._sch_params=dict(
                slice_params=sar_params,
                sampler_params=sampler_master.sch_params,
                sync=True,
                bootstrap=True,
            )
        else:
            self._sch_params = sar_params

    def do_power_multi_power_fill(self, layer_id: int, tr_manager: TrackManager, strap_list: List[Union[WireArray, List[WireArray]]],
                              bound_box: Optional[BBox] = None,
                              x_margin: int = 0, y_margin: int = 0,
                              uniform_grid: bool = False) -> Tuple[List[WireArray], List[WireArray]]:
        """Draw power fill on the given layer."""
        if bound_box is None:
            if self.bound_box is None:
                raise ValueError("bound_box is not set")
            bound_box = self.bound_box
        bound_box = bound_box.expand(dx=-x_margin, dy=-y_margin)
        is_horizontal = (self.grid.get_direction(layer_id) == 0)
        if is_horizontal:
            cl, cu = bound_box.yl, bound_box.yh
            lower, upper = bound_box.xl, bound_box.xh
        else:
            cl, cu = bound_box.xl, bound_box.xh
            lower, upper = bound_box.yl, bound_box.yh
        fill_width = tr_manager.get_width(layer_id, 'sup')
        fill_space = tr_manager.get_sep(layer_id, ('sup', 'sup'))
        sep_margin = tr_manager.get_sep(layer_id, ('sup', ''))
        tr_bot = self.grid.coord_to_track(layer_id, cl, mode=RoundMode.GREATER_EQ)
        tr_top = self.grid.coord_to_track(layer_id, cu, mode=RoundMode.LESS_EQ)
        trs_all = self.get_available_tracks(layer_id, tid_lo=tr_bot, tid_hi=tr_top, lower=lower, upper=upper,
                                        width=fill_width, sep=fill_space*2, sep_margin=sep_margin,
                                        uniform_grid=uniform_grid)
        trs = trs_all
        # trs = []
        # if (layer_id ==4 ):
        #     for tr in trs_all:
        #        if not tr.is_integer:
        #            trs.append(tr)
        # else:
        #     trs = trs_all
        top_vdd: List[WireArray] = []
        top_vss: List[WireArray] = []
        all_wars = [[] for _ in strap_list]
        ret_warrs = []
        htr_sep = HalfInt.convert(fill_space).dbl_value
        
        for ncur, tr_idx in enumerate(trs):
            # tr_idx = (htr0 + ncur * htr_pitch - 1) / 2

            warr = self.add_wires(layer_id, tr_idx, lower, upper, width=fill_width)
            _ncur = HalfInt.convert(tr_idx).dbl_value // htr_sep if uniform_grid else ncur
            all_wars[_ncur % len(strap_list)].append(warr)
        for top_warr, bot_warr in zip(strap_list, all_wars):
            self.draw_vias_on_intersections(top_warr, bot_warr)
        return all_wars
    
    def do_power_multi_power_fill_box(self, layer_id: int, tr_manager: TrackManager, box_list: List[Union[BBox, List[BBox]]],
                              bound_box: Optional[BBox] = None,
                              x_margin: int = 0, y_margin: int = 0,
                              uniform_grid: bool = False) -> Tuple[List[WireArray], List[WireArray]]:
        """Draw power fill on the given layer."""
        if bound_box is None:
            if self.bound_box is None:
                raise ValueError("bound_box is not set")
            bound_box = self.bound_box
        bound_box = bound_box.expand(dx=-x_margin, dy=-y_margin)
        is_horizontal = (self.grid.get_direction(layer_id) == 0)
        if is_horizontal:
            cl, cu = bound_box.yl, bound_box.yh
            lower, upper = bound_box.xl, bound_box.xh
        else:
            cl, cu = bound_box.xl, bound_box.xh
            lower, upper = bound_box.yl, bound_box.yh
        fill_width = tr_manager.get_width(layer_id, 'sup')
        fill_space = tr_manager.get_sep(layer_id, ('sup', 'sup'))
        sep_margin = tr_manager.get_sep(layer_id, ('sup', ''))
        tr_bot = self.grid.coord_to_track(layer_id, cl, mode=RoundMode.GREATER_EQ)
        tr_top = self.grid.coord_to_track(layer_id, cu, mode=RoundMode.LESS_EQ)
        trs_all = self.get_available_tracks(layer_id, tid_lo=tr_bot, tid_hi=tr_top, lower=lower, upper=upper,
                                        width=fill_width, sep=fill_space*2, sep_margin=sep_margin,
                                        uniform_grid=uniform_grid)
        trs = trs_all
        # trs = []
        # if (layer_id ==4 ):
        #     for tr in trs_all:
        #        if not tr.is_integer:
        #            trs.append(tr)
        # else:
        #     trs = trs_all
        top_vdd: List[WireArray] = []
        top_vss: List[WireArray] = []
        all_wars = [[] for _ in box_list]
        ret_warrs = []
        htr_sep = HalfInt.convert(fill_space).dbl_value
        
        for ncur, tr_idx in enumerate(trs):
            # tr_idx = (htr0 + ncur * htr_pitch - 1) / 2

            warr = self.add_wires(layer_id, tr_idx, lower, upper, width=fill_width)
            _ncur = HalfInt.convert(tr_idx).dbl_value // htr_sep if uniform_grid else ncur
            all_wars[_ncur % len(box_list)].append(warr)
        for bot_box, top_warr in zip(box_list, all_wars):
            for warr in top_warr:
                for b in bot_box:
                    track_coord = self.grid.htr_to_coord(layer_id, warr.track_id.base_htr)
                    if self.grid.get_direction(layer_id)==Orient2D.x and b.xl>=bound_box.xl and b.xh<=bound_box.xh \
                        and b.yl<=track_coord and b.yh>=track_coord:
                        self.connect_bbox_to_track_wires(Direction.LOWER, 
                            (f'met{layer_id-1}', 'drawing'), b, warr)
                    if self.grid.get_direction(layer_id)==Orient2D.y and b.yl>=bound_box.yl and b.yh<=bound_box.yh \
                        and b.xl<=track_coord and b.xh>=track_coord:
                        self.connect_bbox_to_track_wires(Direction.LOWER, 
                            (f'met{layer_id-1}', 'drawing'), b, warr)
        return all_wars
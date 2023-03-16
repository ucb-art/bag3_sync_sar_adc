from typing import Any, Optional, Type, Dict

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.routing import TrackID
from bag.layout.template import TemplateDB
from bag.util.immutable import Param
from .digital import NAND2Core, InvChainCore, NOR2Core, InvCore
from .digital import FlopCore
from bag3_sync_sar_adc.layout.util.util import fill_tap
from bag3_sync_sar_adc.layout.vco_cnter_dec import CnterAsync
from pybag.enum import RoundMode, MinLenMode
from xbase.layout.enum import SubPortMode, MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.placement.data import TilePatternElement, TilePattern


class SyncClkGen(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_sync_clk')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            seg_dict='Number of segments.',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row',
            ridx_p='index for pmos row',
            cnter_params='Ring parameters',
            pinfo='Pinfo for unit row strongArm flop',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        cnter_params: Param = self.params['cnter_params']
        seg_dict: Dict[str, Any] = self.params['seg_dict']
        w_n, w_p = self.params['w_n'], self.params['w_p']
        ridx_n, ridx_p = self.params['ridx_n'], self.params['ridx_p']

        cnter_master: MOSBase = self.new_template(CnterAsync, params=cnter_params.copy(
            append=(dict(pinfo=pinfo, export_output=True))))
        cnter_nrows = cnter_master.num_tile_rows
        tile_ele = []
        for idx in range(cnter_nrows + 4):
            tile_ele.append(cnter_master.get_tile_subpattern(0, 1, flip=bool(idx & 1)))
        tile_ele = TilePatternElement(TilePattern(tile_ele))
        self.draw_base((tile_ele, cnter_master.draw_base_info[1]))

        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-1, tile_idx=0)
        pg1_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=-2, tile_idx=0)
        seg_buf_in, seg_buf_out, seg_buf_comp_clk = seg_dict['buf_in'], seg_dict['buf_out'], seg_dict['buf_comp_clk']
        buf_in_params = dict(pinfo=pinfo, seg_list=seg_buf_in, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                             vertical_sup=False, dual_output=True, sig_locs={})
        buf_out_params = dict(pinfo=pinfo, seg_list=seg_buf_out, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                              vertical_sup=False, dual_output=True, sig_locs={})
        buf_comp_clk_params = dict(pinfo=pinfo, seg_list=seg_buf_comp_clk, w_p=w_p, w_n=w_n,
                                   ridx_n=ridx_n, ridx_p=ridx_p, vertical_sup=False, dual_output=True,
                                   sig_locs={'nin0': pg0_tidx, 'nin1': pg1_tidx})

        buf_in_master: MOSBase = self.new_template(InvChainCore, params=buf_in_params)
        buf_out_master: MOSBase = self.new_template(InvChainCore, params=buf_out_params)
        buf_comp_clk_master: MOSBase = self.new_template(InvChainCore, params=buf_comp_clk_params)

        cnter_ncol = cnter_master.num_cols
        nrows = cnter_master.num_tile_rows

        # Floorplan:
        # in buffer
        # comp clk buffer
        #
        # divider
        #
        # out buffer

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xm1_layer = ym_layer + 1
        ym1_layer = xm1_layer + 1
        min_sep = self.min_sep_col

        tile_height = self.get_tile_pinfo(0).height

        # Get track info
        cur_loc = 0
        cnter_col = max(cnter_master.mid_col, buf_in_master.num_cols) - cnter_master.mid_col
        cnter_col += cnter_col & 1

        cnter = self.add_tile(cnter_master, 2, cnter_col+8)
        tap_sep = self.min_sep_col
        tap_sep += tap_sep & 1
        min_tap_ncols = self.tech_cls.min_sub_col + 2 * tap_sep + 4

        cur_loc += int(not (cur_loc & 1))
        cnter_mid_col = max(cnter_col + cnter_master.mid_col + min_sep // 2 + min_sep,
                            cur_loc + buf_in_master.num_cols + min_tap_ncols)
        cnter_mid_col += int(not (cnter_mid_col & 1))

        # Add buffers
        buf_out = self.add_tile(buf_out_master, 1, cnter_mid_col+4)
        buf_comp_clk = self.add_tile(buf_comp_clk_master, cnter_master.num_tile_rows + 2, cnter_mid_col+8)
        buf_in = self.add_tile(buf_in_master, cnter_master.num_tile_rows + 2, cur_loc+8)
        tot_cols = max(cnter_col+cnter_master.num_cols+min_sep+14,
                       cnter_mid_col + buf_comp_clk_master.num_cols + min_sep+14)
        self.set_mos_size(tot_cols)

        # Clock in
        tr_manager = self.tr_manager
        tr_w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        clk_in_vm_tidx = self.grid.coord_to_track(vm_layer, buf_in.bound_box.xl, RoundMode.NEAREST)
        clk_in_vm = self.connect_to_tracks(buf_in.get_pin('nin'), TrackID(vm_layer, clk_in_vm_tidx-1, tr_w_sig_vm),
                                           min_len_mode=MinLenMode.MIDDLE)

        self.connect_to_track_wires([buf_in.get_pin('nout'), buf_in.get_pin('pout')], cnter.get_pin('clkp'))
        self.connect_to_track_wires([buf_in.get_pin('noutb'), buf_in.get_pin('poutb')], cnter.get_pin('clkn'))
        
        for idx in range(1, self.num_tile_rows):
            r0_hm, r1_hm = fill_tap(self, idx, extra_margin=(idx == self.num_tile_rows - 1), port_mode=SubPortMode.ODD)
            self.extend_wires(r0_hm + r1_hm, lower=self.bound_box.xl, upper=self.bound_box.xh)

        self.connect_to_track_wires(buf_in.get_pin('outb'), buf_comp_clk.get_pin('nin'))
        self.connect_to_track_wires(cnter.get_pin('final_outp'), buf_out.get_pin('nin'))

        vdd_xm_list, vss_xm_list = [], []
        # _b, _t = export_xm_sup(self, 0, export_bot=True, export_top=True)
        # vdd_xm_list.append(_t)
        # vss_xm_list.append(_b)
        # _, _t = export_xm_sup(self, self.num_tile_rows - 1, export_top=True)
        # vdd_xm_list.append(_t)
        vdd_xm_list.extend(cnter.get_all_port_pins('VDD_xm'))
        vss_xm_list.extend(cnter.get_all_port_pins('VSS_xm'))
        # vdd_xm_list = self.extend_wires(vdd_xm_list, lower=self.bound_box.xl, upper=self.bound_box.xh)
        # vss_xm_list = self.extend_wires(vss_xm_list, lower=self.bound_box.xl, upper=self.bound_box.xh)

        self.add_pin('VDD', vdd_xm_list, show=self.show_pins, connect=True)
        self.add_pin('VSS', vss_xm_list, show=self.show_pins, connect=True)
        self.add_pin('VDD', buf_in.get_all_port_pins('VDD'), show=self.show_pins, connect=True)
        self.add_pin('VSS', buf_in.get_all_port_pins('VSS'), show=self.show_pins, connect=True)
        self.add_pin('VDD', buf_out.get_all_port_pins('VDD'),show=self.show_pins,  connect=True)
        self.add_pin('VSS', buf_out.get_all_port_pins('VSS'),show=self.show_pins,  connect=True)


        self.add_pin('clk_out', buf_out.get_pin('out'))
        self.add_pin('clk_out_b', buf_out.get_pin('outb'))
        self.add_pin('clk_comp', buf_comp_clk.get_pin('outb'))
        self.add_pin('clk_compn', buf_comp_clk.get_pin('out'))
        self.add_pin('clk_in', clk_in_vm)

        self._sch_params = dict(
            buf_in=buf_in_master.sch_params,
            buf_out=buf_out_master.sch_params,
            buf_comp=buf_comp_clk_master.sch_params,
            div=cnter_master.sch_params,
        )


class SyncDivCounter(MOSBase):
    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'sar_sync_counter')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            seg_dict='Number of segments.',
            w_n='nmos width',
            w_p='pmos width',
            ridx_n='index for nmos row',
            ridx_p='index for pmos row',
            pinfo='Pinfo for unit row strongArm flop',
            substrate_row='True to add substrate row',
            total_cycles='Total number of clock cycles, 2 reserved for sampling'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=4,
            w_p=4,
            ridx_n=0,
            ridx_p=-1,
            substrate_row = False,
        )

    def draw_layout(self) -> None:
        
        # currently just works for 11
        #floorplan
        # buf out
        # x16 

        #         rst on side


        # x2,
        # buf in

        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, Any] = self.params['seg_dict']
        w_n, w_p = self.params['w_n'], self.params['w_p']
        ridx_n, ridx_p = self.params['ridx_n'], self.params['ridx_p']
        substrate_row = self.params['substrate_row']
        bin_code = [int(b) for b in bin(int(self.params['total_cycles']-2))[2:]]
        max_bin = len(bin_code)
        padding = [0 for i in range(4-len(bin_code))]
        bin_code = padding+bin_code

        conn_layer = self.conn_layer
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

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

        seg_buf_in, seg_buf_out, seg_buf_comp_clk = seg_dict['buf_in'], seg_dict['buf_out'], seg_dict['buf_comp_clk']
        buf_in_params = dict(pinfo=pinfo, seg_list=seg_buf_in, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                             vertical_sup=False, dual_output=True, sig_locs={})
        buf_out_params = dict(pinfo=pinfo, seg_list=seg_buf_out, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                              vertical_sup=False, dual_output=True, sig_locs={'nin': ng0_tidx, 'pin': ng2_tidx})
        buf_comp_clk_params = dict(pinfo=pinfo, seg_list=seg_buf_comp_clk, w_p=w_p, w_n=w_n,
                                   ridx_n=ridx_n, ridx_p=ridx_p, vertical_sup=False, dual_output=True,
                                   sig_locs={'nin0': pg0_tidx, 'nin1': pg1_tidx})

        buf_in_master: MOSBase = self.new_template(InvChainCore, params=buf_in_params)
        buf_out_master: MOSBase = self.new_template(InvChainCore, params=buf_out_params)
        buf_comp_clk_master: MOSBase = self.new_template(InvChainCore, params=buf_comp_clk_params)

        seg_flop, seg_inv, seg_nor, seg_nand = seg_dict['flop'], seg_dict['inv'], seg_dict['nor'], seg_dict['nand']
        flop_div_params = dict(pinfo=pinfo, seg=seg_flop, seg_ck=4, resetable=True, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                            vertical_sup=substrate_row, sig_locs={})
        flop_div_master = self.new_template(FlopCore, params=flop_div_params)
        flop_params = dict(pinfo=pinfo, seg=seg_flop, seg_ck=4, resetable=False, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                            vertical_sup=substrate_row, sig_locs={})
        flop_master = self.new_template(FlopCore, params=flop_params)
        inv_div_params = dict(pinfo=pinfo, seg=seg_inv, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                          vertical_sup=substrate_row, sig_locs={'nin': ng0_tidx})
        inv_div_master = self.new_template(InvCore, params=inv_div_params)

        nor_params = dict(pinfo=pinfo, seg=seg_nor, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                                 vertical_sup=substrate_row, vertical_out=False,
                                 sig_locs={'nin1': pg1_tidx, 'nin0': ng1_tidx})
        nor_master = self.new_template(NOR2Core, params=nor_params)
        nand_params = dict(pinfo=pinfo, seg=seg_nand, w_p=w_p, w_n=w_n, ridx_n=ridx_n, ridx_p=ridx_p,
                                 vertical_sup=substrate_row, vertical_out=False,
                                 sig_locs={'nin1': pg1_tidx, 'nin0': ng1_tidx})
        nand_master =self.new_template(NAND2Core, params=nand_params)
        min_sep = self.min_sep_col
        
        tap_sep = self.min_sep_col
        tap_sep += tap_sep & 1
        min_tap_ncols = self.tech_cls.min_sub_col  #+ 2 * tap_sep

        # Row 0 - buffer in and comp clk
        in_col = 10*min_tap_ncols
        buf_in = self.add_tile(buf_in_master, 0, in_col)
        in_col += buf_in_master.num_cols + min_sep
        in_col += in_col & 1
        buf_comp_clk = self.add_tile(buf_comp_clk_master, 0, in_col)
        in_col += buf_comp_clk_master.num_cols + min_sep
        in_col += in_col & 1
        
        # Row 1 - 4 flip flops for clock division
        flop_list = []
        inv_div_list = []
        max_clk_col = 0
        rnum=1
        for r in range(4):
            cur_col = 10*min_tap_ncols
            flop_clk = self.add_tile(flop_div_master, rnum, cur_col)
            cur_col += flop_div_master.num_cols + min_sep
            cur_col += cur_col & 1
            inv_div = self.add_tile(inv_div_master, rnum, cur_col)
            cur_col += inv_div_master.num_cols + min_sep
            cur_col += cur_col & 1
            max_clk_col = max(max_clk_col, cur_col)

            flop_list.append(flop_clk)
            inv_div_list.append(inv_div)
            rnum = rnum+ 1

        # Place reset logic
        rst_col = max_clk_col + min_tap_ncols
        nor1 = self.add_tile(nor_master, 1, rst_col)
        nor2 = self.add_tile(nor_master, 3, rst_col)
        rst_col += nor_master.num_cols + min_sep
        rst_col += rst_col & 1
        nand = self.add_tile(nand_master, 2, rst_col)
        rst_col += nand_master.num_cols + min_sep
        rst_col += rst_col & 1
        inv_rst = self.add_tile(inv_div_master, 2, rst_col)
        rst_col += inv_div_master.num_cols + min_sep
        rst_col += rst_col & 1
        flop_rst = self.add_tile(flop_master, 2, rst_col)
        rst_col += flop_master.num_cols + min_sep
        rst_col += rst_col & 1

        # Row 5 output buffer
        out_col = 10*min_tap_ncols
        buf_out = self.add_tile(buf_out_master, 5, out_col)
        out_col += buf_out_master.num_cols + min_sep
        out_col += out_col & 1

        # set size based on columns
        tot_seg = max(rst_col, out_col, in_col)
        self.set_mos_size(tot_seg)
        
        # add taps  
        for idx in range(1, self.num_tile_rows):
            r0_hm, r1_hm = fill_tap(self, idx, extra_margin=(idx == self.num_tile_rows - 1), port_mode=SubPortMode.ODD)
            self.extend_wires(r0_hm + r1_hm, lower=self.bound_box.xl, upper=self.bound_box.xh)

        # connect input clk
        # -----------------------------------------------------------------------
        tidx_buf_in_conn = self.grid.coord_to_track(conn_layer, buf_in.get_pin('in').middle)
        tidx_buf_comp_in_conn = self.grid.coord_to_track(conn_layer, buf_comp_clk.get_pin('in').middle)
        #clk_bufin = self.connect_to_tracks(buf_in.get_pin('out'), TrackID(conn_layer, tidx_buf_in_conn, 1))
        clk_bufcomp_in = self.connect_to_tracks(buf_comp_clk.get_pin('in'), TrackID(conn_layer, tidx_buf_comp_in_conn, 1))

        tidx_clkinlo_hm = self.grid.coord_to_track(hm_layer, buf_in.bound_box.yl, mode=RoundMode.GREATER_EQ)
        tidx_clkinhi_hm = self.grid.coord_to_track(hm_layer, buf_in.bound_box.yh, mode=RoundMode.LESS_EQ)
        tidx_clkin_hm = self.get_available_tracks(hm_layer, tidx_clkinlo_hm, tidx_clkinhi_hm,
                                              buf_in.bound_box.xl, flop_rst.bound_box.xh, width=tr_w_vm, sep=tr_sp_vm)
        clk_buf = self.connect_to_tracks([buf_in.get_pin('out'), clk_bufcomp_in], 
                                       TrackID(hm_layer, tidx_clkin_hm[len(tidx_clkin_hm)//2], tr_w_hm))
        clk_in = self.connect_to_tracks([buf_in.get_pin('outb'),  flop_rst.get_pin('clk')], 
                                       TrackID(hm_layer, tidx_clkin_hm[len(tidx_clkin_hm)//2 - 2], tr_w_hm))
        # clk b to the latch
        self.connect_to_track_wires( buf_in.get_pin('poutb'), flop_list[0].get_pin('clk'))
        # self.add_pin('buf_in_outb', buf_in.get_pin('outb'))
        # self.add_pin('buf_in_out', buf_in.get_pin('out'))
        # self.add_pin('flop_rst_mid', flop_rst.get_pin('mid'))
        # self.add_pin('flop_rst_clk', flop_rst.get_pin('clk'))

        # connect each feedback latch
        divb_list = [] 
        div_list = []
        for idx, (flop, inv_div) in enumerate(zip(flop_list, inv_div_list)):
            
            tidx_divblo_hm = self.grid.coord_to_track(hm_layer, flop.get_pin('in').lower, mode=RoundMode.GREATER_EQ)
            tidx_divbhi_hm = self.grid.coord_to_track(hm_layer, flop.get_pin('in').upper, mode=RoundMode.LESS_EQ)
            tidx_divb_hm = self.get_available_tracks(hm_layer, tidx_divblo_hm, tidx_divbhi_hm,
                                              flop.bound_box.xl, flop.bound_box.xh, width=tr_w_vm, sep=tr_sp_vm)
            tidx_flop_out_conn = self.grid.coord_to_track(conn_layer, flop.get_pin('nin').middle//43*43)
            #flop_div_conn = self.connect_to_tracks(flop.get_pin('nin'), TrackID(conn_layer, tidx_flop_out_conn, 1))
            divb = self.connect_to_tracks([inv_div.get_pin('out'), flop.get_pin('in')], TrackID(hm_layer, tidx_divb_hm[-1], 1))
            divb_list.append(divb)

            div = self.connect_to_track_wires(flop.get_pin('out'), inv_div.get_pin('in'))
            # self.add_pin(f'divb<{idx}>', divb)
            # self.add_pin(f'div<{idx}>', div)
            # self.add_pin(f'mid<{idx}>', flop.get_pin('mid'))
            # self.add_pin(f'flop_clk<{idx}>', flop.get_pin('clk'))
            div_list.append(div)
        
        # connect the latches together
        tidx_lo_vm = self.grid.coord_to_track(vm_layer, flop_list[0].bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hi_vm = self.grid.coord_to_track(vm_layer, flop_list[0].bound_box.xh, mode=RoundMode.LESS_EQ)
        vert_divb_list = []
        for idx, (divb, flop) in enumerate(zip(divb_list[:-1], flop_list[1:])):
            tidx_divb_vm = self.get_available_tracks(vm_layer, tidx_lo_vm, tidx_hi_vm,
                                              flop_list[0].bound_box.yl, flop_list[-1].bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
            divb_vm = self.connect_to_tracks(divb, TrackID(vm_layer, tidx_divb_vm[0], tr_w_vm))
            # vert_div = self.connect_to_tracks([divb, flop_clk_hm], TrackID(vm_layer, tidx_list_vm[0], tr_w_vm))
            tidx_flopclklo_hm = self.grid.coord_to_track(hm_layer, flop.get_pin('clk').lower, mode=RoundMode.GREATER_EQ)
            tidx_flopclkhi_hm = self.grid.coord_to_track(hm_layer, flop.get_pin('clk').upper, mode=RoundMode.LESS_EQ)
            tidx_flopclk_hm = self.get_available_tracks(hm_layer, tidx_flopclklo_hm, tidx_flopclkhi_hm,
                                              flop.bound_box.xl, inv_div_list[idx+1].bound_box.xh, width=tr_w_vm, sep=tr_sp_vm)
            flop_clk_hm = self.connect_to_tracks(flop.get_pin('clk'), TrackID(hm_layer, tidx_flopclk_hm[0], tr_w_vm))
            vert_divb = self.connect_to_track_wires(divb_vm, flop_clk_hm)
            vert_divb_list.append(divb_vm)

        # connect to buffer output
        tidx_lo_vm = self.grid.coord_to_track(vm_layer, flop_list[-1].bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hi_vm = self.grid.coord_to_track(vm_layer, flop_list[-1].bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_divb_vm = self.get_available_tracks(vm_layer, tidx_lo_vm, tidx_hi_vm,
                                              flop_list[-1].bound_box.yl, buf_out.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        # connect MSB FIXME
        # buf_out_in_vm = self.connect_to_tracks(divb_list[-1], TrackID(vm_layer, tidx_divb_vm[0], tr_w_vm))
        # self.connect_to_track_wires(buf_out_in_vm, buf_out.get_pin('pin'))

        # connect the latches to rst logic 
        # -------------------------------------------------------
        # nor connection
        tidx_lorst_vm = self.grid.coord_to_track(vm_layer, inv_div_list[0].bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hirst_vm = self.grid.coord_to_track(vm_layer, nor1.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_rstnor1_vm = self.get_available_tracks(vm_layer, tidx_lorst_vm, tidx_hirst_vm,
                                              flop_list[0].bound_box.yl, flop_list[1].bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        
        b2_conn = self.connect_to_tracks(nor1.get_pin('nin<1>'), TrackID(vm_layer, tidx_rstnor1_vm[0], tr_w_vm))
        if bin_code[3]:
            self.connect_to_track_wires(b2_conn, divb_list[0])
        else:
            self.connect_to_track_wires(b2_conn, div_list[0])
        # self.add_pin('x2b', xb2)
        # self.add_pin('div_x2b', divb_list[0])
        # self.add_pin('div_x2', div_list[0])
        b4_conn = self.connect_to_tracks(nor1.get_pin('nin<0>'), TrackID(vm_layer, tidx_rstnor1_vm[1], tr_w_vm))
        if bin_code[2]:
            self.connect_to_track_wires(b4_conn, divb_list[1])
        else:
            self.connect_to_track_wires(b4_conn, div_list[1])
        # self.add_pin('x4', x4)
        # self.add_pin('div_x4', div_list[1])
        # self.add_pin('div_x4b', divb_list[1])
        tidx_rstnor2_vm = self.get_available_tracks(vm_layer, tidx_lorst_vm, tidx_hirst_vm,
                                              flop_list[2].bound_box.yl, flop_list[3].bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        b8_conn = self.connect_to_tracks(nor2.get_pin('nin<1>'), TrackID(vm_layer, tidx_rstnor2_vm[0], tr_w_vm))
        if bin_code[1]:
            self.connect_to_track_wires(b8_conn, divb_list[2])
        else:
            self.connect_to_track_wires(b8_conn, div_list[2])

        b16_conn = self.connect_to_tracks(nor2.get_pin('nin<0>'), TrackID(vm_layer, tidx_rstnor2_vm[1], tr_w_vm))
        if bin_code[0]:
            self.connect_to_track_wires(b16_conn, divb_list[3])
        else:
            self.connect_to_track_wires(b16_conn, div_list[3]) 

        # nand connections
        tidx_loin_vm = self.grid.coord_to_track(vm_layer, nor1.bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hiin_vm = self.grid.coord_to_track(vm_layer, nand.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_nandin_vm = self.get_available_tracks(vm_layer, tidx_loin_vm, tidx_hiin_vm,
                                              nor1.bound_box.yl, nor2.bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        
        nand_in1_vm = self.connect_to_tracks([nor1.get_pin('nout'),nor1.get_pin('pout')], 
                                            TrackID(vm_layer, tidx_nandin_vm[0], tr_w_vm))
        self.connect_to_track_wires(nand_in1_vm, nand.get_pin('nin<1>'))
        nand_in2_vm = self.connect_to_tracks([nor2.get_pin('nout'),nor2.get_pin('pout')], 
                                            TrackID(vm_layer, tidx_nandin_vm[2], tr_w_vm))
        self.connect_to_track_wires(nand_in2_vm, nand.get_pin('nin<0>'))

        nand_out_vm = self.connect_to_tracks([nand.get_pin('nout'),nand.get_pin('pout')], 
                                            TrackID(vm_layer, tidx_nandin_vm[-1], tr_w_vm))
        self.connect_to_track_wires(nand_out_vm, inv_rst.get_pin('nin'))
          
        self.connect_to_track_wires(inv_rst.get_pin('out'), flop_rst.get_pin('nin'))  

        # connect rst back to everything else
        hm_hblk, hm_wblk = self.grid.get_block_size(hm_layer, half_blk_x=True, half_blk_y=True)
        tidx_lorst_out_vm = self.grid.coord_to_track(vm_layer, inv_div_list[0].bound_box.xl, mode=RoundMode.GREATER_EQ)
        tidx_hirst_out_vm = self.grid.coord_to_track(vm_layer, nor1.bound_box.xh, mode=RoundMode.LESS_EQ)
        tidx_rstout_vm = self.get_available_tracks(vm_layer, tidx_lorst_vm, tidx_hirst_vm,
                                              flop_list[0].bound_box.yl, flop_list[-1].bound_box.yh, width=tr_w_vm, sep=tr_sp_vm)
        rst_hm_tidx = self.grid.coord_to_track(hm_layer, flop_rst.get_pin('out').middle//hm_wblk * hm_wblk)
        rstout_hm = self.connect_to_tracks(flop_rst.get_pin('out'), TrackID(hm_layer, rst_hm_tidx, tr_w_vm))   
        rst_vm = self.connect_to_tracks(rstout_hm, TrackID(vm_layer, tidx_rstout_vm[0], tr_w_vm))
        for flop in flop_list:
            # self.add_pin('rst', flop.get_pin('rst'))
            self.connect_to_track_wires(rst_vm, flop.get_pin('rst'))
        hm_tracks = self.get_available_tracks(hm_layer, self.grid.coord_to_track(hm_layer, buf_out.bound_box.yl),
                                  self.grid.coord_to_track(hm_layer, buf_out.bound_box.yh), 
                                  lower=buf_out.bound_box.xl, upper=buf_out.bound_box.xh)
        #self.add_pin('PIN', buf_out.get_pin('nin'))
        rst_hm = self.connect_to_tracks(rst_vm, TrackID(hm_layer, hm_tracks[len(hm_tracks)//2], tr_w_hm))
        rst_conn = self.connect_to_tracks(buf_out.get_pin('in'), TrackID(conn_layer, 
                                                        self.grid.coord_to_track(conn_layer, buf_out.get_pin('in').bound_box.xm), 1))
        self.connect_to_track_wires(rst_hm, rst_conn) # FIXME

        # connect supplies
        # -------------------------------------------------------------------------
        vss_list, vdd_list = [], []
        inst_list = [flop_rst, buf_out, buf_in, buf_comp_clk, nor1, nor2, nand] + \
                    flop_list + inv_div_list

        for inst in inst_list:
            vdd_list.append(inst.get_pin('VDD'))
            vss_list.append(inst.get_pin('VSS'))
        vdd_hm = self.connect_wires(vdd_list)
        vss_hm = self.connect_wires(vss_list)

        # add pins
        self.add_pin('clk_in', buf_in.get_pin('in'))
        self.add_pin('comp_clk', buf_comp_clk.get_pin('out'))
        self.add_pin('comp_clkb', buf_comp_clk.get_pin('outb'))
        self.add_pin('clk_out_b', buf_out.get_pin('out'))
        self.add_pin('clk_out', buf_out.get_pin('outb'))
        self.add_pin("VDD", vdd_hm)
        self.add_pin("VSS", vss_hm)

        # schematic parameters
        # -------------------------------------------------------
        sch_params_dict = dict(
            inv_div=inv_div_master.sch_params,
            flop_div=flop_div_master.sch_params,
            nand=nand_master.sch_params,
            rflop=flop_master.sch_params,
            nor=nor_master.sch_params,
            buf_out=buf_out_master.sch_params,
            buf_in=buf_in_master.sch_params,
            buf_comp_clk=buf_comp_clk_master.sch_params,
            total_cycles=self.params['total_cycles']
        )
        self.sch_params = sch_params_dict

from typing import Any, Dict, Type, Optional

from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.routing import TrackID
from bag.layout.template import TemplateDB
from bag.util.immutable import Param
from pybag.enum import RoundMode, MinLenMode
from xbase.layout.mos.base import MOSBase
from .vco_flops import CnterLatch

class CnterDiv(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._mid_col = 0

    @property
    def mid_col(self):
        return self._mid_col

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'vco_cnter_div')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = CnterLatch.get_params_info()
        ans['nbits'] = 'Number of bits'
        ans['shift_clk'] = 'True to shift clock for easier layout routing'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = CnterLatch.get_default_param_values()
        ans['nbits'] = 1
        ans['shift_clk'] = False
        return ans

    def draw_layout(self):
        seg_dict = self.params['seg_dict']
        w_dict = self.params['w_dict']
        pinfo = self.params['pinfo']
        nbits: int = self.params['nbits']
        seg_dict_list, w_dict_list = [], []
        for idx in range(4):
            _seg_dict = dict(
                nin=seg_dict['nin'][idx],
                pin=seg_dict['pin'][idx],
                ntail=seg_dict['ntail'][idx],
                ptail=seg_dict['ptail'][idx],
                nfb=seg_dict['nfb'][idx],
                pfb=seg_dict['pfb'][idx],
            )
            seg_dict_list.append(_seg_dict)
            _w_dict = dict(
                nin=w_dict['nin'][idx],
                pin=w_dict['pin'][idx],
                ntail=w_dict['ntail'][idx],
                ptail=w_dict['ptail'][idx],
                nfb=w_dict['nfb'][idx],
                pfb=w_dict['pfb'][idx],
            )
            w_dict_list.append(_w_dict)

        cnter_master_params = dict(nbits=nbits, seg_dict=seg_dict_list[0], w_dict=w_dict_list[0], pinfo=pinfo)
        master: CnterLatch = self.new_template(CnterLatch, params=cnter_master_params)
        self.draw_base(master.draw_base_info)
        shift_clk: int = self.params['shift_clk']

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        out_vm_tidx = self.arr_info.col_to_track(vm_layer, 3)
        in_vm_tidx = self.arr_info.col_to_track(vm_layer, 1)
        out_final_vm_tidx = self.arr_info.col_to_track(vm_layer, 4)
        sig_locs_orig = {'out': in_vm_tidx, 'in': out_vm_tidx}
        sig_locs_new = {'out': out_vm_tidx, 'in': in_vm_tidx}
        cnter_master_params = dict(nbits=nbits, seg_dict=seg_dict_list[0], w_dict=w_dict_list[0], pinfo=pinfo,
                                   sig_locs=sig_locs_orig)
        master: CnterLatch = self.new_template(CnterLatch, params=cnter_master_params)

        tr_w_clk_vm = tr_manager.get_width(vm_layer, 'clk')
        # Make different masters
        # smallest unit -> double width -> reduce width, double nf -> double nf, double width

        # seg_dict_double, w_dict_half = {}, {}
        # for k, v in seg_dict.items():
        #     seg_dict_double[k] = 2 * v if 'fb' not in k else v
        # for k, v in w_dict.items():
        #     w_dict_half[k] = v // 2 if 'fb' not in k else v

        # placement
        min_sep = self.min_sep_col
        master_w_double_shift_params = dict(nbits=nbits, seg_dict=seg_dict_list[1], w_dict=w_dict_list[1],
                                            pinfo=pinfo, flip_io=True, sig_locs=sig_locs_new)
        master_w_double_shift = self.new_template(CnterLatch, params=master_w_double_shift_params)
        master_seg_double_params = dict(nbits=nbits, seg_dict=seg_dict_list[2], w_dict=w_dict_list[2], pinfo=pinfo,
                                        sig_locs=sig_locs_orig)
        master_seg_double = self.new_template(CnterLatch, params=master_seg_double_params)
        master_both_double_shift_params = dict(nbits=nbits, seg_dict=seg_dict_list[3], w_dict=w_dict_list[3],
                                               pinfo=pinfo, flip_io=True, sig_locs=sig_locs_new)
        master_both_double_shift = self.new_template(CnterLatch, params=master_both_double_shift_params)

        master_list = [master, master_w_double_shift, master_seg_double, master_both_double_shift]
        l_lat_list, r_lat_list = [], []
        nrow = 2 ** (nbits - 1)
        ncol_lat = master.num_cols
        ncol_lat2 = master_list[-1].num_cols
        if nrow <= 2:
            for ridx in range(nrow):
                l_lat_list.append(self.add_tile(master_list[ridx], ridx, ncol_lat if ridx & 1 else 0,
                                                flip_lr=bool(ridx & 1)))

                r_lat_list.append(self.add_tile(master_list[ridx + nrow], nrow - ridx - 1,
                                                ncol_lat2 + ncol_lat + 3 * min_sep if ridx & 1 else ncol_lat + 3 * min_sep,
                                                flip_lr=bool(ridx & 1)))
        # else:
        #     out_final_vm_tidx = self.arr_info.col_to_track(vm_layer, 3)
        #     sig_locs_new = {'out': out_final_vm_tidx}
        #     master_params = self.params.copy(append=dict(flip_io=True, sig_locs=sig_locs_new))
        #     master_final: CnterLatch = self.new_template(CnterLatch, params=master_params)
        #     for ridx in reversed(range(nrow - 1)):
        #         s_lat_list.append(self.add_tile(master_s, ridx, 0 if ridx & 1 else ncol_lat + min_sep))
        #         m_lat_list.append(self.add_tile(master_m, ridx, ncol_lat + min_sep if ridx & 1 else 0))
        #     m_lat_list.append(self.add_tile(master_m, nrow - 1, 0 if nrow & 1 else ncol_lat + min_sep))
        #     s_lat_list.append(self.add_tile(master_final, nrow - 1, ncol_lat + min_sep if nrow & 1 else 0))
        #
        self.set_mos_size()
        #
        # Connect clock signals
        _, clk_tidxs = self.tr_manager.place_wires(vm_layer, ['clk'] * 4,
                                                   center_coord=self.arr_info.col_to_coord(ncol_lat + min_sep))
        nclk_list, pclk_list = [], []
        vdd_list, vss_list = [], []
        for idx in range(nrow):
            pinfo, yb, _ = self.get_tile_info(idx)

            _, io_xm_locs = tr_manager.place_wires(xm_layer, ['sig'] * 4, center_coord=yb + pinfo.height // 2)
            if nrow > 1:
                io_xm_locs_0, io_xm_locs_1 = io_xm_locs[1:-1], io_xm_locs[1:-1]
            else:
                io_xm_locs_0, io_xm_locs_1 = io_xm_locs[1:-1], [io_xm_locs[0], io_xm_locs[-1]]
            if idx == nrow - 1:
                self.connect_matching_tracks([[l_lat_list[idx].get_pin('outp', layer=vm_layer),
                                               r_lat_list[-idx - 1].get_pin('d')],
                                              [l_lat_list[idx].get_pin('outn', layer=vm_layer),
                                               r_lat_list[-idx - 1].get_pin('dn')]],
                                             xm_layer, io_xm_locs_0, width=tr_manager.get_width(xm_layer, 'sig'))
            if idx == 0:
                self.connect_matching_tracks([[l_lat_list[idx].get_pin('d'),
                                               r_lat_list[-idx - 1].get_pin('outn', layer=vm_layer)],
                                              [l_lat_list[idx].get_pin('dn'),
                                               r_lat_list[-idx - 1].get_pin('outp', layer=vm_layer)]],
                                             xm_layer, io_xm_locs_1, width=tr_manager.get_width(xm_layer, 'sig'))
            if nrow > 1:
                pinst, ninst = (r_lat_list[idx], l_lat_list[-1 - idx]) if idx & 1 else (
                    l_lat_list[-1 - idx], r_lat_list[idx])
            else:
                pinst, ninst = (r_lat_list[idx], l_lat_list[-1 - idx])
            nclk_list.extend([ninst.get_pin('clkn'), pinst.get_pin('clkp')])
            pclk_list.extend([ninst.get_pin('clkp'), pinst.get_pin('clkn')])
            vdd_list.extend(self.connect_wires([ninst.get_pin('VDD'), pinst.get_pin('VDD')]))
            vss_list.extend(self.connect_wires([ninst.get_pin('VSS'), pinst.get_pin('VSS')]))

        for idx in range(nrow - 1):
            self.connect_wires([r_lat_list[idx].get_pin('outn', layer=vm_layer), r_lat_list[idx + 1].get_pin('dn')])
            self.connect_wires([r_lat_list[idx].get_pin('outp', layer=vm_layer), r_lat_list[idx + 1].get_pin('d')])
            self.connect_wires([l_lat_list[idx].get_pin('outn', layer=vm_layer), l_lat_list[idx + 1].get_pin('dn')])
            self.connect_wires([l_lat_list[idx].get_pin('outp', layer=vm_layer), l_lat_list[idx + 1].get_pin('d')])
        
        clk_tid_pair = clk_tidxs[1:3] if shift_clk else [clk_tidxs[0], clk_tidxs[-1]]
        nclk, pclk = self.connect_matching_tracks([nclk_list, pclk_list], vm_layer, clk_tid_pair, width=tr_w_clk_vm)

        self.add_pin('VDD', vdd_list, connect=True)
        self.add_pin('VSS', vss_list, connect=True)
        self.add_pin('clkn', nclk)
        self.add_pin('clkp', pclk)

        for idx in range(nrow):
            self.reexport(l_lat_list[idx].get_port('outp'), net_name=f'outp<{idx}>')
            self.reexport(l_lat_list[idx].get_port('outn'), net_name=f'outn<{idx}>')
            self.reexport(r_lat_list[-idx - 1].get_port('outp'), net_name=f'outp<{2 ** nbits - 1 - idx}>')
            self.reexport(r_lat_list[-idx - 1].get_port('outn'), net_name=f'outn<{2 ** nbits - 1 - idx}>')
        #
        self._mid_col = master.num_cols + min_sep // 2
        self.sch_params = dict(
            latch_params_list=[m.sch_params for m in master_list],
            num_stages=2 ** nbits
        )


class CnterAsync(CnterDiv):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._mid_col = 0

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'vco_cnter_async')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = CnterDiv.get_params_info()
        ans['ndivs'] = 'Number of dividers'
        ans['export_output'] = 'True to export final output'
        ans['top_sup_layer'] = 'Top supply layer'
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = CnterDiv.get_default_param_values()
        ans['ndivs'] = 1
        ans['export_output'] = False
        ans['top_sup_layer'] = 6
        return ans

    def draw_layout(self):
        master: CnterDiv = self.new_template(CnterDiv, params=self.params)
        self._mid_col = master.mid_col
        self.draw_base(master.draw_base_info)
        ndivs: int = self.params['ndivs']
        nbits: int = self.params['nbits']

        tr_manager = self.tr_manager
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        xm1_layer = ym_layer + 1
        ym1_layer = xm1_layer + 1

        # placement
        min_sep = self.min_sep_col
        div_nrow = master.num_tile_rows
        params_shift_clk = self.params.copy(append=dict(shift_clk=True))
        master_shift_clk: CnterDiv = self.new_template(CnterDiv, params=params_shift_clk)

        master_list = [master if ridx & 1 else master_shift_clk for ridx in range(ndivs)]
        div_list = []
        for ridx in range(ndivs):
            div_list.append(self.add_tile(master_list[ridx], (ndivs - ridx - 1) * div_nrow, 0))

        self.set_mos_size(max([inst_temp.num_cols for inst_temp in master_list]))

        vdd_list, vss_list, nclk_list, pclk_list = [], [], [], []
        for idx, inst in enumerate(div_list):
            vdd_list.extend(inst.get_all_port_pins('VDD'))
            vss_list.extend(inst.get_all_port_pins('VSS'))
            for b in range(2 ** nbits):
                self.reexport(inst.get_port(f'outn<{b}>'), net_name=f'outn<{b + idx * 2 ** nbits}>')
                self.reexport(inst.get_port(f'outp<{b}>'), net_name=f'outp<{b + idx * 2 ** nbits}>')
            if idx < ndivs - 1:
                outn = div_list[idx].get_pin(f'outn<{2 ** nbits - 1}>', layer=hm_layer)
                outp = div_list[idx].get_pin(f'outp<{2 ** nbits - 1}>', layer=hm_layer)
                clkn = div_list[idx + 1].get_pin('clkn')
                clkp = div_list[idx + 1].get_pin('clkp')
                self.connect_differential_wires(outn, outp, clkn, clkp)
        self.reexport(div_list[0].get_port('clkn'))
        self.reexport(div_list[0].get_port('clkp'))

        tr_w_sup_ym = tr_manager.get_width(ym_layer, 'sup')

        export_output = self.params['export_output']
        if export_output:
            if len(div_list) > 1:
                final_output_tid = [div_list[-2].get_pin('clkn').track_id, div_list[-2].get_pin('clkp').track_id]
            else:
                final_output_tid = [
                    tr_manager.get_next_track(vm_layer, div_list[-2].get_pin('clkn').track_id.base_index,
                                              'clk', 'clk', up=False),
                    tr_manager.get_next_track(vm_layer, div_list[-2].get_pin('clkp').track_id.base_index,
                                              'clk', 'clk', up=True)]
                tr_w_clk_vm = tr_manager.get_width(vm_layer, 'clk')
                final_output_tid = [TrackID(vm_layer, final_output_tid[0], tr_w_clk_vm),
                                    TrackID(vm_layer, final_output_tid[1], tr_w_clk_vm)]

            final_outn = self.connect_to_tracks(div_list[-1].get_pin(f'outn<{2 ** nbits - 1}>', layer=hm_layer),
                                                final_output_tid[0], min_len_mode=MinLenMode.MIDDLE)
            final_outp = self.connect_to_tracks(div_list[-1].get_pin(f'outp<{2 ** nbits - 1}>', layer=hm_layer),
                                                final_output_tid[1], min_len_mode=MinLenMode.MIDDLE)
            self.add_pin('final_outn', final_outn, hide=True)
            self.add_pin('final_outp', final_outp, hide=True)

        self.add_pin('VDD_xm', vdd_list, label='VDD', connect=True)
        self.add_pin('VSS_xm', vss_list, label='VSS', connect=True)
        if self.params['top_sup_layer'] > 4:
            tr_w_sup_xm = tr_manager.get_width(xm_layer, 'sup')
            tr_w_sup_xm1 = tr_manager.get_width(xm1_layer, 'sup')

            ym_tid_l = self.arr_info.col_to_track(ym_layer, 0, mode=RoundMode.GREATER_EQ)
            ym_tid_r = self.arr_info.col_to_track(ym_layer, self.num_cols, mode=RoundMode.LESS_EQ)
            num_ym_sup = tr_manager.get_num_wires_between(ym_layer, 'dum', ym_tid_l, 'dum', ym_tid_r, 'sup')
            _, ym_sup_tidxs = tr_manager.place_wires(ym_layer, ['dum'] + ['sup'] * num_ym_sup + ['dum'],
                                                     center_coord=self.bound_box.w // 2)
            ym_sup_tidxs = ym_sup_tidxs[1:-1]

            vdd_ym = [self.connect_to_tracks(vdd_list, TrackID(ym_layer, tid, tr_w_sup_ym))
                      for tid in ym_sup_tidxs[::2]]
            vss_ym = [self.connect_to_tracks(vss_list, TrackID(ym_layer, tid, tr_w_sup_ym))
                      for tid in ym_sup_tidxs[1::2]]
            self.add_pin('VDD_ym', vdd_ym, label='VDD', connect=True)
            self.add_pin('VSS_ym', vss_ym, label='VSS', connect=True)
            if self.params['top_sup_layer'] > 5:
                vss_xm1_list, vdd_xm1_list = [], []
                for vss in vss_list:
                    xm1_tidx = self.grid.coord_to_track(xm1_layer,
                                                        self.grid.track_to_coord(xm_layer, vss.track_id.base_index),
                                                        mode=RoundMode.NEAREST)
                    vss_xm1 = self.connect_to_tracks(vss_ym, TrackID(xm1_layer, xm1_tidx, tr_w_sup_xm1),
                                                     track_lower=self.bound_box.xl, track_upper=self.bound_box.xh)
                    vss_xm1_list.append(vss_xm1)

                for vdd in vdd_list:
                    xm1_tidx = self.grid.coord_to_track(xm1_layer,
                                                        self.grid.track_to_coord(xm_layer, vdd.track_id.base_index),
                                                        mode=RoundMode.NEAREST)
                    vdd_xm1 = self.connect_to_tracks(vdd_ym, TrackID(xm1_layer, xm1_tidx, tr_w_sup_xm1),
                                                     track_lower=self.bound_box.xl, track_upper=self.bound_box.xh)
                    vdd_xm1_list.append(vdd_xm1)
                self.add_pin('VDD', vdd_xm1_list, connect=True)
                self.add_pin('VSS', vss_xm1_list, connect=True)

        self.sch_params = dict(
            div_params_list=[temp.sch_params for temp in master_list],
            nbits=nbits,
            ndivs=ndivs
        )
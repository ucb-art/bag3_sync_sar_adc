from typing import Any, Optional, Dict, Type, Tuple, cast, List

from bag.design.module import Module
from bag.layout.core import PyLayInstance
from bag.layout.routing import WireArray
from bag.layout.template import TemplateDB, TemplateBase
from bag.util.immutable import Param
from bag.util.importlib import import_class
from pybag.core import BBox, Transform
from pybag.enum import Orient2D
from xbase.layout.enum import MOSWireType, SubPortMode
from xbase.layout.mos.base import MOSBase
from xbase.layout.mos.top import MOSBaseWrapper
from .template import TemplateBaseZL, TrackIDZL


class GenericWrapper(MOSBaseWrapper):
    """A MOSArrayWrapper that works with any given generator class."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBaseWrapper.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='wrapped class name.',
            params='parameters for the wrapped class.',
            export_hidden='True to export hidden pins.',
            export_private='True to bring private pins in BBox',
            half_blk_x='Defaults to True.  True to allow half-block width.',
            half_blk_y='Defaults to True.  True to allow half-block height.',
            top_layer='Top layer override',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(export_hidden=False, half_blk_x=True, half_blk_y=True, export_private=True,
                    top_layer=0)

    def get_layout_basename(self) -> str:
        cls_name: str = self.params.get('cls_name', '')
        if cls_name:
            cls_name = cls_name.split('.')[-1]
            if cls_name.endswith('Core'):
                return cls_name[:-4]
            return cls_name + 'Wrap'
        else:
            # if sub-class of GenericWrapper does not have cls_name parameter,
            # use default base name
            return super(GenericWrapper, self).get_layout_basename()

    def draw_boundaries(self, master: MOSBase, top_layer: int, *,
                        half_blk_x: bool = True, half_blk_y: bool = True) -> PyLayInstance:
        self._core = master

        tech_cls = master.tech_cls
        bbox = master.bound_box
        used_arr = master.used_array

        w_blk, h_blk = self.grid.get_block_size(top_layer,
                                                half_blk_x=half_blk_x, half_blk_y=half_blk_y)

        w_master = bbox.w
        h_master = bbox.h
        w_edge = tech_cls.get_edge_width(w_master, w_blk)
        base_end_info = tech_cls.get_mos_base_end_info(master.place_info, h_blk)

        # get top/bottom boundary delta/height
        num_tiles = used_arr.num_tiles
        idx_bot = int(used_arr.get_flip_tile(0))
        idx_top = int(not used_arr.get_flip_tile(num_tiles - 1))
        dy_bot = base_end_info.h_blk[idx_bot]
        dy_top = base_end_info.h_blk[idx_top]
        h_end_bot = base_end_info.h_mos_end[idx_bot]
        h_end_top = base_end_info.h_mos_end[idx_top]

        self._xform = Transform(w_edge, dy_bot)
        inst = self.add_instance(master, inst_name='X0', xform=self._xform)

        my_used_arr = used_arr.get_copy()
        sd_pitch = tech_cls.sd_pitch
        w_tot = w_edge * 2 + w_master
        h_tot = dy_bot + dy_top + h_master
        self._fill_space(master.grid, tech_cls, w_edge, my_used_arr, sd_pitch, w_tot, h_tot,
                         dy_bot, h_end_bot, h_end_top)

        h_tot = -(-h_tot // h_blk) * h_blk
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot))
        return inst

    def draw_layout(self):
        params = self.params
        cls_name: str = params['cls_name']
        dut_params: Param = params['params']
        export_hidden: bool = params['export_hidden']
        export_private: bool = params['export_private']
        half_blk_x: bool = params['half_blk_x']
        half_blk_y: bool = params['half_blk_y']
        top_layer: int = params['top_layer']

        gen_cls = cast(Type[MOSBase], import_class(cls_name))
        master = self.new_template(gen_cls, params=dut_params)

        top_layer = max(top_layer, master.top_layer)
        self.wrap_mos_base(master, export_hidden, export_private, half_blk_x=half_blk_x, half_blk_y=half_blk_y,
                           top_layer=top_layer)

    def wrap_mos_base(self, master: MOSBase, export_hidden: bool, export_private: bool, half_blk_x: bool = True,
                      half_blk_y: bool = True, top_layer: int = 0):
        grid = self.grid
        inst = self.draw_boundaries(master, top_layer, half_blk_x=half_blk_x, half_blk_y=half_blk_y)

        def private_port_check(lay_id: int) -> bool:
            if lay_id <= top_layer and not grid.is_horizontal(lay_id):
                print(f'{master} WARNING: ports on private layer {lay_id} detected, '
                      f'converting to primitive ports.')
                return True
            return False

        # re-export pins
        for name in inst.port_names_iter():
            if not master.get_port(name).hidden or export_hidden:
                if export_private:
                    self.reexport(inst.get_primitive_port(name, private_port_check))
                else:
                    self.reexport(inst.get_port(name))

        # pass out schematic parameters
        self.sch_params = master.sch_params

        return inst


class MOSBaseTapWithPower(MOSBase, TemplateBaseZL):
    """A MOSArrayWrapper that works with any given generator class."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

        self._sch_cls: Optional[Type[Module]] = None
        self._core: Optional[MOSBase] = None
        self._sup_top_layer = 0

    @property
    def sup_top_layer(self) -> int:
        return self._sup_top_layer

    @property
    def core(self) -> MOSBase:
        return self._core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='wrapped class name.',
            params='parameters for the wrapped class.',
            pwr_gnd_list='List of supply names for each tile.',
            ncols_tot='Total number of columns',
            ndum_side='Dummies at sides',
            export_ym='Export supply to ym layer',
            export_xm1='Export supply to xm1 layer',
            export_ym1='Export supply to ym1 layer',
            fill_tap='True to fill tap',
            sup_top_layer='Top supply layer',
            cover_sup='Cover the entire layout with supply',
            config='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(pwr_gnd_list=None, ncols_tot=0, ndum_side=0, export_ym=False, export_ym1=False, export_xm1=False,
                    fill_tap=True, config={}, sup_top_layer=0, cover_sup=False)

    def get_schematic_class_inst(self) -> Optional[Type[Module]]:
        return self._sch_cls

    def get_layout_basename(self) -> str:
        cls_name: str = self.params['cls_name']
        cls_name = cls_name.split('.')[-1]
        return cls_name + 'Tap'

    def fill_tap(self, tile_idx, ndum_side) -> None:
        """
        This method fill empty region with sub contact
        """
        _, _, flip_tile = self.used_array.get_tile_info(tile_idx)
        intv_list = self.used_array.get_complement(tile_idx, 0, 0, self.num_cols)
        intv_list = [intv_list[0], intv_list[-1]] if len(intv_list) > 1 else intv_list
        port_mode = SubPortMode.ODD if ndum_side & 1 else SubPortMode.EVEN
        tap_sep = self.min_sep_col
        tap_sep += tap_sep & 1
        tap_sep += 2 if self.params.get('extra_tap_margin', False) else 0
        min_fill_ncols = self.tech_cls.min_sub_col + 2 * tap_sep

        def get_diff_port(pmode):
            return SubPortMode.EVEN if pmode == SubPortMode.ODD else SubPortMode.ODD

        intv_list_default = [((-tap_sep, ndum_side-tap_sep), None), ((self.num_cols-ndum_side+tap_sep, self.num_cols+tap_sep) ,None)]
        intv_list = [intv_list[0], intv_list[-1]] if len(intv_list) == 2 else intv_list_default
        for intv in intv_list:
            intv_pair = intv[0]
            nspace = intv_pair[1] - intv_pair[0]
            if nspace < min_fill_ncols:
                continue
            else:
                _port_mode = get_diff_port(port_mode) if (intv_pair[0] + self.min_sep_col) & 1 else port_mode
                if self.params['config'].get('port_mode', None) is not None:
                    _port_mode = SubPortMode.ODD if self.params['config'].get('port_mode', None) else SubPortMode.EVEN

                tap0 = self.add_substrate_contact(0, intv_pair[0] + tap_sep, seg=nspace - 2 * tap_sep,
                                                  tile_idx=tile_idx, port_mode=SubPortMode.EVEN)
            tid0 = self.get_track_id(0, MOSWireType.DS, 'sup', tile_idx=tile_idx)
            self.connect_to_tracks(tap0, tid0)
        intv_list = self.used_array.get_complement(tile_idx, 1, 0, self.num_cols)

        intv_list_default = [((-tap_sep, ndum_side-tap_sep), None), ((self.num_cols-ndum_side+tap_sep, self.num_cols+tap_sep) ,None)]
        intv_list = [intv_list[0], intv_list[-1]] if len(intv_list) == 2 else intv_list_default
        # intv_list = [intv_list[1]] if len(intv_list) < 3 else intv_list
        for intv in intv_list:
            intv_pair = intv[0]
            nspace = intv_pair[1] - intv_pair[0]
            if nspace < min_fill_ncols:
                continue
            else:
                _port_mode = get_diff_port(port_mode) if (intv_pair[0] + self.min_sep_col) & 1 else port_mode
                if self.params['config'].get('port_mode', None) is not None:
                    _port_mode = SubPortMode.ODD if self.params['config'].get('port_mode', None) else SubPortMode.EVEN
                tap1 = self.add_substrate_contact(1, intv_pair[0] + tap_sep, seg=nspace - 2 * tap_sep,
                                                  tile_idx=tile_idx, port_mode=_port_mode)
            tid1 = self.get_track_id(1, MOSWireType.DS, 'sup', tile_idx=tile_idx)
            self.connect_to_tracks(tap1, tid1)

    def draw_layout(self) -> None:
        gen_cls = cast(Type[MOSBase], import_class(self.params['cls_name']))
        pwr_gnd_list: List[Tuple[str, str]] = self.params['pwr_gnd_list']

        master: MOSBase = self.new_template(gen_cls, params=self.params['params'])
        tr_manager = master.tr_manager
        hm_layer = master.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        self._core = master
        self.draw_base(master.draw_base_info)

        num_tiles = master.num_tile_rows
        tap_ncol = max((self.get_tap_ncol(tile_idx=tile_idx) for tile_idx in range(num_tiles)))
        tap_sep_col = self.sub_sep_col
        num_cols = master.num_cols
        tot_ncols = max(num_cols + 2 * self.params['ndum_side'], self.params['ncols_tot'])
        ndum_side = (tot_ncols - num_cols) // 2
        tot_ncols = ndum_side * 2 + num_cols

        self.set_mos_size(tot_ncols, num_tiles=num_tiles)
        if not pwr_gnd_list:
            pwr_gnd_list = [('VDD', 'VSS')] * num_tiles
        elif len(pwr_gnd_list) != num_tiles:
            raise ValueError('pwr_gnd_list length mismatch.')

        inst = self.add_tile(master, 0, ndum_side)
        sup_names = set()

        # vdd_conn_list, vss_conn_list = [], []
        # for tidx in range(num_tiles):
        #     pwr_name, gnd_name = pwr_gnd_list[tidx]
        #     sup_names.add(pwr_name)
        #     sup_names.add(gnd_name)
        #     vdd_list = []
        #     vss_list = []
        #     self.add_tap(0, vdd_list, vss_list, tile_idx=tidx)
        #     self.add_tap(tot_ncols, vdd_list, vss_list, tile_idx=tidx, flip_lr=True)
        #     self.add_pin(pwr_name, vdd_list)
        #     self.add_pin(gnd_name, vss_list)
        #     vdd_conn_list.extend(vdd_list)
        #     vss_conn_list.extend(vss_list)
        #
        # vdd_conn_list = self.connect_wires(vdd_conn_list)[0].to_warr_list()
        # vss_conn_list = self.connect_wires(vss_conn_list)[0].to_warr_list()

        if self.params['fill_tap']:
            for idx in range(num_tiles):
                self.fill_tap(idx, ndum_side)

        if not inst.has_port('VDD_hm') or not inst.has_port('VSS_hm'):
            raise ValueError("instances needs vddhm and vsshm to use this wrapper")
        vdd_hm_list = inst.get_all_port_pins('VDD_hm')
        vss_hm_list = inst.get_all_port_pins('VSS_hm')
        # Connect conn layer and hm layer supply
        # self.connect_differential_wires(vdd_hm_list, vss_hm_list, vdd_conn_list[0], vss_conn_list[0])
        # self.connect_differential_wires(vdd_hm_list, vss_hm_list, vdd_conn_list[1], vss_conn_list[1])

        # Get vm locs at two sides
        side_margin_col = (tot_ncols - master.num_cols) // 2
        sup_bbox_l = BBox(self.bound_box.xl, self.bound_box.yl,
                          self.arr_info.col_to_coord(side_margin_col), self.bound_box.yh)
        sup_bbox_r = BBox(self.arr_info.col_to_coord(tot_ncols - side_margin_col), self.bound_box.yl,
                          self.bound_box.xh, self.bound_box.yh)

        sup_bbox_all_l = BBox(self.bound_box.xl, self.bound_box.yl,
                              (self.bound_box.xl + self.bound_box.xh) // 2, self.bound_box.yh)
        sup_bbox_all_r = BBox((self.bound_box.xl + self.bound_box.xh) // 2, self.bound_box.yl,
                              self.bound_box.xh, self.bound_box.yh)

        sup_vm_l = self.connect_supply_warr(tr_manager, [vdd_hm_list, vss_hm_list], hm_layer, sup_bbox_l,
                                            side_sup=True)
        sup_vm_r = self.connect_supply_warr(tr_manager, [vdd_hm_list, vss_hm_list], hm_layer, sup_bbox_r,
                                            side_sup=True, align_upper=True)
        vdd_vm_list = sup_vm_l[0] + sup_vm_r[0]
        vss_vm_list = sup_vm_l[1] + sup_vm_r[1]

        # Go up to xm_layer
        vdd_xm = inst.get_all_port_pins('VDD', layer=xm_layer)
        if inst.has_port('VSS_xm') or vdd_xm:
            vdd_xm_list = vdd_xm if vdd_xm else inst.get_all_port_pins('VDD_xm')
            for w in vdd_xm_list:
                if vdd_vm_list:
                    w_new = self.connect_to_track_wires(vdd_vm_list, w)
                    w_new = self.extend_wires(w_new, lower=self.bound_box.xl, upper=self.bound_box.xh)
                    self.add_pin('VDD', w_new, connect=False)
        else:
            vdd_xm_list = None

        vss_xm = inst.get_all_port_pins('VSS', layer=xm_layer)
        if inst.has_port('VSS_xm') or vss_xm:
            vss_xm_list = vss_xm if vss_xm else inst.get_all_port_pins('VSS_xm')
            for w in vss_xm_list:
                if vss_vm_list:
                    w_new = self.connect_to_track_wires(vss_vm_list, w)
                    w_new = self.extend_wires(w_new, lower=self.bound_box.xl, upper=self.bound_box.xh)
                    self.add_pin('VSS', w_new, connect=False)
        else:
            vss_xm_list = None

        # Get ym locs at two sides

        self._sup_top_layer = sup_top_layer = self.params['sup_top_layer']
        cover_sup = self.params['cover_sup']
        ym_layer = xm_layer + 1
        xm1_layer = ym_layer + 1
        master_top_layer = self.top_layer

        if sup_top_layer > xm_layer or self.params['export_ym']:
            if not vss_xm_list:
                sup_xm_l = self.connect_supply_warr(tr_manager, [vss_vm_list, vdd_vm_list], vm_layer, sup_bbox_l,
                                                    side_sup=True)
                sup_xm_r = self.connect_supply_warr(tr_manager, [vss_vm_list, vdd_vm_list], vm_layer, sup_bbox_r,
                                                    side_sup=True)
                vdd_xm_list = sup_xm_l[0] + sup_xm_r[0]
                vss_xm_list = sup_xm_l[1] + sup_xm_r[1]

            if not inst.has_port('VDD_xm') or not inst.has_port('VSS_xm'):
                raise ValueError('MOSBaseWithPower: Need xm supply in main block to export to ym')
            bbox_l = sup_bbox_all_l if ym_layer > master_top_layer and cover_sup else sup_bbox_l
            bbox_r = sup_bbox_all_r if ym_layer > master_top_layer and cover_sup else sup_bbox_r
            sup_ym_l = self.connect_supply_warr(tr_manager, [vdd_xm_list, vss_xm_list], xm_layer, bbox_l,
                                                side_sup=False)
            sup_ym_r = self.connect_supply_warr(tr_manager, [vdd_xm_list, vss_xm_list], xm_layer, bbox_r,
                                                side_sup=False, align_upper=True)
            vdd_ym_list = sup_ym_l[0] + sup_ym_r[0]
            vss_ym_list = sup_ym_l[1] + sup_ym_r[1]

        else:
            vdd_ym_list, vss_ym_list = None, None

        if sup_top_layer > ym_layer or self.params['export_xm1']:
            sup_xm1 = self.connect_supply_warr(tr_manager, [vdd_ym_list, vss_ym_list], ym_layer, self.bound_box)
            vdd_xm1_list, vss_xm1_list = sup_xm1[0], sup_xm1[1]
            self.add_pin('VDD', vdd_xm1_list, connect=False)
            self.add_pin('VSS', vss_xm1_list, connect=False)
        else:
            vdd_xm1_list, vss_xm1_list = [], []

        ym1_layer = xm1_layer + 1
        if sup_top_layer > xm1_layer or self.params['export_ym1']:
            bbox_l = sup_bbox_all_l if ym1_layer > master_top_layer and cover_sup else sup_bbox_l
            bbox_r = sup_bbox_all_r if ym1_layer > master_top_layer and cover_sup else sup_bbox_r
            sup_ym1_l = self.connect_supply_warr(tr_manager, [vdd_xm1_list, vss_xm1_list], xm1_layer, bbox_l,
                                                 side_sup=True)
            sup_ym1_r = self.connect_supply_warr(tr_manager, [vdd_xm1_list, vss_xm1_list], xm1_layer, bbox_r,
                                                 side_sup=True, align_upper=True)
            vdd_ym1_list = sup_ym1_l[0] + sup_ym1_r[0]
            vss_ym1_list = sup_ym1_l[1] + sup_ym1_r[1]
            self.add_pin('VDD', vdd_ym1_list, connect=False)
            self.add_pin('VSS', vss_ym1_list, connect=False)
        else:
            vdd_ym1_list, vss_ym1_list = None, None

        if sup_top_layer > ym1_layer:
            vss_list, vdd_list = vss_ym1_list, vdd_ym1_list
            for idx in range(ym1_layer, sup_top_layer):
                idx_dir = self.grid.get_direction(idx)
                if idx_dir == Orient2D.y:
                    vdd_list, vss_list = self.connect_supply_warr(tr_manager, [vdd_list, vss_list], idx,
                                                                  self.bound_box)
                else:
                    sup_l = self.connect_supply_warr(tr_manager, [vdd_list, vss_list], idx, sup_bbox_all_l,
                                                     side_sup=True)
                    sup_r = self.connect_supply_warr(tr_manager, [vdd_list, vss_list], idx, sup_bbox_all_r,
                                                     side_sup=True, align_upper=True)
                    vdd_list = sup_l[0] + sup_r[0]
                    vss_list = sup_l[1] + sup_r[1]
            self.add_pin('VDD', vdd_list, connect=False)
            self.add_pin('VSS', vss_list, connect=False)

        for name in inst.port_names_iter():
            self.reexport(inst.get_port(name))

        self.sch_params = master.sch_params
        self._sch_cls = master.get_schematic_class_inst()


class MOSBaseTapWrapper(GenericWrapper):
    """A MOSArrayWrapper that works with any given generator class."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        GenericWrapper.__init__(self, temp_db, params, **kwargs)
        self._tr_manager = None

    @property
    def tr_manager(self):
        return self._tr_manager

    @property
    def core(self) -> MOSBase:
        real_core = super().core
        return cast(MOSBaseTapWithPower, real_core).core

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        ans = MOSBaseTapWithPower.get_params_info()
        ans.update(**GenericWrapper.get_params_info())
        return ans

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = MOSBaseTapWithPower.get_default_param_values()
        ans.update(**GenericWrapper.get_default_param_values())
        return ans

    def draw_layout(self) -> None:
        master = self.new_template(MOSBaseTapWithPower, params=self.params)
        self._tr_manager = master.core.tr_manager
        top_layer = max(master.sup_top_layer, master.top_layer)
        self.wrap_mos_base(master, False, False, top_layer=top_layer)


class IntegrationWrapper(TemplateBaseZL):
    """
    Connect blocks to higher supply layer and quantized to correct grid
    For symmetric layout makesure block are in the middle """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBaseZL.__init__(self, temp_db, params, **kwargs)
        self._core = None
        self._core_sup_top_layer = None

    def get_schematic_class_inst(self) -> Optional[Type[Module]]:
        return self._sch_cls

    @property
    def core(self):
        return self._core

    @property
    def core_supply_layer(self):
        return self._core_sup_top_layer

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cls_name='wrapped class name.',
            params='parameters for the wrapped class.',
            sup_top_layer='Top supply layer',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(sup_top_layer=0)

    def place_block(self, top_layer, temp: TemplateBase):
        w_blk, h_blk = self.grid.get_block_size(top_layer, half_blk_y=False, half_blk_x=False)
        w_inst, h_inst = temp.bound_box.w, temp.bound_box.h

        w_tot, h_tot = -(-w_inst // w_blk) * w_blk, -(-h_inst // h_blk) * h_blk
        inst = self.add_instance(temp, xform=Transform((w_tot - w_inst) // 2, 0))
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot))
        return inst

    def draw_layout(self) -> None:
        gen_cls = cast(Type[MOSBase], import_class(self.params['cls_name']))
        master: MOSBase = self.new_template(gen_cls, params=self.params['params'])
        sup_top_layer = max(master.top_layer, self.params['sup_top_layer'])
        inst = self.place_block(sup_top_layer, master)
        self._core = master
        tr_manager = master.core.tr_manager

        self._core_sup_top_layer = master.top_layer

        if sup_top_layer > master.top_layer:
            vss_list, vdd_list = inst.get_all_port_pins('VSS', layer=master.top_layer), \
                                 inst.get_all_port_pins('VDD', layer=master.top_layer)
            if not vss_list or not vdd_list:
                raise ValueError("Doesn't have supply routing on its top layer")
            for idx in range(master.top_layer, sup_top_layer):
                sup_w = tr_manager.get_width(idx,'sup')
                sup_sep_ntr = self.get_track_sep(idx, sup_w, sup_w)
                sup_w = self.grid.get_track_info(idx).pitch * sup_sep_ntr
                sup_bbox_all_l = BBox(self.bound_box.xl, self.bound_box.yl,
                                      (self.bound_box.xl + self.bound_box.xh) // 2 - sup_w, self.bound_box.yh)
                sup_bbox_all_r = BBox((self.bound_box.xl + self.bound_box.xh) // 2 + sup_w, self.bound_box.yl,
                                      self.bound_box.xh, self.bound_box.yh)
                idx_dir = self.grid.get_direction(idx)
                if idx_dir == Orient2D.y:
                    vdd_list, vss_list = self.connect_supply_warr(tr_manager, [vdd_list, vss_list], idx,
                                                                  self.bound_box)
                else:
                    sup_l = self.connect_supply_warr(tr_manager, [vdd_list, vss_list], idx, sup_bbox_all_l,
                                                     side_sup=False)
                    sup_r = self.connect_supply_warr(tr_manager, [vdd_list, vss_list], idx, sup_bbox_all_r,
                                                     side_sup=False, align_upper=True)
                    vdd_list = sup_l[0] + sup_r[0]
                    vss_list = sup_l[1] + sup_r[1]
                self.add_pin('VDD', vdd_list)
                self.add_pin('VSS', vss_list)

        else:
            self.reexport(inst.get_port('VDD'), connect=False)
            self.reexport(inst.get_port('VSS'), connect=False)

        for name in inst.port_names_iter():
            # if not 'VDD' == name and not 'VSS' == name:
            self.reexport(inst.get_port(name))

        self.sch_params = master.sch_params
        self._sch_cls = master.get_schematic_class_inst()

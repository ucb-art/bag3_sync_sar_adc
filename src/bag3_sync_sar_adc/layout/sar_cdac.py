import copy

from typing import Any, Dict, Type, Optional, List, Mapping, Union, Tuple

from bag.design.database import ModuleDB, Module
from bag.env import get_tech_global_info
from bag.layout.routing.base import TrackManager, TrackID
from bag.layout.routing.base import WireArray
from bag.layout.template import TemplateDB, TemplateBase
from bag.util.immutable import Param, ImmutableSortedDict
from bag.util.math import HalfInt
from pybag.core import Transform, BBox
from pybag.enum import Orient2D, RoundMode, Direction, PinMode, MinLenMode, Orientation
from xbase.layout.enum import MOSWireType
from xbase.layout.fill.base import DeviceFill
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase, MOSArrayPlaceInfo
from xbase.layout.mos.placement.data import TilePatternElement, TilePattern
from xbase.layout.mos.top import GenericWrapper
from xbase.layout.cap.mim import MIMCap
from xbase.schematic.mimcap import xbase__mimcap
from .util.util import get_available_tracks_reverse


class CapTap(MOSBase):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg='segments dictionary.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=2
        )

    def draw_layout(self):
        pinfo = self.params['pinfo']
        seg = self.params['seg']
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, pinfo)
        self.draw_base(pinfo)

        tap = self.add_substrate_contact(0, 0, seg=seg, tile_idx=0)
        self.set_mos_size()
        self.add_pin('VSS', tap)


class CapUnitCore(TemplateBase):
    """MOMCap core
    Draw a layout has only metal and metal resistor in this shape:
    ----------------|
    --------------  |
    ----------------|
    Horizontal layer is "vertical_layer"
    Top and bottom is connected by "bot_layer"

    Parameters:
        top_w: width of middle horizontal layer
        bot_w: width of top/bot horizontal layer
        bot_y_w: width of vertical layer
        sp: space between top/bot and middle
        sp_le: line-end space between middle horizontal layer
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'cap_unit')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cap_config='MOM cap configuration.',
            width='MOM cap width, in resolution units.',
            tr_w='Track width',
            tr_sp='Track space',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = DeviceFill.get_default_param_values()
        ans.update(
            cap_config={},
            width=0,
            tr_w={},
            tr_sp={},
        )
        return ans

    def draw_layout(self) -> None:
        cap_config: Dict[str, int] = self.params['cap_config']
        tr_w: Dict = self.params['tr_w']
        tr_sp: Dict = self.params['tr_sp']
        width: int = self.params['width']

        tr_manager = TrackManager(self.grid, tr_w, tr_sp)

        grid = self.grid

        # Read cap_info
        top_layer = cap_config['top_layer']
        bot_layer = cap_config['bot_layer']
        top_w = cap_config['top_w']
        bot_w = cap_config['bot_w']
        bot_y_w = cap_config['bot_y_w']
        sp = cap_config['sp']
        sp_le = cap_config['sp_le']

        w_blk, h_blk = grid.get_block_size(max(top_layer, bot_layer), half_blk_x=True, half_blk_y=True)

        # draw cap
        if grid.get_direction(top_layer) == Orient2D.y:
            raise ValueError("Top layer need to be PGD")

        # Get tidx of top/mid/bot horizontal layer
        tidx_l = grid.find_next_track(top_layer, 0, tr_width=top_w, half_track=True)
        tidx_sp = grid.get_sep_tracks(top_layer, ntr1=top_w, ntr2=bot_w)
        tidx_sp = max(tidx_sp, HalfInt(sp))
        tidx_m = tidx_l + tidx_sp
        tidx_h = tidx_m + tidx_sp

        # Add wires
        top_l = self.add_wires(top_layer, tidx_l, 0, width, width=top_w)
        top_h = self.add_wires(top_layer, tidx_h, 0, width, width=top_w)

        height = grid.track_to_coord(top_layer, tidx_h) + grid.get_track_offset(top_layer)
        w_tot = -(-width // w_blk) * w_blk
        h_tot = -(-height // h_blk) * h_blk

        # Connect lower layer
        bot_layer_w = grid.get_track_info(bot_layer).width
        btidx = grid.coord_to_track(bot_layer, width - bot_layer_w, mode=RoundMode.NEAREST, even=True)
        bot = self.add_wires(bot_layer, btidx, 0, height, width=bot_y_w)
        self.add_via_on_grid(bot.track_id, top_l.track_id, extend=True)
        self.add_via_on_grid(bot.track_id, top_h.track_id, extend=True)

        bot_mid_coord = grid.track_to_coord(bot_layer, bot.track_id.base_index)

        top_min_l = grid.get_next_length(top_layer, bot_w, grid.get_wire_total_width(top_layer, bot_w), even=True)
        top_min_le_sp = grid.get_line_end_space(top_layer, bot_w, even=True)
        top_m_len = width - top_min_l - top_min_le_sp
        # top_m_len = grid.get_wire_bounds(bot_layer, btidx, bot_y_w)[0]
        top_m_len_unit = cap_config.get('unit', 1)
        top_m_len = int(top_m_len_unit * (top_m_len - sp_le))
        top_m = self.add_wires(top_layer, tidx_m, 0, top_m_len, width=bot_w)
        _top_m_dum = self.add_wires(top_layer, tidx_m, top_m_len + top_min_le_sp,
                                    grid.get_wire_bounds(bot_layer, btidx, bot_y_w)[1], width=bot_w)

        has_rmetal = cap_config.get('has_rmetal', True)
        if has_rmetal:
            pin_len = grid.get_next_length(top_layer, top_m.track_id.width,
                                           grid.get_wire_total_width(top_layer, top_m.track_id.width), even=True)
            res_top_box = top_m.bound_box
            res_top_box.set_interval(grid.get_direction(top_layer), top_m.bound_box.xh - pin_len,
                                     top_m.bound_box.xh - pin_len // 2)
            res_bot_box = top_l.bound_box
            res_bot_box.set_interval(grid.get_direction(top_layer), top_m.bound_box.xl + pin_len // 2,
                                     top_m.bound_box.xl + pin_len)
            
            self.add_res_metal(top_layer, res_bot_box)
            self.add_res_metal(top_layer, res_top_box)

        # set size
        bnd_box = BBox(0, 0, w_tot, h_tot)
        self.array_box = BBox(0, grid.get_track_offset(top_layer), bot_mid_coord,
                              h_tot - grid.get_track_offset(top_layer))
        self.set_size_from_bound_box(max(top_layer, bot_layer), bnd_box)

        # Fill metal dummy pattern
        for _layer in range(1, min(bot_layer, top_layer)):
            # -- Vertical layers --
            if _layer & 1:
                _tidx_l = self.grid.coord_to_track(_layer, self.array_box.xl, mode=RoundMode.GREATER_EQ)
                _tidx_h = self.grid.coord_to_track(_layer, self.array_box.xh, mode=RoundMode.LESS_EQ)
                _num_dum = tr_manager.get_num_wires_between(_layer, 'dum', _tidx_l, 'dum', _tidx_h, 'dum')
                _tr_w_dum = tr_manager.get_width(_layer, 'dum')
                _, _dum_locs = tr_manager.place_wires(_layer, ['dum'] * _num_dum,
                                                      center_coord=(self.array_box.xh + self.array_box.xl) // 2)
                [self.add_wires(_layer, tidx, self.array_box.yl, self.array_box.yh, width=_tr_w_dum) for tidx in
                 _dum_locs]
            # -- Horizontal layers --
            else:
                _tidx_l = self.grid.coord_to_track(_layer, self.array_box.yl, mode=RoundMode.GREATER_EQ)
                _tidx_h = self.grid.coord_to_track(_layer, self.array_box.yh, mode=RoundMode.LESS_EQ)
                _num_dum = tr_manager.get_num_wires_between(_layer, 'dum', _tidx_l, 'dum', _tidx_h, 'dum')
                _tr_w_dum = tr_manager.get_width(_layer, 'dum')
                _, _dum_locs = tr_manager.place_wires(_layer, ['dum'] * _num_dum,
                                                      center_coord=(self.array_box.yh + self.array_box.yl) // 2)
                [self.add_wires(_layer, tidx, self.array_box.xl, self.array_box.xh, width=_tr_w_dum) for tidx in
                 _dum_locs]

        self.add_pin('minus', bot)
        self.add_pin('plus', top_m, mode=PinMode.LOWER)

        if 'cap' in cap_config and has_rmetal:
            self.sch_params = dict(
                res_plus=dict(layer=top_layer, w=res_top_box.h, l=res_top_box.w),
                res_minus=dict(layer=top_layer, w=res_bot_box.h, l=res_bot_box.w),
                cap=top_m_len_unit * cap_config['cap']
            )
        elif 'cap' in cap_config:
            self.sch_params = dict(cap=top_m_len_unit * cap_config['cap'])
        elif has_rmetal:
            self.sch_params = dict(
                res_plus=dict(layer=top_layer, w=res_top_box.h, l=res_top_box.w),
                res_minus=dict(layer=top_layer, w=res_bot_box.h, l=res_bot_box.w),
            )
        else:
            self.sch_params = dict(
                res_plus=None,
                res_minus=None,
            )


class CapColCore(TemplateBase):
    """Cap core
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        # return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'cap_unit')
        return xbase__mimcap

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            ny='number of unit cap in column',
            ratio='ratio of unit cell',
            cap_config='MOM cap configuration.',
            width='MOM cap width, in resolution units.',
            pin_tr_w='Width for top-plate pin',
            add_tap='Add tap to provides substrate',
            options='Other options, for use in ringamp'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = DeviceFill.get_default_param_values()
        ans.update(
            cap_config={},
            width=0,
            pin_tr_w=1,
            ratio=4,
            add_tap=False,
            options={},
        )
        return ans

    def draw_layout(self) -> None:
        cap_config: ImmutableSortedDict[str, Union[int, float]] = self.params['cap_config']
        options: ImmutableSortedDict[str, Any] = self.params['options']
        width: int = self.params['width']
        ratio: int = self.params['ratio']
        ny: int = self.params['ny']

        if ny & 1:
            raise ValueError("Number of cell must be even number")

        grid = self.grid
        unit_pin_layer = options.get('pin_layer', cap_config['top_layer'] - 1)
        add_tap = options.get('add_tap', False)
        unit_pin_tidx = grid.find_next_track(unit_pin_layer, 0, tr_width=cap_config['top_w'], half_track=True)
        pin_conn_sep = grid.get_sep_tracks(unit_pin_layer, ntr1=cap_config['top_w'], ntr2=1)

        if cap_config['ismim'] == True:
            cap_config_copy = copy.deepcopy(cap_config.to_dict())
            unit_master = self.new_template(CapMIMCore,
                            params=dict(cap_config=cap_config_copy))
            
            mimcap_master = self.new_template(CapMIMCore,
                            params=dict(cap_config=cap_config_copy))
            lay_top_layer = max(unit_pin_layer, mimcap_master.top_layer)
            w_blk, h_blk = grid.get_block_size(lay_top_layer, half_blk_x=True, half_blk_y=True)
    
            unit_x = grid.track_to_coord(unit_pin_layer, unit_pin_tidx + pin_conn_sep)
            unit_x = -(-unit_x // w_blk) * w_blk

            mimcap = self.add_instance(mimcap_master, xform=Transform(unit_x, 0))
            bbox = mimcap.bound_box.extend(x=0, y=0)

            cap_bot = mimcap.get_pin('minus') #just get the minus pin
            cap_top_list = mimcap.get_pin('plus')#just get plus pin

            ideal_cap = unit_master.sch_params.get('cap', 0)
            m = 1 #TODO: the multiple in the schematic - have to edit in schematic

        else:
            cap_half_config = copy.deepcopy(cap_config.to_dict())
            cap_none_config = copy.deepcopy(cap_config.to_dict())
            cap_half_config['unit'] = 0.5
            cap_none_config['unit'] = 0
            unit_master: TemplateBase = self.new_template(CapUnitCore,
                                                          params=dict(cap_config=cap_config, width=width))
            unit_half_master: TemplateBase = self.new_template(CapUnitCore,
                                                               params=dict(cap_config=cap_half_config, width=width))
            unit_none_master: TemplateBase = self.new_template(CapUnitCore,
                                                               params=dict(cap_config=cap_none_config, width=width))

            lay_top_layer = max(unit_pin_layer, unit_master.top_layer)
            w_blk, h_blk = grid.get_block_size(lay_top_layer, half_blk_x=True, half_blk_y=True)

            unit_x = grid.track_to_coord(unit_pin_layer, unit_pin_tidx + pin_conn_sep)
            unit_x = -(-unit_x // w_blk) * w_blk

            if ratio & 8:
                cdac = [self.add_instance(unit_master, xform=Transform(unit_x, unit_master.array_box.h * idx))
                        for idx in range(ny)]
                bbox = cdac[-1].bound_box.extend(x=0, y=0)
                cap_bot = self.connect_wires([c.get_pin('minus') for c in cdac])
                cap_top_list = [c.get_pin('plus') for c in cdac]
                array_bbox = cdac[0].array_box.merge(cdac[-1].array_box)
                ideal_cap = unit_master.sch_params.get('cap', 0)
                m = 4
            elif ratio & 4:
                cdac = [self.add_instance(unit_half_master, xform=Transform(unit_x, unit_half_master.array_box.h * idx))
                        for idx in range(ny)]
                bbox = cdac[-1].bound_box.extend(x=0, y=0)
                cap_bot = self.connect_wires([c.get_pin('minus') for c in cdac])
                cap_top_list = [c.get_pin('plus') for c in cdac]
                array_bbox = cdac[0].array_box.merge(cdac[-1].array_box)
                ideal_cap = unit_half_master.sch_params.get('cap', 0)
                m = 4
            elif ratio & 2:
                cdac = [self.add_instance(unit_half_master, xform=Transform(unit_x, unit_half_master.array_box.h * idx))
                        for idx in range(2)] + \
                       [self.add_instance(unit_none_master, xform=Transform(unit_x, unit_none_master.array_box.h * idx))
                        for idx in range(2, 4)]
                bbox = cdac[-1].bound_box.extend(x=0, y=0)
                cap_bot = self.connect_wires([c.get_pin('minus') for c in cdac])
                cap_top_list = [c.get_pin('plus') for c in cdac]
                array_bbox = cdac[0].array_box.merge(cdac[-1].array_box)
                ideal_cap = unit_half_master.sch_params.get('cap', 0)
                m = 2
            elif ratio & 1:
                cdac = [self.add_instance(unit_half_master, xform=Transform(unit_x, 0))] + \
                       [self.add_instance(unit_none_master, xform=Transform(unit_x, unit_half_master.array_box.h * idx))
                        for idx in range(1, 4)]
                bbox = cdac[-1].bound_box.extend(x=0, y=0)
                cap_bot = self.connect_wires([c.get_pin('minus') for c in cdac])
                cap_top_list = [c.get_pin('plus') for c in cdac]
                array_bbox = cdac[0].array_box.merge(cdac[-1].array_box)
                ideal_cap = unit_half_master.sch_params.get('cap', 0)
                m = 1
            else:
                raise ValueError("Unit is wrong")

        if add_tap:
            tech_global = get_tech_global_info('bag3_digital')
            pinfo = dict(
                lch=tech_global['lch_min'],
                top_layer=MOSArrayPlaceInfo.get_conn_layer(self.grid.tech_info, tech_global['lch_min']) + 1,
                tr_widths={},
                tr_spaces={},
                row_specs=[dict(mos_type='ptap', width=tech_global['w_minn'], threshold='standard',
                                bot_wires=['sup'], top_wires=[])]
            )
            tap_master = self.new_template(CapTap, params=dict(pinfo=pinfo))
            tap = self.add_instance(tap_master, xform=Transform(-tap_master.bound_box.w, 0, Orientation.MY))
            self.reexport(tap.get_port('VSS'))

        self.set_size_from_bound_box(max(cap_config['top_layer'], cap_config['bot_layer']), bbox)
        self.array_box = bbox #array_bbox

        top_pin_list = []
        if (cap_config['ismim'] == True):
            for idx in range(0, ny, 4):
                top_pin_list.append(cap_top_list[0])
                self.add_pin(f'top_xm', cap_top_list[0], hide=True)

        else:
            for idx in range(0, ny, 4):
                _pin = self.connect_to_tracks(cap_top_list[idx: idx + 4],
                                              TrackID(unit_pin_layer, unit_pin_tidx, cap_config['top_w']))
                top_pin_list.append(_pin)
                self.add_pin(f'top_xm', cap_top_list[idx: idx + 4], hide=True)

        connect_top = options.get('connect_top', True)
        if connect_top:
            self.add_pin('top', self.connect_wires(top_pin_list))
        else:
            [self.add_pin(f'top', _pin) for _pin in top_pin_list]
        array_box_l = self.grid.track_to_coord(top_pin_list[0].layer_id, top_pin_list[0].track_id.base_index)
        self.array_box.extend(x=array_box_l)

        self.add_pin('bot', cap_bot)

        if cap_config['ismim']:
            parameters = copy.deepcopy(cap_config.to_dict())
            params_dum = copy.deepcopy(cap_config.to_dict())
            self.sch_params = \
                unit_master.sch_params.copy(copy.deepcopy(cap_config.to_dict()))
    
        else:
            new_sch_params = dict(m=m, plus_term='top', minus_term='bot')
            if ideal_cap:
                new_sch_params['cap'] = ideal_cap
            self.sch_params = \
                unit_master.sch_params.copy(append=new_sch_params)

class CapDrvCore(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'cap_drv')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            min_height='Height to match capdac',
            pinfo='placement information object.',
            seg='segments dictionary.',
            sp='dummy seperation',
            w='widths.',
            ny='number of rows',
            dum_row_idx='Index of dummy rows',
            sw_type='Type of switch',
            nx='number of columns',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w=4,
            sp=2,
            ny=5,
            min_height=0,
            dum_row_idx=[],
            nx=3,
            sw_type='nch',
        )

    def draw_layout(self):
        min_height: int = self.params['min_height']
        sw_type: str = self.params['sw_type']
        ny: int = self.params['ny']
        nx: int = self.params['nx']
        w: int = self.params['w']
        pinfo_dict = self.params['pinfo'].to_yaml()

        if min_height > 0:
            pinfo_dict['tile_specs']['place_info']['drv_tile']['min_height'] = min_height
        pinfo_dict['tile_specs']['place_info']['drv_tile']['row_specs'][0]['mos_type'] = sw_type
        pinfo_dict['tile_specs']['place_info']['drv_tile']['row_specs'][0]['width'] = w
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, pinfo_dict)
        pinfo0 = [TilePatternElement(pinfo[1]['drv_tile'])] * ny
        self.draw_base((TilePattern(pinfo0), pinfo[1]))

        dum_row_idx: List[int] = self.params['dum_row_idx']
        seg: int = self.params['seg']
        sp: int = self.params['sp']
        w: int = self.params['w']

        tr_manager = self.tr_manager
        conn_layer = self.conn_layer
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1
        tr_sup_vm_w = tr_manager.get_width(vm_layer, 'sup')
        tr_sup_xm_w = tr_manager.get_width(xm_layer, 'sup')
        tr_sup_ym_w = tr_manager.get_width(ym_layer, 'sup')

        sw_list_list, ctrl_list_list, vref_list_list = [], [], []
        cap_bot_list = []

        pin_lower = self.arr_info.col_to_coord(0)
        vdd_list, vss_list = [], []
        sup_bot_tid_list = []
        xm_tid_list = []
        tap_ncol = self.get_tap_ncol(tile_idx=0)
        tap_sep_col = self.sub_sep_col
        tap_ncol += tap_sep_col
        tile_height = self.get_tile_info(0)[0].height
        num_xm_per_tile = tr_manager.get_num_wires_between(xm_layer, 'sup',
                                                           self.grid.coord_to_track(xm_layer, 0, RoundMode.NEAREST),
                                                           'sup',
                                                           self.grid.coord_to_track(xm_layer, tile_height,
                                                                                    RoundMode.NEAREST),
                                                           'sup')
        if not num_xm_per_tile & 1:
            num_xm_per_tile += 1
        for idx in range(ny):
            if dum_row_idx and idx in dum_row_idx:
                continue
            self.add_tap(0, vdd_list, vss_list, tile_idx=idx)
            sw_list, ctrl_list, vref_list = [], [], []
            tid_bot = self.get_track_id(0, MOSWireType.DS, wire_name='sig', wire_idx=2, tile_idx=idx)
            tid_ref = self.get_track_id(0, MOSWireType.DS, wire_name='sig', wire_idx=0, tile_idx=idx)
            sw_col = tap_ncol

            # if nx != 2:
            tid_list = []
            for jdx in range(nx):
                sw_list.append(self.add_mos(0, sw_col, seg, w=w, tile_idx=idx))
                sw_col += seg + sp
                tid_list.append(self.get_track_index(0, MOSWireType.G, wire_name='sig',
                                                     wire_idx=-jdx - 1 if nx != 2 else jdx*2, tile_idx=idx))
                vref_list.append(self.connect_to_tracks(sw_list[-1].d, tid_ref, min_len_mode=MinLenMode.MIDDLE))

            ctrl_list.extend(self.connect_matching_tracks([sw.g for sw in sw_list], hm_layer,
                                                          tid_list, track_lower=pin_lower,
                                                          min_len_mode=MinLenMode.MIDDLE))
            cap_bot_list.append(self.connect_to_tracks([sw.s for sw in sw_list], tid_bot))
            self.add_tap(sw_col - sp + tap_ncol, vdd_list, vss_list, tile_idx=idx, flip_lr=True)

            # supply_hm
            # sup_bot_tid_list.append(self.get_track_id(0, MOSWireType.G, wire_name='sup', tile_idx=idx))
            sup_bot_tid_list.append(self.get_track_id(0, MOSWireType.DS, wire_name='sup', tile_idx=idx))
            sup_bot_tid_list.append(self.get_track_id(0, MOSWireType.DS, wire_name='sup', wire_idx=-1, tile_idx=idx))
            sw_list_list.append(sw_list)
            ctrl_list_list.append(ctrl_list)
            vref_list_list.append(vref_list)

        self.set_mos_size(self.num_cols, max(ny, dum_row_idx[-1]))
        for idx in range(ny):
            tile_info, yb, _ = self.get_tile_info(idx)
            xm_locs = self.get_available_tracks(xm_layer, self.grid.coord_to_track(xm_layer, yb, RoundMode.NEAREST),
                                                self.grid.coord_to_track(xm_layer, yb + tile_height, RoundMode.NEAREST),
                                                self.bound_box.xl, self.bound_box.xh, tr_sup_xm_w,
                                                tr_manager.get_sep(xm_layer, ('sup', 'sup')), False)
            if not len(xm_locs) & 1:
                xm_locs.pop(-1)
            xm_tid_list.append(xm_locs)
        sup_hm_list = []
        for tid in sup_bot_tid_list:
            sup_conn_list = vdd_list if sw_type == 'pch' else vss_list
            sup_hm_list.append(self.connect_to_tracks(sup_conn_list, tid))

        vref_vm_list = []
        for idx in range(nx):
            vref_vm_tidx = self.grid.coord_to_track(vm_layer, vref_list_list[0][idx].middle,
                                                    mode=RoundMode.LESS if seg & 1 else RoundMode.NEAREST)
            vref_vm_list.append(self.connect_to_tracks([vref_list[idx] for vref_list in vref_list_list],
                                                       TrackID(vm_layer, vref_vm_tidx, tr_sup_vm_w),
                                                       track_upper=self.bound_box.yh, track_lower=self.bound_box.yl))

        sup_vm_locs = self.get_available_tracks(vm_layer,
                                                self.arr_info.col_to_track(vm_layer, 0),
                                                self.arr_info.col_to_track(vm_layer, tap_ncol),
                                                self.bound_box.yl, self.bound_box.yh,
                                                tr_manager.get_width(vm_layer, 'sup'),
                                                tr_manager.get_sep(vm_layer, ('sup', 'sup')),
                                                include_last=True)[::2]
        sup_vm_locs += get_available_tracks_reverse(self, vm_layer,
                                                    self.arr_info.col_to_track(vm_layer, self.num_cols - tap_ncol,
                                                                               RoundMode.NEAREST),
                                                    self.arr_info.col_to_track(vm_layer, self.num_cols,
                                                                               RoundMode.NEAREST),
                                                    self.bound_box.yl, self.bound_box.yh,
                                                    tr_manager.get_width(vm_layer, 'sup'),
                                                    tr_manager.get_sep(vm_layer, ('sup', 'sup')),
                                                    include_last=True)[::2]

        sup_vm_list = []
        for tid in sup_vm_locs:
            sup_vm_list.append(self.connect_to_tracks(sup_hm_list, TrackID(vm_layer, tid, tr_sup_vm_w)))
        
        sup_xm_list = []
        for tid_list in xm_tid_list:
            mid_tid = tid_list[num_xm_per_tile // 2]
            for idx, vref in enumerate(vref_vm_list):
                #self.connect_to_tracks(vref, TrackID(xm_layer, mid_tid, tr_sup_xm_w))
                self.add_pin(f'vref{idx}_xm', vref_vm_list[idx]) #self.connect_to_tracks(vref, TrackID(xm_layer, mid_tid, tr_sup_xm_w)))
            tid_list.pop(num_xm_per_tile // 2)

        if sw_type == 'nch':
            for tid_list in xm_tid_list:
                for tid in tid_list[::2]:
                    sup_xm_list.append(self.connect_to_tracks(sup_vm_list, TrackID(xm_layer, tid, tr_sup_xm_w)))
            self.add_pin('VSS_xm', sup_xm_list)
        else:
            for tid_list in xm_tid_list:
                for tid in tid_list[1::2]:
                    sup_xm_list.append(self.connect_to_tracks(sup_vm_list, TrackID(xm_layer, tid, tr_sup_xm_w)))
            self.add_pin('VDD_xm', sup_xm_list)

        for idx in range(nx):
            self.add_pin(f'vref{idx}', vref_vm_list[idx])
            self.add_pin(f'ctrl{idx}', [ctrl_list[idx] for ctrl_list in ctrl_list_list])

        if vdd_list:
            self.add_pin('VDD', sup_vm_list)
        if vss_list:
            self.add_pin('VSS', sup_vm_list)
        self.add_pin(f'bot', cap_bot_list, mode=PinMode.UPPER)
        self.sch_params = dict(
            lch=self.arr_info.lch,
            w=w,
            seg=seg,
            intent=self.get_row_info(0, 0).threshold
        )


class CMSwitch(MOSBase):
    """A inverter with only transistors drawn, no metal connections
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='placement information object.',
            seg='segments dictionary.',
            w='widths.',
            ncols_tot='Total number of fingersa',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w=4,
            ncols_tot=0,
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)
        tr_manager = self.tr_manager

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1

        seg: int = self.params['seg']
        w: int = self.params['w']
        tap_ncol = self.get_tap_ncol(tile_idx=0)
        tap_sep_col = self.sub_sep_col
        tap_ncol += tap_sep_col

        vdd_list, vss_list = [], []
        tot_cols = max(self.params['ncols_tot'], seg + 2 * tap_ncol)
        self.add_tap(0, vdd_list, vss_list, tile_idx=0)
        sw = self.add_mos(0, (tot_cols - seg) // 2, seg, w=w)
        self.add_tap(tot_cols, vdd_list, vss_list, tile_idx=0, flip_lr=True)
        self.set_mos_size()

        tid_g = self.get_track_id(0, MOSWireType.G, wire_name='sig', wire_idx=0)
        tid_sig = self.get_track_id(0, MOSWireType.DS, wire_name='sig', wire_idx=1)
        tid_ref = self.get_track_id(0, MOSWireType.DS, wire_name='sig', wire_idx=0)

        sam_hm = self.connect_to_tracks(sw.g, tid_g)
        ref_hm = self.connect_to_tracks(sw.d, tid_ref)
        sig_hm = self.connect_to_tracks(sw.s, tid_sig)

        # get middle track for sample signal
        mid_vm_tidx = self.arr_info.col_to_track(vm_layer, tot_cols // 2, RoundMode.NEAREST)
        sam_vm = self.connect_to_tracks(sam_hm, TrackID(vm_layer, mid_vm_tidx, tr_manager.get_width(vm_layer, 'ctrl')))
        tid_l = self.arr_info.col_to_track(vm_layer, tap_ncol, mode=RoundMode.NEAREST)
        tid_r = self.arr_info.col_to_track(vm_layer, self.num_cols - tap_ncol, mode=RoundMode.NEAREST)

        tr_w_sup_vm = tr_manager.get_width(vm_layer, 'sup')
        tr_w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        vref_vm_locs = self.get_available_tracks(vm_layer, tid_l, mid_vm_tidx, self.bound_box.yl, self.bound_box.yh,
                                                 tr_manager.get_width(vm_layer, 'sup'),
                                                 tr_manager.get_sep(vm_layer, ('sup', 'sup')))
        sig_vm_locs = get_available_tracks_reverse(self, vm_layer, mid_vm_tidx, tid_r, self.bound_box.yl,
                                                   self.bound_box.yh, tr_manager.get_width(vm_layer, 'sig'),
                                                   tr_manager.get_sep(vm_layer, ('sig', 'sig')))
        vref_vm = [self.connect_to_tracks(ref_hm, TrackID(vm_layer, _tid, tr_w_sup_vm)) for _tid in vref_vm_locs]
        sig_vm = [self.connect_to_tracks(sig_hm, TrackID(vm_layer, _tid, tr_w_sig_vm)) for _tid in sig_vm_locs]
        vm_warrs = vref_vm + sig_vm
        vm_warrs_max_coord, vm_warrs_min_coord = max([v.upper for v in vm_warrs]), min([v.lower for v in vm_warrs])
        vref_vm = self.extend_wires(vref_vm, upper=vm_warrs_max_coord, lower=vm_warrs_min_coord)
        sig_vm = self.extend_wires(sig_vm, upper=vm_warrs_max_coord, lower=vm_warrs_min_coord)

        tr_w_sup_xm = tr_manager.get_width(xm_layer, 'sup')
        tr_w_sig_xm = tr_manager.get_width(xm_layer, 'sig')

        # Connect supplies
        tr_sup_hm_w = tr_manager.get_width(hm_layer, 'sup')
        tr_sup_vm_w = tr_manager.get_width(vm_layer, 'sup')
        tr_sup_xm_w = tr_manager.get_width(xm_layer, 'sup')
        sup_hm_tids = [self.get_track_id(0, MOSWireType.G, wire_name='sup'),
                       self.get_track_id(0, MOSWireType.DS, wire_name='sup')]

        sup_hm_list = []
        for tid in sup_hm_tids:
            sup_hm_list.append(self.connect_to_tracks(vss_list, tid))

        sup_vm_locs = self.get_available_tracks(vm_layer,
                                                self.arr_info.col_to_track(vm_layer, 0),
                                                self.arr_info.col_to_track(vm_layer, tap_ncol),
                                                self.bound_box.yl, self.bound_box.yh,
                                                tr_manager.get_width(vm_layer, 'sup'),
                                                tr_manager.get_sep(vm_layer, ('sup', 'sup')),
                                                include_last=True)[::2]
        sup_vm_locs += get_available_tracks_reverse(self, vm_layer,
                                                    self.arr_info.col_to_track(vm_layer, self.num_cols - tap_ncol,
                                                                               RoundMode.NEAREST),
                                                    self.arr_info.col_to_track(vm_layer, self.num_cols,
                                                                               RoundMode.NEAREST),
                                                    self.bound_box.yl, self.bound_box.yh,
                                                    tr_manager.get_width(vm_layer, 'sup'),
                                                    tr_manager.get_sep(vm_layer, ('sup', 'sup')),
                                                    include_last=True)[::2]

        sup_vm_list = []
        for tid in sup_vm_locs:
            sup_vm_list.append(self.connect_to_tracks(sup_hm_list, TrackID(vm_layer, tid, tr_sup_vm_w)))

        tile_info, yb, _ = self.get_tile_info(0)
        tile_height = tile_info.height
        xm_locs = self.get_available_tracks(xm_layer, self.grid.coord_to_track(xm_layer, yb, RoundMode.NEAREST),
                                            self.grid.coord_to_track(xm_layer, yb + tile_height, RoundMode.NEAREST),
                                            self.bound_box.xl, self.bound_box.xh, tr_sup_xm_w,
                                            tr_manager.get_sep(xm_layer, ('sup', 'sup')), False)
        if not len(xm_locs):
            xm_locs = xm_locs[:-1]
        # y_mid_coord = (self.bound_box.yl + self.bound_box.yh) // 2
        # xm_mid_tidx = self.grid.coord_to_track(xm_layer, y_mid_coord, mode=RoundMode.NEAREST)
        vref_xm = self.connect_to_tracks(vref_vm, TrackID(xm_layer, xm_locs[len(xm_locs)//2], tr_w_sup_xm))
        sig_xm = self.connect_to_tracks(sig_vm, TrackID(xm_layer, xm_locs[len(xm_locs)//2], tr_w_sup_xm))
        xm_locs.pop(len(xm_locs)//2)
        sup_xm_list = []
        for tid in xm_locs[::2]:
            sup_xm_list.append(self.connect_to_tracks(sup_vm_list, TrackID(xm_layer, tid, tr_sup_xm_w)))
        self.add_pin('sam', sam_vm)
        self.add_pin('ref', vref_xm)
        self.add_pin('sig', sig_xm)
        self.add_pin('VSS', sup_xm_list)

        self.sch_params = dict(
            l=self.arr_info.lch,
            nf=seg,
            w=w,
            intent=self.place_info.get_row_place_info(0).row_info.threshold,
        )


class CapDacColCore(TemplateBase):
    """MOMCap core
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)
        self._actual_width = 0

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'cdac_array_bot')

    @property
    def actual_width(self) -> int:
        return self._actual_width

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            nbits='Number of bits',
            ny_list='list of ny',
            ratio_list='list of ratio',
            sw_type='switch type list',
            diff_idx='differential cap index',
            seg='segments dictionary.',
            seg_cm='segments dictionary.',
            sp='segments dictionary.',
            w_n='widths dictionary.',
            w_p='widths dictionary.',
            w_cm='widths dictionary.',
            cap_config='MOM cap configuration.',
            width='MOM cap width, in resolution units.',
            pinfo='placement information object.',
            pinfo_cm='placement information object.',
            remove_cap='True to remove capacitor, use it when doesnt have rmetal',
            lower_layer_routing='only use up to m4',
            tr_widths='Track width dictionary',
            tr_spaces='Track space dictionary',
            has_cm_sw='has a cm sw in the cdac layout', 
            row_list='number of rows',
            col_list='number of columns'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = DeviceFill.get_default_param_values()
        ans.update(
            cap_config={},
            width=0,
            w_n=4,
            w_p=4,
            w=4,
            remove_cap=False,
            has_cm_sw = True,
            row_list =[],
            col_list =[]
        )
        return ans

    def draw_layout(self) -> None:
        cap_config: ImmutableSortedDict[str, int] = self.params['cap_config']
        width: int = self.params['width']
        nbits: int = self.params['nbits']
        seg: int = self.params['seg']
        sp: int = self.params['sp']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        seg_cm: int = self.params['seg_cm']
        w_cm: int = self.params['w_cm']
        diff_idx: int = self.params['diff_idx']
        ny_list: List[int] = self.params['ny_list'].to_list()
        ratio_list: List[int] = self.params['ratio_list'].to_list()
        row_list: List[int] = self.params['row_list'].to_list()
        col_list: List[int] = self.params['col_list'].to_list()
        sw_type: List[str] = self.params['sw_type'].to_list()
        tr_widths: Dict[str, Any] = self.params['tr_widths']
        tr_spaces: Mapping[Tuple[str, str], Mapping[int, Union[float, HalfInt]]] = self.params['tr_spaces']
        grid = self.grid
        tr_manager = TrackManager(self.grid, tr_widths, tr_spaces)
        has_cm_sw = self.params['has_cm_sw']

        if nbits < 3:
            raise ValueError("[CDAC layout]: Less than 3-bit is not supported")

        # organize the cap ratios and placement
        # diff_idx: index from which need 2 of the cap (differential on top and bot)
        # ny: number of unit components in each cap/4
        # ratio: units come in full, half, none flavors
        
        # complete full ny list with differential caps (ny[5:end] flipped + ny)
        ny_list = ny_list[diff_idx:][::-1] + ny_list
        #do same to ratio list
        ratio_list = ratio_list[diff_idx:][::-1] + ratio_list
        
        #bit list lists out which cap belongs to which total cap ratio (ex from 0 to 8)
        bit_list = [1, 0, 2, 3] + list(range(4, diff_idx-1))+ list(range(diff_idx - 1, nbits + 1))
        bit_list = bit_list[diff_idx:][::-1] + bit_list
        
        #compute the number of units
        if not row_list:
            tot_col = width//cap_config['unit_width']
            if (cap_config['ismim']==True):    
                row_list = []
                col_list = []
                dum_col_list = []
                h_idx = 0
                for idx in range(0, max(bit_list)+1):
                    if (idx==0):
                        row_list.append(1)
                        col_list.append(1)
                        dum_col_list.append(tot_col-1)
                    else:
                        if (width/(2**(idx-1)) >= cap_config['unit_width']):
                            if ((idx-1)>=h_idx):
                                h_idx = idx-1
                            col_list.append(2**(idx-1))
                            dum_col_list.append(tot_col-2**(idx-1))
                            row_list.append(1)
                        else:
                            col_list.append(2**h_idx)
                            dum_col_list.append(tot_col -2**h_idx )
                            row_list.append(2**(idx-h_idx-1))
                    if (idx >= diff_idx):
                        row_list[-1] = -(-row_list[-1]//2)
        dum_col_list =[4-col for col in col_list]
        row_list = row_list[diff_idx:][::-1] + \
                        [row_list[1], row_list[0]] + row_list[2:diff_idx] + row_list[diff_idx:]
        col_list = col_list[diff_idx:][::-1] + \
                        [col_list[1], col_list[0]] + col_list[2:diff_idx] + col_list[diff_idx:]
        dum_col_list = dum_col_list[diff_idx:][::-1] + \
                            [dum_col_list[1], dum_col_list[0]] + dum_col_list[2:diff_idx] + dum_col_list[diff_idx:]

        # Place control signals
        conn_layer = MOSArrayPlaceInfo.get_conn_layer(self.grid.tech_info,
                                                      self.params['pinfo']['tile_specs']['arr_info']['lch'])
        hm_layer = conn_layer + 1
        vm_layer = hm_layer + 1
        xm_layer = vm_layer + 1
        ym_layer = xm_layer + 1

        tr_w_sup_ym = tr_manager.get_width(ym_layer, 'sup')

        # -- first track --
        tr_w_ctrl_vm = tr_manager.get_width(vm_layer, 'ctrl')

        ctrl_tidx_start = grid.find_next_track(vm_layer, 0, tr_width=tr_w_ctrl_vm)
        ctrl_tidx_used, ctrl_tidx_locs = \
            tr_manager.place_wires(vm_layer, ['ctrl'] * (3 * nbits + 1), align_idx=0, align_track=ctrl_tidx_start)
        sw_x = self.grid.track_to_coord(vm_layer, ctrl_tidx_used)
        routing_bnd = sw_x

        # Setup templates for size calculation
        if (cap_config['ismim'] == True):
            cap_config_mim = copy.deepcopy(cap_config.to_dict())
            cap_config_mim['total_width'] = self.params['width']
            cap_config_mim['num_rows'] = sum(row_list) #+sum(bit_list)*cap_config['cap_sp']
            cap_config_mim['num_cols'] = max(col_list)
            cap_config_mim['dum_col_l'] = min(dum_col_list)
            cap_master = self.new_template(CapColCore, params=dict(cap_config=cap_config_mim, ny=4 * sum(ny_list)))
            unit_cap_master = self.new_template(CapColCore, params=dict(cap_config=cap_config_mim, ny=4))
        else:
            cap_master = self.new_template(CapColCore, params=dict(cap_config=cap_config, width=width, ny=4 * sum(ny_list)))
            unit_cap_master = self.new_template(CapColCore, params=dict(cap_config=cap_config, width=width, ny=4))
        
        unit_cap_height = unit_cap_master.array_box.h
        w_cap, h_cap = cap_master.bound_box.w, cap_master.bound_box.h
        sw_params = dict(
            cls_name=CapDrvCore.get_qualified_name(),
            params=dict(pinfo=self.params['pinfo'], seg=seg, ny=sum(ny_list), w=w_n, sp=sp,
                        dum_row_idx=[sum(ny_list[:nbits - diff_idx + 1]) + 1], min_height=unit_cap_height)
        )
        sw_master = self.new_template(GenericWrapper, params=sw_params)
        top_layer = max(cap_master.top_layer, sw_master.top_layer)
        w_blk, h_blk = self.grid.get_block_size(top_layer)
        cm_sw_params = dict(pinfo=self.params['pinfo_cm'], seg=seg_cm, w=w_cm)
        cm_sw_dum_master = self.new_template(CMSwitch, params=cm_sw_params)
        w_master = cm_sw_dum_master.bound_box.w
        w_edge = cm_sw_dum_master.tech_cls.get_edge_width(w_master, w_blk)
        ncols_tot = (cap_master.bound_box.w - 2 * w_edge) // cm_sw_dum_master.sd_pitch
        cm_sw_gen_params = dict(
            cls_name=CMSwitch.get_qualified_name(),
            params=dict(pinfo=self.params['pinfo_cm'], seg=seg_cm, w=w_cm,
                        ncols_tot=ncols_tot - (ncols_tot & 1))
        )
        
        cm_sw_master = self.new_template(GenericWrapper, params=cm_sw_gen_params)
        y_cm_sw_top = -(-cm_sw_master.bound_box.h // h_blk) * h_blk

        capmim_y = []
        if (cap_config['ismim']):
            if sw_type.count('n') == 3:
                w_sw, h_sw = sw_master.bound_box.w, sw_master.bound_box.h
                sw_y = y_cm_sw_top if has_cm_sw else 0
                cap_y = sw_y + (h_sw - h_cap) // 2
                sw_x = -(-sw_x // w_blk) * w_blk
                sw = self.add_instance(sw_master, inst_name='XSW', xform=Transform(sw_x, sw_y))
                # Get sorted ctrl pins
                sw_ctrl_n: List[Union[WireArray, None]] = sw.get_all_port_pins('ctrl0')
                sw_ctrl_m: List[Union[WireArray, None]] = sw.get_all_port_pins('ctrl1')
                sw_ctrl_p: List[Union[WireArray, None]] = sw.get_all_port_pins('ctrl2')
                sw_bot: List[Union[WireArray, None]] = sw.get_all_port_pins('bot')
                self.reexport(sw.get_port('vref0'), net_name='vref<2>')
                self.reexport(sw.get_port('vref1'), net_name='vref<1>')
                self.reexport(sw.get_port('vref2'), net_name='vref<1>')
                vrefm, vrefm_pin = sw.get_port('vref1'), sw.get_pin('vref1')
                sw_right_coord = sw.bound_box.xh
                sw_params_list = [sw_master.sch_params for _ in range(nbits)]
                sw_vss_bbox: List[BBox] = sw.get_all_port_pins('VSS')
            elif sw_type.count('n') == 2:
                sw_n_params = dict(
                    cls_name=CapDrvCore.get_qualified_name(),
                    draw_taps=True,
                    params=dict(pinfo=self.params['pinfo'], seg=seg, ny=sum(ny_list), w=w_n, sp=sp, nx=2, sw_type='nch',
                                dum_row_idx=[sum(ny_list[:nbits - diff_idx + 1]) + 1], min_height=unit_cap_height)
                )
                sw_p_params = dict(
                    cls_name=CapDrvCore.get_qualified_name(),
                    draw_taps=True,
                    params=dict(pinfo=self.params['pinfo'], seg=seg, ny=sum(ny_list), w=w_p, sp=sp, nx=1, sw_type='pch',
                                dum_row_idx=[sum(ny_list[:nbits - diff_idx + 1]) + 1], min_height=unit_cap_height)
                )
                sw_n_master = self.new_template(GenericWrapper, params=sw_n_params)
                sw_p_master = self.new_template(GenericWrapper, params=sw_p_params)
                # back = self.add_instance(sw_p_master)
                top_layer = max(cap_master.top_layer, sw_n_master.top_layer, sw_p_master.top_layer)
                w_blk, h_blk = self.grid.get_block_size(top_layer)
                w_sw_p, h_sw = sw_p_master.bound_box.w, sw_p_master.bound_box.h
                w_sw_n, h_sw = sw_n_master.bound_box.w, sw_n_master.bound_box.h 
                sw_y = (y_cm_sw_top + h_blk) if has_cm_sw else 0
                cap_y = sw_y #+ (h_sw - h_cap) // 2
                capmim_y.append(cap_y)
                sw_x = -(-sw_x // w_blk) * w_blk

                sw_type_dict = dict(
                    XN=sw_type[0],
                    XM=sw_type[1],
                    XP=sw_type[2],
                )
                sw_params_list = [sw_n_master.sch_params.copy(append=dict(sw_type_dict=sw_type_dict)) for _ in range(nbits)]
                swn_x = -(-(sw_x + w_sw_p) // w_blk) * w_blk 
 
                # Get sorted ctrl pins
                sw_ctrl_m: List[Union[WireArray, None]] = [] 
                sw_ctrl_n: List[Union[WireArray, None]] = [] 
                sw_ctrl_p: List[Union[WireArray, None]] = [] 
                sw_bot = [] # List[Union[WireArray, None]] = []
                vref0_xm = []
                vref1_xm = []
                vref2_xm = []
                vrefm_single, vrefm_pin_single = [], []
                vdd_xm, vss_xm = [], []
                vdd_vm, vss_vm = [], []

                cap_config_dum = copy.deepcopy(cap_config.to_dict()) 
                for idx, (ny, row, bit) in enumerate(zip(ny_list, row_list, bit_list)):
                    cap_config_dum['num_rows'] = row_list[idx]
                    cap_config_dum['num_cols'] = col_list[idx]
                    cap_config_dum['dum_col_l'] = dum_col_list[idx]
                    cap_master = self.new_template(CapColCore, params=dict(cap_config=cap_config_dum, ny=4,
                                                                       ratio=1))

                    dums = ny                   
                    unit_cap_height = cap_master.array_box.yh // ny #int(h/self.grid.resolution) // ny
                    sw_n_params = dict(
                    cls_name=CapDrvCore.get_qualified_name(),
                    draw_taps=True,
                    params=dict(pinfo=self.params['pinfo'], seg=seg, ny=ny, w=w_n, sp=sp, nx=2, sw_type='nch',
                                dum_row_idx=[dums], min_height=unit_cap_height)
                    )
                    if idx == 0:
                        dums = sum(ny_list)
                    sw_p_params = dict(
                        cls_name=CapDrvCore.get_qualified_name(),
                        draw_taps=True,
                        params=dict(pinfo=self.params['pinfo'], seg=seg, ny=ny, w=w_p, sp=sp, nx=1, sw_type='pch',
                                dum_row_idx=[dums], min_height=unit_cap_height)
                    )
                    sw_n_master = self.new_template(GenericWrapper, params=sw_n_params)
                    sw_p_master = self.new_template(GenericWrapper, params=sw_p_params)

                    unit_drv = 0
                    if(bit>0):
                        sw_p = self.add_instance(sw_p_master, inst_name='XSWP', xform=Transform(sw_x, sw_y))
                        sw_n = self.add_instance(sw_n_master, inst_name='XSWN', xform=Transform(swn_x, sw_y))

                        sw_ctrl_m = sw_ctrl_m + sw_n.get_all_port_pins('ctrl0')
                        sw_ctrl_n = sw_ctrl_n + sw_n.get_all_port_pins('ctrl1')
                        sw_ctrl_p = sw_ctrl_p + sw_p.get_all_port_pins('ctrl0')

                        self.reexport(sw_n.get_port('vref0'), net_name='vref<1>')
                        self.reexport(sw_n.get_port('vref1'), net_name='vref<0>')
                        self.reexport(sw_p.get_port('vref0'), net_name='vref<2>')

                        vref0_xm = vref0_xm + sw_n.get_all_port_pins('vref0_xm')
                        vref1_xm = vref1_xm + sw_n.get_all_port_pins('vref1_xm')
                        vref2_xm = vref2_xm + sw_p.get_all_port_pins('vref0_xm')

                        sw_right_coord = sw_n.bound_box.xh
                    
                        for botn, botp in zip(sw_n.get_all_port_pins('bot'), sw_p.get_all_port_pins('bot')):
                            sw_bot.append(self.connect_wires([botn, botp])[0]) #FIXME
                        vrefm_single.append(sw_n.get_port('vref0'))
                        vrefm_pin_single.append(sw_n.get_pin('vref0'))
                        #vrefm_single.append(sw_n.get_pin('vref0'))

                        self.reexport(sw_p.get_port('VDD'), connect=True)
                        self.reexport(sw_n.get_port('VSS'), connect=True)
                        vdd_vm = vdd_vm + sw_p.get_all_port_pins('VDD')
                        vss_vm = vss_vm + sw_n.get_all_port_pins('VSS')
                        vdd_xm_ = sw_p.get_all_port_pins('VDD_xm')
                        vss_xm_ = sw_n.get_all_port_pins('VSS_xm')
                        vdd_xm = vdd_xm + vdd_xm_
                        if (bit==1):
                            unit_drv = sw_n.bound_box.yh-sw_n.bound_box.yl
                    sw_y = int(sw_y + max( (sw_n.bound_box.yh-sw_n.bound_box.yl), unit_drv,
                                    int(cap_master.array_box.yh) ))
                    sw_y = -(-sw_y//h_blk)*h_blk 
                    capmim_y.append(sw_y)
                
                # Ports on vertical layers are returned as boxes
                vref1_tidx = self.grid.coord_to_track(vm_layer, -(-vref0_xm[0].xl//(w_blk//2))*w_blk//2)
                self.add_rect_array((f'met{vm_layer}', 'drawing'), BBox(vref0_xm[0].xl, vref0_xm[0].yl, vref0_xm[-1].xh, vref0_xm[-1].yh))
                self.add_rect_array((f'met{vm_layer}', 'drawing'), BBox(vref1_xm[0].xl, vref1_xm[0].yl, vref1_xm[-1].xh, vref1_xm[-1].yh))
                self.add_rect_array((f'met{vm_layer}', 'drawing'), BBox(vref2_xm[0].xl, vref2_xm[0].yl, vref2_xm[-1].xh, vref2_xm[-1].yh))

                for n in range(4):  #always have 4 columns of VSS and 4 columns of VDD
                    self.add_rect_array((f'met{vm_layer}', 'drawing'), BBox(vss_vm[n].xl, vss_vm[n].yl, 
                                                            vss_vm[len(vss_vm)-4+n].xh, vss_vm[len(vss_vm)-4+n].yh))
                    self.add_rect_array((f'met{vm_layer}', 'drawing'), BBox(vdd_vm[n].xl, vdd_vm[n].yl, 
                                                            vdd_vm[len(vdd_vm)-4+n].xh, vdd_vm[len(vdd_vm)-4+n].yh))
                

                if not self.params['lower_layer_routing']:
                    vref_ym_list = []
                    for vref in [vref0_xm, vref1_xm, vref2_xm]:
                        mid_coord = vref[0].middle
                        tid = self.grid.coord_to_track(ym_layer, mid_coord, RoundMode.NEAREST)
                        vref_ym_list.append(self.connect_to_tracks(vref, TrackID(ym_layer, tid, tr_w_sup_ym)))
                    ym_sup_locs = self.get_available_tracks(ym_layer, self.grid.coord_to_track(ym_layer, sw_p.bound_box.xl,
                                                                                               RoundMode.NEAREST),
                                                            self.grid.coord_to_track(ym_layer, sw_n.bound_box.xh,
                                                                                     RoundMode.LESS_EQ),
                                                            upper=sw_n.bound_box.yh, lower=sw_n.bound_box.yl,
                                                            width=tr_w_sup_ym,
                                                            sep=tr_manager.get_sep(ym_layer, ('sup', 'sup')))
                    vdd_ym_list, vss_ym_list = [], []
                    xm_vdd_ret_list, xm_vss_ret_list = [], []
                    for tid in ym_sup_locs[::2]:
                        vdd_ym_list.append(
                            self.connect_to_tracks(vdd_xm, TrackID(ym_layer, tid, tr_w_sup_ym), ret_wire_list=xm_vdd_ret_list))
                    for tid in ym_sup_locs[1::2]:
                        vss_ym_list.append(
                            self.connect_to_tracks(vss_xm, TrackID(ym_layer, tid, tr_w_sup_ym), ret_wire_list=xm_vss_ret_list))

                    xm_sup_list = xm_vdd_ret_list + xm_vss_ret_list
                    xm_sup_max_coord, xm_sup_min_coord = max([x.upper for x in xm_sup_list]), \
                                                         min([x.lower for x in xm_sup_list])
                    self.extend_wires(xm_sup_list, upper=xm_sup_max_coord, lower=xm_sup_min_coord)
                    ym_sup_list = vdd_ym_list + vss_ym_list
                    ym_sup_max_coord, ym_sup_min_coord = max([y.upper for y in ym_sup_list]), 0

                    vdd_ym_list = self.extend_wires(vdd_ym_list, upper=ym_sup_max_coord, lower=ym_sup_min_coord)
                    vss_ym_list = self.extend_wires(vss_ym_list, upper=ym_sup_max_coord, lower=ym_sup_min_coord)
            else:
                sw_n, sw_p = None, None
                raise NotImplementedError
        else:
            if sw_type.count('n') == 3:
                w_sw, h_sw = sw_master.bound_box.w, sw_master.bound_box.h
                sw_y = y_cm_sw_top if has_cm_sw else 0
                cap_y = sw_y + (h_sw - h_cap) // 2
                sw_x = -(-sw_x // w_blk) * w_blk
                sw = self.add_instance(sw_master, inst_name='XSW', xform=Transform(sw_x, sw_y))
                # Get sorted ctrl pins
                sw_ctrl_n: List[Union[WireArray, None]] = sw.get_all_port_pins('ctrl0')
                sw_ctrl_m: List[Union[WireArray, None]] = sw.get_all_port_pins('ctrl1')
                sw_ctrl_p: List[Union[WireArray, None]] = sw.get_all_port_pins('ctrl2')
                sw_bot: List[Union[WireArray, None]] = sw.get_all_port_pins('bot')
                self.reexport(sw.get_port('vref0'), net_name='vref<2>')
                self.reexport(sw.get_port('vref1'), net_name='vref<1>')
                self.reexport(sw.get_port('vref2'), net_name='vref<1>')
                vrefm, vrefm_pin = sw.get_port('vref1'), sw.get_pin('vref1')
                sw_right_coord = sw.bound_box.xh
                sw_params_list = [sw_master.sch_params for _ in range(nbits)]
                sw_vss_bbox: List[BBox] = sw.get_all_port_pins('VSS')
            elif sw_type.count('n') == 2:
                sw_n_params = dict(
                    cls_name=CapDrvCore.get_qualified_name(),
                    draw_taps=True,
                    params=dict(pinfo=self.params['pinfo'], seg=seg, ny=sum(ny_list), w=w_n, sp=sp, nx=2, sw_type='nch',
                                dum_row_idx=[sum(ny_list[:nbits - diff_idx + 1]) + 1], min_height=unit_cap_height)
                )
                sw_p_params = dict(
                    cls_name=CapDrvCore.get_qualified_name(),
                    draw_taps=True,
                    params=dict(pinfo=self.params['pinfo'], seg=seg, ny=sum(ny_list), w=w_p, sp=sp, nx=1, sw_type='pch',
                                dum_row_idx=[sum(ny_list[:nbits - diff_idx + 1]) + 1], min_height=unit_cap_height)
                )
                sw_n_master = self.new_template(GenericWrapper, params=sw_n_params)
                sw_p_master = self.new_template(GenericWrapper, params=sw_p_params)
                top_layer = max(cap_master.top_layer, sw_n_master.top_layer, sw_p_master.top_layer)
                w_blk, h_blk = self.grid.get_block_size(top_layer)
                w_sw_p, h_sw = sw_p_master.bound_box.w, sw_p_master.bound_box.h
                w_sw_n, h_sw = sw_n_master.bound_box.w, sw_n_master.bound_box.h
                sw_y = y_cm_sw_top if has_cm_sw else 0
                cap_y = sw_y + (h_sw - h_cap) // 2
                sw_x = -(-sw_x // w_blk) * w_blk
                sw_p = self.add_instance(sw_p_master, inst_name='XSWP', xform=Transform(sw_x, sw_y))
                sw_x = -(-(sw_x + w_sw_p) // w_blk) * w_blk
                sw_n = self.add_instance(sw_n_master, inst_name='XSWN', xform=Transform(sw_x, sw_y))
                # Get sorted ctrl pins
                sw_ctrl_m: List[Union[WireArray, None]] = sw_n.get_all_port_pins('ctrl0')
                sw_ctrl_n: List[Union[WireArray, None]] = sw_n.get_all_port_pins('ctrl1')
                sw_ctrl_p: List[Union[WireArray, None]] = sw_p.get_all_port_pins('ctrl0')
                sw_bot: List[Union[WireArray, None]] = []
                for botn, botp in zip(sw_n.get_all_port_pins('bot'), sw_p.get_all_port_pins('bot')):
                    sw_bot.append(self.connect_wires([botn, botp])[0])
                self.reexport(sw_n.get_port('vref0'), net_name='vref<1>')
                self.reexport(sw_n.get_port('vref1'), net_name='vref<0>')
                self.reexport(sw_p.get_port('vref0'), net_name='vref<2>')
                vref0_xm = sw_n.get_all_port_pins('vref0_xm')
                vref1_xm = sw_n.get_all_port_pins('vref1_xm')
                vref2_xm = sw_p.get_all_port_pins('vref0_xm')

                vrefm, vrefm_pin = sw_n.get_port('vref0'), sw_n.get_pin('vref0')
                sw_right_coord = sw_n.bound_box.xh
                sw_type_dict = dict(
                    XN=sw_type[0],
                    XM=sw_type[1],
                    XP=sw_type[2],
                )
                sw_params_list = [sw_n_master.sch_params.copy(append=dict(sw_type_dict=sw_type_dict)) for _ in range(nbits)]
                self.reexport(sw_p.get_port('VDD'), connect=True)
                vdd_xm = sw_p.get_all_port_pins('VDD_xm')
                vss_xm = sw_n.get_all_port_pins('VSS_xm')
                vdd_xm = self.extend_wires(vdd_xm, upper=vss_xm[0].upper)
                vss_xm = self.extend_wires(vss_xm, lower=vdd_xm[0].lower)

                if not self.params['lower_layer_routing']:
                    vref_ym_list = []
                    for vref in [vref0_xm, vref1_xm, vref2_xm]:
                        mid_coord = vref[0].middle
                        tid = self.grid.coord_to_track(ym_layer, mid_coord, RoundMode.NEAREST)
                        vref_ym_list.append(self.connect_to_tracks(vref, TrackID(ym_layer, tid, tr_w_sup_ym)))
                    ym_sup_locs = self.get_available_tracks(ym_layer, self.grid.coord_to_track(ym_layer, sw_p.bound_box.xl,
                                                                                               RoundMode.NEAREST),
                                                            self.grid.coord_to_track(ym_layer, sw_n.bound_box.xh,
                                                                                     RoundMode.LESS_EQ),
                                                            upper=sw_n.bound_box.yh, lower=sw_n.bound_box.yl,
                                                            width=tr_w_sup_ym,
                                                            sep=tr_manager.get_sep(ym_layer, ('sup', 'sup')))
                    vdd_ym_list, vss_ym_list = [], []
                    xm_vdd_ret_list, xm_vss_ret_list = [], []
                    for tid in ym_sup_locs[::2]:
                        vdd_ym_list.append(
                            self.connect_to_tracks(vdd_xm, TrackID(ym_layer, tid, tr_w_sup_ym), ret_wire_list=xm_vdd_ret_list))
                    for tid in ym_sup_locs[1::2]:
                        vss_ym_list.append(
                            self.connect_to_tracks(vss_xm, TrackID(ym_layer, tid, tr_w_sup_ym), ret_wire_list=xm_vss_ret_list))

                    xm_sup_list = xm_vdd_ret_list + xm_vss_ret_list
                    xm_sup_max_coord, xm_sup_min_coord = max([x.upper for x in xm_sup_list]), \
                                                         min([x.lower for x in xm_sup_list])
                    self.extend_wires(xm_sup_list, upper=xm_sup_max_coord, lower=xm_sup_min_coord)
                    ym_sup_list = vdd_ym_list + vss_ym_list
                    ym_sup_max_coord, ym_sup_min_coord = max([y.upper for y in ym_sup_list]), 0

                    vdd_ym_list = self.extend_wires(vdd_ym_list, upper=ym_sup_max_coord, lower=ym_sup_min_coord)
                    vss_ym_list = self.extend_wires(vss_ym_list, upper=ym_sup_max_coord, lower=ym_sup_min_coord)
            else:
                sw_n, sw_p = None, None
                raise NotImplementedError
        # Place input signal
        tr_w_sig_vm = tr_manager.get_width(vm_layer, 'sig')
        tr_sp_sig_vm = tr_manager.get_sep(vm_layer, ('sig', 'sig'))
        sig_tidx_start = grid.find_next_track(vm_layer, sw_right_coord, tr_width=tr_w_sig_vm)
        sig_tidx_used, sig_tidx_locs = tr_manager.place_wires(vm_layer, ['sig'] * nbits*2, align_idx=0,
                                                              align_track=sig_tidx_start)
        sig_tidx_used, sig_tidx_locs = tr_manager.place_wires(vm_layer, ['sig'] * nbits*2 + ['cap'], align_idx=0,
                                                              align_track=sig_tidx_start)
        cap_x = self.grid.track_to_coord(vm_layer, sig_tidx_locs[-1])

        cap_x = -(-cap_x // w_blk) * w_blk
        cap_config_copy = copy.deepcopy(cap_config.to_dict())

        cap_list = []
        cap_master_list = [cap_master] * (nbits + 1)

        if (cap_config['ismim'] == True):
            cap_ext_x = []
            max_pin = 0
            for idx in range(0, len(bit_list)):
                cap_config_copy['num_rows'] = row_list[idx]
                cap_config_copy['num_cols'] = col_list[idx]
                cap_config_copy['dum_col_l'] = dum_col_list[idx]
                cap_master = self.new_template(CapColCore, params=dict(cap_config=cap_config_copy, ny=4,
                                                                       ratio=1))
                cap_master_list[bit_list[idx]] = cap_master  
                cap = self.add_instance(cap_master, inst_name='XCAP', xform=Transform(cap_x, -(-capmim_y[idx] // h_blk) * h_blk))
                cap_list.append(cap)
                cap_ext_x.append(cap.array_box.xl)
                cap_y += cap_master.array_box.yh
            
            # Get cap dac pins
            cap_bot = [pin for inst in cap_list for pin in inst.get_all_port_pins('top')]

        else:
            for idx, (ny, ratio) in enumerate(zip(ny_list, ratio_list)):
                cap_master = self.new_template(CapColCore, params=dict(cap_config=cap_config_copy, width=width, ny=4 * ny,
                                                                       ratio=ratio))
                cap_master_list[bit_list[idx]] = cap_master
                cap = self.add_instance(cap_master, inst_name='XCAP', xform=Transform(cap_x, -(-cap_y // h_blk) * h_blk))
                cap_list.append(cap)
                cap_y += cap_master.array_box.h
            
            # Get cap dac pins
            cap_bot = [pin for inst in cap_list for pin in inst.get_all_port_pins('top')]

            #sort by track_id.base_index
            cap_bot.sort(key=lambda x: x.track_id.base_index)

        ntr_margin = self.grid.get_sep_tracks(vm_layer, tr_manager.get_width(vm_layer, 'sup'),
                                              cap_list[0].get_pin('top').track_id.width)
        coord_margin = self.grid.track_to_coord(vm_layer, ntr_margin)
        if self.params['lower_layer_routing']:
            cm_sw_x = cap_x-coord_margin
            cm_sw_x = -(-cm_sw_x//w_blk)*w_blk
        else:
            cm_sw_x = cap_x

        if has_cm_sw:
            cm_sw = self.add_instance(cm_sw_master, inst_name='XSW_CM', xform=Transform(cm_sw_x, 0))

        # left space for clock routing
        num_tr, _ = tr_manager.place_wires(vm_layer, ['cap', 'clk', 'clk'], align_idx=0)
        coord_tr = self.grid.track_to_coord(vm_layer, num_tr)

        w_tot = -(-(cap_x + w_cap + coord_tr) // w_blk) * w_blk
        h_tot = -(-max(cap.bound_box.yh, sw_n.bound_box.yh) // h_blk) * h_blk
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot))

        for pin_list in [sw_ctrl_m, sw_ctrl_n, sw_ctrl_p]:
            pin_list.sort(key=lambda x: x.track_id.base_index)

        # Get sorted bottom pin
        sw_bot.sort(key=lambda x: x.track_id.base_index)

        #making the dummy cap not connectable
        for idx in [sum(ny_list[:nbits - diff_idx + 1]) + 1]:
            sw_bot.insert(idx, None)
            sw_ctrl_m.insert(idx, None)
            sw_ctrl_n.insert(idx, None)
            sw_ctrl_p.insert(idx, None)

        tr_w_cap_hm = tr_manager.get_width(hm_layer, 'cap')
        tr_w_cap_vm = tr_manager.get_width(vm_layer, 'cap')
        tr_w_cap_xm = tr_manager.get_width(xm_layer, 'cap')    
        if (len(sw_bot) > len(cap_bot)):
            cap_bot_copy = cap_bot
            ext_cap = []
            for idx in range(0, len(cap_bot_copy)):
                ext_cap = ext_cap + [cap_bot_copy[idx] for n in range(0, ny_list[idx])]
            cap_bot = ext_cap
        for _sw, _cap in zip(sw_bot, cap_bot):
            if _sw and _cap:
                lay_diff = _sw.track_id.layer_id - _cap.track_id.layer_id
                if abs(lay_diff) >1:
                    self.extend_wires(_sw, upper=_cap.upper)
                    vm_tidx = self.grid.coord_to_track(_sw.track_id.layer_id+1, 
                                               -(((-(_cap.upper+_cap.lower)//2)//(w_blk//2)))*(w_blk//2))
                    vm_conn = self.connect_to_tracks(_sw, TrackID(_sw.track_id.layer_id+1, vm_tidx, tr_w_cap_hm))
                    # xm_tidx = self.grid.coord_to_track(hm_conn.track_id.layer_id+1, 
                    #                            -(-((hm_conn.upper+hm_conn.lower)//2)//86)*86)
                    # xm_conn = self.connect_to_tracks(hm_conn, TrackID(hm_conn.track_id.layer_id+1, xm_tidx, tr_w_cap_hm))
                    self.connect_to_track_wires(vm_conn, _cap)
                else:
                    self.connect_to_track_wires(_sw, _cap) 

        
        # cap top  
        #cap_top = self.connect_wires([pin for inst in cap_list for pin in inst.get_all_port_pins('bot')], upper=-(-(width//w_blk)*w_blk))
        top_layer = cap_list[0].get_port('bot').get_single_layer()
        cap_top = self.add_wires(top_layer, 
                                self.grid.coord_to_track(top_layer, cap_list[0].get_port('bot').get_bounding_box().xm), 
                                lower= cap_list[0].get_port('bot').get_bounding_box().yl, 
                                upper= cap_list[-1].get_port('bot').get_bounding_box().yh, width=4)

        # Connect to common-mode switch
        if has_cm_sw:
            if (cap_config['ismim']):
                for (vrefm, vrefm_pin) in zip(vrefm_single, vrefm_pin_single):
                    self.connect_bbox_to_track_wires(Direction.LOWER, (vrefm.get_single_layer(), 'drawing'),
                                            vrefm_pin, cm_sw.get_all_port_pins('ref'))
                vrefm = vrefm_single[0]
                vrefm_pin = vrefm_pin_single[0]
            else:
                self.connect_bbox_to_track_wires(Direction.LOWER, (vrefm.get_single_layer(), 'drawing'),
                                            vrefm_pin, cm_sw.get_all_port_pins('ref'))

        if has_cm_sw:
            if(cap_top[0].layer_id > 4):
                pins = cm_sw.get_all_port_pins('sig')
                wire = self.add_wires(pins[0].track_id.layer_id, pins[0].track_id.base_index, lower=(cap_list[0].array_box.xh-w_blk),
                                        upper=cap_list[0].array_box.xh, width=6)
                self.extend_wires(cm_sw.get_all_port_pins('sig'), upper=cap_list[0].array_box.xh)
                self.connect_to_track_wires(wire, cap_top)
            else:
                self.connect_to_track_wires(cm_sw.get_all_port_pins('sig'), cap_top)

        # Group pins for each bit
        ctrl_bit_temp = dict(
            ctrl_m=[],
            ctrl_n=[],
            ctrl_p=[],
        )
        bit_pin_dict_list = [copy.deepcopy(ctrl_bit_temp) for _ in range(nbits)]
        bit_cap_list_list = [copy.deepcopy([]) for _ in range(nbits)]
        for idx, bit_idx in enumerate(bit_list):
            start_idx, stop_idx = sum(ny_list[:idx]), sum(ny_list[:idx + 1])
            if bit_idx:
                bit_pin_dict_list[bit_idx - 1]['ctrl_m'].extend(sw_ctrl_m[start_idx: stop_idx])
                bit_pin_dict_list[bit_idx - 1]['ctrl_n'].extend(sw_ctrl_n[start_idx: stop_idx])
                bit_pin_dict_list[bit_idx - 1]['ctrl_p'].extend(sw_ctrl_p[start_idx: stop_idx])
                bit_cap_list_list[bit_idx - 1].extend(sw_bot[start_idx: stop_idx])

        # Connect control signal to vm-layer
        ctrl_hm_ret_list = []
        ctrl_m_vm_list, ctrl_n_vm_list, ctrl_p_vm_list = [], [], []
        for idx in range(nbits):
            _bit_pins = bit_pin_dict_list[idx]
            ctrl_m_vm_list.append(self.connect_to_tracks(_bit_pins['ctrl_m'],
                                                         TrackID(vm_layer, ctrl_tidx_locs[3 * idx], tr_w_ctrl_vm),
                                                         track_lower=self.bound_box.yl, ret_wire_list=ctrl_hm_ret_list))
            ctrl_n_vm_list.append(self.connect_to_tracks(_bit_pins['ctrl_n'],
                                                         TrackID(vm_layer, ctrl_tidx_locs[3 * idx + 1], tr_w_ctrl_vm),
                                                         track_lower=self.bound_box.yl, ret_wire_list=ctrl_hm_ret_list))
            ctrl_p_vm_list.append(self.connect_to_tracks(_bit_pins['ctrl_p'],
                                                         TrackID(vm_layer, ctrl_tidx_locs[3 * idx + 2], tr_w_ctrl_vm),
                                                         track_lower=self.bound_box.yl, ret_wire_list=ctrl_hm_ret_list))
        ctrl_hm_ret_min_coord, ctrl_hm_ret_max_coord = min([x.lower for x in ctrl_hm_ret_list]), \
                                                       max([x.upper for x in ctrl_hm_ret_list])
        self.extend_wires(ctrl_hm_ret_list, lower=ctrl_hm_ret_min_coord, upper=ctrl_hm_ret_max_coord)

        cap_cm_list = cap_bot[sum(ny_list[:nbits - diff_idx + 1]) + 1: sum(ny_list[:nbits - diff_idx + 1]) + 2]

        for _cap_cm in cap_cm_list:

            if has_cm_sw:
                coord = self.grid.track_to_coord(_cap_cm.track_id.layer_id, _cap_cm.track_id.base_index)
                hm_idx = self.grid.coord_to_track(hm_layer, coord)
                conn = self.connect_bbox_to_tracks(Direction.UPPER, (vrefm.get_single_layer(), 'drawing'),
                                             vrefm_pin, TrackID(hm_layer, hm_idx, 1))
                self.extend_wires(conn, upper=_cap_cm.upper)
                hor_tidx = self.grid.coord_to_track(conn.track_id.layer_id+1, 
                                               -(-(((_cap_cm.upper+_cap_cm.lower)//2)//(w_blk//2)))*(w_blk//2))
                hm_conn = self.connect_to_tracks(conn, TrackID(conn.track_id.layer_id+1, hor_tidx, tr_w_cap_hm))
                self.connect_to_track_wires(hm_conn, _cap_cm)

            elif (not has_cm_sw and cap_config['ismim']==True):
                coord = self.grid.track_to_coord(_cap_cm.track_id.layer_id, _cap_cm.track_id.base_index)
                vm_idx = self.grid.coord_to_track(vm_layer, -(-(((_cap_cm.upper+_cap_cm.lower)//2)//(w_blk//2)))*(w_blk//2))
                conn = self.connect_to_tracks(_cap_cm, TrackID(vm_layer, vm_idx, 1)) 
                hor_tidx = self.grid.coord_to_track(conn.track_id.layer_id-1, coord)
                hm_conn = self.connect_to_tracks(conn, TrackID(conn.track_id.layer_id-1, hor_tidx, tr_w_cap_hm))
                self.connect_to_tracks(hm_conn, TrackID(vm_layer, vref1_tidx, tr_w_cap_hm))

            else:
                self.connect_bbox_to_track_wires(Direction.UPPER, (vrefm_single.get_single_layer(), 'drawing'),
                                             vrefm_pin_single[0],_cap_cm)

        bot_vm_list: List[WireArray] = []
        for idx in range(nbits):
            bot_vm_list.append(self.connect_to_tracks(bit_cap_list_list[idx],
                                                      TrackID(vm_layer, sig_tidx_locs[idx*2]+1, tr_w_sig_vm),
                                                      track_upper=self.bound_box.yh))
        bot_vm_list_bot_coord = sw_y
        if (cap_config['ismim']):
            bot_vm_list = self.extend_wires(bot_vm_list, lower=(-(-capmim_y[0] // h_blk) * h_blk))
        else:
            bot_vm_list = self.extend_wires(bot_vm_list, lower=bot_vm_list_bot_coord)
        for idx, bot_wire in enumerate(bot_vm_list):
            self.add_pin(f'bot<{idx}>', bot_wire, mode=PinMode.UPPER)

        # flip n and p control, just because comparator output and differential ...
        ctrl_top_coord = max([c.upper for c in ctrl_n_vm_list + ctrl_p_vm_list + ctrl_m_vm_list])
        ctrl_n_vm_list = self.extend_wires(ctrl_n_vm_list, upper=ctrl_top_coord)
        ctrl_p_vm_list = self.extend_wires(ctrl_p_vm_list, upper=ctrl_top_coord)
        ctrl_m_vm_list = self.extend_wires(ctrl_m_vm_list, upper=ctrl_top_coord)

        for idx, (n, m, p) in enumerate(zip(ctrl_n_vm_list, ctrl_m_vm_list, ctrl_p_vm_list)):
            self.add_pin(f'ctrl_m<{idx}>', m, mode=PinMode.LOWER)
            self.add_pin(f'ctrl_n<{idx}>', p, mode=PinMode.LOWER)
            self.add_pin(f'ctrl_p<{idx}>', n, mode=PinMode.LOWER)

        if has_cm_sw:
            tr_sp_sig_cap_vm = tr_manager.get_sep(vm_layer, ('sig', 'cap'))
            vm_tidx_stop = self.grid.coord_to_track(vm_layer, cm_sw.bound_box.xh, mode=RoundMode.NEAREST)
            vm_tidx_start = self.grid.coord_to_track(vm_layer, cm_sw.bound_box.xl, mode=RoundMode.NEAREST)

        if not self.params['lower_layer_routing']:
            self.connect_to_track_wires(cm_sw.get_all_port_pins('VSS'), vss_ym_list)

        # TODO: fix VSS
        self.reexport(sw_n.get_port('VSS'), connect=True)
        if has_cm_sw: 
            self.reexport(cm_sw.get_port('sam'))
        # for vss_bbox in sw_vss_bbox + cm_bbox:
        #     self.add_pin_primitive('VSS', f'm{conn_layer}', vss_bbox, connect=True)

        self.add_pin('top', cap_top)

        m_list = [len(_l) for _l in bit_cap_list_list]
        sw_list = m_list
        m_list = [1 if d < diff_idx-1 else 2 for d in range(nbits)]
        unit_params_list = [master.sch_params for master in cap_master_list[1:]]

        self._actual_width = self.bound_box.w - routing_bnd
        self.sch_params = dict(
            sw_params_list=sw_params_list,
            unit_params_list=unit_params_list,
            cm_unit_params=cap_master_list[0].sch_params,
            bot_probe=True,
            cap_m_list=m_list,
            sw_m_list=sw_list,
            cm=ny_list[nbits - 1],
            cm_sw=cm_sw_master.sch_params, 
            has_cm_sw = has_cm_sw,
            remove_cap=self.params['remove_cap'],
        )

class CapMIMCore(TemplateBase):
    """MIMCap core
    Draw a layout has only metal and metal resistor in a rectangle
    Horizontal layer is "vertical_layer"
    Top and bottom is connected by "bot_layer"

    Parameters:
        top_w: width of middle horizontal layer
        bot_w: width of top/bot horizontal layer
        bot_y_w: width of vertical layer
        sp: space between top/bot and middle
        sp_le: line-end space between middle horizontal layer
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return xbase__mimcap
        # return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'cap_unit')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            cap_config='MIM cap configuration.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        ans = DeviceFill.get_default_param_values()
        ans.update(
            cap_config={},
        )
        return ans

    def draw_layout(self) -> None:
        grid = self.grid
        master = self.new_template(MIMCap, params=self.params['cap_config'])
        capMIM =self.add_instance(master, inst_name='XMIM')

        self.set_size_from_bound_box(self.params['cap_config']['top_layer'],
                     BBox(0,0, master.bound_box.w, master.bound_box.h))
        
        cap_config = self.params['cap_config']
        top_layer = max(cap_config['top_layer'], cap_config['bot_layer'])
        bot_layer = min(cap_config['top_layer'], cap_config['bot_layer'])
        # lay_top = capMIM.get_port('TOP').get_single_layer()
        # self.add_pin('minus', capMIM.get_pin('TOP'), show=True)
        # self.add_pin('plus', capMIM.get_pin('BOT'), show=True)

        minus_tidx = self.grid.coord_to_track(top_layer, (capMIM.get_pin('TOP').xm//86)*86)
        minus_pin = self.add_wires(top_layer, minus_tidx, capMIM.get_pin('TOP').yl, capMIM.get_pin('TOP').yh)

        plus_tidx = self.grid.coord_to_track(bot_layer, (capMIM.get_pin('BOT').ym//86)*86)
        plus_pin = self.add_wires(bot_layer, plus_tidx, capMIM.get_pin('BOT').xl, capMIM.get_pin('BOT').xh)

        self.add_pin('minus', minus_pin, show=True)
        self.add_pin('plus', plus_pin, show=True)

        # Set schematic parameters
        has_rmetal = cap_config.get('has_rmetal', True)
        if has_rmetal:
            res_top_box = capMIM.get_pin('TOP')
            res_bot_box = capMIM.get_pin('BOT')

        if 'cap' in cap_config and has_rmetal:
            self.sch_params = dict(
                res_plus=dict(layer=top_layer, w=res_top_box.h, l=res_top_box.w),
                res_minus=dict(layer=top_layer, w=res_bot_box.h, l=res_bot_box.w),
                cap=cap_config.get('unit', 1) * cap_config['cap']
            )
        elif 'cap' in cap_config:
            self.sch_params = dict(cap=cap_config.get('unit', 1) * cap_config['cap'])
        elif has_rmetal:
            self.sch_params = dict(
                res_plus=dict(layer=top_layer, w=res_top_box.h, l=res_top_box.w),
                res_minus=dict(layer=top_layer, w=res_bot_box.h, l=res_bot_box.w),
            )
        else:
            self.sch_params = dict(
                res_plus=None,
                res_minus=None
            )

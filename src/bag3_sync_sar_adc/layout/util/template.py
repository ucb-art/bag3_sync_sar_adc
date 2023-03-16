import abc
from os.path import exists
from typing import Any, Dict, Tuple
from typing import (
    Union, List, Optional
)

from bag.io import read_yaml
from bag.layout.routing import RoutingGrid
from bag.layout.routing.base import TrackID, WireArray, TrackManager
from bag.layout.template import TemplateDB, TemplateBase
from bag.typing import TrackType
from bag.util.immutable import Param
from bag.util.math import HalfInt
from pybag.core import BBox
from pybag.core import (
    COORD_MIN, COORD_MAX, PyTrackID
)
from pybag.enum import (
    MinLenMode, Orient2D, RoundMode, Direction
)


class TrackIDZL(TrackID):
    """A class that represents locations of track(s) on the routing grid.

    Parameters
    ----------
    layer_id : int
        the layer ID.
    track_idx : TrackType
        the smallest middle track index in the array.  Multiples of 0.5
    width : int
        width of one track in number of tracks.
    num : int
        number of tracks in this array.
    pitch : TrackType
        pitch between adjacent tracks, in number of track pitches.
    grid: Optional[RoutingGrid]
        the routing grid associated with this TrackID object.
    """

    def __init__(self, layer_id: int, track_idx: TrackType, width: int = 1, num: int = 1,
                 pitch: TrackType = 0, grid: Optional[RoutingGrid] = None) -> None:
        if num < 1:
            raise ValueError('TrackID must have 1 or more tracks.')

        if width > 0:
            PyTrackID.__init__(self, layer_id, int(round(2 * track_idx)), width, num,
                               int(round(2 * pitch)))
            self._grid = grid
        else:
            if exists('bag3_sar_vco_adc_data/grid.yaml'):
                proj_grid = read_yaml('bag3_sar_vco_adc_data/grid.yaml')
            else:
                proj_grid = dict(tr_widths=dict(layer_id=1), tr_spaces=dict(layer_id=0))
            num = abs(width)
            width = proj_grid['tr_widths'][layer_id]  # read from preset grid, if not successful, use 1
            pitch = grid.get_sep_tracks(layer_id, width, width) + proj_grid['tr_spaces'][layer_id]
            pitch.up_even(True)
            self._grid = grid
            totw = (num - 1) * pitch.dbl_value // 2
            PyTrackID.__init__(self, layer_id, int(round(2 * track_idx - totw)), width, num, int(round(2 * pitch)))


class TemplateBaseZL(TemplateBase, abc.ABC):

    def add_wires(self, layer_id: int, track_idx: TrackType, lower: int, upper: int, *,
                  width: int = 1, num: int = 1, pitch: TrackType = 1) -> WireArray:
        tid = TrackIDZL(layer_id, track_idx, width=width, num=num, pitch=pitch, grid=self._grid)
        warr = WireArray(tid, lower, upper)
        self._layout.add_warr(tid, lower, upper)
        self._use_color = True
        return warr

    def connect_matching_tracks(self, warr_list_list: List[Union[WireArray, List[WireArray]]],
                                tr_layer_id: int, tr_idx_list: List[TrackType], *,
                                width: int = 1,
                                track_lower: Optional[int] = None,
                                track_upper: Optional[int] = None,
                                min_len_mode: MinLenMode = MinLenMode.NONE
                                ) -> List[Optional[WireArray]]:
        # simple error checking
        num_tracks = len(tr_idx_list)  # type: int
        if num_tracks != len(warr_list_list):
            raise ValueError('Connection list parameters have mismatch length.')
        if num_tracks == 0:
            raise ValueError('Connection lists are empty.')

        if track_lower is None:
            track_lower = COORD_MAX
        if track_upper is None:
            track_upper = COORD_MIN

        wbounds = [[COORD_MAX, COORD_MIN], [COORD_MAX, COORD_MIN]]
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            tid = TrackIDZL(tr_layer_id, tr_idx, width=width, grid=self._grid)
            for warr in WireArray.wire_grp_iter(warr_list):
                cur_lay_id = warr.layer_id
                if cur_lay_id == tr_layer_id + 1:
                    wb_idx = 1
                elif cur_lay_id == tr_layer_id - 1:
                    wb_idx = 0
                else:
                    raise ValueError(
                        'WireArray layer {} cannot connect to layer {}'.format(cur_lay_id,
                                                                               tr_layer_id))

                bnds = self._layout.connect_warr_to_tracks(warr.track_id, tid,
                                                           warr.lower, warr.upper)
                wbounds[wb_idx][0] = min(wbounds[wb_idx][0], bnds[wb_idx][0])
                wbounds[wb_idx][1] = max(wbounds[wb_idx][1], bnds[wb_idx][1])
                track_lower = min(track_lower, bnds[1 - wb_idx][0])
                track_upper = max(track_upper, bnds[1 - wb_idx][1])

        # fix min_len_mode
        track_lower, track_upper = self.fix_track_min_length(tr_layer_id, width, track_lower,
                                                             track_upper, min_len_mode)
        # extend wires
        ans = []
        for warr_list, tr_idx in zip(warr_list_list, tr_idx_list):
            for warr in WireArray.wire_grp_iter(warr_list):
                wb_idx = (warr.layer_id - tr_layer_id + 1) // 2
                self._layout.add_warr(warr.track_id, wbounds[wb_idx][0], wbounds[wb_idx][1])

            cur_tid = TrackIDZL(tr_layer_id, tr_idx, width=width, grid=self._grid)
            warr = WireArray(cur_tid, track_lower, track_upper)
            self._layout.add_warr(cur_tid, track_lower, track_upper)
            ans.append(warr)

        self._use_color = True
        return ans

    def get_track_sep(self, layer, ntr1, ntr2) -> HalfInt:
        if ntr1 > 0 and ntr2 > 0:
            return self.grid.get_sep_tracks(layer, ntr1, ntr2)
        else:
            tid = TrackIDZL(layer, HalfInt(0), -2, grid=self._grid)
            neg_pitch = tid.pitch
            if ntr1 < 0 and ntr2 < 0:
                tot_w = (abs(ntr1 + ntr2)) * neg_pitch.dbl_value // 2
            else:
                neg_ntr = ntr1 if ntr2 > 0 else ntr2
                pos_ntr = ntr2 if ntr2 > 0 else ntr1
                tot_w = (abs(neg_ntr) - 1) * neg_pitch.dbl_value // 2 + \
                        self.grid.get_sep_tracks(layer, tid.width, pos_ntr).dbl_value
            return HalfInt(tot_w)

    def get_available_wires(self, warr_list, bbox):
        ans_list = []
        layer_id = warr_list[0].layer_id
        margin = self.get_track_sep(layer_id, 1, 1)
        tr_specs = self.grid.get_track_info(layer_id)
        margin = margin*tr_specs.pitch

        xmargin = 0 if self.grid.get_direction(layer_id) == Orient2D.x else margin
        ymargin = 0 if self.grid.get_direction(layer_id) == Orient2D.y else margin

        sp_list = [0, 0]
        sp_list[self.grid.get_direction(layer_id).value ^ 1] = margin
        spx, spy = sp_list
        for warr in warr_list:
            if self.grid.get_direction(layer_id) == Orient2D.x:
                bbox = BBox(bbox.xl, warr.bound_box.yl, bbox.xh, warr.bound_box.yh)
            else:
                bbox = BBox(warr.bound_box.xl, bbox.yl, warr.bound_box.xh, bbox.yh)
            if not self._layout.get_intersect(warr.layer_id, bbox, spx, spy, False):
                ans_list.append(warr)
        return ans_list

    def get_available_tracks(self, layer_id: int, tid_lo: TrackType, tid_hi: TrackType,
                             lower: int, upper: int, width: int = 1, sep: HalfInt = HalfInt(1),
                             include_last: bool = False, sep_margin: Optional[HalfInt] = None,
                             align_to_higer=False, uniform_grid=False) -> List[HalfInt]:
        grid = self.grid

        orient = grid.get_direction(layer_id)
        tr_info = grid.get_track_info(layer_id)
        if width < 0:
            tid = TrackIDZL(layer_id, 0, width, grid=self.grid)
            if sep_margin is None:
                sep_margin = self.get_track_sep(layer_id, width, 1)
            bl, bu = grid.get_wire_bounds_htr(layer_id, 0, tid.width)
            tr_w2 = (bu - bl) // 2 + (abs(width) - 1) * tr_info.pitch * tid.pitch
        else:
            if sep_margin is None:
                sep_margin = grid.get_sep_tracks(layer_id, width, 1, same_color=False)
            bl, bu = grid.get_wire_bounds_htr(layer_id, 0, width)
            tr_w2 = (bu - bl) // 2
        margin = tr_info.pitch * sep_margin - (tr_info.width // 2) - tr_w2

        sp_list = [0, 0]
        sp_list[orient.value ^ 1] = margin
        spx, spy = sp_list

        htr0 = HalfInt.convert(tid_lo).dbl_value
        htr1 = HalfInt.convert(tid_hi).dbl_value
        if include_last:
            htr1 += 1
        htr_sep = HalfInt.convert(sep).dbl_value
        ans = []
        if align_to_higer:
            cur_htr = htr1
            while cur_htr > htr0:
                mid = grid.htr_to_coord(layer_id, cur_htr)
                box = BBox(orient, lower, upper, mid - tr_w2, mid + tr_w2)
                if not self._layout.get_intersect(layer_id, box, spx, spy, False):
                    ans.append(HalfInt(cur_htr))
                    cur_htr -= htr_sep
                else:
                    cur_htr -= 1
        else:
            cur_htr = htr0
            while cur_htr < htr1:
                mid = grid.htr_to_coord(layer_id, cur_htr)
                box = BBox(orient, lower, upper, mid - tr_w2, mid + tr_w2)
                if not self._layout.get_intersect(layer_id, box, spx, spy, False):
                    ans.append(HalfInt(cur_htr))
                    cur_htr += htr_sep
                else:
                    cur_htr += 1

        return ans

    def get_tids_between(self, layer_id, tid_lo, tid_hi, width, sep, sep_margin, include_last,
                         align_to_higher=False, center=True, mod=None):
        grid = self.grid
        actual_sep = sep if width > 0 else self.get_track_sep(layer_id, width, width)
        if actual_sep <= 0:
            raise ValueError('Sep need to be set when using positive width')

        if sep_margin is None:
            sep_margin = self.get_track_sep(layer_id, width, 1)

        htr0 = HalfInt.convert(tid_lo).dbl_value
        htr1 = HalfInt.convert(tid_hi).dbl_value
        if include_last:
            if align_to_higher:
                htr0 -= 1
            else:
                htr1 += 1
        htr_sep = HalfInt.convert(actual_sep).dbl_value
        htr_sep_margin = HalfInt.convert(sep_margin).dbl_value
        ans = []
        cur_htr = htr0 + htr_sep_margin
        while cur_htr + htr_sep_margin < htr1:
            ans.append(cur_htr)
            cur_htr += htr_sep
        if mod:
            ans = ans[:len(ans) // mod * mod]

        if center:
            htr_ofst = (htr0 + htr1 - ans[0] - ans[-1] + int(align_to_higher)) // 2
            ans = [idx + htr_ofst for idx in ans]
        ans = [TrackIDZL(layer_id, HalfInt(tidx), width, grid=grid) for tidx in ans]

        return ans

    def via_up(self, tr_manager: TrackManager, warr: Union[WireArray, List[WireArray]], layer_id, w_type: str = 'sig',
               alignment: RoundMode = RoundMode.NEAREST, bbox=None, align_higher=False):
        warr_dir = self.grid.get_direction(layer_id)
        next_layer = layer_id + 1
        next_w = tr_manager.get_width(next_layer, w_type)
        warr = warr if isinstance(warr, WireArray) else WireArray.list_to_warr(warr)
        lower_alignment = alignment
        if alignment == RoundMode.NEAREST:
            upper_alignment = alignment
        elif alignment == RoundMode.LESS:
            upper_alignment = RoundMode.GREATER
        elif alignment == RoundMode.LESS_EQ:
            upper_alignment = RoundMode.GREATER_EQ
        elif alignment == RoundMode.GREATER:
            upper_alignment = RoundMode.LESS
        else:
            upper_alignment = RoundMode.LESS_EQ

        if warr_dir == Orient2D.x:
            lower = bbox.xl if bbox else warr.lower
            upper = bbox.xh if bbox else warr.upper
        else:
            lower = bbox.yl if bbox else warr.lower
            upper = bbox.yh if bbox else warr.upper

        tid_lo = self.grid.coord_to_track(next_layer, lower, lower_alignment)
        tid_hi = self.grid.coord_to_track(next_layer, upper, upper_alignment)
        sep = 0 if next_w < 0 else tr_manager.get_sep(next_layer, (w_type, w_type))
        tid_list = self.get_tids_between(next_layer, tid_lo, tid_hi, next_w, sep, 0, True, align_to_higher=align_higher)

        next_warr = [self.connect_to_tracks(warr, tid) for tid in tid_list]
        next_warr = WireArray.list_to_warr(next_warr)
        return next_warr

    def via_stack_up(self, tr_manager: TrackManager, warr: Union[WireArray, List[WireArray]],
                     layer_id: int, target_layer_id: int, w_type: str,
                     alignment: RoundMode = RoundMode.NEAREST, bbox=None,
                     align_higher_x=False, align_higher_y=False):
        warr_dict = dict()
        ret_warr = warr
        for layer in range(layer_id, target_layer_id):
            lay_dir = self.grid.get_direction(layer_id)
            align_higher = (lay_dir == Orient2D.x and align_higher_x) or (lay_dir == Orient2D.y and align_higher_y)
            ret_warr = self.via_up(tr_manager, ret_warr, layer, w_type, alignment, bbox, align_higher=align_higher)
            warr_dict[layer + 1] = ret_warr

        return warr_dict

    def connect_supply_bbox(self, tr_manager, vdd_bbox, vss_bbox, layer_id, bbox):
        layer_name = f'm{layer_id}'
        next_layer = layer_id + 1
        sup_w = tr_manager.get_width(next_layer, 'sup')
        sup_sp = tr_manager.get_sep(next_layer, ('sup', 'sup'))
        layer_dir = self.grid.get_direction(next_layer)

        coord_tid_lo = bbox.yl if layer_dir == Orient2D.x else bbox.xl
        coord_tid_hi = bbox.yh if layer_dir == Orient2D.x else bbox.xh
        coord_lo = bbox.xl if layer_dir == Orient2D.x else bbox.yl
        coord_hi = bbox.xh if layer_dir == Orient2D.x else bbox.yh
        locs = self.get_available_tracks(next_layer,
                                         self.grid.coord_to_track(next_layer, coord_tid_lo, RoundMode.GREATER),
                                         self.grid.coord_to_track(next_layer, coord_tid_hi, RoundMode.LESS),
                                         coord_lo, coord_hi, sup_w, sup_sp)
        vdd_xm_locs = locs[::2]
        vss_xm_locs = locs[1::2]
        vdd_next_list_list, vss_next_list_list = [], []

        power_bbox_xh = coord_hi
        power_bbox_xl = coord_lo
        for tid in vdd_xm_locs:
            vdd_xm_list = []
            for vdd in vdd_bbox:
                vdd_xm_list.append(
                    self.connect_bbox_to_tracks(Direction.LOWER, (layer_name, 'drawing'),
                                                vdd, TrackID(next_layer, tid, sup_w),
                                                track_lower=power_bbox_xl, track_upper=power_bbox_xh))
            vdd_next_list_list.append(self.connect_wires(vdd_xm_list)[0])
        for tid in vss_xm_locs:
            vss_xm_list = []
            for vss in vss_bbox:
                vss_xm_list.append(
                    self.connect_bbox_to_tracks(Direction.LOWER, (layer_name, 'drawing'),
                                                vss, TrackID(next_layer, tid, sup_w),
                                                track_lower=power_bbox_xl, track_upper=power_bbox_xh))
            vss_next_list_list.append(self.connect_wires(vss_xm_list)[0])

        for bbox in vdd_bbox + vss_bbox:
            if layer_dir == Orient2D.x:
                bbox.extend(y=coord_tid_lo)
                bbox.extend(y=coord_tid_hi)
            else:
                bbox.extend(x=coord_tid_lo)
                bbox.extend(x=coord_tid_hi)

        return vdd_next_list_list, vss_next_list_list

    def connect_supply_warr(self, tr_manager, warr_list_list, layer_id, bbox,
                            up=True, side_sup=False, align_upper=False, width_override=None,
                            min_len=False, extend_lower_layer=True):
        if (warr_list_list[0][0].layer_id - layer_id) > 0:
            raise ValueError("warr list doesn't match layer")
        next_layer = layer_id + 1 if up else layer_id - 1
        num_sup = len(warr_list_list)
        sup_w = width_override if width_override else tr_manager.get_width(next_layer, 'sup')
        # special handling with negative supply
        if sup_w > 0:
            sup_sp = tr_manager.get_sep(next_layer, ('sup', 'sup'))
        else:
            sup_sp = self.get_track_sep(next_layer, sup_w, sup_w)
        layer_dir = self.grid.get_direction(next_layer)

        coord_tid_lo = bbox.yl if layer_dir == Orient2D.x else bbox.xl
        coord_tid_hi = bbox.yh if layer_dir == Orient2D.x else bbox.xh
        coord_lo = bbox.xl if layer_dir == Orient2D.x else bbox.yl
        coord_hi = bbox.xh if layer_dir == Orient2D.x else bbox.yh

        sep_margin = sup_sp.div2(False)  # Make sure wire is within the box
        if side_sup:
            locs = self.get_tids_between(next_layer,
                                         self.grid.coord_to_track(next_layer, coord_tid_lo, RoundMode.GREATER),
                                         self.grid.coord_to_track(next_layer, coord_tid_hi, RoundMode.LESS),
                                         sup_w, sup_sp, sep_margin, False,
                                         align_to_higher=align_upper and layer_dir == Orient2D.y)
        else:
            locs = self.get_available_tracks(next_layer,
                                             self.grid.coord_to_track(next_layer, coord_tid_lo,
                                                                      RoundMode.GREATER_EQ) + sep_margin,
                                             self.grid.coord_to_track(next_layer, coord_tid_hi,
                                                                      RoundMode.LESS_EQ) - sep_margin,
                                             coord_lo, coord_hi, sup_w, sup_sp, include_last=True,
                                             align_to_higer=align_upper and layer_dir == Orient2D.y)
            locs = [TrackIDZL(next_layer, tid, sup_w, grid=self.grid) for tid in locs]

        if align_upper and layer_dir == Orient2D.y:
            locs = locs[::-1]
        next_warr_list_list = []
        ret_wire = []
        for idx in range(num_sup):
            warr_list = warr_list_list[idx]
            next_loc = locs[idx::num_sup]
            # vss_xm_locs = locs[1::2]
            # vdd_next_list_list, vss_next_list_list = [], []
            power_bbox_xh = coord_hi
            power_bbox_xl = coord_lo
            next_list_list = []
            for tid in next_loc:
                next_list = []
                for warr in warr_list:
                    if min_len:
                        next_list.append(self.connect_to_tracks(warr, tid, ret_wire_list=ret_wire,
                                                                min_len_mode=MinLenMode.MIDDLE))
                    else:
                        next_list.append(self.connect_to_tracks(warr, tid, track_lower=power_bbox_xl,
                                                                track_upper=power_bbox_xh, ret_wire_list=ret_wire,
                                                                min_len_mode=MinLenMode.MIDDLE))

                next_list_list.append(self.connect_wires(next_list)[0])
            next_warr_list_list.append(next_list_list)

        ret_wre_upper = max([w.upper for w in ret_wire])
        ret_wre_lower = min([w.lower for w in ret_wire])
        coord_tid_hi = max(coord_tid_hi, ret_wre_upper)
        coord_tid_lo = min(coord_tid_lo, ret_wre_lower)
        if extend_lower_layer:
            for warr_list in warr_list_list:
                self.extend_wires(warr_list, lower=coord_tid_lo, upper=coord_tid_hi)

        return next_warr_list_list

    def connect_supply_stack_warr(self, tr_manager, warr_list_list, layer_id, target_layer_id,
                                  bbox_list: Union[BBox, List[BBox]], side_sup=False, align_upper=False,
                                  extend_lower_layer=False):
        next_warr_dict = [dict() for _ in range(len(warr_list_list))]
        # vdd_warr_dict = dict()
        # vss_warr_dict = dict()
        # vdd_list, vss_list = vdd_warr, vss_warr
        if isinstance(bbox_list, BBox):
            bbox_list = [bbox_list] * (target_layer_id - layer_id)
        elif len(bbox_list) != target_layer_id - layer_id:
            raise ValueError("length of bbox doesn't match layer stack")

        for idx in range(layer_id, target_layer_id):
            # if align_upper and self.grid.get_direction(idx) == Orient2D.x:
            #     warr_list_list = warr_list_list[::-1]
            warr_list_list = self.connect_supply_warr(tr_manager, warr_list_list, idx, bbox_list[idx - layer_id],
                                                      side_sup=side_sup, align_upper=align_upper,
                                                      extend_lower_layer=extend_lower_layer)
            # if align_upper and self.grid.get_direction(idx) == Orient2D.x:
            #     warr_list_list = warr_list_list[::-1]
            for jdx in range(len(warr_list_list)):
                next_warr_dict[jdx][idx + 1] = warr_list_list[jdx]
            # vdd_warr_dict[idx + 1] = vdd_list
            # vss_warr_dict[idx + 1] = vss_list

        return next_warr_dict

    def quentize_to_track_pitch(self, coord, layer_id, round_mdoe=RoundMode.NEAREST):
        tid = self.grid.coord_to_track(layer_id, coord, round_mdoe)
        return self.grid.track_to_coord(layer_id, tid)

    def match_warr_length(self, warr_list: List[Union[WireArray, List[WireArray]]]):
        if not isinstance(warr_list, List):
            raise Warning("Should be a list, didn't do anything")
        else:
            warr_upper, warr_lower = COORD_MIN, COORD_MAX
            for warr in warr_list:
                if isinstance(warr, WireArray):
                    warr_upper = max(warr_upper, warr.upper)
                    warr_lower = min(warr_lower, warr.lower)
                else:
                    for subwarr in warr:
                        warr_upper = max(warr_upper, subwarr.upper)
                        warr_lower = min(warr_lower, subwarr.lower)
            ret_list = []
            for warr in warr_list:
                new_warr = self.extend_wires(warr, lower=warr_lower, upper=warr_upper)
                ret_list.append(new_warr if len(new_warr) > 1 else new_warr[0])
            return ret_list

    def connect_warr_to_tids(self, warr_list: Union[WireArray, List[WireArray]], tid_list: List[TrackID],
                             bnd: Union[None, Tuple[int, int]] = None, min_len_mode: Union[None, MinLenMode] = None):
        if bnd:
            return [self.connect_to_tracks(warr_list, tid, track_lower=bnd[0], track_upper=bnd[1],
                                           min_len_mode=min_len_mode) for tid in tid_list]
        else:
            return [self.connect_to_tracks(warr_list, tid, min_len_mode=min_len_mode) for tid in tid_list]

    def connect_warr_to_tids_stack(self, warr: Union[WireArray, List[WireArray]],
                                   tid_list_list: List[List[TrackID]],
                                   bnd_list: Union[None, Union[None, List[Tuple[int, int]]]] = None,
                                   min_len_mode: Union[None, MinLenMode] = None):
        warr_list = [warr]
        for idx, tid_list in enumerate(tid_list_list):
            warr_list.append(self.connect_warr_to_tids(warr_list[-1], tid_list,
                                                       bnd=bnd_list[idx] if bnd_list else None,
                                                       min_len_mode=min_len_mode))
        return warr_list

    def export_tap_hm(self, tr_manager, warr: WireArray, hm_layer, target_layer, bbox=None, align_upper=False):
        next_warr = [warr]
        if bbox:
            x_upper = min(warr.upper, bbox[1])
            x_lower = max(warr.lower, bbox[0])
        else:
            x_upper, x_lower = warr.upper, warr.lower
        for idx in range(hm_layer + 1, target_layer + 1):
            lay_dir = self.grid.get_direction(idx)
            if lay_dir == Orient2D.y:
                # Use -1 to make sure via space is ok
                tid_temp = TrackIDZL(idx, 0, -1, grid=self.grid)
                v_min_len = 2 * self.grid.get_next_length(idx, tid_temp.width, 0)
                bbox = BBox(x_lower, warr.bound_box.yl - v_min_len, x_upper, warr.bound_box.yh + v_min_len)
                next_warr = self.connect_supply_warr(tr_manager, [next_warr], idx - 1, bbox, width_override=-1,
                                                     align_upper=align_upper, min_len=True)[0]
            else:
                w = tr_manager.get_width(idx, 'sup')
                tid = self.grid.coord_to_track(idx, (warr.bound_box.yh + warr.bound_box.yl) // 2, RoundMode.NEAREST)
                next_warr = [self.connect_to_tracks(next_warr, TrackIDZL(idx, tid, w, grid=self.grid))]
        return next_warr

    def dummy_fill(self, layer_id, tr_w, tr_sup, bbox, side_sup=False, align_upper=False):
        layer_dir = self.grid.get_direction(layer_id)

        coord_tid_lo = bbox.xl if layer_dir == Orient2D.x else bbox.yl
        coord_tid_hi = bbox.xh if layer_dir == Orient2D.x else bbox.yh
        coord_lo = bbox.yl if layer_dir == Orient2D.x else bbox.xl
        coord_hi = bbox.yh if layer_dir == Orient2D.x else bbox.xh

        sep_margin = tr_sup // 2  # Make sure wire is within the box
        locs = self.get_available_tracks(layer_id,
                                         self.grid.coord_to_track(layer_id, coord_tid_lo,
                                                                  RoundMode.GREATER) + sep_margin,
                                         self.grid.coord_to_track(layer_id, coord_tid_hi,
                                                                  layer_id.LESS) - sep_margin,
                                         coord_lo, coord_hi, tr_w, tr_w)


class WideRoutingTest(TemplateBaseZL):

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        TemplateBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            bot_layer='',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(bot_layer=3)

    def draw_layout(self) -> None:
        bot_layer = self.params['bot_layer']
        top_layer = bot_layer + 1
        if self.grid.get_direction(bot_layer) is Orient2D.x:
            hlayer = bot_layer
            vlayer = top_layer
        else:
            hlayer = top_layer
            vlayer = bot_layer

        num_w = 4
        for tdx in range(2, num_w + 2):
            bdx = self.grid.get_min_track_width(bot_layer, top_ntr=tdx)
            print(f'For top layer track_w = {tdx}, minimum bot layer track_w = {bdx}')
        cur_idx_h, cur_idx_v = 0, 0
        for hdx in range(1, num_w + 1):
            h_min_len = self.grid.get_next_length(hlayer, hdx, 0, even=True)
            v_coord = self.grid.track_to_coord(hlayer, 0)
            h_wire = self.add_wires(hlayer, cur_idx_h, lower=v_coord - h_min_len // 2, upper=v_coord + h_min_len // 2,
                                    width=-hdx)
            cur_idx_v = 0
            for vdx in range(1, num_w + 1):
                v_min_len = self.grid.get_next_length(vlayer, vdx, 0, even=True)
                h_coord = self.grid.track_to_coord(hlayer, cur_idx_h)
                v_wire = self.add_wires(vlayer, cur_idx_v, lower=h_coord - v_min_len // 2,
                                        upper=h_coord + v_min_len // 2, width=-vdx)
                try:
                    self.connect_to_track_wires(h_wire, v_wire)
                except RuntimeError:
                    print(f'No possible via between hlayer track_w = {hdx} and vlayer track_w = {vdx}')
                cur_idx_v = self.get_track_sep(vlayer, -vdx, -vdx - 1) + cur_idx_v
            cur_idx_h = self.get_track_sep(hlayer, -hdx, -hdx - 1) + cur_idx_h

        for hdx in range(1, num_w + 1):
            h_min_len = self.grid.get_next_length(hlayer, hdx, 0, even=True)
            v_coord = self.grid.track_to_coord(hlayer, 0)
            h_wire = self.add_wires(hlayer, cur_idx_h, lower=v_coord - h_min_len // 2, upper=v_coord + h_min_len // 2,
                                    width=-hdx)
            cur_idx_v = 0
            for vdx in range(1, num_w + 1):
                v_min_len = self.grid.get_next_length(vlayer, vdx, 0, even=True)
                h_coord = self.grid.track_to_coord(hlayer, cur_idx_h)
                v_wire = self.add_wires(vlayer, cur_idx_v, lower=h_coord - v_min_len // 2,
                                        upper=h_coord + v_min_len // 2, width=vdx)
                try:
                    self.connect_to_track_wires(h_wire, v_wire)
                except RuntimeError:
                    print(f'No possible via between hlayer track_w = {hdx} and vlayer track_w = {vdx}')
                cur_idx_v = self.get_track_sep(vlayer, vdx, vdx + 1) + cur_idx_v
            cur_idx_h = self.get_track_sep(hlayer, -hdx, -hdx - 1) + cur_idx_h
        #
        # for hdx in range(1, num_w + 1):
        #     h_min_len = self.grid.get_next_length(hlayer, hdx, 0, even=True)
        #     v_coord = self.grid.track_to_coord(hlayer, 0)
        #     h_wire = self.add_wires(hlayer, cur_idx_h, lower=v_coord - h_min_len // 2, upper=v_coord + h_min_len // 2,
        #                             width=hdx)
        #     cur_idx_v = 0
        #     for vdx in range(1, num_w + 1):
        #         v_min_len = self.grid.get_next_length(vlayer, vdx, 0, even=True)
        #         h_coord = self.grid.track_to_coord(hlayer, cur_idx_h)
        #         v_wire = self.add_wires(vlayer, cur_idx_v, lower=h_coord - v_min_len // 2,
        #                                 upper=h_coord + v_min_len // 2, width=-vdx)
        #         try:
        #             self.connect_to_track_wires(h_wire, v_wire)
        #         except RuntimeError:
        #             print(f'No possible via between hlayer track_w = {hdx} and vlayer track_w = {vdx}')
        #         cur_idx_v = 2*self.get_track_sep(vlayer, -vdx, -vdx - 1) + cur_idx_v
        #     cur_idx_h = 2*self.get_track_sep(hlayer, -hdx, -hdx - 1) + cur_idx_h

        for hdx in range(1, 4):
            h_min_len = self.grid.get_next_length(hlayer, hdx, 0, even=True)
            v_coord = self.grid.track_to_coord(hlayer, 0)
            h_wire = self.add_wires(hlayer, cur_idx_h, lower=v_coord - h_min_len // 2, upper=v_coord + h_min_len // 2,
                                    width=hdx if hdx & 1 else -hdx)
            cur_idx_v = 0
            for vdx in range(1, 4):
                v_min_len = self.grid.get_next_length(vlayer, vdx, 0, even=True)
                h_coord = self.grid.track_to_coord(hlayer, cur_idx_h)
                v_wire = self.add_wires(vlayer, cur_idx_v, lower=h_coord - v_min_len // 2,
                                        upper=h_coord + v_min_len // 2, width=-vdx if vdx & 1 else vdx)
                try:
                    self.connect_to_track_wires(h_wire, v_wire)
                except RuntimeError:
                    print(f'No possible via between hlayer track_w = {hdx} and vlayer track_w = {vdx}')
                cur_idx_v = self.get_track_sep(vlayer, -vdx if vdx & 1 else vdx,
                                               vdx + 1 if vdx & 1 else -vdx - 1) + cur_idx_v
            cur_idx_h = self.get_track_sep(hlayer, hdx if hdx & 1 else -hdx,
                                           -hdx - 1 if hdx & 1 else hdx + 1) + cur_idx_h
            print(cur_idx_h)

        w_tot = self.grid.track_to_coord(vlayer, cur_idx_v)
        h_tot = self.grid.track_to_coord(hlayer, cur_idx_h)
        self.set_size_from_bound_box(top_layer, BBox(0, 0, w_tot, h_tot), round_up=True)

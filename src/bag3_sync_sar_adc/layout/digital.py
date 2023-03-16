
"""This module contains layout generators for digital gates and memory cells layout generator,
"""

from typing import Any, Dict, Sequence, Optional, Union, Tuple, Mapping, Type, List

from itertools import chain

from pybag.enum import MinLenMode, RoundMode

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.database import ModuleDB
from bag.design.module import Module
from bag.layout.template import TemplateDB, PyLayInstance
from bag.layout.routing.base import TrackID, WireArray

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.data import MOSPorts, NAND2Ports

from bag3_digital.schematic.inv import bag3_digital__inv
from bag3_digital.schematic.passgate import bag3_digital__passgate


class InvCore(MOSBase):
    """A single inverter.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__inv

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='segments of transistors',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            w_p='pmos width, can be list or integer if all widths are the same.',
            w_n='pmos width, can be list or integer if all widths are the same.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            is_guarded='True if it there should be guard ring around the cell',
            sig_locs='Optional dictionary of user defined signal locations',
            vertical_out='True to draw output on vertical metal layer.',
            vertical_sup='True to have supply unconnected on conn_layer.',
            vertical_in='False to not draw the vertical input wire when is_guarded = True.',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            tr_manager='override track manager',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            is_guarded=False,
            sig_locs={},
            vertical_out=True,
            vertical_sup=False,
            vertical_in=True,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            tr_manager=None
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        grid = self.grid

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack_p: int = self.params['stack_p']
        stack_n: int = self.params['stack_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        is_guarded: bool = self.params['is_guarded']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        vertical_in: bool = self.params['vertical_in']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # set is_guarded = True if both rows has same orientation
        rpinfo_n = self.get_row_info(ridx_n)
        rpinfo_p = self.get_row_info(ridx_p)
        is_guarded = is_guarded or rpinfo_n.flip == rpinfo_p.flip

        # Placement
        nports = self.add_mos(ridx_n, 0, seg_n, w=w_n, stack=stack_n)
        pports = self.add_mos(ridx_p, 0, seg_p, w=w_p, stack=stack_p)

        self.set_mos_size()

        # get wire_indices from sig_locs
        tr_manager = self.params['tr_manager'] if self.params['tr_manager'] else self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        nout_tidx = sig_locs.get('nout', self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=0))
        pout_tidx = sig_locs.get('pout', self.get_track_index(ridx_p, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=-1))
        nout_tid = TrackID(hm_layer, nout_tidx, tr_w_h)
        pout_tid = TrackID(hm_layer, pout_tidx, tr_w_h)

        pout = self.connect_to_tracks(pports.d, pout_tid, min_len_mode=mlm.get('pout', MinLenMode.NONE))
        nout = self.connect_to_tracks(nports.d, nout_tid, min_len_mode=mlm.get('nout', MinLenMode.NONE))

        if vertical_out:
            vm_tidx = sig_locs.get('out', grid.coord_to_track(vm_layer, max(pout.middle, nout.middle),
                                                              mode=RoundMode.NEAREST))
            vm_tid = TrackID(vm_layer, vm_tidx, width=tr_w_v)
            self.add_pin('out', self.connect_to_tracks([pout, nout], vm_tid))
        else:
            self.add_pin('out', [pout, nout], connect=True)
            vm_tidx = None

        if is_guarded:
            nin_tidx = sig_locs.get('nin', self.get_track_index(ridx_n, MOSWireType.G,
                                                                wire_name='sig', wire_idx=0))
            pin_tidx = sig_locs.get('pin', self.get_track_index(ridx_p, MOSWireType.G,
                                                                wire_name='sig', wire_idx=-1))

            nin = self.connect_to_tracks(nports.g, TrackID(hm_layer, nin_tidx, width=tr_w_h))
            pin = self.connect_to_tracks(pports.g, TrackID(hm_layer, pin_tidx, width=tr_w_h))
            self.add_pin('pin', pin, hide=True)
            self.add_pin('nin', nin, hide=True)
            if vertical_in:
                in_tidx = self.grid.find_next_track(vm_layer, nin.lower,
                                                    tr_width=tr_w_v, mode=RoundMode.GREATER_EQ)
                if vm_tidx is not None:
                    in_tidx = min(in_tidx,
                                  self.tr_manager.get_next_track(vm_layer, vm_tidx, 'sig', 'sig',
                                                                 up=False))

                in_tidx = sig_locs.get('in', in_tidx)
                self.add_pin('in', self.connect_to_tracks([pin, nin],
                                                          TrackID(vm_layer, in_tidx, width=tr_w_v)))
            else:
                self.add_pin('in', [pin, nin], connect=True)
        else:
            in_tidx = sig_locs.get('in', None)
            if in_tidx is None:
                in_tidx = sig_locs.get('nin', None)
                if in_tidx is None:
                    default_tidx = self.get_track_index(ridx_n, MOSWireType.G,
                                                        wire_name='sig', wire_idx=0)
                    in_tidx = sig_locs.get('pin', default_tidx)

            in_warr = self.connect_to_tracks([nports.g, pports.g], TrackID(hm_layer, in_tidx, width=tr_w_h),
                                             min_len_mode=mlm.get('nin', MinLenMode.NONE))
            self.add_pin('in', in_warr)
            self.add_pin('pin', in_warr, hide=True)
            self.add_pin('nin', in_warr, hide=True)

        self.add_pin(f'pout', pout, hide=True)
        self.add_pin(f'nout', nout, hide=True)

        xr = self.bound_box.xh
        if vertical_sup:
            self.add_pin('VDD', pports.s, connect=True)
            self.add_pin('VSS', nports.s, connect=True)
        else:
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sup')
            vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)
            vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        default_wp = self.place_info.get_row_place_info(ridx_p).row_info.width
        default_wn = self.place_info.get_row_place_info(ridx_n).row_info.width
        thp = self.place_info.get_row_place_info(ridx_p).row_info.threshold
        thn = self.place_info.get_row_place_info(ridx_n).row_info.threshold
        lch = self.place_info.lch
        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=lch,
            w_p=default_wp if w_p == 0 else w_p,
            w_n=default_wn if w_n == 0 else w_n,
            th_n=thn,
            th_p=thp,
            stack_p=stack_p,
            stack_n=stack_n,
        )


class InvChainCore(MOSBase):
    """An inverter chain.

    Assumes:

    1. PMOS row above NMOS row.
    2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._out_invert = False

    @property
    def out_invert(self) -> bool:
        return self._out_invert

    @property
    def num_stages(self) -> int:
        return len(self.params['seg_list']) if self.params['seg_list'] else \
            len(self.params['segn_list'])

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'inv_chain')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_list='List of segments per stage.',
            segp_list='List of pmos segments per stage',
            segn_list='List of nmos segments per stage',
            stack_list='List of stacks per stage',
            w_p='pmos width, can be list or integer if all widths are the same.',
            w_n='pmos width, can be list or integer if all widths are the same.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            dual_output='Whether to export complementary outputs.  Works only if nstage > 1',
            is_guarded='True if it there should be guard ring around the cell',
            sig_locs='Optional dictionary of user defined signal locations',
            vertical_out='True to draw output on vertical metal layer.',
            vertical_sup='True to have supply unconnected on conn_layer.',
            export_pins='True to export simulation pins.',
            sep_stages='True to separate the stages and not share source/drain.',
            buf_col_list='List of inverter column indices.',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            tr_manager='override track manager',
            mid_int_vm='True to place intermediate vm at the boundary'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            stack_list=None,
            dual_output=False,
            is_guarded=False,
            sig_locs={},
            vertical_out=True,
            vertical_sup=False,
            export_pins=False,
            sep_stages=False,
            seg_list=[],
            segp_list=[],
            segn_list=[],
            buf_col_list=None,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            tr_manager=None,
            mid_int_vm=False,
        )

    def draw_layout(self) -> None:
        params = self.params
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, params['pinfo'])
        self.draw_base(pinfo)

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        dual_output: bool = params['dual_output']
        export_pins: bool = params['export_pins']
        is_guarded: bool = params['is_guarded']
        sep_stages: bool = params['sep_stages']
        sig_locs: Mapping[str, Union[float, HalfInt]] = params['sig_locs']
        buf_col_list: Optional[Sequence[int]] = params['buf_col_list']

        if export_pins and dual_output:
            raise ValueError('export_pins and dual_output cannot be True at the same time')
        tr_manager = self.params['tr_manager'] if self.params['tr_manager'] else self.tr_manager
        inv_masters = self._create_masters(pinfo, tr_manager)
        nstage = len(inv_masters)
        if nstage == 1:
            dual_output = False
        if nstage % 2 == 1:
            self._out_invert = True

        # Placement and connect on the way
        min_sep = self.min_sep_col
        cur_col = 0
        wn_prev = wp_prev = -1
        sup_last_prev = False
        out_prev: Optional[List[WireArray]] = None
        nout_prev: Optional[WireArray] = None
        pout_prev: Optional[WireArray] = None
        vdd_list = []
        vss_list = []
        sch_params_list = []

        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        for idx, master in enumerate(inv_masters):
            sch_params_cur = master.sch_params
            wp_cur = sch_params_cur['w_p']
            wn_cur = sch_params_cur['w_n']
            segp_cur = sch_params_cur['seg_p']
            segn_cur = sch_params_cur['seg_n']
            stackp_cur = sch_params_cur['stack_n']
            stackn_cur = sch_params_cur['stack_n']
            fgp_cur = segp_cur * stackp_cur
            fgn_cur = segn_cur * stackn_cur

            # get instance column index
            if buf_col_list:
                cur_col = buf_col_list[idx]
            else:
                # NOTE: place on even columns only to preserve supply parity
                if (cur_col != 0 and
                        (sep_stages or wn_cur != wn_prev or wp_cur != wp_prev or
                         not sup_last_prev or (cur_col & 1) == 1)):
                    # we can abut with previous stage if widths are the same, the last
                    # source/drain of the previous stage is the supply, and we're on even column now
                    cur_col += min_sep
                cur_col += (cur_col & 1)

            # add instance
            inst = self.add_tile(master, 0, cur_col, commit=False)
            # update vertical track locations from sig_locs
            map_list = []
            if idx == 0 and is_guarded:
                # update input track
                map_list.append(('in', 'in'))
            if idx == nstage - 2 and dual_output and master.params['vertical_out']:
                # update last2 out
                map_list.append(('out' if self._out_invert else 'outb', 'out'))
            elif idx == nstage - 1:
                # update last out and potentially last2 out
                if dual_output and is_guarded:
                    # update last2 out
                    map_list.append(('out' if self._out_invert else 'outb', 'in'))
                if master.params['vertical_out']:
                    # update last out
                    map_list.append(('outb' if self._out_invert else 'out', 'out'))

            if map_list:
                self._update_inst_sig_locs(sig_locs, map_list, hm_layer, inst)
            inst.commit()

            vdd_list.append(inst.get_pin('VDD'))
            vss_list.append(inst.get_pin('VSS'))
            sch_params_list.append(sch_params_cur)
            # NOTE: use get_all_port_pins() because if vertical_out = False,
            # we have more than one outputs
            out_cur = inst.get_all_port_pins('out')
            pout_cur = inst.get_pin('pout')
            nout_cur = inst.get_pin('nout')

            # perform connections
            if idx == 0:
                self.reexport(inst.get_port('in'))
                self.reexport(inst.get_port('pin'))
                self.reexport(inst.get_port('nin'))
            else:
                if is_guarded:
                    warr = self.connect_to_track_wires([nout_prev, pout_prev], inst.get_pin('in'))
                    if dual_output and idx == nstage - 1:
                        # export second output on vm layer
                        suf = '' if self._out_invert else 'b'
                        self.add_pin('out' + suf, warr)
                else:
                    self.connect_to_track_wires(inst.get_pin('in'), out_prev)

            if idx == nstage - 1:
                suf = 'b' if self._out_invert else ''
                self.reexport(inst.get_port('out'), net_name='out' + suf, label='out'+suf)
                self.add_pin('pout' + suf, pout_cur, hide=True)
                self.add_pin('nout' + suf, nout_cur, hide=True)
            elif dual_output and (idx == nstage - 2):
                suf = '' if self._out_invert else 'b'
                if not is_guarded:
                    # out_cur is on vm layer only if there's no guard ring
                    self.add_pin('out' + suf, out_cur)
                self.add_pin('pout' + suf, pout_cur, hide=True)
                self.add_pin('nout' + suf, nout_cur, hide=True)
            elif export_pins:
                if idx == 0 and nstage == 2:
                    self.add_pin('mid', out_cur)
                else:
                    self.add_pin(f'mid<{idx}>', out_cur)

            # out_cur_vm
            if self.params['mid_int_vm'] and not out_cur[0].layer_id > 2:
                out_cur_vm_tidx = self.grid.coord_to_track(vm_layer, inst.bound_box.xh, RoundMode.LESS)
                out_cur = self.connect_to_tracks(out_cur, TrackID(vm_layer, out_cur_vm_tidx, tr_w_v))
                if idx == nstage-1:
                    suf = 'b' if self._out_invert else ''
                    self.add_pin('out' + suf, out_cur)
                elif dual_output and (idx == nstage - 2):
                    suf = '' if self._out_invert else 'b'
                    if not is_guarded:
                        # out_cur is on vm layer only if there's no guard ring
                        self.add_pin('out' + suf, out_cur)

            out_prev = out_cur
            pout_prev = pout_cur
            nout_prev = nout_cur
            wn_prev = wn_cur
            wp_prev = wp_cur
            if fgp_cur > fgn_cur:
                sup_last_prev = (segp_cur % 2 == 0)
                cur_col += fgp_cur
            elif fgp_cur < fgn_cur:
                sup_last_prev = (segn_cur % 2 == 0)
                cur_col += fgn_cur
            else:
                sup_last_prev = (segp_cur % 2 == 0) and (segn_cur % 2 == 0)
                cur_col += fgn_cur

        self.set_mos_size()

        # export supplies
        self.add_pin('VDD', self.connect_wires(vdd_list))
        self.add_pin('VSS', self.connect_wires(vss_list))

        self.sch_params = dict(
            inv_params=sch_params_list,
            export_pins=export_pins,
            dual_output=dual_output,
        )

    def _create_masters(self, pinfo: MOSBasePlaceInfo, tr_manager) -> Sequence[InvCore]:
        params = self.params
        seg_list: Sequence[int] = params['seg_list']
        segp_list: Optional[Sequence[int]] = params.get('segp_list', None)
        segn_list: Optional[Sequence[int]] = params.get('segn_list', None)
        stack_list: Optional[Sequence[int]] = params['stack_list']
        w_p: Union[int, Sequence[int]] = params['w_p']
        w_n: Union[int, Sequence[int]] = params['w_n']
        ridx_p: int = params['ridx_p']
        ridx_n: int = params['ridx_n']
        is_guarded: bool = params['is_guarded']
        sig_locs: Mapping[str, Union[float, HalfInt]] = params['sig_locs']
        vertical_out: bool = params['vertical_out']
        vertical_sup: bool = params['vertical_sup']

        nout_tidx = get_adj_tidx_list(self, ridx_n, sig_locs, MOSWireType.DS, 'nout', False)
        pout_tidx = get_adj_tidx_list(self, ridx_p, sig_locs, MOSWireType.DS, 'pout', True)
        nin_tidx = get_adj_tidx_list(self, ridx_n, sig_locs, MOSWireType.G, 'nin', True)
        pin_tidx = get_adj_tidx_list(self, ridx_p, sig_locs, MOSWireType.G, 'pin', False)

        sig_locs_list = [
            dict(
                nout=nout_tidx[0],
                pout=pout_tidx[0],
                nin=nin_tidx[0],
                pin=pin_tidx[0],
            ),
            dict(
                nout=nout_tidx[1],
                pout=pout_tidx[1],
                nin=nin_tidx[1],
                pin=pin_tidx[1],
            ),
        ]

        nstage = len(seg_list) if seg_list else len(segn_list)
        if nstage == 0:
            raise ValueError('Must have at least one inverter.')

        if isinstance(w_p, int):
            w_p = [w_p] * nstage
        elif len(w_p) != nstage:
            raise ValueError(f'length of w_p list is not equal to {nstage}')

        if isinstance(w_n, int):
            w_n = [w_n] * nstage
        elif len(w_n) != nstage:
            raise ValueError(f'length of w_n list is not equal to {nstage}')

        if stack_list is None:
            stack_list = [1] * nstage
        elif len(stack_list) != nstage:
            raise ValueError(f'length of stack_list is not equal to {nstage}')

        if seg_list and not segp_list and not segn_list:
            segp_list = seg_list
            segn_list = seg_list
        elif segp_list and segn_list:
            if len(segp_list) != len(segn_list):
                raise ValueError("segn_list and segp_list must be the same length")
        else:
            raise ValueError("Provide only seg_list or segp_list and segn_list")

        master_list = []
        for idx, (segp, segn, wp, wn, stack) in enumerate(zip(segp_list, segn_list,
                                                              w_p, w_n, stack_list)):
            is_last = (idx == nstage - 1)
            # last stage vertical_out flag controlled by user parameters
            # for all others, draw vertical output if no guard ring
            vout = ((not is_guarded) and ((not is_last) or vertical_out) or
                    (is_last and vertical_out))
            params = dict(
                pinfo=pinfo,
                seg_p=segp,
                seg_n=segn,
                stack_p=stack,
                stack_n=stack,
                w_p=wp,
                w_n=wn,
                ridx_p=ridx_p,
                ridx_n=ridx_n,
                is_guarded=is_guarded,
                sig_locs=sig_locs_list[idx % 2],
                vertical_out=False if self.params['mid_int_vm'] else vout,
                vertical_sup=vertical_sup,
                tr_manager=tr_manager,
            )
            master_list.append(self.new_template(InvCore, params=params))

        return master_list

    def _update_inst_sig_locs(self, sig_locs: Mapping[str, Union[float, HalfInt]],
                              sig_map_list: Sequence[Tuple[str, str]],
                              layer: int, inst: PyLayInstance) -> None:
        grid = self.grid
        xform = inst.transformation.get_inverse()
        append = {}
        for sig_name, inst_sig_name in sig_map_list:
            test = sig_locs.get(sig_name, None)
            if test is not None:
                append[inst_sig_name] = grid.transform_track(layer, test, xform)

        if append:
            sig_locs_inst = inst.master.params['sig_locs'].copy(append=append)
            inst.new_master_with(sig_locs=sig_locs_inst)


class InvTristateCore(MOSBase):
    """A gated inverter with two enable signals.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'inv_tristate')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            seg_p='Number of segments of pmos.',
            seg_n='Number of segments of nmos.',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            vertical_out='True to draw output on vertical metal layer.',
            vertical_sup='True to have supply unconnected on conn_layer.',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            tr_manager='override track manager',
            flip_ds='True to use drain conn for supply, used for abut gates'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            sig_locs=None,
            vertical_out=True,
            vertical_sup=False,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            tr_manager=None,
            flip_ds=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack_p: int = self.params['stack_p']
        stack_n: int = self.params['stack_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        vertical_out: bool = self.params['vertical_out']
        vertical_sup: bool = self.params['vertical_sup']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']

        if w_p == 0:
            w_p = self.place_info.get_row_place_info(ridx_p).row_info.width
        if w_n == 0:
            w_n = self.place_info.get_row_place_info(ridx_n).row_info.width

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if vertical_out and self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        if stack_p != stack_n:
            raise ValueError(f'The layout generator does not support stack_n = {stack_n} != '
                             f'stack_p = {stack_p}')

        # place instances and set size
        self.set_mos_size(2 * max(seg_p * stack_p, seg_n * stack_n))
        nports = self.add_nand2(ridx_n, 0, seg_n, w=w_n, stack=stack_n)
        pports = self.add_nand2(ridx_p, 0, seg_p, w=w_p, stack=stack_p)

        # get track information
        tr_manager = self.params['tr_manager'] if self.params['tr_manager'] else self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        if sig_locs is None:
            sig_locs = {}
        en_tidx = sig_locs.get('nen', self.get_track_index(ridx_n, MOSWireType.G,
                                                           wire_name='sig', wire_idx=0))
        in_key = 'nin' if 'nin' in sig_locs else 'pin'
        in_tidx = sig_locs.get(in_key, self.get_track_index(ridx_n, MOSWireType.G,
                                                            wire_name='sig', wire_idx=1))
        enb_tidx = sig_locs.get('pen', self.get_track_index(ridx_p, MOSWireType.G,
                                                            wire_name='sig', wire_idx=-1))
        pout_tidx = sig_locs.get('pout', self.get_track_index(ridx_p, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=0))
        nout_tidx = sig_locs.get('nout', self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=-1))

        # connect wires
        tid = TrackID(hm_layer, in_tidx, width=tr_w_h)
        in_warr_list = []

        flip_ds = self.params['flip_ds']
        if flip_ds:
            nports = NAND2Ports(nports.g1, nports.g0, nports.s, nports.d, nports.m)
            pports = NAND2Ports(pports.g1, pports.g0, pports.s, pports.d, pports.m)

        hm_in = self.connect_to_tracks(list(chain(nports.g0, pports.g0)), tid,
                                       ret_wire_list=in_warr_list)
        self.add_pin('nin', hm_in, hide=True)
        self.add_pin('pin', hm_in, hide=True)
        self.add_pin('in', in_warr_list)
        tid = TrackID(hm_layer, en_tidx, width=tr_w_h)
        self.add_pin('en', self.connect_to_tracks(nports.g1, tid, min_len_mode=mlm.get('en', MinLenMode.NONE)))
        tid = TrackID(hm_layer, enb_tidx, width=tr_w_h)
        self.add_pin('enb', self.connect_to_tracks(pports.g1, tid, min_len_mode=mlm.get('enb', MinLenMode.NONE)))
        tid = TrackID(hm_layer, nout_tidx, width=tr_w_h)
        nout = self.connect_to_tracks(nports.d, tid)
        tid = TrackID(hm_layer, pout_tidx, width=tr_w_h)
        pout = self.connect_to_tracks(pports.d, tid)

        # connect output
        if vertical_out:
            tr_w_v = tr_manager.get_width(vm_layer, 'sig')
            out_tidx = sig_locs.get('out', self.grid.coord_to_track(vm_layer, pout.middle,
                                                                    mode=RoundMode.NEAREST))
            tid = TrackID(vm_layer, out_tidx, width=tr_w_v)
            self.add_pin('out', self.connect_to_tracks([pout, nout], tid))

        # connect supplies
        xr = self.bound_box.xh
        if vertical_sup:
            self.add_pin('VDD', pports.s, connect=True)
            self.add_pin('VSS', nports.s, connect=True)
        else:
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
            vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
            vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        self.add_pin('pout', pout, label='out:', hide=vertical_out)
        self.add_pin('nout', nout, label='out:', hide=vertical_out)
        # set properties
        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=self.place_info.lch,
            w_n=w_n,
            w_p=w_p,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            stack_p=stack_p,
            stack_n=stack_n,
        )


class NAND2Core(MOSBase):
    """
    A 2-input NAND gate.

    'out' and 'in' pin direction is determined by vertical_out and vertical_in flags, respectively
    but regardless, in0_vm, in1_vm, out_vm, and out_hm are always available, and also if
    connect_inputs is True in0_hm, in1_hm will also be available.

    Assumes:

        1. PMOS row above NMOS row.
        2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'nand')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            seg_p='Number of segments of pmos.',
            seg_n='Number of segments of nmos.',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            is_guarded='True if it there should be guard ring around the cell',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            vertical_out='True to have output pin on vertical layer',
            vertical_in='True to have input pins on vertical layer.  Only used if is_guarded=True',
            vertical_sup='True to have supply unconnected on conn_layer.',
            tr_manager='override track manager',
            flip_ds='True to use drain conn for supply, used for abut gates'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            sig_locs={},
            is_guarded=False,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            vertical_out=True,
            vertical_in=True,
            vertical_sup=False,
            tr_manager=None,
            flip_ds=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack_p: int = self.params['stack_p']
        stack_n: int = self.params['stack_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        is_guarded: bool = self.params['is_guarded']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']
        vertical_out: bool = self.params['vertical_out']
        vertical_in: bool = self.params['vertical_in']
        vertical_sup: bool = self.params['vertical_sup']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        tot_col = 2 * max(seg_p * stack_p, seg_n * stack_n)
        self.set_mos_size(tot_col)

        tr_manager = self.params['tr_manager'] if self.params['tr_manager'] else self.tr_manager

        if is_guarded is False and stack_n != stack_p:
            raise ValueError(f'If is_guarded is False, then the layout generator requires that '
                             f'stack_n = {stack_n} == stack_p = {stack_p}.')

        # if guard ring is true ridx_p is automatically mapped to the new row indices
        pports = self.add_nand2(ridx_p, 0, seg_p, w=w_p, stack=stack_p, other=True)
        nports = self.add_nand2(ridx_n, 0, seg_n, w=w_n, stack=stack_n)

        flip_ds = self.params['flip_ds']
        if flip_ds:
            nports = NAND2Ports(nports.g1, nports.g0, nports.s, nports.d, nports.m)
            pports = NAND2Ports(pports.g1, pports.g0, pports.s, pports.d, pports.m)

        xr = self.bound_box.xh
        if vertical_sup:
            self.add_pin('VDD', pports.s, connect=True)
            self.add_pin('VSS', nports.s, connect=True)
        else:
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
            vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
            vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        mlm_in0 = mlm.get('in0', None)
        mlm_in1 = mlm.get('in1', None)
        nin_tid = get_adj_tid_list(self, ridx_n, sig_locs, MOSWireType.G, 'nin', True, tr_w_h)
        pin_tid = get_adj_tid_list(self, ridx_p, sig_locs, MOSWireType.G, 'pin', False, tr_w_h)

        if is_guarded:
            n_in0 = self.connect_to_tracks(nports.g0, nin_tid[0], min_len_mode=mlm_in0)
            n_in1 = self.connect_to_tracks(nports.g1, nin_tid[1], min_len_mode=mlm_in1)
            p_in0 = self.connect_to_tracks(pports.g0, pin_tid[0], min_len_mode=mlm_in0)
            p_in1 = self.connect_to_tracks(pports.g1, pin_tid[1], min_len_mode=mlm_in1)
            in0_tidx = sig_locs.get('in0', self.grid.coord_to_track(vm_layer, self.bound_box.xl,
                                                                    RoundMode.GREATER_EQ))
            in1_tidx = sig_locs.get('in1', self.grid.coord_to_track(vm_layer, self.bound_box.xh,
                                                                    RoundMode.LESS_EQ))
            if vertical_in:
                in0_vm = self.connect_to_tracks([n_in0, p_in0],
                                                TrackID(vm_layer, in0_tidx, width=tr_w_v))
                in1_vm = self.connect_to_tracks([n_in1, p_in1],
                                                TrackID(vm_layer, in1_tidx, width=tr_w_v))
                self.add_pin('in<0>', in0_vm)
                self.add_pin('in<1>', in1_vm)
                self.add_pin('nin<0>', n_in0, hide=True)
                self.add_pin('nin<1>', n_in1, hide=True)
                self.add_pin('pin<0>', p_in0, hide=True)
                self.add_pin('pin<1>', p_in1, hide=True)
            else:
                self.add_pin('nin<0>', n_in0, label='in<0>:')
                self.add_pin('nin<1>', n_in1, label='in<1>:')
                self.add_pin('pin<0>', p_in0, label='in<0>:')
                self.add_pin('pin<1>', p_in1, label='in<1>:')
        else:
            key_name = 'nin' if any(k in sig_locs for k in ['nin', 'nin0']) else 'pin'
            in_tid = get_adj_tid_list(self, ridx_n, sig_locs, MOSWireType.G, key_name, True, tr_w_h)
            in0_vm, in1_vm = [], []
            in0 = self.connect_to_tracks(list(chain(nports.g0, pports.g0)), in_tid[0],
                                         min_len_mode=mlm_in0, ret_wire_list=in0_vm)
            in1 = self.connect_to_tracks(list(chain(nports.g1, pports.g1)), in_tid[1],
                                         min_len_mode=mlm_in0, ret_wire_list=in1_vm)

            self.add_pin('nin<0>', in0, hide=True)
            self.add_pin('pin<0>', in0, hide=True)
            self.add_pin('nin<1>', in1, hide=True)
            self.add_pin('pin<1>', in1, hide=True)
            self.add_pin('in<0>', in0_vm)
            self.add_pin('in<1>', in1_vm)

        nd_tidx = sig_locs.get('nout',
                               self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                    wire_name='sig', wire_idx=-1))
        nd_tid = TrackID(hm_layer, nd_tidx, width=tr_w_h)
        mlm_out = mlm.get('out', MinLenMode.MIDDLE)
        if self.can_short_adj_tracks and not is_guarded:
            # we can short adjacent source/drain tracks together, so we can avoid going to vm_layer
            out = self.connect_to_tracks([pports.d, nports.d], nd_tid, min_len_mode=mlm_out)
            self.add_pin('nout', out, hide=True)
            self.add_pin('out', pports.d)
        else:
            # need to use vm_layer to short adjacent tracks.
            pd_tidx = sig_locs.get('pout',
                                   self.get_track_index(ridx_p, MOSWireType.DS_GATE,
                                                        wire_name='sig'))
            pd_tid = TrackID(hm_layer, pd_tidx, width=tr_w_h)
            pout = self.connect_to_tracks(pports.d, pd_tid, min_len_mode=mlm_out)
            nout = self.connect_to_tracks(nports.d, nd_tid, min_len_mode=mlm_out)
            vm_tidx = sig_locs.get('out', self.grid.coord_to_track(vm_layer, pout.middle,
                                                                   mode=RoundMode.GREATER_EQ))
            if vertical_out:
                out = self.connect_to_tracks([pout, nout], TrackID(vm_layer, vm_tidx,
                                                                   tr_manager.get_width(vm_layer, 'sig')))
                self.add_pin('pout', pout, hide=True)
                self.add_pin('nout', nout, hide=True)
                self.add_pin('out', out)
            else:
                self.add_pin('pout', pout, label='out:')
                self.add_pin('nout', nout, label='out:')

        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            stack_p=stack_p,
            stack_n=stack_n,
        )


class NANDNOR3Core(MOSBase):
    """
    A 3-input NAND/NOR gate.

    'out' and 'in' pin direction is determined by vertical_out and vertical_in flags, respectively
    but regardless, in0_vm, in1_vm, in2_vm, out_vm, and out_hm are always available

    Assumes:

        1. PMOS row above NMOS row.
        2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._num_in = 3
        self._pdn_series = None

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            seg_p='Number of segments of pmos.',
            seg_n='Number of segments of nmos.',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            is_guarded='True if it there should be guard ring around the cell',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            vertical_out='True to have output pin on vertical layer',
            vertical_in='True to have input pins on vertical layer.  Only used if is_guarded=True',
            vertical_sup='True to have supply unconnected on conn_layer.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            sig_locs={},
            is_guarded=False,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                in2=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            vertical_out=True,
            vertical_in=True,
            vertical_sup=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack_p: int = self.params['stack_p']
        stack_n: int = self.params['stack_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        is_guarded: bool = self.params['is_guarded']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']
        vertical_out: bool = self.params['vertical_out']
        vertical_in: bool = self.params['vertical_in']
        vertical_sup: bool = self.params['vertical_sup']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        pun_wider = seg_p * stack_p > seg_n * stack_n
        seg_equal = seg_n == seg_p
        stack_equal = stack_n == stack_p
        stack_n_odd = stack_n & 1
        stack_p_odd = stack_p & 1

        if self._pdn_series:
            draw_pdn = self.draw_series_network
            draw_pun = self.draw_parallel_network

            pdn_kwargs = dict(w=w_n, seg=seg_n, stack=stack_n, ridx=ridx_n, is_nmos=True)
            pun_kwargs = dict(w=w_p, seg=seg_p, stack=stack_p, ridx=ridx_p,
                              split=seg_equal and stack_equal)
        else:
            draw_pdn = self.draw_parallel_network
            draw_pun = self.draw_series_network

            pdn_kwargs = dict(w=w_n, seg=seg_n, stack=stack_n, ridx=ridx_n,
                              split=seg_equal and stack_equal)
            pun_kwargs = dict(w=w_p, seg=seg_p, stack=stack_p, ridx=ridx_p, is_nmos=False)

        if pun_wider:
            p_warrs, p_tot_col = draw_pun(**pun_kwargs)
            n_warrs, _ = draw_pdn(**pdn_kwargs, ref_width=p_tot_col)
        else:
            n_warrs, n_tot_col = draw_pdn(**pdn_kwargs)
            p_warrs, _ = draw_pun(**pun_kwargs, ref_width=n_tot_col)

        pg_warrs, pout_warrs, vdd_warrs = p_warrs
        ng_warrs, nout_warrs, vss_warrs = n_warrs

        self.set_mos_size()

        xr = self.bound_box.xh

        if vertical_sup:
            self.add_pin('VDD', vdd_warrs, connect=True)
            self.add_pin('VSS', vss_warrs, connect=True)
        else:
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
            vss = self.connect_to_tracks(vss_warrs, ns_tid, track_lower=0, track_upper=xr)
            vdd = self.connect_to_tracks(vdd_warrs, ps_tid, track_lower=0, track_upper=xr)

            self.add_pin('VSS', vss)
            self.add_pin('VDD', vdd)

        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')

        mlm_ins = [mlm.get(f'in{i}', None) for i in range(self._num_in)]

        # Use vm_layer for in connection for the following conditions:
        # 1. There should be a guard ring around the cell
        # 2. stack_n != stack_p
        # 3. stack_n == stack_p, but seg_n != seg_p for odd stack
        use_in_vm = is_guarded or not stack_equal or (stack_n_odd and seg_n != seg_p)

        vm_tidxs = {k: v for k, v in sig_locs.items() if
                    k.startswith('in') and use_in_vm or k == 'out'}
        vm_tidx_min = self.grid.coord_to_track(vm_layer, self.bound_box.xl, RoundMode.GREATER_EQ)
        vm_tidx_max = self.grid.coord_to_track(vm_layer, self.bound_box.xh, RoundMode.LESS_EQ)

        nd_tidx = sig_locs.get('nout',
                               self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig',
                                                    wire_idx=-1))
        nd_tid = TrackID(hm_layer, nd_tidx, width=tr_w_h)

        pd_tidx = sig_locs.get('pout',
                               self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig'))
        pd_tid = TrackID(hm_layer, pd_tidx, width=tr_w_h)

        mlm_out = mlm.get('out', MinLenMode.MIDDLE)
        nout = self.connect_to_tracks(nout_warrs, nd_tid, min_len_mode=mlm_out)
        pout = self.connect_to_tracks(pout_warrs, pd_tid, min_len_mode=mlm_out)
        if vertical_out:
            if 'out' not in vm_tidxs:
                if self._pdn_series:
                    out_coord = nout.middle if stack_n_odd or not pun_wider else pout.middle
                else:
                    out_coord = pout.middle if stack_p_odd or pun_wider else nout.middle
                init_out_vm_tidx = self.grid.coord_to_track(vm_layer, out_coord,
                                                            mode=RoundMode.GREATER_EQ)
                out_vm_tidx = self._get_closest_unused_tidx(vm_tidxs, vm_layer, init_out_vm_tidx,
                                                            tidx_min=vm_tidx_min,
                                                            tidx_max=vm_tidx_max)
                vm_tidxs['out'] = out_vm_tidx
            out = self.connect_to_tracks([pout, nout], TrackID(vm_layer, vm_tidxs['out']))
            self.add_pin('pout', pout, hide=True)
            self.add_pin('nout', nout, hide=True)
            self.add_pin('out', out)
        else:
            self.add_pin('pout', pout, label='out:')
            self.add_pin('nout', nout, label='out:')

        nin_tid = self._get_hm_tid_list(ridx_n, sig_locs, 'nin', stack_n, hm_layer, tr_w_h, True)
        if use_in_vm:
            tr_w_v = tr_manager.get_width(vm_layer, 'sig')
            pin_tid = self._get_hm_tid_list(ridx_p, sig_locs, 'pin', stack_p, hm_layer, tr_w_h,
                                            False)

            for i in range(self._num_in):
                if f'in{i}' in vm_tidxs:
                    continue
                coord = round(self.bound_box.xl + (self.bound_box.xh - self.bound_box.xl) * (
                        i / (self._num_in - 1)))
                init_in_vm_tidx = self.grid.coord_to_track(vm_layer, coord, RoundMode.NEAREST)
                in_vm_tidx = self._get_closest_unused_tidx(vm_tidxs, vm_layer, init_in_vm_tidx,
                                                           tidx_min=vm_tidx_min,
                                                           tidx_max=vm_tidx_max)
                vm_tidxs[f'in{i}'] = in_vm_tidx

            for i, (n_g, p_g, ntid, ptid, mlm_in) in enumerate(
                    zip(ng_warrs, pg_warrs, nin_tid, pin_tid, mlm_ins)):
                n_in = self.connect_to_tracks(n_g, ntid, min_len_mode=mlm_in)
                p_in = self.connect_to_tracks(p_g, ptid, min_len_mode=mlm_in)

                if vertical_in:
                    self.add_pin(f'nin<{i}>', n_in, hide=True)
                    self.add_pin(f'pin<{i}>', p_in, hide=True)
                    in_vm = self.connect_to_tracks([n_in, p_in],
                                                   TrackID(vm_layer, vm_tidxs[f'in{i}'],
                                                           width=tr_w_v))
                    self.add_pin(f'in<{i}>', in_vm)
                else:
                    self.add_pin(f'nin<{i}>', n_in, label=f'in<{i}>:')
                    self.add_pin(f'pin<{i}>', p_in, label=f'in<{i}>:')
        else:
            in_vms = [[] for _ in range(self._num_in)]
            in_warrs = []
            for i, (n_g, p_g, tid, mlm_in, in_vm) in enumerate(
                    zip(ng_warrs, pg_warrs, nin_tid, mlm_ins, in_vms)):
                in_warrs.append(self.connect_to_tracks(n_g + p_g, tid, min_len_mode=mlm_in,
                                                       ret_wire_list=in_vm))

            for i, (in_warr, in_vm) in enumerate(zip(in_warrs, in_vms)):
                self.add_pin(f'nin<{i}>', in_warr, hide=True)
                self.add_pin(f'pin<{i}>', in_warr, hide=True)
                self.add_pin(f'in<{i}>', in_vm)

        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            stack_p=stack_p,
            stack_n=stack_n,
            num_in=self._num_in
        )

    def draw_series_network(self, w: int, seg: int, stack: int, ridx: int, **kwargs: Any
                            ) -> Tuple[Tuple[List[List[WireArray]], List[WireArray],
                                             List[WireArray]], int]:
        if stack & 1:
            return self.draw_series_network_stack_odd(w, seg, stack, ridx, **kwargs)
        else:
            return self.draw_series_network_stack_even(w, seg, stack, ridx)

    def draw_parallel_network(self, w: int, seg: int, stack: int, ridx: int, **kwargs: Any
                              ) -> Tuple[Tuple[List[List[WireArray]], List[WireArray],
                                               List[WireArray]], int]:
        if stack & 1:
            return self.draw_parallel_network_stack_odd(w, seg, stack, ridx, **kwargs)
        else:
            return self.draw_parallel_network_stack_even(w, seg, stack, ridx)

    def draw_series_network_stack_even(self, w: int, seg: int, stack: int, ridx: int
                                       ) -> Tuple[Tuple[List[List[WireArray]], List[WireArray],
                                                        List[WireArray]], int]:
        ports = self.add_mos(ridx, 0, seg, w=w, g_on_s=False, stack=self._num_in * stack,
                             sep_g=True)
        g_warrs = self._get_g_idx_list_stack_even(ports.g, seg, stack)
        return (g_warrs, [ports.d], [ports.s]), self._num_in * seg * stack

    def draw_parallel_network_stack_even(self, w: int, seg: int, stack: int, ridx: int
                                         ) -> Tuple[Tuple[List[List[WireArray]], List[WireArray],
                                                          List[WireArray]], int]:
        ports = self.add_mos(ridx, 0, self._num_in * seg, w=w, g_on_s=False, stack=stack,
                             sep_g=True)
        g_warrs = self._get_g_idx_list_stack_even(ports.g, seg, stack)
        return (g_warrs, [ports.d], [ports.s]), self._num_in * seg * stack

    def draw_series_network_stack_odd(self, w: int, seg: int, stack: int, ridx: int,
                                      is_nmos: bool, ref_width: int = 0
                                      ) -> Tuple[Tuple[List[List[WireArray]], List[WireArray],
                                                       List[WireArray]], int]:
        ports_l, ports_r, g_warrs, tot_col = self.draw_network_shared_stack_odd(w, seg, stack, ridx,
                                                                                ref_width)

        if seg & 1:
            ext_ports_l = ports_l.s
            ext_ports_r = ports_r.s
            int_ports_l = ports_l.d
            int_ports_r = ports_r.d
            num_ext_ports_l = ports_l.num_s
            num_ext_ports_r = ports_r.num_s
            num_int_ports_l = ports_l.num_d
        else:
            ext_ports_l = ports_l.d
            ext_ports_r = ports_r.d
            int_ports_l = ports_l.s
            int_ports_r = ports_r.s
            num_ext_ports_l = ports_l.num_d
            num_ext_ports_r = ports_r.num_d
            num_int_ports_l = ports_l.num_s

        mid0_warrs = [int_ports_l]
        mid1_warrs = [int_ports_r]
        sup_warrs = []
        out_warrs = []

        num_sup_out_wires = seg - seg // 2

        for i in range(num_ext_ports_l):
            if i < num_sup_out_wires:
                sup_warrs.append(ext_ports_l[i])
            else:
                mid1_warrs.append(ext_ports_l[i])
        for i in range(num_ext_ports_r):
            if i < num_sup_out_wires:
                out_warrs.append(ext_ports_r[i])
            else:
                mid0_warrs.append(ext_ports_r[i])

        mid1_tid = self.get_track_id(ridx, MOSWireType.DS, wire_name='sig',
                                     wire_idx=0 if is_nmos else -1)
        self.connect_to_tracks(mid1_warrs, mid1_tid)

        if len(mid0_warrs) > 1 or num_int_ports_l > 1:
            mid0_tid = self.get_track_id(ridx, MOSWireType.DS, wire_name='sig',
                                         wire_idx=-1 if is_nmos else 0)
            self.connect_to_tracks(mid0_warrs, mid0_tid)

        return (g_warrs, out_warrs, sup_warrs), tot_col

    def draw_parallel_network_stack_odd(self, w: int, seg: int, stack: int, ridx: int,
                                        split: bool = True, ref_width: int = 0
                                        ) -> Tuple[Tuple[List[List[WireArray]], List[WireArray],
                                                         List[WireArray]], int]:
        if (seg & 1) or split:
            ports_l, ports_r, g_warrs, tot_col = self.draw_network_shared_stack_odd(w, seg, stack,
                                                                                    ridx, ref_width)
            sup_ports_l = ports_l.d
            out_ports_l = ports_l.s
            if seg & 1:
                sup_ports_r = ports_r.s
                out_ports_r = ports_r.d
            else:
                sup_ports_r = ports_r.d
                out_ports_r = ports_r.s

            sup_warrs = [sup_ports_l, sup_ports_r]
            out_warrs = [out_ports_l, out_ports_r]
        else:
            tot_col = self._num_in * seg * stack
            offset_col = (ref_width - tot_col) // 2 if tot_col < ref_width else 0
            ports = self.add_mos(ridx, offset_col, self._num_in * seg, w=w, g_on_s=False,
                                 stack=stack, sep_g=True)
            num_gs_per_inp = stack * seg // 2
            g_warrs = [list() for _ in range(3)]
            for i in range(ports.num_g):
                q, _ = divmod(i, num_gs_per_inp)
                g_warrs[q].append(ports.g[i])
            sup_warrs = [ports.s]
            out_warrs = [ports.d]

        return (g_warrs, out_warrs, sup_warrs), tot_col

    def draw_network_shared_stack_odd(self, w: int, seg: int, stack: int, ridx: int,
                                      ref_width: int = 0) \
            -> Tuple[MOSPorts, MOSPorts, List[List[WireArray]], int]:
        seg_r = self._num_in * seg // 2
        seg_l = self._num_in * seg - seg_r
        tot_col = (seg_l + seg_r) * stack + self.min_sep_col
        if tot_col < ref_width:
            col_l = (ref_width - tot_col) // 2
            col_r = tot_col + col_l
        else:
            col_l = 0
            col_r = tot_col
        g_on_s = bool(seg & 1)

        ports_l = self.add_mos(ridx, col_l, seg_l, w=w, g_on_s=g_on_s, stack=stack, sep_g=True)
        ports_r = self.add_mos(ridx, col_r, seg_r, w=w, g_on_s=g_on_s, stack=stack, sep_g=True,
                               flip_lr=True)

        num_gs_per_inp = stack * seg - stack * seg // 2
        g0, g1, g2 = [], [], []
        for i in range(ports_l.num_g):
            if i < num_gs_per_inp:
                g0.append(ports_l.g[i])
            else:
                g1.append(ports_l.g[i])
        for i in range(ports_r.num_g):
            if i < num_gs_per_inp:
                g2.append(ports_r.g[i])
            else:
                g1.append(ports_r.g[i])

        g_warrs = [g0, g1, g2]
        return ports_l, ports_r, g_warrs, tot_col

    def _get_gate_in_stack_even(self, idx: int, stack: int) -> int:
        """
        Left to right
        0, 1, 2, ..., nports-1, nports-1, nports-2, ..., 0, 0, 1, ...
        each element is repeated by number of stacks
        """
        q, r = divmod(2 * idx // stack, self._num_in)
        if q & 1:  # q is odd
            return self._num_in - 1 - r
        else:
            return r

    def _get_g_idx_list_stack_even(self, g_warr: WireArray, seg: int, stack: int
                                   ) -> List[List[WireArray]]:
        rv: List[List[WireArray]] = [[] for _ in range(self._num_in)]
        for idx in range(self._num_in * seg * stack // 2):
            rv[self._get_gate_in_stack_even(idx, stack)].append(g_warr[idx])
        return rv

    def _get_closest_unused_tidx(self, sig_locs: Mapping[str, Union[float, HalfInt]], layer: int,
                                 tidx_init: HalfInt, tidx_min: HalfInt = None,
                                 tidx_max: HalfInt = None) -> HalfInt:
        if tidx_min is None and tidx_max is None:
            raise ValueError(f"Either tidx_min or tidx_max has to be defined")

        if tidx_min is not None:
            tidx_init = max(tidx_init, tidx_min)
        elif tidx_max is not None:
            tidx_init = min(tidx_init, tidx_max)

        if tidx_min is not None and tidx_max is not None:
            tidx_high = tidx_init
            tidx_low = tidx_init
            while tidx_low >= tidx_min and tidx_high <= tidx_max:
                if not self._check_tidx_unavailable(tidx_low, sig_locs):
                    return tidx_low
                tidx_low = self.tr_manager.get_next_track(layer, tidx_low, 'sig', 'sig', up=False)
                if not self._check_tidx_unavailable(tidx_high, sig_locs):
                    return tidx_high
                tidx_high = self.tr_manager.get_next_track(layer, tidx_high, 'sig', 'sig', up=True)
            if tidx_low < tidx_min and tidx_high > tidx_max:
                raise ValueError("Cannot find unused track")
            if tidx_low < tidx_min:
                tidx_min = None
                tidx_init = tidx_high
            else:
                tidx_max = None
                tidx_init = tidx_low

        tidx = tidx_init
        if tidx_min is None:
            while tidx <= tidx_max:
                if not self._check_tidx_unavailable(tidx, sig_locs):
                    return tidx
                tidx = self.tr_manager.get_next_track(layer, tidx, 'sig', 'sig', up=True)
            raise ValueError("Cannot find unused track")

        elif tidx_max is None:
            while tidx >= tidx_min:
                if not self._check_tidx_unavailable(tidx, sig_locs):
                    return tidx
                tidx = self.tr_manager.get_next_track(layer, tidx, 'sig', 'sig', up=False)
            raise ValueError("Cannot find unused track")

    @staticmethod
    def _check_tidx_unavailable(tidx: HalfInt,
                                sig_locs: Mapping[str, Union[float, HalfInt]]) -> bool:
        return any([abs(tidx - used_tidx) < 1 for used_tidx in sig_locs.values()])

    def _get_hm_tid_list(self, ridx: int, sig_locs: Mapping[str, Union[float, HalfInt]],
                         key_name: str, stack: int, hm_layer: int, tr_w_h: int, up: bool
                         ) -> List[TrackID]:
        in_tidx = list(get_adj_tidx_list(self, ridx, sig_locs, MOSWireType.G, key_name, up))
        if stack & 1:
            in_tidx.append(in_tidx[0])
        else:
            if f'{key_name}2' not in sig_locs:
                in_tidx.append(
                    self.tr_manager.get_next_track(hm_layer, in_tidx[1], 'sig', 'sig', up=up))
            else:
                in2_tidx = sig_locs[f'{key_name}2']
                if in2_tidx in in_tidx:
                    raise ValueError('in0, in1, and in2 must all be on different tracks')
                in_tidx.append(in2_tidx)
        in_tid = [TrackID(hm_layer, tidx, width=tr_w_h) for tidx in in_tidx]
        return in_tid


class NAND3Core(NANDNOR3Core):
    """
    A 3-input NAND gate.
    Due to the similarities between NAND and NOR, please look at NANDNOR3Core for most of
    the implementation.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        NANDNOR3Core.__init__(self, temp_db, params, **kwargs)
        self._pdn_series = True

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'nand')


class NOR3Core(NANDNOR3Core):
    """
    A 3-input NOR gate.
    Due to the similarities between NAND and NOR, please look at NANDNOR3Core for most of
    the implementation.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        NANDNOR3Core.__init__(self, temp_db, params, **kwargs)
        self._pdn_series = False

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'nor')


class NOR2Core(MOSBase):
    """
    A 2-input NOR gate.

    'out' and 'in' pin direction is determined by vertical_out and vertical_in flags, respectively
    but regardless, in0_vm, in1_vm, out_vm, and out_hm are always available, and also if
    connect_inputs is True in0_hm, in1_hm will also be available.

    Assumes:

        1. PMOS row above NMOS row.
        2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'nor')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            seg_p='Number of segments of pmos.',
            seg_n='Number of segments of nmos.',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            is_guarded='True if it there should be guard ring around the cell',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            vertical_out='True to have output pin on vertical layer',
            vertical_in='True to have input pins on vertical layer.  Only used if is_guarded=True',
            vertical_sup='True to have supply unconnected on conn_layer.',
            tr_manager='override track manager',
            flip_ds='True to use drain conn for supply, used for abut gates'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            sig_locs={},
            is_guarded=False,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            vertical_out=True,
            vertical_in=True,
            vertical_sup=False,
            tr_manager=None,
            flip_ds=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack_p: int = self.params['stack_p']
        stack_n: int = self.params['stack_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Dict[str, float] = self.params['sig_locs']
        is_guarded: bool = self.params['is_guarded']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']
        vertical_out: bool = self.params['vertical_out']
        vertical_in: bool = self.params['vertical_in']
        vertical_sup: bool = self.params['vertical_sup']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        tot_col = 2 * max(seg_p * stack_p, seg_n * stack_n)
        self.set_mos_size(tot_col)

        tr_manager = self.params['tr_manager'] if self.params['tr_manager'] else self.tr_manager

        if is_guarded is False and stack_n != stack_p:
            raise ValueError(f'If is_guarded is False, then the layout generator requires that '
                             f'stack_n = {stack_n} == stack_p = {stack_p}.')

        pports = self.add_nand2(ridx_p, 0, seg_p, w=w_p, stack=stack_p)
        nports = self.add_nand2(ridx_n, 0, seg_n, w=w_n, stack=stack_n, other=True)

        flip_ds = self.params['flip_ds']
        if flip_ds:
            nports = NAND2Ports(nports.g1, nports.g0, nports.s, nports.d, nports.m)
            pports = NAND2Ports(pports.g1, pports.g0, pports.s, pports.d, pports.m)


        xr = self.bound_box.xh
        if vertical_sup:
            self.add_pin('VDD', pports.s, connect=True)
            self.add_pin('VSS', nports.s, connect=True)
        else:
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
            vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
            vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        mlm_in0 = mlm.get('in0', None)
        mlm_in1 = mlm.get('in1', None)
        nin_tid = get_adj_tid_list(self, ridx_n, sig_locs, MOSWireType.G, 'nin', True, tr_w_h)
        pin_tid = get_adj_tid_list(self, ridx_p, sig_locs, MOSWireType.G, 'pin', False, tr_w_h)

        if is_guarded:
            n_in0 = self.connect_to_tracks(nports.g0, nin_tid[0], min_len_mode=mlm_in0)
            n_in1 = self.connect_to_tracks(nports.g1, nin_tid[1], min_len_mode=mlm_in1)
            p_in0 = self.connect_to_tracks(pports.g0, pin_tid[0], min_len_mode=mlm_in0)
            p_in1 = self.connect_to_tracks(pports.g1, pin_tid[1], min_len_mode=mlm_in1)
            in0_tidx = sig_locs.get('in0', self.grid.coord_to_track(vm_layer, self.bound_box.xl,
                                                                    RoundMode.GREATER_EQ))
            in1_tidx = sig_locs.get('in1', self.grid.coord_to_track(vm_layer, self.bound_box.xh,
                                                                    RoundMode.LESS_EQ))
            if vertical_in:
                in0_vm = self.connect_to_tracks([n_in0, p_in0],
                                                TrackID(vm_layer, in0_tidx, width=tr_w_v))
                in1_vm = self.connect_to_tracks([n_in1, p_in1],
                                                TrackID(vm_layer, in1_tidx, width=tr_w_v))
                self.add_pin('in<0>', in0_vm)
                self.add_pin('in<1>', in1_vm)
                self.add_pin('nin<0>', n_in0, hide=True)
                self.add_pin('nin<1>', n_in1, hide=True)
                self.add_pin('pin<0>', p_in0, hide=True)
                self.add_pin('pin<1>', p_in1, hide=True)
            else:
                self.add_pin('nin<0>', n_in0, label='in<0>:')
                self.add_pin('nin<1>', n_in1, label='in<1>:')
                self.add_pin('pin<0>', p_in0, label='in<0>:')
                self.add_pin('pin<1>', p_in1, label='in<1>:')
        else:
            key_name = 'nin' if any(k in sig_locs for k in ['nin', 'nin0']) else 'pin'
            in_tid = get_adj_tid_list(self, ridx_n, sig_locs, MOSWireType.G, key_name, True, tr_w_h)
            in0_vm, in1_vm = [], []
            in0 = self.connect_to_tracks(list(chain(nports.g0, pports.g0)), in_tid[0],
                                         min_len_mode=mlm_in0, ret_wire_list=in0_vm)
            in1 = self.connect_to_tracks(list(chain(nports.g1, pports.g1)), in_tid[1],
                                         min_len_mode=mlm_in0, ret_wire_list=in1_vm)
            self.add_pin('nin<0>', in0, hide=True)
            self.add_pin('pin<0>', in0, hide=True)
            self.add_pin('nin<1>', in1, hide=True)
            self.add_pin('pin<1>', in1, hide=True)
            self.add_pin('in<0>', in0_vm)
            self.add_pin('in<1>', in1_vm)

        pd_tidx = sig_locs.get('pout',
                               self.get_track_index(ridx_p, MOSWireType.DS_GATE,
                                                    wire_name='sig', wire_idx=0))
        pd_tid = TrackID(hm_layer, pd_tidx, width=tr_w_h)
        mlm_out = mlm.get('out', MinLenMode.MIDDLE)
        if self.can_short_adj_tracks and not is_guarded:
            # we can short adjacent source/drain tracks together, so we can avoid going to vm_layer
            out = self.connect_to_tracks([pports.d, nports.d], pd_tid, min_len_mode=mlm_out)
            self.add_pin('pout', out, hide=True)
            self.add_pin('out', nports.d)
        else:
            # need to use vm_layer to short adjacent tracks.
            nd_tidx = sig_locs.get('nout',
                                   self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                        wire_name='sig', wire_idx=-1))
            nd_tid = TrackID(hm_layer, nd_tidx, width=tr_w_h)
            nout = self.connect_to_tracks(nports.d, nd_tid, min_len_mode=mlm_out)
            pout = self.connect_to_tracks(pports.d, pd_tid, min_len_mode=mlm_out)
            vm_tidx = sig_locs.get('out', self.grid.coord_to_track(vm_layer, nout.middle,
                                                                   mode=RoundMode.GREATER_EQ))
            if vertical_out:
                out = self.connect_to_tracks([pout, nout], TrackID(vm_layer, vm_tidx,
                                                                   tr_manager.get_width(vm_layer, 'sig')))
                self.add_pin('pout', pout, hide=True)
                self.add_pin('nout', nout, hide=True)
                self.add_pin('out', out)
            else:
                self.add_pin('pout', pout, label='out:')
                self.add_pin('nout', nout, label='out:')

        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            stack_p=stack_p,
            stack_n=stack_n,
        )


class XORCore(MOSBase):
    """
    A 2-input XNOR gate.

    'out' and 'in' pin direction is determined by vertical_out and vertical_in flags, respectively
    but regardless, in0_vm, in1_vm, out_vm, and out_hm are always available, and also if
    connect_inputs is True in0_hm, in1_hm will also be available.

    Assumes:

        1. PMOS row above NMOS row.
        2. PMOS gate connections on bottom, NMOS gate connections on top.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'xor')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            seg_p='Number of segments of pmos.',
            seg_n='Number of segments of nmos.',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            is_guarded='True if it there should be guard ring around the cell',
            min_len_mode='A Dictionary specfiying min_len_mode for connections',
            vertical_out='True to have output pin on vertical layer',
            vertical_in='True to have input pins on vertical layer.  Only used if is_guarded=True',
            vertical_sup='True to have supply unconnected on conn_layer.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
            sig_locs={},
            is_guarded=False,
            min_len_mode=dict(
                in0=MinLenMode.NONE,
                in1=MinLenMode.NONE,
                out=MinLenMode.MIDDLE,
            ),
            vertical_out=True,
            vertical_in=True,
            vertical_sup=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack_p: int = self.params['stack_p']
        stack_n: int = self.params['stack_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Dict[str, float] = self.params['sig_locs']
        is_guarded: bool = self.params['is_guarded']
        mlm: Dict[str, MinLenMode] = self.params['min_len_mode']
        vertical_out: bool = self.params['vertical_out']
        vertical_in: bool = self.params['vertical_in']
        vertical_sup: bool = self.params['vertical_sup']

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        min_sep = self.min_sep_col
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        tot_col = 4 * max(seg_p * stack_p, seg_n * stack_n) + min_sep
        self.set_mos_size(tot_col)

        if is_guarded is False and stack_n != stack_p:
            raise ValueError(f'If is_guarded is False, then the layout generator requires that '
                             f'stack_n = {stack_n} == stack_p = {stack_p}.')

        pports0 = self.add_nand2(ridx_p, 0, seg_p, w=w_p, stack=stack_p)
        nports0 = self.add_nand2(ridx_n, 0, seg_n, w=w_n, stack=stack_n)
        pports1 = self.add_nand2(ridx_p, tot_col, seg_p, w=w_p, stack=stack_p, flip_lr=True)
        nports1 = self.add_nand2(ridx_n, tot_col, seg_n, w=w_n, stack=stack_n, flip_lr=True)

        xr = self.bound_box.xh
        if vertical_sup:
            self.add_pin('VDD', [pports0.s, pports1.s], connect=True)
            self.add_pin('VSS', [nports0.s, nports1.s], connect=True)
        else:
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
            vdd = self.connect_to_tracks([pports0.s, pports1.s], ps_tid, track_lower=0, track_upper=xr)
            vss = self.connect_to_tracks([nports0.s, nports1.s], ns_tid, track_lower=0, track_upper=xr)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        mlm_in0 = mlm.get('in0', None)
        mlm_in1 = mlm.get('in1', None)
        nin_tid = get_adj_tid_list(self, ridx_n, sig_locs, MOSWireType.G, 'nin', True, tr_w_h)
        pin_tid = get_adj_tid_list(self, ridx_p, sig_locs, MOSWireType.G, 'pin', False, tr_w_h)

        if is_guarded:
            n_in0 = self.connect_to_tracks(nports.g0, nin_tid[0], min_len_mode=mlm_in0)
            n_in1 = self.connect_to_tracks(nports.g1, nin_tid[1], min_len_mode=mlm_in1)
            p_in0 = self.connect_to_tracks(pports.g0, pin_tid[0], min_len_mode=mlm_in0)
            p_in1 = self.connect_to_tracks(pports.g1, pin_tid[1], min_len_mode=mlm_in1)
            in0_tidx = sig_locs.get('in0', self.grid.coord_to_track(vm_layer, self.bound_box.xl,
                                                                    RoundMode.GREATER_EQ))
            in1_tidx = sig_locs.get('in1', self.grid.coord_to_track(vm_layer, self.bound_box.xh,
                                                                    RoundMode.LESS_EQ))
            if vertical_in:
                in0_vm = self.connect_to_tracks([n_in0, p_in0],
                                                TrackID(vm_layer, in0_tidx, width=tr_w_v))
                in1_vm = self.connect_to_tracks([n_in1, p_in1],
                                                TrackID(vm_layer, in1_tidx, width=tr_w_v))
                self.add_pin('in<0>', in0_vm)
                self.add_pin('in<1>', in1_vm)
                self.add_pin('nin<0>', n_in0, hide=True)
                self.add_pin('nin<1>', n_in1, hide=True)
                self.add_pin('pin<0>', p_in0, hide=True)
                self.add_pin('pin<1>', p_in1, hide=True)
            else:
                self.add_pin('nin<0>', n_in0, label='in<0>:')
                self.add_pin('nin<1>', n_in1, label='in<1>:')
                self.add_pin('pin<0>', p_in0, label='in<0>:')
                self.add_pin('pin<1>', p_in1, label='in<1>:')
        else:
            _, in_mid_tidx = tr_manager.place_wires(vm_layer, ['sig']*2,
                                                    center_coord=self.arr_info.col_to_coord(self.num_cols//2))
            nin_tid = get_adj_tid_list(self, ridx_n, sig_locs, MOSWireType.G,  'nin', True, tr_w_h)
            pin_tid = get_adj_tid_list(self, ridx_p, sig_locs, MOSWireType.G, 'pin', False, tr_w_h)
            in0 = self.connect_to_tracks(nports0.g0+pports0.g0, nin_tid[0], min_len_mode=mlm_in0)
            in0b = self.connect_to_tracks(nports1.g0+pports1.g0, pin_tid[0], min_len_mode=mlm_in0)
            in1b_l = self.connect_to_tracks(pports0.g1, pin_tid[1], min_len_mode=mlm_in0)
            in1_l = self.connect_to_tracks(nports0.g1, nin_tid[1], min_len_mode=mlm_in0)
            in1b_r = self.connect_to_tracks(nports1.g1, nin_tid[1], min_len_mode=mlm_in0)
            in1_r = self.connect_to_tracks(pports1.g1, pin_tid[1], min_len_mode=mlm_in0)
            _, in_l_tidx = tr_manager.place_wires(vm_layer, ['sig']*2, center_coord=in1_l.middle)
            _, in_r_tidx = tr_manager.place_wires(vm_layer, ['sig']*2, center_coord=in1_r.middle)
            p_d_tidx = self.get_track_id(ridx_p, MOSWireType.DS, wire_name='sig', wire_idx=-1)
            n_d_tidx = self.get_track_id(ridx_n, MOSWireType.DS, wire_name='sig', wire_idx=0)
            in1b_l_vm, in1_l_vm = self.connect_differential_tracks(in1b_l, in1_l, vm_layer, in_l_tidx[0], in_l_tidx[1])
            in1b_r_vm, in1_r_vm = self.connect_differential_tracks(in1b_r, in1_r, vm_layer, in_r_tidx[0], in_r_tidx[1])
            self.connect_to_tracks([in1b_l_vm, in1b_r_vm], p_d_tidx)
            self.connect_to_tracks([in1_l_vm, in1_r_vm], n_d_tidx)
            self.add_pin('in<0>', in0)
            self.add_pin('inb<0>', in0b)
            self.add_pin('in1_r', in1_r, label='in<1>')
            self.add_pin('in1_l', in1_l, label='in<1>')
            self.add_pin('in1b_r', in1b_r, label='inb<1>')
            self.add_pin('in1b_l', in1b_l, label='inb<1>')

        pd_tidx = sig_locs.get('pout', self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0))
        pd_tid = TrackID(hm_layer, pd_tidx, width=tr_w_h)
        mlm_out = mlm.get('out', MinLenMode.MIDDLE)
        if self.can_short_adj_tracks and not is_guarded:
            # we can short adjacent source/drain tracks together, so we can avoid going to vm_layer
            out = self.connect_to_tracks([pports.d, nports.d], pd_tid, min_len_mode=mlm_out)
            self.add_pin('pout', out, hide=True)
            self.add_pin('out', nports.d)
        else:
            # need to use vm_layer to short adjacent tracks.
            nd_tidx = sig_locs.get('nout',
                                   self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                        wire_name='sig', wire_idx=-1))
            nd_tid = TrackID(hm_layer, nd_tidx, width=tr_w_h)
            nout = self.connect_to_tracks([nports0.d, nports1.d], nd_tid, min_len_mode=mlm_out)
            pout = self.connect_to_tracks([pports0.d, pports1.d], pd_tid, min_len_mode=mlm_out)
            vm_tidx = sig_locs.get('out', self.grid.coord_to_track(vm_layer, nout.middle,
                                                                   mode=RoundMode.GREATER_EQ))
            if vertical_out:
                out = self.connect_to_tracks([pout, nout], TrackID(vm_layer, vm_tidx))
                self.add_pin('pout', pout, hide=True)
                self.add_pin('nout', nout, hide=True)
                self.add_pin('out', out)
            else:
                self.add_pin('pout', pout, label='out:')
                self.add_pin('nout', nout, label='out:')

        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=self.place_info.lch,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            stack_p=stack_p,
            stack_n=stack_n,
        )


class PassGateCore(MOSBase):
    """CMOS pass gate
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_digital__passgate

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='Number of segments.',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
            vertical_out='whether output pin (d) on vm_layer',
            vertical_in='whether input pin (s) on vm_layer, only relevant if is_guarded = True.',
            vertical_sup='True to remove hm supply',
            is_guarded='True if it there should be guard ring around the cell',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs={},
            vertical_out=True,
            vertical_in=True,
            vertical_sup=True,
            is_guarded=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']
        vertical_out: bool = self.params['vertical_out']
        vertical_in: bool = self.params['vertical_in']
        is_guarded: bool = self.params['is_guarded']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if vertical_out and self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        pports = self.add_mos(ridx_p, 0, seg_p, w=w_p)
        nports = self.add_mos(ridx_n, 0, seg_n, w=w_n)
        self.set_mos_size()

        # VDD/VSS wires
        if not self.params['vertical_sup']:
            xr = self.bound_box.xh
            ns_tid = self.get_track_id(ridx_n, MOSWireType.DS_GATE, wire_name='sup')
            ps_tid = self.get_track_id(ridx_p, MOSWireType.DS_GATE, wire_name='sup')
            vss = self.add_wires(hm_layer, ns_tid.base_index, 0, xr, width=ns_tid.width)
            vdd = self.add_wires(hm_layer, ps_tid.base_index, 0, xr, width=ps_tid.width)
            self.add_pin('VDD', vdd)
            self.add_pin('VSS', vss)

        # input will connect on the track id aligned with transistor's source
        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        en_tidx = sig_locs.get('en', self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig'))
        enb_tidx = sig_locs.get('enb', self.get_track_index(ridx_p, MOSWireType.G,
                                                            wire_name='sig', wire_idx=-1))
        nd_tidx = sig_locs.get('nd',
                               self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                    wire_name='sig', wire_idx=-1))
        pd_tidx = sig_locs.get('pd',
                               self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig'))

        en_warr = self.connect_to_tracks(nports.g, TrackID(hm_layer, en_tidx, width=tr_w_h))
        enb_warr = self.connect_to_tracks(pports.g, TrackID(hm_layer, enb_tidx, width=tr_w_h))
        nd_warr = self.connect_to_tracks(nports.d, TrackID(hm_layer, nd_tidx, width=tr_w_h))
        pd_warr = self.connect_to_tracks(pports.d, TrackID(hm_layer, pd_tidx, width=tr_w_h))

        self.add_pin('nd', nd_warr, hide=True)
        self.add_pin('pd', pd_warr, hide=True)
        if vertical_out:
            in_tid = TrackID(vm_layer,
                             self.grid.coord_to_track(vm_layer, nd_warr.middle,
                                                      mode=RoundMode.NEAREST),
                             width=tr_w_v)

            in_warr = self.connect_to_tracks([nd_warr, pd_warr], track_id=in_tid)
            self.add_pin('d', in_warr)  # pin on vm layer
        else:
            self.add_pin('d', [nd_warr, pd_warr])

        if is_guarded:
            tr_man = self.tr_manager
            ns_tidx = sig_locs.get('ns', tr_man.get_next_track(hm_layer, nd_tidx, 'sig', 'sig'))
            ps_tidx = sig_locs.get('ps', tr_man.get_next_track(hm_layer, pd_tidx, 'sig', 'sig',
                                                               up=False))

            ps_warr = self.connect_to_tracks(nports.s, TrackID(hm_layer, ns_tidx, width=tr_w_h))
            ns_warr = self.connect_to_tracks(pports.s, TrackID(hm_layer, ps_tidx, width=tr_w_h))

            if vertical_in:
                s_tidx = self.grid.find_next_track(vm_layer, ns_warr.lower,
                                                   tr_width=tr_w_v, mode=RoundMode.GREATER_EQ)
                s_tidx = sig_locs.get('s', s_tidx)
                s_warr = self.connect_to_tracks([ps_warr, ns_warr],
                                                TrackID(vm_layer, s_tidx, width=tr_w_v))
                self.add_pin('s', s_warr)

            self.add_pin('ns', ns_warr, hide=True)
            self.add_pin('ps', ps_warr, hide=True)
        else:
            s_tidx = sig_locs.get('s', None)
            if s_tidx is None:
                s_tidx = sig_locs.get('ns', None)
                if s_tidx is None:
                    s_tidx = sig_locs.get('ps', self.grid.get_middle_track(en_tidx, enb_tidx))
            s_warr = self.connect_to_tracks([nports.s, pports.s],
                                            TrackID(hm_layer, s_tidx, width=tr_w_h))
            self.add_pin('ns', s_warr, hide=True)
            self.add_pin('ps', s_warr, hide=True)
            self.add_pin('s', s_warr)

        self.add_pin('en', en_warr)
        self.add_pin('enb', enb_warr)

        self.sch_params = dict(
            lch=self.place_info.lch,
            seg_p=seg_p,
            seg_n=seg_n,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
        )


class LatchCore(MOSBase):
    """A transmission gate based latch."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @property
    def seg_in(self) -> int:
        return self.sch_params['seg_dict']['tin']

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output inverter.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='input stage fanout.',
            fanout_kp='keeper stage fanout.',
            vertical_sup='True to have vertical supply',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
            vertical_sup=False
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        # setup floorplan
        self.draw_base(pinfo)

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']
        vertical_sup: bool = self.params['vertical_sup']

        # compute track locations
        tr_manager = self.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        if sig_locs is None:
            sig_locs = {}
        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        t0_in_tidx = sig_locs.get(key, self.get_track_index(ridx_p, MOSWireType.G,
                                                            wire_name='sig', wire_idx=0))
        t0_enb_tidx = sig_locs.get('pclkb', self.get_track_index(ridx_p, MOSWireType.G,
                                                                 wire_name='sig', wire_idx=1))
        t0_en_tidx = sig_locs.get('nclk', self.get_track_index(ridx_n, MOSWireType.G,
                                                               wire_name='sig', wire_idx=0))
        t1_en_tidx = sig_locs.get('nclkb', self.get_track_index(ridx_n, MOSWireType.G,
                                                                wire_name='sig', wire_idx=1))
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        nd1_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)

        seg_t1 = max(1, int(round(seg / (2 * fanout_kp))) * 2)
        seg_t0 = max(2 * seg_t1, max(2, int(round(seg / (2 * fanout_in))) * 2))
        params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      sig_locs={'nin': t0_en_tidx, 'pout': pd1_tidx, 'nout': nd1_tidx}, vertical_sup=vertical_sup)
        inv_master = self.new_template(InvCore, params=params)

        params['seg'] = seg_t0
        params['vertical_out'] = False
        params['sig_locs'] = {'nin': t0_enb_tidx, 'pout': pd0_tidx, 'nout': nd0_tidx,
                              'nen': t0_en_tidx, 'pen': t0_in_tidx}
        t0_master = self.new_template(InvTristateCore, params=params)
        params['seg'] = seg_t1
        params['sig_locs'] = {'nin': t0_in_tidx, 'pout': pd0_tidx, 'nout': nd0_tidx,
                              'nen': t1_en_tidx, 'pen': t0_enb_tidx}
        params['min_len_mode'] = {'en': MinLenMode.MIDDLE, 'enb': MinLenMode.MIDDLE}
        t1_master = self.new_template(InvTristateCore, params=params)

        # set size
        blk_sp = self.min_sep_col
        t0_ncol = t0_master.num_cols
        t1_ncol = t1_master.num_cols
        inv_ncol = inv_master.num_cols
        num_cols = t0_ncol + t1_ncol + inv_ncol + blk_sp * 2
        self.set_mos_size(num_cols)

        # add instances
        t1_col = t0_ncol + blk_sp
        inv_col = num_cols - inv_ncol
        t0 = self.add_tile(t0_master, 0, 0)
        t1 = self.add_tile(t1_master, 0, t1_col)
        inv = self.add_tile(inv_master, 0, inv_col)

        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in (t0, t1, inv):
            vss_list.append(inst.get_pin('VSS'))
            vdd_list.append(inst.get_pin('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # export input
        self.reexport(t0.get_port('nin'), net_name='nin', hide=True)
        self.reexport(t0.get_port('pin'), net_name='pin', hide=True)
        self.reexport(t0.get_port('in'))

        # connect output
        out = inv.get_pin('out')
        in2 = t1.get_pin('nin')
        self.connect_to_track_wires(in2, out)
        self.add_pin('out', out)
        self.add_pin('nout', in2, hide=True)
        self.add_pin('pout', in2, hide=True)

        # connect middle node
        col = inv_col - max(1, blk_sp // 2)
        mid_tid = TrackID(vm_layer, self.arr_info.get_source_track(col), width=tr_w_v)
        warrs = [t0.get_pin('pout'), t0.get_pin('nout'), t1.get_pin('pout'), t1.get_pin('nout'),
                 inv.get_pin('nin')]
        self.connect_to_tracks(warrs, mid_tid)
        self.add_pin('outb', inv.get_pin('in'))
        self.add_pin('noutb', inv.get_pin('nin'), hide=True)
        self.add_pin('poutb', inv.get_pin('nin'), hide=True)

        # connect clocks
        clk_tidx = sig_locs.get('clk', self.arr_info.get_source_track(t1_col + 1))
        clkb_tidx = sig_locs.get('clkb', self.arr_info.get_source_track(t1_col - blk_sp - 1))
        clk_tid = TrackID(vm_layer, clk_tidx, width=tr_w_v)
        clkb_tid = TrackID(vm_layer, clkb_tidx, width=tr_w_v)
        t0_en = t0.get_pin('en')
        t1_en = t1.get_pin('en')
        t1_enb = t1.get_pin('enb')
        t0_enb = t0.get_pin('enb')
        clk = self.connect_to_tracks([t0_en, t1_enb], clk_tid)
        clkb = self.connect_to_tracks([t0_enb, t1_en], clkb_tid)
        self.add_pin('clk', clk)
        self.add_pin('clkb', clkb)
        self.add_pin('nclk', t0_en, hide=True)
        self.add_pin('pclk', t1_enb, hide=True)
        self.add_pin('nclkb', t1_en, hide=True)
        self.add_pin('pclkb', t0_enb, hide=True)

        # set properties
        self._sch_params = dict(
            lch=self.place_info.lch,
            w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
            w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
            th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
            th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
            seg_dict=dict(tin=seg_t0, tfb=seg_t1, buf=seg),
        )


class FlopCore(MOSBase):
    """A transmission gate based flip-flop."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        self._cntr_col_clk = None

    @property
    def seg_in(self):
        return self.sch_params['seg_m']['in']

    @property
    def cntr_col_clk(self):
        return self._cntr_col_clk

    def get_schematic_class_inst(self) -> Optional[Type[Module]]:
        rst: bool = self.params['resetable']
        scan: bool = self.params['scanable']
        if scan:
            raise ValueError('See Developer')
        if rst:
            # noinspection PyTypeChecker
            return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'rst_flop')
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_digital', 'flop')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output inverter.',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='latch input stage fanout.',
            fanout_kp='latch keeper stage fanout.',
            fanout_lat='fanout between latches.',
            fanout_mux='fanout of scan mux, if present.',
            seg_ck='number of segments for clock inverter.  0 to disable.',
            seg_mux='Dictionary of segments for scan mux, if scanable',
            resetable='True if flop is resetable, default is False',
            scanable='True if flop needs to have scanability',
            extra_sp='This parameter is added to the min value of one of the separations '
                     '(mostly used to make power vertical stripes aligned)',
            vertical_sup='True to have vertical supply',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
            fanout_lat=4,
            fanout_mux=4,
            seg_ck=0,
            seg_mux=None,
            resetable=False,
            scanable=False,
            extra_sp=0,
            vertical_sup=False
        )

    def draw_layout(self):
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])

        seg: int = self.params['seg']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']
        fanout_lat: float = self.params['fanout_lat']
        fanout_mux: float = self.params['fanout_mux']
        seg_ck: int = self.params['seg_ck']
        seg_mux: Optional[Dict[str, int]] = self.params['seg_mux']
        rst: bool = self.params['resetable']
        scan: bool = self.params['scanable']
        extra_sp: int = self.params['extra_sp']
        vertical_sup: bool = self.params['vertical_sup']

        # setup floorplan
        self.draw_base(pinfo)
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # compute track locations
        if sig_locs is None:
            sig_locs = {}
        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        in_tidx = sig_locs.get(key, self.get_track_index(ridx_p, MOSWireType.G,
                                                         wire_name='sig', wire_idx=0))
        pclkb_tidx = sig_locs.get('pclkb', self.get_track_index(ridx_p, MOSWireType.G,
                                                                wire_name='sig', wire_idx=0))
        nclk_idx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        nclkb_idx = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        pclk_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        clk_idx = sig_locs.get('clk', None)
        clkb_idx = sig_locs.get('clkb', None)

        # make masters
        lat_params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                          sig_locs={'nclk': nclk_idx, 'nclkb': nclkb_idx, 'pclk': pclk_tidx,
                                    'nin': pclk_tidx, 'pclkb': pclkb_tidx,
                                    'pout': sig_locs.get('pout', pclkb_tidx)},
                          fanout_in=fanout_in, fanout_kp=fanout_kp, vertical_sup=vertical_sup)

        s_master = self.new_template(RstLatchCore if rst else LatchCore, params=lat_params)
        seg_m = max(2, int(round(s_master.seg_in / (2 * fanout_lat))) * 2)
        lat_params['seg'] = seg_m
        lat_params['sig_locs'] = lat_sig_locs = {'nclk': nclkb_idx, 'nclkb': nclk_idx,
                                                 'pclk': pclkb_tidx, 'nin': in_tidx,
                                                 'pclkb': pclk_tidx}
        if clk_idx is not None:
            lat_sig_locs['clkb'] = clk_idx
        if clkb_idx is not None:
            lat_sig_locs['clk'] = clkb_idx

        m_master = self.new_template(RstLatchCore if rst else LatchCore, params=lat_params)

        cur_col = 0
        blk_sp = self.min_sep_col
        pd0_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1_tidx = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        nd0_tidx = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pg0_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1_tidx = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)

        inst_list = []
        mux_inst = None
        mux_master = None
        if scan:
            if seg_mux is None:
                raise ValueError('Please specify segments for scan mux.')
            mux_params = dict(pinfo=pinfo, seg=seg_mux['seg'], w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                              ridx_n=ridx_n, sel_seg=seg_mux['sel_seg'], fout=fanout_mux,
                              sig_locs={'pselb': pd1_tidx, 'pin1': pg0_tidx, 'penb': pg1_tidx},
                              vertical_out=False, vertical_sup=vertical_sup)
            mux_master = self.new_template(Mux2to1Core, params=mux_params)
            mux_inst = self.add_tile(mux_master, 0, cur_col)
            mux_ncol = mux_master.num_cols
            cur_col += mux_ncol + blk_sp
            inst_list.append(mux_inst)

        m_ncol = m_master.num_cols
        s_ncol = s_master.num_cols
        m_inv_sp = blk_sp # if rst else 0
        inv_master = None
        if seg_ck > 0:
            params = dict(pinfo=pinfo, seg=seg_ck, w_p=w_p, w_n=w_n, ridx_p=ridx_p,
                          ridx_n=ridx_n, sig_locs={'nin': nclk_idx, 'pout': pd0_tidx,
                                                   'nout': nd0_tidx}, vertical_sup=vertical_sup)

            inv_master = self.new_template(InvCore, params=params)
            ncol = cur_col + m_ncol + s_ncol + blk_sp + inv_master.num_cols + m_inv_sp
            scol = cur_col + m_ncol + inv_master.num_cols + blk_sp + m_inv_sp + extra_sp
            b_inst = self.add_tile(inv_master, 0, cur_col + m_ncol + m_inv_sp)
            self._cntr_col_clk = scol - (blk_sp + extra_sp) // 2
            inst_list.append(b_inst)
        else:
            ncol = cur_col + m_ncol + s_ncol + blk_sp
            scol = cur_col + m_ncol + blk_sp + extra_sp
            self._cntr_col_clk = scol - (blk_sp + extra_sp) // 2
            b_inst = None

        # set size
        self.set_mos_size(ncol)

        # add instances
        m_inst = self.add_tile(m_master, 0, cur_col)
        s_inst = self.add_tile(s_master, 0, scol)
        inst_list.append(m_inst)
        inst_list.append(s_inst)

        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in inst_list:
            vss_list.extend(inst.get_all_port_pins('VSS'))
            vdd_list.extend(inst.get_all_port_pins('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # connect intermediate node
        mid = self.connect_wires([s_inst.get_pin('nin'), m_inst.get_pin('nout')])
        self.add_pin('mid', mid)
        # connect clocks
        pclkb = self.connect_wires([s_inst.get_pin('pclkb'), m_inst.get_pin('pclk')])
        if b_inst is None:
            self.connect_wires([s_inst.get_pin('nclk'), m_inst.get_pin('nclkb')])
            self.reexport(m_inst.get_port('clk'), net_name='clkb')
            self.reexport(m_inst.get_port('nclk'), net_name='nclkb')
        else:
            self.connect_wires([s_inst.get_pin('nclk'), m_inst.get_pin('nclkb'),
                                b_inst.get_pin('nin')])
            self.connect_to_track_wires(pclkb, b_inst.get_pin('out'))

        # connect rst if rst is True
        if rst:
            rst_warr = self.connect_wires([s_inst.get_pin('nrst'), m_inst.get_pin('nrst')])
            self.add_pin('nrst', rst_warr, hide=True)
            self.add_pin('prst', rst_warr, hide=True)
            self.add_pin('rst', rst_warr, label='rst:') #[s_inst.get_pin('rst'), m_inst.get_pin('rst')], label='rst:')

        # connect mux output to flop input if scan is true
        # hm_layer = self.conn_layer + 1
        # vm_layer = hm_layer + 1
        if scan:
            flop_in = m_inst.get_pin('nin')
            mux_out_vm_tidx = self.grid.coord_to_track(vm_layer, flop_in.lower,
                                                       mode=RoundMode.NEAREST)
            mux_out = [mux_inst.get_pin('pout'), mux_inst.get_pin('nout'), flop_in]
            flop_in_vm = self.connect_to_tracks(mux_out, TrackID(vm_layer, mux_out_vm_tidx))
            self.add_pin('flop_in', flop_in_vm, hide=True)

        # add pins
        # NOTE: use reexport so hidden pins propagate correctly
        if scan:
            self.reexport(mux_inst.get_port('in<0>'), net_name='in', hide=False)
            self.reexport(mux_inst.get_port('in<1>'), net_name='scan_in', hide=False)
            self.reexport(mux_inst.get_port('sel'), net_name='scan_en', hide=False)
        else:
            self.reexport(m_inst.get_port('in'))
        self.reexport(m_inst.get_port('nin'))
        self.reexport(m_inst.get_port('pin'))
        self.reexport(s_inst.get_port('out'))
        self.reexport(s_inst.get_port('nout'))
        self.reexport(s_inst.get_port('pout'))
        self.reexport(m_inst.get_port('clkb'), net_name='clk')
        self.reexport(m_inst.get_port('nclkb'), net_name='nclk')
        self.reexport(s_inst.get_port('outb'), net_name='outb')
        self.reexport(s_inst.get_port('noutb'), net_name='noutb')
        self.reexport(s_inst.get_port('poutb'), net_name='poutb')
        if rst:
            self.reexport(m_inst.get_port('mid_vm'), net_name='mid_vm_m', hide=True)
            self.reexport(s_inst.get_port('mid_vm'), net_name='mid_vm_s', hide=True)

        # set properties
        if rst:
            self.sch_params = dict(
                m_params=m_master.sch_params,
                s_params=s_master.sch_params,
                inv_params=inv_master.sch_params if inv_master else None
            )
        else:
            self.sch_params = dict(
                lch=self.place_info.lch,
                w_n=self.place_info.get_row_place_info(ridx_n).row_info.width if w_n == 0 else w_n,
                w_p=self.place_info.get_row_place_info(ridx_p).row_info.width if w_p == 0 else w_p,
                th_n=self.place_info.get_row_place_info(ridx_n).row_info.threshold,
                th_p=self.place_info.get_row_place_info(ridx_p).row_info.threshold,
                seg_m=m_master.sch_params['seg_dict'],
                seg_s=s_master.sch_params['seg_dict'],
                seg_ck=seg_ck,
            )

        if scan:
            self.sch_params = dict(
                rst_flop_params=self.sch_params,
                mux_params=mux_master.sch_params,
            )


class RstLatchCore(MOSBase):
    """A transmission gate based latch with reset pin."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)
        #self._seg_in = None

    @property
    def seg_in(self) -> int:
        return self._seg_in
        #return self.sch_params['seg_dict']['tin']

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_sync_sar_adc', 'rst_latch')

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='number of segments of output NOR.',
            seg_dict='Dictionary of number of segments',
            w_p='pmos width.',
            w_n='nmos width.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Signal track location dictionary.',
            fanout_in='input stage fanout.',
            fanout_kp='keeper stage fanout.',
            vertical_clk='True to have vertical clk and clkb',
            vertical_sup='True to have vertical supply',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs=None,
            fanout_in=4,
            fanout_kp=8,
            seg_dict=None,
            seg=1,
            vertical_clk=True,
            vertical_sup=2,
    )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        # setup floorplan
        self.draw_base(pinfo)

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')
        

        seg: int = self.params['seg']
        seg_dict: Optional[Dict[str, int]] = self.params['seg_dict']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Optional[Dict[str, float]] = self.params['sig_locs']
        fanout_in: float = self.params['fanout_in']
        fanout_kp: float = self.params['fanout_kp']
        vertical_clk: bool = self.params['vertical_clk']
        vertical_sup: bool = self.params['vertical_sup']

        # compute track locations and create masters
        tr_manager = self.tr_manager
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        if sig_locs is None:
            sig_locs = {}

        ng0 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=0)
        ng1 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg0 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=0)
        pg1 = self.get_track_index(ridx_p, MOSWireType.G, wire_name='sig', wire_idx=1)
        pg2 = self.get_track_index(ridx_n, MOSWireType.G, wire_name='sig', wire_idx=2) #why does p but not n wor
        nd0 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        nd1 = self.get_track_index(ridx_n, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        pd0 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        pd1 = self.get_track_index(ridx_p, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        if seg_dict is None:
            seg_t1 = max(1, int(round(seg / (2 * fanout_kp))) * 2)
            seg_t0 = self._seg_in = max(2 * seg_t1, max(2, int(round(seg / (2 * fanout_in))) * 2))
        else:
            seg = seg_dict['nor']
            seg_t1 = seg_dict['keep']
            seg_t0 = seg_dict['in']

        nor_sig_locs = dict(
            nin0=sig_locs.get('rst', pg2),
            nin1=sig_locs.get('nclk', ng0),
            pout=pd1,
            nout=nd1,
        )
        params = dict(pinfo=pinfo, seg=seg, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      sig_locs=nor_sig_locs, vertical_sup=vertical_sup)
        nor_master = self.new_template(NOR2Core, params=params)

        key = 'in' if 'in' in sig_locs else ('nin' if 'nin' in sig_locs else 'pin')
        t0_sig_locs = dict(
            nin=sig_locs.get(key, pg0),
            pout=pd0,
            nout=nd0,
            nen=sig_locs.get('nclk', ng0),
            pen=sig_locs.get('pclkb', pg1),
        )
        params = dict(pinfo=pinfo, seg=seg_t0, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=t0_sig_locs, vertical_sup=vertical_sup)
        t0_master = self.new_template(InvTristateCore, params=params)

        t1_sig_locs = dict(
            nin=sig_locs.get('pout', pg1),
            pout=pd0,
            nout=nd0,
            nen=sig_locs.get('nclkb', ng1),
            pen=sig_locs.get('pclk', pg0),
        )
        mlm_dict = dict(
            enb= MinLenMode.MIDDLE,
            en=MinLenMode.MIDDLE,
        )
        params = dict(pinfo=pinfo, seg=seg_t1, w_p=w_p, w_n=w_n, ridx_p=ridx_p, ridx_n=ridx_n,
                      vertical_out=False, sig_locs=t1_sig_locs, vertical_sup=vertical_sup,
                      min_len_mode=mlm_dict)
        t1_master = self.new_template(InvTristateCore, params=params)

        # set size
        blk_sp = max(self.min_sep_col, self.get_hm_sp_le_sep_col())
        t0_ncol = t0_master.num_cols
        t1_ncol = t1_master.num_cols
        nor_ncol = nor_master.num_cols
        num_cols = t0_ncol + t1_ncol + nor_ncol + blk_sp * 2
        self.set_mos_size(num_cols)

        # add instances
        t1_col = t0_ncol + blk_sp
        nor_col = num_cols - nor_ncol
        t0 = self.add_tile(t0_master, 0, 0)
        t1 = self.add_tile(t1_master, 0, t1_col)
        nor = self.add_tile(nor_master, 0, nor_col)

        # routing
        # connect/export VSS/VDD
        vss_list, vdd_list = [], []
        for inst in (t0, t1, nor):
            vss_list.extend(inst.get_all_port_pins('VSS'))
            vdd_list.extend(inst.get_all_port_pins('VDD'))
        self.add_pin('VSS', self.connect_wires(vss_list))
        self.add_pin('VDD', self.connect_wires(vdd_list))

        # export input
        self.reexport(t0.get_port('nin'), net_name='nin', hide=True)
        self.reexport(t0.get_port('pin'), net_name='pin', hide=True)
        self.reexport(t0.get_port('in'))

        # connect output
        out = nor.get_pin('out')
        in2 = t1.get_pin('nin')
        self.connect_to_track_wires(in2, out)
        self.add_pin('out', out)
        self.add_pin('nout', in2, hide=True)
        self.add_pin('pout', in2, hide=True)

        # connect middle node
        mid_coord = (t1.bound_box.xh + nor.bound_box.xl) // 2
        mid_tidx = self.grid.coord_to_track(vm_layer, mid_coord, RoundMode.NEAREST)
        mid_tid = TrackID(vm_layer, mid_tidx, width=tr_w_v)
        if out.layer_id == vm_layer:
            next_tidx = tr_manager.get_next_track(vm_layer, mid_tidx, 'sig', 'sig', up=True)
            if next_tidx >= out.track_id.base_index:
                raise ValueError('oops!')

        warrs = [t0.get_pin('pout'), t0.get_pin('nout'), t1.get_pin('pout'), t1.get_pin('nout'),
                 nor.get_pin('nin<1>')]
        mid_vm_warr = self.connect_to_tracks(warrs, mid_tid)

        # connect clocks
        clk_tidx = sig_locs.get('clk', self.arr_info.get_source_track(t1_col + 1))
        clkb_tidx = sig_locs.get('clkb', self.arr_info.get_source_track(t1_col - blk_sp - 1))
        clk_tid = TrackID(vm_layer, clk_tidx, width=tr_w_v)
        clkb_tid = TrackID(vm_layer, clkb_tidx, width=tr_w_v)
        t0_en = t0.get_pin('en')
        t1_en = t1.get_pin('en')
        t1_enb = t1.get_pin('enb')
        t0_enb = t0.get_pin('enb')
        clk_hm = [t0_en, t1_enb]
        clkb_hm = [t0_enb, t1_en]
        if vertical_clk:
            clk = self.connect_to_tracks(clk_hm, clk_tid)
            clkb = self.connect_to_tracks(clkb_hm, clkb_tid)
            self.add_pin('clk', clk)
            self.add_pin('clkb', clkb)

        self.add_pin('outb', [nor.get_pin('in<1>'), mid_vm_warr])
        self.add_pin('noutb', nor.get_pin('nin<1>'), hide=True)
        self.add_pin('poutb', nor.get_pin('nin<1>'), hide=True)
        self.add_pin('rst', nor.get_pin('in<0>'))
        self.add_pin('nrst', nor.get_pin('nin<0>'), hide=True)
        self.add_pin('prst', nor.get_pin('nin<0>'), hide=True)
        self.add_pin('mid_vm', mid_vm_warr, hide=True)

        self.add_pin('nclk', t0_en, label='clk:', hide=vertical_clk)
        self.add_pin('pclk', t1_enb, label='clk:', hide=vertical_clk)
        self.add_pin('nclkb', t1_en, label='clkb:', hide=vertical_clk)
        self.add_pin('pclkb', t0_enb, label='clkb:', hide=vertical_clk)

        # set properties
        self.sch_params = dict(
            tin=t0_master.sch_params,
            tfb=t1_master.sch_params,
            nor=nor_master.sch_params,
        )


def get_adj_tidx_list(layout: MOSBase, ridx: int, sig_locs: Mapping[str, Union[float, HalfInt]],
                      wtype: MOSWireType, prefix: str, up: bool) -> Tuple[HalfInt, HalfInt]:
    """Helper method that gives two adjacent signal wires, with sig_locs override."""
    hm_layer = layout.conn_layer + 1

    out0 = sig_locs.get(prefix + '0', None)
    if out0 is not None:
        # user specify output track index for both parities
        out1 = sig_locs[prefix + '1']
    else:
        out0 = sig_locs.get(prefix, None)
        if out0 is not None:
            # user only specify one index.
            out1 = layout.tr_manager.get_next_track(hm_layer, out0, 'sig', 'sig', up=up)
        else:
            # use default track indices
            widx = 0 if up else -1
            out0 = layout.get_track_index(ridx, wtype, wire_name='sig', wire_idx=widx)
            out1 = layout.tr_manager.get_next_track(hm_layer, out0, 'sig', 'sig', up=up)

    if out0 == out1:
        raise ValueError(f'{prefix}0 and {prefix}1 must be on different tracks.')

    return out0, out1


def get_adj_tid_list(layout: MOSBase, ridx: int, sig_locs: Mapping[str, Union[float, HalfInt]],
                     wtype: MOSWireType, prefix: str, up: bool, tr_w_h: int
                     ) -> Tuple[TrackID, TrackID]:
    hm_layer = layout.conn_layer + 1
    out0, out1 = get_adj_tidx_list(layout, ridx, sig_locs, wtype, prefix, up)
    return TrackID(hm_layer, out0, width=tr_w_h), TrackID(hm_layer, out1, width=tr_w_h)

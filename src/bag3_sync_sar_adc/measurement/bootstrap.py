from typing import Any, Union, Tuple, Optional, Mapping, cast, Dict, Type

# import matplotlib.pyplot as plt
import numpy as np
import math
import copy

from pathlib import Path

from bag.simulation.core import TestbenchManager
from bag.simulation.data import AnalysisType
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.math.interpolate import LinearInterpolator

from bag.concurrent.util import GatherHelper
import matplotlib

from bag3_testbenches.measurement.data.tran import interp1d_no_nan
from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.base import TranTB

class BootstrapMM(MeasurementManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tbm_info: Optional[Tuple[TranTB, Mapping[str, Any]]] = None

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, sim_results: Union[SimResults, MeasureResult], 
                       Fs: int, Fin: int, Vdm: float, numbits: int, base_per: float) -> Tuple[bool, MeasInfo]:
        data = cast(SimResults, sim_results).data
        tvec=data['time']
        
        # --------------- Calculate swept frequency/amplitude parameters used for reference later
        data_shape = tvec.shape
        if len(data_shape) <= 3:
            in_feat_shape: Tuple = (data_shape[1],)
        else:
            in_feat_shape = data_shape[1:-1]
        num_swps = int(np.prod(data_shape[1:-1]))
        num_samp_swp= int(np.prod(data_shape[1:-2]))
        fs, fin, vdm = [], [], []
        if 'num_samp' in data:
            fs = np.array([[d for d in data['num_samp']] for i in range(num_samp_swp)])
            fs = fs.reshape(in_feat_shape)
            fin = np.array([Fin for i in range(num_swps)]).reshape(data_shape[1:-1])
            vdm = np.array([Vdm for i in range(num_swps)]).reshape(data_shape[1:-1])
        elif 'num_sig' in data:
            num_samp_swp = int(np.prod(data_shape[1:-2]))
            fin = np.array([[d for d in data['num_sig']] for i in range(num_samp_swp)])
            fin = np.array(fin).reshape(in_feat_shape)
            fs = np.array([Fs for i in range(num_swps)]).reshape(data_shape[1:-1])
            vdm = np.array([Vdm for i in range(num_swps)]).reshape(data_shape[1:-1])
        elif 'vdm' in data:
            num_samp_swp = int(np.prod(data_shape[1:-2]))
            vdm = np.array([[d for d in data['vdm']] for i in range(num_samp_swp)])
            vdm = np.array(vdm).reshape(in_feat_shape)
            fin = np.array([Fin for i in range(num_swps)]).reshape(data_shape[1:-1])
            fs =  np.array([Fs for i in range(num_swps)]).reshape(data_shape[1:-1])

        num_samp_vec = fs.reshape(in_feat_shape)


        # data length per sample
        # ---------- Flatten and take FFT
        sndr_dict = dict()
        sfdr_dict = dict()
        enob_dict = dict()
    
        for b in range(numbits):
            sndr, sfdr, enob = [] , [], []
            out = data[f'out<{b}>']
            shape_by_samp = (np.prod(data_shape[1:-1]), data_shape[-1])
            tvec_rs = tvec.reshape(shape_by_samp)
            out_rs = out.reshape(shape_by_samp)
            
            for i, _ in enumerate(tvec_rs):
                num_samples = int(num_samp_vec[i])
                per = base_per*num_samp_vec[0]/num_samples
                time = []
                val = []
                for v, t in zip(out_rs[i], tvec_rs[i]):
                        if (not math.isnan(t)) and (not math.isnan(v)):
                            time.append(t)
                            val.append(v)
                interp_out = interp1d_no_nan(time, val)
                sample_time = [time[-1]-idx*per for idx in range(num_samples)]
                sample_out = interp_out(sample_time)
                fft = np.abs(np.fft.fft(sample_out)) / num_samples
                fft = 2*fft[1:num_samples // 2 + 1]
                fft_db = 20 * np.log10(fft)
                fft_db_sorted = np.sort(fft_db)
                fft_sorted = np.sort(fft)
                sfdr.append(fft_db_sorted[-1] - fft_db_sorted[-2])
                noise_pwr = np.sum(np.square(fft_sorted[:-1]))
                sig_pwr = fft_sorted[-1] ** 2
                _sndr = 10 * np.log10(sig_pwr / noise_pwr)
                sndr.append(_sndr)
                enob.append((_sndr - 1.76) / 6.02)

            sndr = np.array(sndr).reshape(data_shape[1:-1])
            sfdr = np.array(sfdr).reshape(data_shape[1:-1])
            enob = np.array(enob).reshape(data_shape[1:-1])

            sndr_dict[f'bit<{b}>'] = sndr
            sfdr_dict[f'bit<{b}>'] = sfdr
            enob_dict[f'bit<{b}>'] = enob
        
            
        return MeasInfo('done', {'sfdr': sfdr_dict, 'sndr': sndr_dict, 'enob': enob_dict,
                                 'fs': fs, 'fin': fin, 'vdm': vdm})

    def setup_tbm(self, sim_db: SimulationDB, dut: DesignInstance, analysis: TranTB, specs: Dict) -> TranTB:
        tbm_specs = copy.deepcopy(specs)
        tbm_specs['dut_pins'] = list(dut.sch_master.pins.keys())
        swp_info = []
        for key, v in specs.get('swp_info', dict()).items():
            v = tbm_specs['swp_info'][key]
            if isinstance(v, list):
                swp_info.append((key, dict(type='LIST', values=v)))
            else:
                _type = v['type']
                if _type == 'LIST':
                    swp_info.append((key, dict(type='LIST', values=v['values'])))
                elif _type == 'LINEAR':
                    swp_info.append((key, dict(type='LINEAR', start=v['start'], stop=v['stop'], num=v['num'])))
                elif _type == 'LOG':
                    swp_info.append((key, dict(type='LOG', start=v['start'], stop=v['stop'], num=v['num'])))
                else:
                    raise RuntimeError
        tbm_specs['swp_info'] = swp_info
        tbm = cast(analysis, sim_db.make_tbm(analysis, tbm_specs))
        return tbm
    
    @staticmethod
    async def _run_sim(name: str, sim_db: SimulationDB, sim_dir: Path, dut: DesignInstance,
                       tbm: TranTB):
        sim_id = f'{name}'
        sim_results = await sim_db.async_simulate_tbm_obj(sim_id, sim_dir / sim_id,
                                                          dut, tbm, {}, tb_name=sim_id)

        return sim_results

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        """ Measures sampling bandwidth, input frequency bandwidth, and dynamic range
        """
        
        results = dict()
        # ---------- Max sampling frequency ---------------
        num_samp_base = self.specs['tbm_specs']['sim_params']['base_sample']
        num_sig_base= self.specs['tbm_specs']['sim_params']['num_sig']
        vdm_base = self.specs['tbm_specs']['sim_params']['vdm']
        numbits = self.specs['tbm_specs']['sim_params']['numbits']

        base_per = self.specs['tbm_specs']['sim_params']['base_per']
        if 'num_samp' in self.specs['swp_info'].keys():
            samp_specs = copy.deepcopy(dict(**self.specs['tbm_specs']))
            freq = []
            for k in self.specs['swp_info'].keys():
                if not('num_sig' in k) and not('vdm' in k):
                    samp_specs['swp_info'][k] = self.specs['swp_info'][k]
            # make sure num_samp is the last sweep
            samp_specs['swp_info']['num_samp'] = samp_specs['swp_info'].pop('num_samp')
            samp_specs['src_list'] = list(samp_specs['src_list'])
            tbm_fsamp = self.setup_tbm(sim_db, dut, TranTB, samp_specs)
            samp_results = await self._run_sim(name + '_fsamp', sim_db, sim_dir, dut, tbm_fsamp)
            results['Fsamp'] = self.process_output(samp_results, num_samp_base,
                                                   num_sig_base, vdm_base, numbits, base_per).prev_results

            samp_3db_list = []
            for (sndr_k, sndr_v), (sfdr_k, sfdr_v) in \
                  zip(results['Fsamp']['sndr'].items(), results['Fsamp']['sfdr'].items()):
                swp_shape = sndr_v.shape
                num_swps = int(np.prod(swp_shape[:-1]))
                sndr_fs = sndr_v.reshape(num_swps, swp_shape[-1])
                sfdr_fs = sfdr_v.reshape(num_swps, swp_shape[-1])
                sndr_fs = sndr_fs[0]
                sfdr_fs = sfdr_fs[0]
                for idx, (sndr, sfdr) in enumerate(zip(sndr_fs, sndr_fs)):
                    if sndr_fs[0]-sndr > 3 or sfdr_fs[0]-sfdr >3:
                        _samp_3db = results['Fsamp']['fs'][idx]
                    elif idx == (len(sndr_fs)-1):
                        _samp_3db = results['Fsamp']['fs'][idx]
                    else:
                        continue
                samp_3db_list.append(_samp_3db)
            samp_3db = min(samp_3db_list)
            
            # swp_shape = results['Fsamp']['sndr'].shape
            # num_swps = int(np.prod(swp_shape[:-1]))
            # sndr_fs = results['Fsamp']['sndr'].reshape(num_swps, swp_shape[-1])
            # sfdr_fs = results['Fsamp']['sfdr'].reshape(num_swps, swp_shape[-1])
            # sndr_fs = sndr_fs[0]
            # sfdr_fs = sfdr_fs[0]
            # for idx, (sndr, sfdr) in enumerate(zip(sndr_fs, sndr_fs)):
            #     if sndr_fs[0]-sndr > 3 or sfdr_fs[0]-sfdr >3:
            #         samp_3db = results['Fsamp']['fs'][idx]
            #     elif idx == (len(sndr_fs)-1):
            #         samp_3db = results['Fsamp']['fs'][idx]
            #     else:
            #         continue
            

        # ---------- Input Frequency BW ---------------
        if 'num_sig' in self.specs['swp_info'].keys():
            if not('num_samp' in self.specs['swp_info'].keys()) or  \
                self.specs['swp_info']['num_samp'] == samp_3db:
                specs_sig = copy.deepcopy(dict(**self.specs['tbm_specs']))
                specs_sig['swp_info']['num_sig'] = self.specs['swp_info']['num_sig']
            else:
                specs_sig = copy.deepcopy(copy.deepcopy(dict(**self.specs['tbm_specs'])))
                specs_sig['sim_params']['num_samp']=samp_3db
                num_sig_swp_info = self.specs['swp_info']['num_sig']
                v = []
                for nsig in list(np.linspace(float(num_sig_swp_info['start']), 
                                float(num_sig_swp_info['stop']), num_sig_swp_info['num'])):
                    nsig_new = nsig*(samp_3db//num_samp_base)-1
                    nsig_new = nsig_new if nsig_new%2==1 else nsig_new-1
                    v.append(nsig_new)
                swp_info = dict(type='LIST', values=v)
                specs_sig['swp_info']['num_sig'] = swp_info
            specs_sig['src_list'] = list(specs_sig['src_list'])
            tbm_fsig = self.setup_tbm(sim_db, dut, TranTB, specs_sig)
            sig_results = await self._run_sim(name + '_fsig', sim_db, sim_dir, dut, tbm_fsig)
            data = cast(SimResults, sig_results).data
            results['Fin'] = self.process_output(sig_results, samp_3db, num_sig_base, 
                                                 vdm_base, numbits, base_per*num_samp_base/samp_3db).prev_results

        # ---------- Dynamic Range ---------------
        if 'vdm' in self.specs['swp_info'].keys():
            specs_vdm = copy.deepcopy(dict(**self.specs['tbm_specs']))
            specs_vdm['swp_info']['vdm'] = self.specs['swp_info']['vdm']
            specs_vdm['src_list'] = list(specs_vdm['src_list'])
            tbm_vdm = self.setup_tbm(sim_db, dut, 
            
            TranTB, specs_vdm)
            vdm_results = await self._run_sim(name + '_vdm', sim_db, sim_dir, dut, tbm_vdm)
            data = cast(SimResults, vdm_results).data
            results['vdm'] = self.process_output(vdm_results, num_samp_base, num_sig_base, 
                                                 vdm_base, numbits, base_per).prev_results


        # should clean up the fin, fsamp, and vdm results for final reporting, can 
        # use intermediate for plotting or smth
        print(results)
        res = dict()
        if len(results['Fsamp']['enob'][f'bit<{numbits-1}>'].shape) > 1:
            res['max_enob'] = np.array([max(r) for r in results['Fsamp']['enob'][f'bit<{numbits-1}>']])
        else :
            res['max_enob'] = max(results['Fsamp']['enob'][f'bit<{numbits-1}>'])
        # res['switch_resistance']
        # res['dynamic_range'] = fit_dr()

        return res

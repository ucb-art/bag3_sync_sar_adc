import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Any, Union, Tuple, Optional, Mapping, List, cast, Dict
import copy

from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimData
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag3_liberty.data import parse_cdba_name
from bag3_testbenches.measurement.data.tran import interp1d_no_nan
from bag3_testbenches.measurement.tran.base import TranTB
from bag.io.file import write_yaml, read_yaml

class SarSliceMM(MeasurementManager):
    """
    This testbench implement a static simulation of the adc. It can support
    1. Give a single vdm and look at the quantized result. Only ADC is required, DAC is implemented in python
    2. Give a sweep list of vdm
        2.1 A small set of vdm can help figuring out the max conversion
        2.2 Sweep the full range to see the code thres and the "simulated" low-freq fft
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tbm_info: Optional[Tuple[TranTB, Mapping[str, Any]]] = None
        #self._dut_specs = read_yaml(self.specs['gen_specs_file']).to_dict()
        # self._dut = None

    @classmethod
    def plot_sig(cls, sim_data, sig, x_sig, axis):
        x_vec = sim_data[x_sig]
        axis.plot(x_vec, sim_data[sig][0])
        axis.grid()

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')
    #return self._tbm_info, True
    
    @staticmethod
    async def _run_sim(name: str, sim_db: SimulationDB, sim_dir: Path, dut: DesignInstance,
                       tbm: TranTB):
        sim_id = f'{name}'
        sim_results = await sim_db.async_simulate_tbm_obj(sim_id, sim_dir / sim_id,
                                                          dut, tbm, {}, tb_name=sim_id)

        return sim_results
    
    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs

        tbm_specs = copy.deepcopy(dict(**specs['tbm_specs']))
        tbm_specs['dut_pins'] = list(dut.sch_master.pins.keys())
        tbm_specs['pwr_domain'] = tbm_specs.get('pwr_domain', dict())
        tbm_specs['pwr_domain'].update(
            {parse_cdba_name(p)[0]: ('VSS', 'VDD') for p in list(dut.sch_master.pins.keys())})
        
        tbm_specs['pin_values'] = tbm_specs.get('pin_values', {})
        swp_info = []
        for k, v in specs.get('swp_info', dict()).items():
            if isinstance(v, list):
                swp_info.append((k, dict(type='LIST', values=v)))
            else:
                _type = v['type']
                if _type == 'LIST':
                    swp_info.append((k, dict(type='LIST', values=v['val'])))
                elif _type == 'LINEAR':
                    swp_info.append((k, dict(type='LINEAR', start=v['start'], stop=v['stop'], num=v['num'])))
                elif _type == 'LOG':
                    swp_info.append((k, dict(type='LOG', start=v['start'], stop=v['stop'], num=v['num'])))
                else:
                    raise RuntimeError
        tbm_specs['swp_info'] = swp_info
        tbm = cast(TranTB, sim_db.make_tbm(TranTB, tbm_specs))
        self._tbm_info = tbm, {}
        self._dut = dut
        self._dut_specs = self._dut.lay_master.params
        self._dut_specs_cdac = read_yaml(self._dut_specs['cdac_params'])['params']  \
                                if self._dut_specs['directory'] else self._dut_specs['cdac_params']
        nbit = self._dut_specs_cdac['nbits']
        row_list = self._dut_specs_cdac['row_list']
        col_list = self._dut_specs_cdac['col_list']
        diff_idx = self._dut_specs_cdac['diff_idx']
        weight_list = [r*c for r,c in zip(row_list[1:diff_idx], col_list[1:diff_idx])] 
        if diff_idx <= nbit:
            weight_list = weight_list + [2*r*c for r,c in zip(row_list[diff_idx:], col_list[diff_idx:])]
        self.weight_list = weight_list

        sig_results = await self._run_sim(name, sim_db, sim_dir, dut, tbm)
        fig_list, results = self.process_output(sig_results)
        
        with PdfPages(str(self.specs['results_files']+'.pdf')) as pdf:
            for f in fig_list:
                pdf.savefig(f)
        return results #MeasInfo('sim', {})  # ???: function of MeasInfo???
    
    def process_output(self, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        # sim_params = self.specs['tbm_specs']['sim_params']
        data = cast(SimResults, sim_results).data
        # Simulation has swp variables other than corner and time
        tmax_list = []
        if len(data.sweep_params) > 2:
            result_list = []
            swp_vdm = data.sweep_params[1]
            code_hist_list = []
            dout_hist_list = []
            for idx, val in enumerate(data[swp_vdm]):
                # find the location of last bit comparison, help figure out the max speed.
                vsup = self.specs['tbm_specs']['sim_params']['vdd']
                tvec_r = data['time'][0, idx, :][::-1]
                pvec_r = data['comp_p'][0, idx, :][::-1]
                nvec_r = data['comp_n'][0, idx, :][::-1]
                ptvec = np.where(pvec_r < 0.9 * vsup)[0]
                ntvec = np.where(nvec_r < 0.9 * vsup)[0]
                ptmax = 0 if ptvec.size == 0 else tvec_r[ptvec[0]]
                ntmax = 0 if ntvec.size == 0 else tvec_r[ntvec[0]]
                tmax_list.append(max(ptmax, ntmax))
                result_swp = self._process_output_helper(data, val, idx)
                result_list.append(result_swp)

                if self.specs['hist_analysis']:
                    code_hist_swp, dout_hist_swp = self._process_hist_helper(data, idx)
                    code_hist_list = code_hist_list + code_hist_swp
                    dout_hist_list = dout_hist_list + dout_hist_swp
                
            self.log_result(dict(max_conversion_time=max(tmax_list)))
            # save the vdm and output code list
            vdm_list, code_list = [], []
            for res in result_list:
                vdm_list.append(res['vdm'])
                code_list.append(res['code'])

            results_dict = dict()
            if self.specs['make_static_code_map']:
                fig_trans, fig_sndr, ncode, sndr, enob = self.fit_code_map(vdm_list, code_list)
                results_dict['static_ncode'] = ncode
                results_dict['static_sndr'] = sndr
                results_dict['static_enob'] = enob
                self.log_result(dict(num_codes=ncode, sndr_low_freq=sndr, enob_low_freq=enob))
                self.warn("This testbench implements the same simulation as old LAYGO generator static testbench. "
                          "It has calibration function when use sub-radix2."
                          "Make sure you have enough sweep points to get enough code_th. "
                          "Due to limited number of sweep points, "
                          "the performance might be worse than dynamic testbench.")
            if self.specs['hist_analysis']:
                # print("CODE HIST: ", code_hist_list)
                fig_hist, result_hist = self.process_hist(code_hist_list)
                results_dict.update(result_hist)
                print(result_hist)
            write_yaml(self.specs['results_files']+'.yaml', results_dict)
        else:
            result = self._process_output_helper(data, self.specs['tbm_specs']['sim_params']['vdm'], -1)
            self.log_result(result)
            self.warn("Not enough points to make static code map or complete histogram analysis for INL/DNL."
                      "You must have >2 sweep points and set make_static_code_map and hist_analysis, respectively"
                      "for these analysis")
        return [fig_trans, fig_sndr, fig_hist],  MeasInfo('done', {})

    def _process_output_helper(self, data: SimData, vdm: float, idx: int):
        # Get some simulation params
        nbit = self._dut_specs_cdac['nbits']
        row_list = self._dut_specs_cdac['row_list']
        val_sup = self.specs['tbm_specs']['sim_params']['vdd']
        val_range = self.specs['tbm_specs']['sim_params']['vrefp'] - self.specs['tbm_specs']['sim_params']['vrefn']
        val_th = val_sup / 2

        # remove nan
        def find_last_non_nan(vec):
            for num in vec[::-1]:
                if not np.isnan(num):
                    return num

        # bit processing
        bit_list = []
        if idx > -1:
            for jdx in range(1, nbit+1):
                data_out_final_value = find_last_non_nan(data[f'data_out<{jdx}>'][0, idx, :])
                bit_list.append(data_out_final_value > val_th)
        else:
            for jdx in range(1, nbit+1):
                data_out_final_value = find_last_non_nan(data[f'data_out<{jdx}>'][0])
                bit_list.append(data_out_final_value > val_th)
        #Might be something wrong here?
        
        # if len(row_list)<nbit:
        #     bit_list = [str(int(not b)) for b in bit_list]
        #     bit_str = '0b' + ''.join(bit_list[::-1])
        #     code = int(bit_str, 2)
        #     dout = code / 2 ** nbit * 2 * val_range - val_range
        # else:
        bit_list = [int(not b) for b in bit_list]
        # col_list = self._dut_specs['cdac_params']['col_list']
        # diff_idx = self._dut_specs_cdac['diff_idx']
        # weight_list = [r*c for r,c in zip(row_list[1:diff_idx], col_list[1:diff_idx])] 
        # if diff_idx < nbit:
        #     weight_list = weight_list + [2*r*c for r,c in zip(row_list[diff_idx:], col_list[diff_idx:])]
        code = sum([b*w for b,w in zip(bit_list, self.weight_list)])
        dout = code/sum(self.weight_list) * 2*val_range - val_range

        return dict(vdm=vdm, dout=dout, code=code) #all_codes=all_codes_list

    def _process_hist_helper(self, data: SimData, idx: int):
        # Collect the bits for histogram test

        self._dut_specs = self._dut.lay_master.params
        nbit = self._dut_specs_cdac['nbits']
        val_sup = self.specs['tbm_specs']['sim_params']['vdd']
        val_range = self.specs['tbm_specs']['sim_params']['vrefp'] - self.specs['tbm_specs']['sim_params']['vrefn']
        val_th = val_sup / 2

        row_list = self._dut_specs_cdac['row_list']
        

        def find_last_non_nan_idx(vec):
            for idx, num in enumerate(vec[::-1]):
                if not np.isnan(num):
                    return len(vec)-1-idx
                
        # bit processing
        data_out_list = [] # reorganize so not dependent on idx
        data_time = []
        #data_out_fin_idx=[]
        if idx > -1:
            for jdx in range(1, nbit+1):
                data_out_list.append(data[f'data_out<{jdx}>'][0, idx, :])
                data_time.append(data['time'][0,idx,:])
                #data_out_fin_idx.append(find_last_non_nan_idx(data[f'data_out<{jdx}>'][0, idx, :]))
        else:
            for jdx in range(1, nbit+1):
                data_out_list.append(data[f'data_out<{jdx}>'][0])
                data_time.append(data['time'][0])
                #data_out_fin_idx.append(find_last_non_nan_idx(data[f'data_out<{jdx}>'][0]))

        # bit_list = [int(not b) for b in bit_list]
        # bit_str = '0b' + ''.join(bit_list[::-1])
        # code = sum([b*w for b,w in zip(bit_list, weight)])
        # dout = code / 2 ** nbit * 2 * val_range - val_range

        # Count the number of codes in this sample for histogram test
        num_samples = self.specs['tbm_specs']['sim_params']['num_samples']
        per=self.specs['tbm_specs']['sim_params']['t_per']*(nbit+2)
        all_codes_matrix = []
        for n in range(nbit):
            time = []
            val = []
            for v, t in zip(data_out_list[n], data_time[n]):
                if (not np.isnan(t)) and (not np.isnan(v)):
                    time.append(t)
                    val.append(v)
            interp_out = interp1d_no_nan(time, val)
            sample_time = [time[-1]-i*per for i in range(num_samples)]
            sample_out = interp_out(sample_time)
            bit_sam = ['0' if b>val_th else '1' for b in sample_out]
            all_codes_matrix.append(bit_sam)
        
        all_codes_matrix = np.array(all_codes_matrix)
        # print("ALL CODES ARRAY SHAPE: ", all_codes_matrix.shape)
        # print(all_codes_matrix)
        all_codes_list = []
        all_dout_list = []
        for ns in range(num_samples):
            a = all_codes_matrix[:,ns]
            # if len(row_list) < nbit:
            #     code = int('0b'+''.join(a[::-1]),2)
            #     all_codes_list.append(code)
            #     dout = code / 2 ** nbit * 2 * val_range - val_range
            #     all_dout_list.append(dout)
            # else:
            # col_list = self._dut_specs_cdac['col_list']
            # diff_idx = self._dut_specs_cdac['diff_idx']
            # weight_list = [r*c for r,c in zip(row_list[1:diff_idx], col_list[1:diff_idx])] 
            # if diff_idx < nbit:
            #     weight_list = weight_list + [2*r*c for r,c in zip(row_list[diff_idx:], col_list[diff_idx:])]
            
            a = [int(i) for i in a]
            # print(a, self.weight_list)
            code = sum([bit*w for bit, w in zip(a, self.weight_list)]) #a[::-1]
            all_codes_list.append(code)
            dout = code / sum(self.weight_list) * 2 * val_range - val_range
            all_dout_list.append(dout)

        return all_codes_list, all_dout_list
    
    def process_hist(self, code_list: List[Union[float, int]]):
        # Given code list, produce histogram, INL and DNL

        nbit = self._dut_specs_cdac['nbits']
        row_list = self._dut_specs_cdac['row_list']
        # if len(row_list) < nbit:
        #     num_bins = 2**nbit
        # else:
        # col_list = self._dut_specs_cdac['col_list']
        # diff_idx = self._dut_specs_cdac['diff_idx']
        # weight_list = [r*c for r,c in zip(row_list[1:diff_idx], col_list[1:diff_idx])] 
        # if diff_idx < nbit:
        #     weight_list = weight_list + [2*r*c for r,c in zip(row_list[diff_idx:], col_list[diff_idx:])]
        num_bins = sum(self.weight_list)

        hist, bin_edges = np.histogram(code_list, bins=num_bins)

        bins = hist[1:-1] # ignore first and last histogram points
        avg_bins = sum(bins)/len(bins)
        dnl_list = [(b/avg_bins)-1 for b in bins]
        print("DNL_LIST: ", dnl_list)
        # dnl = [(t - avg_codew)/avg_codew for t in time_delta]
        inl_list = [sum(dnl_list[0:i]) for i, _ in enumerate(dnl_list)]
        inl_list.append(sum(dnl_list))
        
        fig, ax = plt.subplots(2, 1)
        labelsize=20
        ticksize=18
        ax[0].plot(np.arange(1, len(inl_list)+1, 1), inl_list, 'o--', label='INL', )
        # plt.legend(loc='upper right')
        # plt.title("INL - 8 bit")
        ax[0].set_xlabel('Code', fontsize = labelsize, fontweight='bold')
        ax[0].grid()
        ax[0].set_ylabel('INL (LSB)', fontsize = labelsize, fontweight='bold')
        ax[0].set_ylim([-(max(inl_list)+0.05), (max(inl_list)+0.05)])
        plt.setp(ax[0].get_xticklabels(), fontsize=ticksize)
        plt.setp(ax[0].get_yticklabels(), fontsize=ticksize)
        # plt.figure(2)
        ax[1].plot(np.arange(1, len(dnl_list)+1, 1), dnl_list, '^--', label='DNL', )
        # plt.legend(loc='upper right')
        # plt.title("DNL - 8 bit")
        ax[1].set_xlabel('Code', fontsize = labelsize, fontweight='bold')
        ax[1].grid()
        ax[1].set_ylabel('DNL (LSB)', fontsize = labelsize, fontweight='bold')
        ax[1].set_ylim([-(max(dnl_list)+0.05), (max(dnl_list)+0.05)])
        plt.setp(ax[1].get_xticklabels(), fontsize=ticksize)
        plt.setp(ax[1].get_yticklabels(), fontsize=ticksize)
        plt.tight_layout()

        return fig, dict(inl=max(inl_list), dnl=max(dnl_list))

    
    @classmethod
    def plot_transfer_func(cls, vin, code, ax):
        length = len(vin)
        if len(code) != length:
            raise ValueError('Vin and Code length dont match')

        for idx in range(length - 1):
            ax.plot([vin[idx], vin[idx + 1]], [code[idx], code[idx]], 'r')
            ax.plot([vin[idx + 1], vin[idx + 1]], [code[idx], code[idx + 1]], 'r')

    #need another def for figuring out weights
    # TODO make this work for non binary weighting
    def fit_code_map(self, vin, code):
        #nbit = self._dut_specs_cdac['nbits'] #FIXME 4 #self._dut.sch_master.params['nbits']
        nbit_cal = int(np.floor(np.log2(sum(self.weight_list)))) #nbit
        _vcm = self.specs['tbm_specs']['sim_params']['v_vcm']

        code_th = 0
        code_list = [] #binary codes
        vth_list = []  #threshold list
        # --- Extract code map ---
        for i, v in enumerate(vin):
            if code[i] > code_th:
                code_list.append(int(code_th))
                vth_list.append(0.5 * (vin[i - 1] + vin[i]))
                code_th = code[i]
        # print(code_list, len(code_list))
        # print("VTH LIST: ", vth_list, len(vth_list))
        # estimate input range calibration due to gain error
        delta_avg = (vth_list[-1] - vth_list[0]) / (code_list[-1]-code_list[0]) #2 ** (nbit - 1)
        print("DELTA_AVG: ", delta_avg)
        v_in = 0.5 * (-vth_list[0] + vth_list[-1]) # - delta_avg
        print("your adjusted input swing is:", v_in)
        print(f"number of code{len(code_list)}")
        fig, ax = plt.subplots(1)
        self.plot_transfer_func(vth_list, code_list, ax)

        ax.grid()
        ax.set_xlabel('V-dm sweep')
        ax.set_ylabel('Quantized output')
        # ----------

        # --- make a calibration map ---
        # First find the voltage step size based on v_in range (adjusted)
        v_bit_cal = 2 * v_in / 2 ** nbit_cal 
        calmap_in_th = []
        calmap_out = []
        for i, c in enumerate(code_list):
            calout = 2 ** nbit_cal - 1
            v_th = vth_list[i]
            trig = 0
            # section finds the largest code (given bit cal), and sends to calmap_out
            for c2 in range(2 ** nbit_cal):
                v_th_cal = -v_in + v_bit_cal * (c2 + 1)
                if v_th < v_th_cal and trig == 0:
                    calout = c2
                    trig = 1
            while len(calmap_in_th) - 1 < c:
                calmap_in_th.append(len(calmap_in_th))
                calmap_out.append(calout)

        # evaluate low freq ENOB
        # make a sine signal
        sin_freq = 1 / (2 ** (nbit_cal + 2 + 2)) * 3
        sin_time_bin = np.arange(2 ** (nbit_cal + 2 + 2))
        # input signal
        vsin = 0.99 * v_in * np.sin(2 * np.pi * sin_freq * sin_time_bin)
        freq_array = np.fft.fftfreq(sin_time_bin.shape[-1])

        # --- ideal n_bit_cal adc ---
        # map sine wave onto ideal calibration steps
        sinq_ideal = []
        for v in vsin:
            sinq_val = 0
            for c in range(2 ** nbit_cal):
                v_comp = -v_in + v_bit_cal * c
                if v >= v_comp:
                    sinq_val = c
            sinq_ideal.append(sinq_val)
        sinq_ideal_fft = np.fft.fft(np.array(sinq_ideal) - np.average(sinq_ideal))
        sinq_ideal_fft_abs = np.absolute(sinq_ideal_fft)
        sinq_ideal_fft_db = 20 * np.log10(np.absolute(sinq_ideal_fft))
        sinq_ideal_fft_dbc = sinq_ideal_fft_db - max(sinq_ideal_fft_db)

        # sndr&enob
        sinq_ideal_fft_argmax = np.argmax(sinq_ideal_fft_abs)
        sinq_ideal_fft_sigpwr = 2 * sinq_ideal_fft_abs[sinq_ideal_fft_argmax] ** 2  # negative freq)
        sinq_ideal_fft_totpwr = np.sum(np.square(sinq_ideal_fft_abs))
        sinq_ideal_sndr = 20 * np.log10(
            sinq_ideal_fft_sigpwr / (sinq_ideal_fft_totpwr - sinq_ideal_fft_sigpwr)) / 2  # /2 from sine
        sinq_ideal_enob = (sinq_ideal_sndr - 1.76) / 6.02
        print('ideal SNDR', sinq_ideal_sndr)
        print('ideal ENOB', sinq_ideal_enob)
        # --- simulated adc ---
        # map calibrated codes onto the sine wave
        sinq_vth = []
        sinq_code_raw = []
        sinq_code = []
        for v in vsin:
            sinq_vth_val = min(vth_list)
            sinq_code_raw_val = 0
            for i, vth in enumerate(vth_list):
                if v >= vth:
                    sinq_vth_val = vth
                    sinq_code_raw_val = code_list[i]+1 #Take the next code if v>=vth
            sinq_vth.append(sinq_vth_val)
            sinq_code_raw.append(sinq_code_raw_val)
            # convert to calibrated code
            sinq_code_val = 0
            for i, c in enumerate(calmap_in_th):
                if sinq_code_raw_val > c:
                    sinq_code_val = calmap_out[i]
            sinq_code.append(sinq_code_val)
        sinq_fft = np.fft.fft(np.array(sinq_code) - np.average(sinq_code))
        sinq_fft_abs = np.absolute(sinq_fft)
        sinq_fft_db = 20 * np.log10(np.absolute(sinq_fft))
        sinq_fft_dbc = sinq_fft_db - max(sinq_fft_db)
        # sndr&enob
        sinq_fft_argmax = np.argmax(sinq_fft_abs)
        sinq_fft_sigpwr = 2 * sinq_fft_abs[sinq_fft_argmax] ** 2  # negative freq)
        sinq_fft_totpwr = np.sum(np.square(sinq_fft_abs))
        sinq_sndr = 20 * np.log10(sinq_fft_sigpwr / (sinq_fft_totpwr - sinq_fft_sigpwr)) / 2  # /2 from sine
        sinq_enob = (sinq_sndr - 1.76) / 6.02
        print('SNDR', sinq_sndr)
        print('ENOB', sinq_enob)

        # --- Plot time-domain and freq-domain ---
        fft_n = np.size(freq_array) // 2
        fig_sndr, ax = plt.subplots(2, 1)
        ax[0].plot(sin_time_bin, sinq_ideal, 'r', label=f'ideal {nbit_cal}-bit')
        ax[0].plot(sin_time_bin, sinq_code, 'b', label='design')
        ax[0].legend()
        ax[0].grid()
        strtitle = 'Time domain, SNDR:' + "{:.2f}".format(sinq_sndr) + ', ENOB:' "{:.2f}".format(sinq_enob)
        ax[0].set_title(strtitle)
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('code')

        ax[1].plot(freq_array[:fft_n], sinq_ideal_fft_dbc[:fft_n], 'r', label=f'ideal {nbit_cal}-bit')
        ax[1].plot(freq_array[:fft_n], sinq_fft_dbc[:fft_n], 'b', label='design')
        ax[1].legend()
        ax[1].grid()
        strtitle = 'Frequency domain, SNDR:' + "{:.2f}".format(sinq_sndr) + ', ENOB:' "{:.2f}".format(sinq_enob)
        ax[1].set_title(strtitle)
        ax[1].set_xlabel('f/fs')
        ax[1].set_ylabel('dB')
        plt.tight_layout()
       
        return fig, fig_sndr, len(code_list), sinq_sndr, sinq_enob

    def log_result(self, new_result: Mapping[str, Any]) -> None:
        fmt = '{:.5g}'
        msg = []
        for k, v in new_result.items():
            msg.append(f'{k} = {fmt.format(v)}')
        self.log('\n'.join(msg))


class SarSliceDynamicMM(SarSliceMM):
    """
    This testbench replace dc input in the static testbench with a sin input
    The signal frequency is calculated by how many signal cycles (num_sig) in simulation period
        freq_sig = (num_sig/num_sample)*(1/t_per)
    """
    #FIXME Need to add DR measurement and max sampling speed characterization
    def process_output(self, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        data = cast(SimResults, sim_results).data
        tvec = data['time']
        nbit = self._dut_specs_cdac['nbits']
        sim_params = self.specs['tbm_specs']['sim_params']
        num_sample = sim_params['num_sample']
        num_sig = sim_params['num_sig']
        t_per = sim_params['t_per']
        row_list = self._dut_specs_cdac['row_list']

        calc = self._tbm_info[0].get_calculator(data)
        t_sam = calc.eval(sim_params['t_sam'])

        val_th = sim_params['vdd'] / 2

        bit_list = []
        for idx in range(1, nbit+1):
            yvec = data[f'data_out<{idx}>'][0]
            bit_list.append(interp1d_no_nan(tvec, yvec))

        dout_list = []
        tvec = []
        for idx in range(4, num_sample+4):
            _t = t_per * idx + t_sam
            _binary_list = []
            for jdx in range(nbit):
                _bit = bit_list[jdx]
                _binary_list.append(_bit(_t) > val_th)

            # if len(row_list)<nbit:
            #     _binary_list = [str(int(not b)) for b in _binary_list]
            #     bit_str = '0b' + ''.join(_binary_list[::-1])
            #     code = int(bit_str, 2)
            #     # dout = code
            #     dout = code / 2 ** nbit 
            # else:
            _binary_list = [int(not b) for b in _binary_list]
            # col_list = self._dut_specs_cdac['col_list']
            # diff_idx = self._dut_specs_cdac['diff_idx']
            # weight_list = [r*c for r,c in zip(row_list[1:diff_idx], col_list[1:diff_idx])] 
            # if diff_idx < nbit:
            #     weight_list = weight_list + [2*r*c for r,c in zip(row_list[diff_idx:], col_list[diff_idx:])]
            code = sum([b*w for b,w in zip(_binary_list, self.weight_list)])
            dout = code/sum(self.weight_list) 
          
            dout_list.append(dout)
            tvec.append(_t[0])
        print("WEIGHT LIST: ", self.weight_list)
        fig_fft, sndr, sfdr, enob = self.process_fft(np.array(tvec), np.array(dout_list), 
                                            file=self.specs['results_files'], plot=True )
        self.log_result(dict(freq=calc.eval(sim_params['freq_sig'])[0], sndr=sndr, sfdr=sfdr, enob=enob))
        results_dict = dict()
        results_dict['sndr']=float(sndr)
        results_dict['sfdr']=float(sfdr)
        results_dict['enob']=float(enob)
        print(results_dict)

        figures_list = [fig_fft]
        if self.specs['sine_hist']:
            fig_lin, lin_dict = self.process_hist(dout_list)
            results_dict = dict(results_dict, **lin_dict)
            figures_list = figures_list + fig_lin
        write_yaml(self.specs['results_files']+'.yaml', results_dict)
        return figures_list, MeasInfo('done', {})

    @classmethod
    def process_fft(cls, tvec: np.ndarray, yvec: np.ndarray, file: str, plot: bool = True):
        time_vec = tvec
        sampled = yvec
        sampled_diff = sampled - np.mean(sampled)
        n_points = len(sampled_diff)
        fft = np.abs(np.fft.fft(sampled_diff)) / n_points
        fft = fft[1:n_points // 2 + 1]
        fft_db = 20 * np.log10(fft)
        fft_db_sorted = np.sort(fft_db)
        fft_sorted = np.sort(fft)
        sfdr = (fft_db_sorted[-1] - fft_db_sorted[-2])

        noise_pwr = np.sum(np.square(fft_sorted[:-1]))
        sig_pwr = fft_sorted[-1] ** 2
        sndr = 10 * np.log10(sig_pwr / noise_pwr)
        enob = (sndr - 1.76) / 6.02

        f, ax = plt.subplots(2)
        if plot:
            ax[0].set_title('Output Voltage vs Time')
            ax[0].set_ylabel('Voltage (V)')
            ax[0].set_xlabel('Time (s)')
            
            ax[1].set_title('Output FFT')
            ax[1].set_ylabel('dB')
            ax[1].set_xlabel('fs (GHz)')

            freq_axis = np.arange(n_points)[range(int(n_points // 2))] / n_points

            _freq_axis = []
            _fft_db = []
            for freq, fft in zip(freq_axis, fft_db):
                if fft>-150 and fft<float('inf'): # Even noise floor shouldn't be -150dB
                    _fft_db.append(fft)
                    _freq_axis.append(freq)

            ax[0].plot(time_vec, sampled)
            ax[1].stem(_freq_axis, _fft_db, bottom=min(_fft_db)-2, use_line_collection=True )

            at = AnchoredText('\n'.join([f"sfdr={sfdr:.2f}", 
                                        f"sndr={sndr:.2f}", 
                                        f"enob={enob:.2f}"]), loc='upper right', frameon=True)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax[1].add_artist(at)
            [x.grid(True) for x in ax]
            plt.tight_layout()

        return f, sndr, sfdr, enob

    def process_hist(self, code_list: List[Union[float, int]]):
        # Given code list, produce histogram, INL and DNL

        # nbit = self._dut_specs_cdac['nbits']
        # row_list = self._dut_specs_cdac['row_list']
        # if len(row_list) < nbit:
        #     num_bins = 2**nbit
        # else:
        # col_list = self._dut_specs_cdac['col_list']
        # diff_idx = self._dut_specs_cdac['diff_idx']
        # weight_list = [r*c for r,c in zip(row_list[1:diff_idx], col_list[1:diff_idx])] 
        # if diff_idx < nbit:
        #     weight_list = weight_list + [2*r*c for r,c in zip(row_list[diff_idx:], col_list[diff_idx:])]
        num_bins = sum(self.weight_list)+1 
        print(num_bins)
        # x = np.linspace(0, 2*np.pi, 25600)
        # code_list = np.sin(x) 

        hist, bin_edges = np.histogram(code_list, bins=num_bins)
        #correct for sine wave
        # amp = 4*self.specs['tbm_specs']['sim_params']['vdm']
        # lsb = amp/num_bins
        # hist = [h*(1/np.pi())*(np.arcsin((n+1)*lsb/amp) - np.arcsin((n)*lsb/amp)) for n, h in enumerate(hist[:-1])] 
        # fig_hist, ax_hist = plt.subplots(1,1)
        # ax_hist.plot(np.arange(0, len(hist), 1), hist)
        # ax_hist.set_title("Raw Histogram")

        bins = hist[1:-1] # ignore first and last histogram points
        avg_bins = sum(bins)/len(bins)
        dnl_list = [(b/avg_bins)-1 for b in bins]
        print(hist)
        print("bins: ", bins)
        # dnl = [(t - avg_codew)/avg_codew for t in time_delta]
        inl_list = [sum(dnl_list[0:i]) for i, _ in enumerate(dnl_list)]
        inl_list.append(sum(dnl_list))
        
        fig, ax = plt.subplots(2, 1)
        labelsize=20
        ticksize=18
        ax[0].plot(np.arange(1, len(inl_list)+1, 1), inl_list, 'o--', label='INL', )
        # plt.legend(loc='upper right')
        # plt.title("INL - 8 bit")
        ax[0].set_xlabel('Code', fontsize = labelsize, fontweight='bold')
        ax[0].grid()
        ax[0].set_ylabel('INL (LSB)', fontsize = labelsize, fontweight='bold')
        ax[0].set_ylim([-(max(inl_list)+0.05), (max(inl_list)+0.05)])
        plt.setp(ax[0].get_xticklabels(), fontsize=ticksize)
        plt.setp(ax[0].get_yticklabels(), fontsize=ticksize)
        # plt.figure(2)
        ax[1].plot(np.arange(1, len(dnl_list)+1, 1), dnl_list, '^--', label='DNL', )
        # plt.legend(loc='upper right')
        # plt.title("DNL - 8 bit")
        ax[1].set_xlabel('Code', fontsize = labelsize, fontweight='bold')
        ax[1].grid()
        ax[1].set_ylabel('DNL (LSB)', fontsize = labelsize, fontweight='bold')
        ax[1].set_ylim([-(max(dnl_list)+0.05), (max(dnl_list)+0.05)])
        plt.setp(ax[1].get_xticklabels(), fontsize=ticksize)
        plt.setp(ax[1].get_yticklabels(), fontsize=ticksize)
        plt.tight_layout()

        ############### Correction of sine wave
        factor = num_bins/2
        print(len(code_list))
        print(factor)
        hist_sine = [len(code_list)/np.pi*(np.arcsin((n+1-factor)*1.6/((num_bins-1)*.825)) - np.arcsin((n-factor)*1.6/((num_bins-1)*.825))) for n in range(0, num_bins)] 
        print(hist_sine, len(hist_sine), len(bins))
        fig_hist_corr, ax_hist_corr = plt.subplots(1,1)
        ax_hist_corr.plot(np.arange(0, len(hist_sine), 1), hist_sine)
        ax_hist_corr.plot(np.arange(0, len(hist), 1), hist)
        ax_hist_corr.set_title("Expected Sine Histogram")

        #bins = hist[1:-1] # ignore first and last histogram points
        #avg_bins = sum(bins)/len(bins)
        # print("hist_sine: ", bins, hist_sine[1:-1])
        # print(len(bins), len(hist_sine[1:-1]))
        print([h*(sum(bins)/sum(hist_sine[1:-1])) for h in hist_sine[1:-1]])
        dnl_list = [b/(h*sum(bins)/sum(hist_sine[1:-1]))-1 for b,h in zip(bins, hist_sine[1:-1])] #[(b/avg_bins)-1 for b in bins]
        # print("DNL_LIST: ", dnl_list)
        # dnl = [(t - avg_codew)/avg_codew for t in time_delta]
        inl_list = [sum(dnl_list[0:i]) for i, _ in enumerate(dnl_list)]
        inl_list.append(sum(dnl_list))
        
        fig_corr, ax_corr = plt.subplots(2, 1)
        labelsize=20
        ticksize=18
        ax_corr[0].plot(np.arange(1, len(inl_list)+1, 1), inl_list, 'o--', label='INL', )
        # plt.legend(loc='upper right')
        # plt.title("INL - 8 bit")
        ax_corr[0].set_xlabel('Code', fontsize = labelsize, fontweight='bold')
        ax_corr[0].grid()
        ax_corr[0].set_ylabel('INL (LSB)', fontsize = labelsize, fontweight='bold')
        ax_corr[0].set_ylim([(min(inl_list)-0.05), (max(inl_list)+0.05)])
        plt.setp(ax_corr[0].get_xticklabels(), fontsize=ticksize)
        plt.setp(ax_corr[0].get_yticklabels(), fontsize=ticksize)
        # plt.figure(2)
        ax_corr[1].plot(np.arange(1, len(dnl_list)+1, 1), dnl_list, '^--', label='DNL', )
        # plt.legend(loc='upper right')
        # plt.title("DNL - 8 bit")
        ax_corr[1].set_xlabel('Code', fontsize = labelsize, fontweight='bold')
        ax_corr[1].grid()
        ax_corr[1].set_ylabel('DNL (LSB)', fontsize = labelsize, fontweight='bold')
        ax_corr[1].set_ylim([(min(dnl_list)-0.05), (max(dnl_list)+0.05)])
        plt.setp(ax_corr[1].get_xticklabels(), fontsize=ticksize)
        plt.setp(ax_corr[1].get_yticklabels(), fontsize=ticksize)
        plt.tight_layout()

        return [fig_hist_corr, fig, fig_corr], dict(inl=max(inl_list), dnl=max(dnl_list))
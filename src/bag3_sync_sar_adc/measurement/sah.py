from matplotlib.offsetbox import AnchoredText
from typing import Any, Union, Tuple, Optional, Mapping, cast, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_testbenches.measurement.tran.analog import AnalogTranTB
from bag3_testbenches.measurement.data.tran import interp1d_no_nan


class SampleHoldMM(MeasurementManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tbm_info: Optional[Tuple[AnalogTranTB, Mapping[str, Any]]] = None

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        specs = self.specs

        tbm_specs = dict(**specs['tbm_specs'])
        tbm_specs['dut_pins'] = list(dut.sch_master.pins.keys())
        swp_info = []
        for k, v in tbm_specs.get('swp_info', dict()).items():
            swp_info.append((k, dict(type='LIST', values=v)))
        tbm_specs['swp_info'] = swp_info

        tbm = cast(AnalogTranTB, sim_db.make_tbm(AnalogTranTB, tbm_specs))
        self._tbm_info = tbm, {}
        return False, MeasInfo('sim', {})  # ???: function of MeasInfo???

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        return self._tbm_info, True

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        sim_params = self.specs['tbm_specs']['sim_params']

        data = cast(SimResults, sim_results).data
        # Simulation has swp variables other than corner and time
        if len(data.sweep_params) > 2:
            result_list = []
            swp_freq = data.sweep_params[1]
            for idx, val in enumerate(data[swp_freq]):
                data_swp = dict()
                data_swp.update({'time': data['time'][:, idx, :]})
                for signame in data.signals:
                    data_swp.update({signame: data[signame][:, idx, :]})

                out_vec = data_swp['out_p'] - data_swp['out_n'] if 'out_p' in data.signals else data_swp['out']
                result_swp = self._process_output_helper(val, data_swp['time'][0], out_vec[0])
                self.log_result(result_swp)
                result_list.append(result_swp)
            plot_result = False
            result = self.plot_swp(result_list, plot_result)
            if plot_result:
                plt.show()
        else:
            out_vec = data['out_p'] - data['out_n'] if 'out_p' in data.signals else data['out']
            result = self._process_output_helper(sim_params['num_sig'], data['time'], out_vec[0])
            self.log_result(result)

        return True, MeasInfo('done', result)

    def _process_output_helper(self, freq: float, tvec: np.ndarray, yvec: np.ndarray):
        sim_params = self.specs['tbm_specs']['sim_params']
        num_sample = sim_params['num_sample']
        freq_sample = sim_params['freq_sample']
        tsettle = sim_params['t_settle']
        tper = 1 / sim_params['freq_sample']

        freq = freq / num_sample * freq_sample
        sig_out = interp1d_no_nan(tvec, yvec)

        tsim = tsettle + (num_sample + 1) / freq_sample
        t_sampled = np.arange(tsettle + tper, tsim, tper)
        sig_sampled = sig_out(t_sampled)
        sndr, sfdr = self.process_fft(t_sampled, sig_sampled, False)

        return dict(freq=freq, sndr=sndr, sfdr=sfdr)

    def log_result(self, new_result: Mapping[str, Any]) -> None:
        fmt = '{:.5g}'
        msg = []
        for k, v in new_result.items():
            msg.append(f'{k} = {fmt.format(v)}')
        self.log('\n'.join(msg))

    @classmethod
    def process_fft(cls, tvec: np.ndarray, yvec: np.ndarray, plot: bool = True):
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

        if plot:
            f, ax = plt.subplots(2)
            ax[0].set_title(f'Input Voltage vs Time')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_xlabel('fs (GHz)')

            freq_axis = np.arange(n_points)[range(int(n_points // 2))] / n_points

            ax[0].plot(time_vec, sampled)
            ax[1].plot(freq_axis, fft_db, '--o', markerfacecolor="None",
                       # color='red', markeredgecolor='red',
                       linewidth=0.5)
            [x.grid(True) for x in ax]
            at = AnchoredText(f"sfdr={sfdr}, sndr={sndr}",
                              loc='upper left', prop=dict(size=8), frameon=True,
                              )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax[1].add_artist(at)
            plt.tight_layout()
            plt.show()
        return sndr, sfdr

    @classmethod
    def plot_result(cls, sfdr: List[float], sndr: List[float], freq_list: List[float],
                    lengend: str = '', color_idx=0, ax=None):

        color = plt.cm.get_cmap('Set1')(color_idx)
        if not ax:
            f, ax = plt.subplots(1)
        ax.set_title('Input frequency (GHz)')
        ax.set_ylabel('Signal Power (dB)')
        # freq_list = [freq/1e9 for freq in freq_list]
        ax.plot(freq_list, sfdr, '--o', markerfacecolor="None",
                color=color, markeredgecolor=color,
                linewidth=0.5, label='SFDR'+lengend)
        ax.plot(freq_list, sndr, '--x', markerfacecolor="None",
                color=color, markeredgecolor=color,
                linewidth=0.5, label='SNDR'+lengend)

        # ax.set_ylim(30, 90)
        # plt.yticks(np.arange(30, 90, 5))
        ax.legend()

    def plot_swp(self, tb_results: List[Mapping[str, Any]], plot=True) -> Dict:
        sfdr_list, sndr_list, freq_list = [], [], []
        for result in tb_results:
            sfdr_list.append(result['sfdr'])
            sndr_list.append(result['sndr'])
            freq_list.append(result['freq'])
        if plot:
            self.plot_result(sfdr_list, sndr_list, freq_list)
        return dict(sfdr=sfdr_list, sndr=sndr_list, freq=freq_list)

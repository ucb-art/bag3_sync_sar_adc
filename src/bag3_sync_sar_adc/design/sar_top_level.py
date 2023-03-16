# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Mapping, Any, Dict, List, Union
import numpy as np
import argparse
import math
import copy
from copy import deepcopy

from pybag.enum import LogLevel
from bag.simulation.cache import DesignInstance, MeasureResult, SimResults
from bag.simulation.core import MeasurementManager
from bag.concurrent.util import GatherHelper
from bag.io.sim_data import save_sim_results, load_sim_file
from bag.io.file import read_yaml, write_yaml
from bag.core import BagProject
from bag.util.misc import register_pdb_hook

from bag.simulation.design import DesignerBase

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate cell from spec file.')
    parser.add_argument('specs', help='Design specs file name.')
    parser.add_argument('-x', '--extract', action='store_true', default=False,
                        help='Run extracted simulation')
    parser.add_argument('-f', '--force_extract', action='store_true', default=False,
                        help='Force RC extraction even if layout/schematic are unchanged')
    parser.add_argument('-s', '--force_sim', action='store_true', default=False,
                        help='Force simulation even if simulation netlist is unchanged')
    parser.add_argument('-c', '--gen_cell', action='store_true', default=False,
                        help='Generate testbench schematics and DUT for debugging.')
    parser.add_argument('-cd', '--gen_cell_dut', action='store_true', default=False,
                        help='Generate only DUT for debugging.')
    parser.add_argument('-ct', '--gen_cell_tb', action='store_true', default=False,
                        help='Generate only testbench schematics for debugging.')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
                        help='Print only warning messages or above.')
    args = parser.parse_args()
    return args


async def run_dsn(prj: BagProject, args: argparse.Namespace) -> None:
    specs: Mapping[str, Any] = read_yaml(args.specs)

    log_level = LogLevel.WARN if args.quiet else LogLevel.INFO
    DesignerBase.design_cell(prj, specs, extract=args.extract, force_sim=args.force_sim,
                             force_extract=args.force_extract, gen_cell=args.gen_cell, gen_cell_dut=args.gen_cell_dut,
                             gen_cell_tb=args.gen_cell_tb, log_level=log_level)

def run_gen_cell(prj: BagProject, args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    prj.generate_cell(specs, raw=args.raw, gen_lay=args.gen_lay, run_drc=args.run_drc,
                      gen_sch=args.gen_sch, run_lvs=args.run_lvs, run_rcx=args.run_rcx,
                      gen_lef=args.gen_lef, flat=args.flat, sim_netlist=args.gen_sim,
                      gen_hier=args.gen_hier, gen_model=args.gen_mod,
                      gen_shell=args.gen_shell, export_lay=args.export_lay,
                      gen_netlist=args.gen_netlist)

def run_meas_cell(prj: BagProject, args: argparse.Namespace) -> None:
    specs: Mapping[str, Any] = read_yaml(args.specs)

    log_level = LogLevel.WARN if args.quiet else LogLevel.INFO
    prj.measure_cell(specs, extract=args.extract, force_sim=args.force_sim, force_extract=args.force_extract,
                     gen_cell=args.gen_cell, gen_cell_dut=args.gen_cell_dut, gen_cell_tb=args.gen_cell_tb,
                     log_level=log_level, fake=args.fake)

def budget_noise(tot_noise: float, clk_jitter:float, samp_freq: Union[float, int], 
                 fs_range: float, res: Union[float, int]):
    """sets the noise budget for each component
        Arguments:
            tot_noise: total allowable noise
            clk_jitter: clock jitter, in units of time
            samp_freq: the sampling frequency, in Hzz
            fs_range: full scale range
            res: resistance, for RC feasibility calculation
    """  

    jitter_budget = clk_jitter*fs_range*(np.sin(1/samp_freq)-np.sin(-1/samp_freq))
    c_large = samp_freq*2*np.pi*res
    kT = 9.83e-22
    if np.sqrt(kT/c_large) > tot_noise-jitter_budget:
        print("Budget not feasible")
    cdac_budget = np.sqrt(kT/c_large)*8
    comp_budget = tot_noise - cdac_budget - jitter_budget

    return cdac_budget, comp_budget, jitter_budget

def set_redundancy(resolution: int, throughput: int, mismatch: float):
    """Set the redundancy scheme

        Arguments:
            resolution: the number of requested bits
            throughput: number of evaluation cycles
            mismatch: capacitor mismatch, given as percentage
    """
    num_codes = pow(2, resolution) * (1+mismatch)
    bits_redun = num_codes - pow(2, resolution)
    if num_codes>pow(2, throughput): 
        # I think this should be an error
        print("Warning: Resolution not responsible given throughput")
    
    return [pow(2, n) for n in range(resolution)]

def set_cdac_params(cdac_params: Mapping[str, Any], dest_file: str, redun_config: List[int], 
                    unit_cap: float, capparea: float=2.2e-6, cap_cols:int=4, lay_res:float=0.005):
    """
        Calculates the capacitor rows and writes to file
        Arguments:
            cdac_params: parameters, use to write new parameters
            dest_file: yaml file path to write modified cdac parameters
            redun_config: cap sizing at each decision step
            unit_cap: size of the LSB cap in farads (F)
            capparea: cap size in farads per 1um^2 area 
            cal_cols: number of intended columns for each row
            lay_res: layout resolution. Used to translate cap size to BAG usable format
    """
    nbits = len(redun_config)
    row_list = [1] + [math.ceil(b/cap_cols) for b in redun_config]
    col_list = [1] + [b%cap_cols for b in redun_config]
    ratio_list = [1] + [math.ceil(b/cap_cols) if idx>3 else 8 \
                        for idx, b in enumerate(redun_config)]
    ny_list = row_list

    sq_area = unit_cap/capparea
    wl = math.sqrt(sq_area)
    unit_width = math.ceil(wl/lay_res)
    unit_height = unit_width
    width = unit_width * cap_cols

    write_params = copy.deepcopy(cdac_params)
    write_params['nbits'] = nbits
    write_params['row_list'] = row_list
    write_params['col_list'] = col_list
    write_params['ratio_list'] = ratio_list
    write_params['ny_list'] = ny_list
    write_params['width'] = width
    write_params['cap_config']['unit_width'] = unit_width
    write_params['cap_config']['unit_height'] = unit_height

    write_yaml(dest_file, write_params)


def set_logic_params(logic_params: Mapping[str, Any], dest_file: str, nbits: int):
    """
        Calculates the logic parameters and writes to file
        Arguments:
            logic_params: Mapping of parameters
            dest_file: str of destination yaml path to write to
            nbits: number of ADC decision steps
    """
    logic_unit_row_arr = [1, 2, 2]
    idx = 2
    for i in range(nbits-4):
        logic_unit_row_arr[idx] +=1
        if idx > 0:
            idx = idx-1
        else:
            idx = 2
    flop_out_unit_row_arr = [3, 2]
    idx = 0
    for i in range (nbits-4):
        flop_out_unit_row_arr[idx] +=1
        idx = 1 if idx==0 else 0
    logic_scale_list = [1 for i in range(nbits+1)]

    write_params = copy.deepcopy(logic_params)
    write_params['logic_array']['logic_unit_row_arr']=logic_unit_row_arr
    write_params['logic_array']['flop_out_unit_row_arr']=flop_out_unit_row_arr
    write_params['logic_array']['seg_dict']['logic_scale_list']=logic_scale_list

    write_yaml(dest_file, write_params)

def set_adc_params(adc_params: Mapping[str, Any], dest_file_list: List[str]):
    write_params = copy.deepcopy(adc_params)
    # fill in all the other files
    write_yaml(dest_file_list['sar'], write_params)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['bprj']

    specs = read_yaml(_args.specs)
    
    # read parameters
    unit_res = specs['process_info']['unit_res_ohms']
    c_mismatch = specs['process_info']['cap_mismatch_percentage']
    input_range = specs['top_specs']['input_range']
    N = specs['top_specs']['resolution_bits']
    samp_freq = specs['top_specs']['samp_freq']
    throughput = specs['top_specs']['throughput']
    clk_jitter = specs['app_specs']['clk_in_jitter']
    dest_file_list = specs['dest_files']

    # read generator yaml
    gen_specs = read_yaml(specs['gen_specs'])
    gen_specs_cdac = read_yaml(gen_specs['cdac_params'])
    gen_specs_logic_array = read_yaml(gen_specs['logic_params'])
    

    # feedforward path
    lsb_size = input_range/pow(2, N)
    max_noise = lsb_size/2
    redun_config = set_redundancy(N, throughput, c_mismatch)

    noise_cdac, noise_comp, noise_jitter = budget_noise(max_noise, clk_jitter, 
                                                        samp_freq, input_range, unit_res)

    # # async the optimizer runs
    # #comparator
    run_dsn(_prj, specs['opt_components']['comp'])
    
    kT = 9.83e-22
    unit_cap = (kT/noise_cdac)/sum(redun_config)
    cap_cols = 4 #we lay out in sets of 4 columns
    set_cdac_params()
        # need to calculate the dummy rows and cols somewhere 
    set_logic_params()

    # # sampler
    run_dsn(_prj, specs['opt_components']['bootstrap'])

    # final synthesize ADC
    run_gen_cell(_prj, specs['dest_files']['sar_top'])

    # simulate ADC
    run_meas_cell(_prj, specs['top_verification_tbm']['static'])
    run_meas_cell(_prj, specs['top_verification_tbm']['dynamic'])

    # read results

    # #enumerate how the ADC failed:
    # while (pass_specs !=0 and iterations<num_iter):
    #     edit_specs(pass_specs)
    #     # synthesize, simulate, check    

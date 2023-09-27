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
import asyncio
import copy
from copy import deepcopy
import time

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
    # design
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
    
    # measure
    parser.add_argument('--fake', action='store_true', default=False,
                        help='Enable fake measurement.')
    
    # generate
    parser.add_argument('-d', '--drc', dest='run_drc', action='store_true', default=False,
                        help='run DRC.')
    parser.add_argument('-v', '--lvs', dest='run_lvs', action='store_true', default=False,
                        help='run LVS.')
    parser.add_argument('--rcx', dest='run_rcx', action='store_true', default=False,
                        help='run RCX.')
    parser.add_argument('-raw', dest='raw', action='store_true', default=False,
                        help='generate GDS/netlist files instead of OA cellviews.')
    parser.add_argument('-flat', dest='flat', action='store_true', default=False,
                        help='generate flat netlist.')
    parser.add_argument('-lef', dest='gen_lef', action='store_true', default=False,
                        help='generate LEF.')
    parser.add_argument('-hier', '--gen-hier', dest='gen_hier', action='store_true',
                        default=False, help='generate Hierarchy.')
    parser.add_argument('-mod', '--gen-model', dest='gen_mod', action='store_true', 
                        default=False, help='generate behavioral model files.')
    parser.add_argument('-sim', dest='gen_sim', action='store_true', default=False,
                        help='generate simulation netlist instead.')
    parser.add_argument('-shell', dest='gen_shell', action='store_true', default=False,
                        help='generate verilog shell file.')
    parser.add_argument('-lay', dest='export_lay', action='store_true', default=False,
                        help='export layout file.')
    parser.add_argument('-netlist', dest='gen_netlist', action='store_true', default=False,
                        help='generate netlist file.')
    parser.add_argument('--no-layout', dest='gen_lay', action='store_false', default=True,
                        help='disable layout.')
    parser.add_argument('--no-sch', dest='gen_sch', action='store_false', default=True,
                        help='disable schematic.')
    
    args = parser.parse_args()
    return args

def run_dsn(prj: BagProject, spec_file: str, args: argparse.Namespace, dest_file: str) -> None:
    specs: Mapping[str, Any] = read_yaml(spec_file)
    specs['dsn_params']['dest_file'] = dest_file
    specs['dsn_params']['result_dest_file'] = dest_file.replace('.yaml', '_result.yaml')

    log_level = LogLevel.WARN if args.quiet else LogLevel.INFO
    DesignerBase.design_cell(prj, specs,extract=args.extract, force_sim=args.force_sim,
                             force_extract=args.force_extract, gen_cell=args.gen_cell, 
                             gen_cell_dut=args.gen_cell_dut,
                             gen_cell_tb=args.gen_cell_tb, log_level=log_level)

def run_gen_cell(prj: BagProject, spec_file: str, args: argparse.Namespace) -> None:
    specs = read_yaml(spec_file)
    prj.generate_cell(specs, raw=args.raw, gen_lay=args.gen_lay, run_drc=args.run_drc,
                      gen_sch=args.gen_sch, run_lvs=args.run_lvs, run_rcx=args.run_rcx,
                      gen_lef=args.gen_lef, flat=args.flat, sim_netlist=args.gen_sim,
                      gen_hier=args.gen_hier, gen_model=args.gen_mod,
                      gen_shell=args.gen_shell, export_lay=args.export_lay,
                      gen_netlist=args.gen_netlist)

def run_meas_cell(prj: BagProject, spec_file: str, args: argparse.Namespace,
                  gen_specs_file: str, nbits: int) -> None:
    specs: Mapping[str, Any] = read_yaml(spec_file)
    specs['gen_specs_file'] = gen_specs_file

    # Adjust the testbench to correspond with the number of decisions made
    _save_list = specs['meas_params']['tbm_specs']['save_outputs']
    _save_list = [f'data_out<{nbits}:0>' if 'data_out' in s else s for s in _save_list]
    specs['meas_params']['tbm_specs']['save_outputs'] = _save_list
    specs['meas_params']['tbm_specs']['sim_params']['num_channel'] = nbits
    if nbits % 2:
        specs['meas_params']['tbm_specs']['sim_params']['rst_cycle'] = 1
    else:
        specs['meas_params']['tbm_specs']['sim_params']['rst_cycle'] = 2

    log_level = LogLevel.WARN if args.quiet else LogLevel.INFO
    prj.measure_cell(specs, extract=args.extract, force_sim=args.force_sim, 
                     force_extract=args.force_extract, gen_cell=args.gen_cell, 
                     gen_cell_dut=args.gen_cell_dut, gen_cell_tb=args.gen_cell_tb,
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

def non_binary_redundancy(resolution, R, mismatch, cap_cols: int=4, rowsplit: int = 5):

    p_list = [1]
    diff_flag = 0
    pbyq_max = R  #0.5*math.exp(0.7*resolution/R)
    finish_flag = 0
    while(not finish_flag):
        remwt = pow(2, resolution) - 1 - sum(p_list)
        
        p_temp = 2*p_list[-1]
        q_temp = -p_temp + sum(p_list) + 1
        if q_temp==0:
            w_pq = 1 if p_temp > 4 else 0

        elif (p_temp/q_temp>pbyq_max):
            w_pq = (p_temp-pbyq_max*q_temp)/(pbyq_max+1)
        else:
            w_pq = 0
        w_mismatch = 0 #1 if p_temp>2 else 0 #mismatch*0.01*p_temp/pow(2, resolution)
        w_adj = math.ceil(max(w_pq, w_mismatch))
        
        valid_list = [i for i in range(1, cap_cols+1)] + [(cap_cols-1)*i for i in range(2, rowsplit+2)] \
                         +  [cap_cols*i for i in range(2, rowsplit+1)]
        valid_list = list(set(sorted(valid_list)))
        p_temp = min(valid_list, key=lambda x: abs(x - (p_temp-w_adj))) \
                    if math.ceil((p_temp-w_adj)/cap_cols) <= rowsplit else p_temp-w_adj
        
        # Reweight first few indices to make the weight determination in later steps easier
        if math.ceil(p_temp/4) > rowsplit : 
            if not diff_flag:
                sum_lower = sum(p_list)
                offset = (pow(2, resolution) - 1 - sum_lower)%(cap_cols*2)
                print("OFFSET: ", offset)
                offset = offset-(cap_cols*2) if offset>cap_cols else offset
                print("OFFSET ADJ: ", offset)
                remwt_short = sum_lower + offset  #_short indicates that it pertains to the list of indices that dont get split
                if offset> 0:
                    pbyq_max_short = R*2 #0.5*math.exp(0.7*resolution/R)
                    print("pbyq: ", pbyq_max_short)
                    finish_flag_short = 0
                    ps_list=[1]
                elif offset<0:
                    pbyq_max_short = pbyq_max #0.5*math.exp(0.7*resolution/R)
                    print("pbyq: ", pbyq_max_short)
                    finish_flag_short = 0
                    ps_list=[1]
                else:
                    finish_flag_short = 1
                while(not finish_flag_short):
                    # p is next value, q is sum of existing values (?)
                    # want to have a p/q that hits the redundancy requirements
                    # basically reconstruct the first few weights in list to satisfy remwt_short
                    remwts = remwt_short - sum(ps_list)
                    ps_temp = 2*ps_list[-1]
                    qs_temp = -ps_temp + sum(ps_list) + 1
                    print("qs_temp: ", qs_temp)
                    if qs_temp==0: # in case q is 0, dont get undefined
                        ws_pq = 1 if ps_temp > cap_cols else 0 

                    elif (ps_temp/qs_temp>pbyq_max_short):
                        # adjust p by ws_pq
                        ws_pq = (ps_temp-pbyq_max_short*qs_temp)/(pbyq_max_short+1) if ps_temp > cap_cols else 0
                    else:
                        ws_pq = 0
                    ws_mismatch = mismatch/100 if (ps_temp>cap_cols) else 0 
                    ws_adj = math.ceil(max(ws_pq, ws_mismatch)) # final adjustment is max of ws_pq or ws_mismatch
                    ps_temp = min(valid_list, key=lambda x: abs(x - (ps_temp-ws_adj)))
                    
                    # Sort out case where the remaining weight isn't a nice "valid number"
                    if remwts<ps_temp:
                        if remwts in valid_list:
                            ps_list.append(remwts)
                        else: 
                            sorted_list = sorted(valid_list, key=lambda x: abs(x - remwts))
                            # get closest number that is larger than remwts
                            for s in sorted_list:
                                if s > remwts:
                                    closest_number = s
                                    break
                            # closest_number = sorted_list[0] if sorted_list[0]>remwts \
                            #                     else sorted_list[1]

                            qs_list = [0] + [-p + sum(ps_list[0:j]) + 1 for j, p in enumerate(ps_list)]
                            sub_off = closest_number - remwts
                            print(closest_number, sub_off)
                            print(ps_list[::-1])
                            for i, rev_p in enumerate(ps_list[::-1]):
                                if sub_off > 0 and rev_p <=cap_cols:
                                    print(ps_list[len(ps_list)-1-i])
                                    ps_list[len(ps_list)-1-i] = ps_list[len(ps_list)-1-i]-1
                                    sub_off = sub_off - 1
                            ps_list.append(closest_number)
                        finish_flag_short = 1
                    else:
                        ps_list.append(ps_temp)
                        finish_flag_short = 0
                
                if offset != 0:
                    p_list = sorted(ps_list) 
                diff_flag = 1

                p_temp = 2*p_list[-1]
                q_temp = -p_temp + sum(p_list) + 1
                if q_temp==0:
                    p_temp = p_temp - 1
                elif (p_temp/q_temp>pbyq_max):
                    w_pq = (p_temp-pbyq_max*q_temp)/(pbyq_max+1)
                else:
                    w_pq = 0

                w_mismatch = mismatch/100 if (p_temp>cap_cols and q_temp <1) else 0 #mismatch*0.01*p_temp/pow(2, resolution)
                w_adj = math.ceil(max(w_pq, w_mismatch))

                p_temp = p_temp-w_adj
                p_temp = p_temp - p_temp%(cap_cols*2) if p_temp%(cap_cols*2) < cap_cols or \
                         (-(p_temp + cap_cols*2 - p_temp%(cap_cols*2)) + sum(p_list) + 1) <1 \
                        else p_temp + (cap_cols*2) - p_temp%(cap_cols*2)

            else:
                p_temp = p_temp - p_temp%(cap_cols*2) if p_temp%(cap_cols*2) < cap_cols or \
                         (-(p_temp + cap_cols*2 - p_temp%(cap_cols*2)) + sum(p_list) + 1) <1 \
                        else p_temp + (cap_cols*2) - p_temp%(cap_cols*2)
            
        if remwt<p_temp:
            p_list.append(remwt)
            finish_flag = 1
        else:
            p_list.append(p_temp)
            finish_flag = 0
  
    final_plist = sorted(p_list)
    qs_list = [0] + [-p + sum(final_plist[0:j]) + 1 for j, p in enumerate(final_plist)]
    return sorted(p_list)

def set_redundancy(resolution: int, throughput: int, mismatch: float):
    """Set the redundancy scheme

        Arguments:
            resolution: the number of requested bits
            throughput: number of evaluation cycles
            mismatch: capacitor mismatch, given as percentage
    """
    # num_codes = pow(2, resolution) * (1+mismatch*0.001)
    # bits_redun = num_codes - pow(2, resolution)
    if pow(2, resolution)>pow(2, throughput): 
        # I think this should be an error
        print("Warning: Resolution not possible given throughput")
        return [pow(2, n) for n in range(resolution)]
    elif resolution==throughput:
        return [pow(2, n) for n in range(resolution)]
    else:
        R = 0.5*math.exp(0.7*resolution/3)
        if R>1 and throughput>resolution:
            max_cycles = 0
            p_list_prev = []
            while R>=1 and R<200 and max_cycles==0:
                p_list = non_binary_redundancy(resolution, R, mismatch)
                max_cycles = 1 if len(p_list)<=throughput else 0 
                R = R*2
        else:
            print("Warning: no settling improvement achieved through \
                  non-binary weighting, binary output returned")
            p_list = [pow(2, n) for n in range(resolution)]
        return p_list
        
def set_cdac_params(cdac_params: Mapping[str, Any], dest_file: str, redun_config: List[int], 
                    unit_cap: float, capparea: float=2.2e-6, cap_cols:int=4, rowsplit: int = 5, 
                    lay_res:float=0.005, min_cap: float=400):
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
    # nbits = 9 #len(redun_config)
    # ny_list=   [1, 1, 1, 1, 1, 1, 2, 4, 4, 8]
    # ratio_list= [1, 1, 2, 4, 8, 8, 8, 8, 8, 8] 
    # col_list= [1, 1, 2, 3, 3, 4, 4, 3, 4, 4]
    # row_list= [1, 1, 1, 1, 2, 3, 5, 6, 8, 14]

    nbits=len(redun_config)

    row_list = [1] 
    col_list = [1]
    ny_list = [1]
    ratio_list = [1]

    valid_ny_sizes = [pow(2, i) for i in range(int(math.log(sum(redun_config))))] #[1,2,4] + [cap_cols*2*i for i in range(1, redun_config[-1]//(cap_cols*2))]
    print("VALID NY SIZES: ", valid_ny_sizes)
    diff_flag = 0
    diff_idx = nbits+1
    for idx, size in enumerate(redun_config):
        if size<=cap_cols:
            _col = size
            _row = 1
            _ny = 1
            _ratio = size
        else:
            if not(size%cap_cols):
                _col = cap_cols
                _row = size//(cap_cols) 
                # _ny
                _ratio = cap_cols
                if size/cap_cols > rowsplit:
                    diff_idx = idx+1 if not(diff_flag) else diff_idx
                    diff_flag = 1 if not(diff_flag) else 0
                    _row = size//(2*cap_cols) 
                    #_ny = int(round(_row/(2*cap_cols))*2*cap_cols)
                    _ratio = cap_cols*2
            else:
                _col = cap_cols-1
                _row = size//(cap_cols-1)
                #_ny = _row if _row<=cap_cols else cap_cols
                _ratio = cap_cols
            sorted_list = sorted(valid_ny_sizes, key=lambda x: abs(x - _row*3//4))
            _ny = sorted_list[0]

        row_list.append(_row)
        col_list.append(_col)
        ny_list.append(_ny)
        ratio_list.append(_ratio)
    print(ny_list) 
    sq_area = unit_cap/capparea  
    wl = math.sqrt(sq_area)
    unit_width = math.ceil(wl/lay_res) if math.ceil(wl/lay_res)>min_cap else min_cap
    unit_height = unit_width
    width = unit_width * cap_cols

    write_params = copy.deepcopy(cdac_params)
    write_params['params']['nbits'] = nbits
    write_params['params']['diff_idx'] = diff_idx
    write_params['params']['row_list'] = row_list
    write_params['params']['col_list'] = col_list
    write_params['params']['ratio_list'] = ratio_list
    write_params['params']['ny_list'] = ny_list
    write_params['params']['width'] = width
    write_params['params']['cap_config']['unit_width'] = unit_width
    write_params['params']['cap_config']['unit_height'] = unit_height
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
    max_idx = 2
    idx = 2
    for i in range(nbits-4):
        if logic_unit_row_arr[idx] >= 4:
            logic_unit_row_arr = [3, 3, 3, 3]
            max_idx += 1
            idx = max_idx
        else:   
            logic_unit_row_arr[idx] +=1
            if idx > 0:
                idx = idx-1
            else:
                idx = max_idx

    flop_out_unit_row_arr = [3, 2]
    idx = 1
    for i in range (nbits-4):
        flop_out_unit_row_arr[idx] +=1
        idx = 1 if idx==0 else 0
    logic_scale_list = [1 for i in range(nbits+1)]

    write_params = copy.deepcopy(logic_params)
    write_params['params']['logic_array']['logic_unit_row_arr']=logic_unit_row_arr
    write_params['params']['logic_array']['flop_out_unit_row_arr']=flop_out_unit_row_arr
    write_params['params']['logic_array']['seg_dict']['logic_scale_list']=logic_scale_list

    write_yaml(dest_file, write_params)

def set_clk_params(clk_params: Mapping[str, Any], dest_file: str, nbits: int):
    """
        Calculates the clk divider parameters and writes to file
        Arguments:
            logic_params: Mapping of parameters
            dest_file: str of destination yaml path to write to
            nbits: number of ADC decision steps
    """
    write_params = copy.deepcopy(clk_params)
    write_params['params']['total_cycles']=nbits
    write_yaml(dest_file, write_params)

def set_sample_opt_params(opt_file_src: str, opt_file_dest: str, sig_sampler_wrfile: str,
                          gen_specs_sampler: Mapping[str, Any], meas_wrfile: str, len_redun: int):
    
    sig_sampler_params = copy.deepcopy(read_yaml(gen_specs_sampler['params']['sig_sampler']))
    default_list = sig_sampler_params['params']['nmos_params']['sampler_params']['m_list']
    if len(default_list) != len_redun:
        #breakpoint()
        if len_redun>len(default_list):
            new_list = default_list + [max(default_list) for i in range(len_redun-len(default_list))]
        else:
            new_list = default_list[0:len_redun]
        print("NEW LIST: ", new_list, '******************************************************')
        sig_sampler_params['params']['nmos_params']['sampler_params']['m_list'] = new_list
        sig_sampler_params['params']['nmos_params']['seg_dict']['sampler'] = new_list
        write_yaml(sig_sampler_wrfile, sig_sampler_params)

        write_dsn_params = copy.deepcopy(read_yaml(opt_file_src))
        meas_params = copy.deepcopy(read_yaml( write_dsn_params['dsn_params']['meas_params']))
        meas_params['meas_params']['tbm_specs']['save_outputs']
        _save_list = meas_params['meas_params']['tbm_specs']['save_outputs']
        _save_list = [f'out<{len_redun-1}:0>' if 'out<' in s else s for s in _save_list]
        meas_params['meas_params']['tbm_specs']['save_outputs'] = _save_list
        write_yaml(meas_wrfile, meas_params)

        write_dsn_params['dsn_params']['gen_specs'] = sig_sampler_wrfile
        write_dsn_params['dsn_params']['meas_params'] = meas_wrfile
        write_yaml(opt_file_dest, write_dsn_params)

        return opt_file_dest
    else:
        return opt_file_src

def set_sampletop_params(sampler_params: Mapping[str, Any], dest_files):
    """
        Calculates the clk divider parameters and writes to file
        Arguments:
            logic_params: Mapping of parameters
            dest_file: str of destination yaml path to write to
            nbits: number of ADC decision steps
    """
    write_top_params = copy.deepcopy(sampler_params)
    write_top_params['params']['sig_sampler']=dest_files['sig_sampler']
    write_top_params['params']['vcm_sampler']=dest_files['vcm_sampler']
    write_top_params['params']['vcm_mid_sw']=dest_files['vcm_mid_sw']
    write_yaml(dest_files['top'], write_top_params)

    write_vcm_params = copy.deepcopy(read_yaml(sampler_params['params']['vcm_sampler']))
    write_yaml(dest_files['vcm_sampler'], write_vcm_params)

    write_vcm_mid_params = copy.deepcopy(read_yaml(sampler_params['params']['vcm_mid_sw']))
    write_yaml(dest_files['vcm_mid_sw'], write_vcm_mid_params)

def set_sar_top_params(adc_params: Mapping[str, Any], dest_file_list: List[str]):
    write_params = copy.deepcopy(adc_params)
    write_params['params']['sampler_params']=dest_file_list['sampler']['top']
    write_params['params']['logic_params']=dest_file_list['logic']
    write_params['params']['comp_params']=dest_file_list['comp']
    write_params['params']['clkgen_params']=dest_file_list['clkgen']
    write_params['params']['cdac_params']=dest_file_list['cdac']
    write_yaml(dest_file_list['sar_top'], write_params)

if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['bprj']

    t_start = time.time()
    specs = read_yaml(_args.specs)

    # read parameters
    unit_res = specs['process_info']['unit_res_ohms']
    c_mismatch = specs['process_info']['cap_mismatch_percentage']
    input_range = specs['top_specs']['input_range']
    N = specs['top_specs']['resolution_bits']
    samp_freq = specs['top_specs']['samp_freq']
    throughput = specs['top_specs']['throughput']
    clk_jitter = specs['app_specs']['clk_in_jitter']

    # write all the destination file names
    dest_file_list = specs['dest_files']
    for key, val in dest_file_list.items():
        if not('directory' in key or 'sampler' in key):
            dest_file_list[key] = dest_file_list['directory'] + val
        if 'sampler' in key:
            for k, v in dest_file_list['sampler'].items():
                dest_file_list['sampler'][k] = dest_file_list['directory'] + v

    # read generator yaml
    gen_specs = read_yaml(specs['gen_specs'])
    gen_specs_cdac = read_yaml(gen_specs['params']['cdac_params'])
    gen_specs_logic_array = read_yaml(gen_specs['params']['logic_params'])
    gen_specs_clkgen = read_yaml(gen_specs['params']['clkgen_params'])
    gen_specs_sampler = read_yaml(gen_specs['params']['sampler_params'])

    # feedforward path
    lsb_size = input_range/pow(2, N)
    max_noise = lsb_size/2
    redun_config = set_redundancy(N, throughput, c_mismatch)
    print(redun_config)

    noise_cdac, noise_comp, noise_jitter = budget_noise(max_noise, clk_jitter, 
                                                        samp_freq, input_range, unit_res)

    # TODO: parallelize the optimizer run
    run_dsn(_prj, specs['opt_components']['comp'], _args, dest_file_list['comp'])
    comp_opt_perf = read_yaml(dest_file_list['comp'].replace('.yaml', '_result.yaml'))
    if comp_opt_perf['noise'][0] > noise_comp:
        print("*****WARNING: Comparator noise budgeted unacheivable for topology and sizing range")
    else:
        print(f"----Comparator noise {comp_opt_perf['noise'][0]} less than budgeted {noise_comp} ----- :)")
        
    kT = 9.83e-22
    unit_cap = (kT/noise_cdac)/sum(redun_config)
    set_cdac_params(gen_specs_cdac, dest_file_list['cdac'], redun_config, 
                    unit_cap)
    set_logic_params(gen_specs_logic_array, dest_file_list['logic'], len(redun_config))
    set_clk_params(gen_specs_clkgen, dest_file_list['clkgen'], 
                  len(redun_config)+(1 if len(redun_config)%2 else 2))

    # # sampler
    sampler_opt_file = set_sample_opt_params(specs['opt_components']['bootstrap'],
                                dest_file_list['directory']+specs['opt_components']['bootstrap_copy'], 
                                dest_file_list['directory']+specs['opt_components']['sig_sampler_dsn'],
                                gen_specs_sampler,
                                dest_file_list['directory']+specs['opt_components']['sig_sampler_meas'],
                                len(redun_config))
    print("SIMMING THIS: ", sampler_opt_file, '-----------------------------------------------------')
    run_dsn(_prj, sampler_opt_file, _args, 
            dest_file_list['sampler']['sig_sampler'])
    set_sampletop_params(gen_specs_sampler, dest_file_list['sampler'])
    # #write to the final ADC
    set_sar_top_params(gen_specs, dest_file_list)


    # # simulate ADC
    # run_meas_cell(_prj, specs['top_verification_tbm']['static'], _args)
    run_meas_cell(_prj, specs['top_verification_tbm']['dynamic'], _args, 
                  dest_file_list['sar_top'], len(redun_config))

    # read results
    performance = read_yaml('results_dynamic.yaml')
    print(performance)

    # #enumerate how the ADC failed:
    # while (pass_specs !=0 and iterations<num_iter):
    #     edit_specs(pass_specs)
    #     # simulate, check    
    # final synthesize ADC
    run_gen_cell(_prj, specs['dest_files']['sar_top'], _args)

    t_end = time.time()

    print(f"TOTAL TIME: {t_end-t_start}")

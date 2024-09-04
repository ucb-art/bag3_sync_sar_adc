import math
from typing import Mapping, Dict, Any, Tuple, Optional, List, Type, \
                    Sequence, cast, Union, Coroutine, Iterable, Callable
from bag.util.immutable import Param
from pprint import pprint

import os, sys
import time
from pathlib import Path
import pickle
# from asyncio import create_task
import asyncio
from asyncio import create_task
from copy import deepcopy

import pprint

from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.ltisys import freqresp
from bag.simulation.cache import DesignInstance, MeasureResult, SimResults
from bag.simulation.core import MeasurementManager
from bag.simulation.cache import SimulationDB
from bag.io.sim_data import save_sim_results, load_sim_file
from bag.io.file import write_yaml, read_yaml
from bag.math.dfun import DiffFunction
from bag.util.search import BinaryIterator, FloatBinaryIterator, BinaryIteratorInterval

from xbase.layout.mos.placement.data import TileInfoTable

from bag3_testbenches.design.optimize.base import OptDesigner, OptimizationError
from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB
from bag.simulation.design import DesignerBase

from bag3_digital.layout.stdcells.util import STDCellWrapper
from bag3_digital.layout.stdcells.levelshifter import LevelShifter, LevelShifterCore

from bag3_sync_sar_adc.layout.sar_comp import SARComp
from bag3_sync_sar_adc.measurement.comparator import ComparatorMM

from bag.concurrent.util import GatherHelper

from bag.env import get_tech_global_info
from .util import parse_params_file, get_dut_cls, get_param, todict

import datetime

class CompDesigner(OptDesigner):
    "Will design a comparator "
    def __init__(self, root_dir: Path, sim_db: SimulationDB, dsn_specs: Mapping[str, Any]) -> None:
        super().__init__(root_dir, sim_db, dsn_specs)

    def commit(self):
        super().commit()
        base_gen_specs = parse_params_file(self.dsn_specs['gen_specs'])
        self._dut_class = get_dut_cls(base_gen_specs)
        self._base_gen_specs = self._dut_class.process_params(base_gen_specs['params'])[0]
        self._base_gen_specs = self.get_dut_gen_specs(False, self._base_gen_specs, self._base_gen_specs)
        self._meas_params = parse_params_file(self.dsn_specs['meas_params'])['meas_params']
        self._constraints = self.dsn_specs['opt_specs']
        
        self.result_dest = self.dsn_specs['result_dest_file']
        self._constraints = {k: (v[0] if v[0] is not None else 0, 
                                 v[1] if  v[1] is not None else math.inf) \
                                 for k, v in self.dsn_specs['opt_specs']['spec_constraints'].items()}
        #FIXME
        self._is_lay = True

    @classmethod
    def get_dut_gen_specs(cls, is_lay: bool, base_gen_specs: Param,
                          gen_params: Mapping[str, Any]) -> Union[Param, Dict[str, Any]]:
        lut = cls.dsn_to_gen_spec_map('SA')
        comp_segs = {k: get_param(k, gen_params, base_gen_specs, lut[k], dtype=int)
                     for k in ['nfb', 'pfb', 'tail', 'in', 'sw']}

        # if is_lay:
        #     raise NotImplementedError
        # else:
        seg_dict = base_gen_specs['seg_dict'].copy(append={
            'nfb': comp_segs['nfb'],
            'pfb': comp_segs['pfb'],
            'tail': comp_segs['tail'],
            'in': comp_segs['in'],
            'sw': comp_segs['sw']
        })
        change_specs = todict(base_gen_specs.copy().to_dict())
        change_specs['seg_dict'] = seg_dict
        return change_specs #base_gen_specs.copy(append=dict(seg=seg_dict))

    @staticmethod
    def dsn_to_gen_spec_map(comp_cls):
        "Figures out which comparator specs to use"
        if comp_cls == 'SA':
            dict = {
                'nfb': ['seg_dict', 'nfb'],
                'pfb': ['seg_dict', 'pfb'],
                'tail': ['seg_dict', 'tail'],
                'in': ['seg_dict', 'in'],
                'sw': ['seg_dict', 'sw']
            }
        
        return dict

    async def pre_setup(self, dsn_params: Dict[str, Any]):
        return dict(**dsn_params)

    @classmethod
    def get_meas_var_list(cls):
        return ['delay', 'noise', 'reward', 'avg_power']
    
    async def async_design(self,  **kwargs: Any) -> Mapping[str, Any]:
        char_start_time = time.perf_counter()
        await self.characterize_designs()
        char_end_time = time.perf_counter()
        print("Characterization Time: ", char_end_time-char_start_time)
        
        db_path = self._out_dir / 'db.hdf5'
        db_data = load_sim_file(str(db_path)) #FIXME

        model_start_time = time.perf_counter()
        fn_table, swp_order = self.make_models()
        model_end_time = time.perf_counter()
        print("Modeling Time: ", model_end_time-model_start_time)
        # self.plot_specs(swp_order, db_data, fn_table)
        opt_specs = self._dsn_specs['opt_specs']
        spec_constraints = {k: (0 if v[0] is None else v[0]*(-1e6), math.inf if v[1] is None else v[1]*1e6) for k, v in opt_specs['spec_constraints'].items()}
        var_constraints = opt_specs['var_constraints']
        c_load_arr = [] #self.sim_load_swp.get_swp_values('c_load')
        self.run_opt_sweep(*opt_specs['opt'], '', c_load_arr, fn_table, swp_order,
                           var_constraints, spec_constraints)
        
        return fn_table

    def run_opt_sweep(self, opt_var: str, opt_maximize: bool, swp_var: str, swp_vals: Union[List[float], np.ndarray],
                      fn_table: Dict[str, List[DiffFunction]], swp_order: List[str],
                      var_constraints: Dict[str, Any], spec_constraints: Dict[str, Any]):
        opt_start_time = time.perf_counter()
        if swp_var == '':
            size = 1
            opt_x = {}
            opt_y = np.full(size, np.nan)
            spec_vals = {}
            num_envs = len(self.env_list)
            success_idx_list = []
            self.log(f"single opt sweep...")
            try:
                opt_x, opt_y, spec_vals = self.optimize(
                    opt_var, fn_table, swp_order, maximize=opt_maximize, reduce_fn=np.min if opt_maximize else np.max,
                    var_constraints={**var_constraints},
                    spec_constraints=spec_constraints
                )
            except OptimizationError as e:
                self.warn(f"Error occurred while running: {e}")
            else:
                success_idx_list.append(1)
        else:
            size = len(swp_vals)
            opt_x = {}
            opt_y = np.full(size, np.nan)
            spec_vals = {}
            num_envs = len(self.env_list)
            success_idx_list = []
            for i, swp_val in enumerate(swp_vals):
                self.log(f"Performing {opt_var} optimization for {swp_var} = {swp_val}...")
                try:
                    sub_opt_x, sub_opt_y, sub_spec_vals = self.optimize(
                        opt_var, fn_table, swp_order, maximize=opt_maximize, reduce_fn=np.min if opt_maximize else np.max,
                        var_constraints={**var_constraints, swp_var: swp_val},
                        spec_constraints=spec_constraints
                    )
                except OptimizationError as e:
                    self.warn(f"Error occurred while running: {e}")
                    continue
                else:
                    success_idx_list.append(i)
                    if len(success_idx_list) == 1:
                        for k, sub_v in sub_opt_x.items():
                            opt_x[k] = np.full((size, *np.array(sub_v).shape), np.nan)
                        for k in sub_spec_vals:
                            spec_vals[k] = np.full((size, num_envs), np.nan)
                    for k, v in sub_opt_x.items():
                        opt_x[k][i] = np.array(v)
                    opt_y[i] = sub_opt_y
                    for k, v in sub_spec_vals.items():
                        spec_vals[k][i] = v
        print("OPT_X: ", opt_x)
        print("OPT_Y: ", opt_y)
        print("SPEC_VALS: ", spec_vals)

        opt_end_time = time.perf_counter()
        print("Optimization Time: ", opt_end_time-opt_start_time)
        
        ####
        plt_key = 'avg_power'
        params = self._dsn_specs['dsn_swp_params']
        # Generate linspace arrays based on the dictionary
        linspace_arrays = {key: np.linspace(value['start'], value['stop'], value['num'])
                        for key, value in params.items()}

        # Extract keys and linspace arrays
        keys = list(linspace_arrays.keys())
        arrays = list(linspace_arrays.values())

        # Create meshgrid
        meshgrids = np.meshgrid(*arrays, indexing='ij')
        # Create a dictionary to hold the meshgrid results with corresponding keys
        meshgrid_dict = {key: meshgrid for key, meshgrid in zip(keys, meshgrids)}

        values = []
        x_plot = []
        y_plot = []
        for i in meshgrid_dict['in'][:,0,0,0]:
            for j in meshgrid_dict['tail'][0,0,0,:]:
                x_plot.append(i)
                y_plot.append(j)
                values.append(fn_table[plt_key][0].__call__([i,opt_x['nfb'], opt_x['pfb'],j]))
        #values = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

                # Generate linspace arrays based on the dictionary
        linspace_interp_arrays = {key: np.arange(value['start'], value['stop']+2, 2)
                        for key, value in params.items()}

        # Extract keys and linspace arrays
        arrays_interp = list(linspace_interp_arrays.values())

        # Create meshgrid
        meshgrids_interp= np.meshgrid(*arrays_interp, indexing='ij')

        # Create a dictionary to hold the meshgrid results with corresponding keys
        meshgrid_interp_dict = {key: meshgrid for key, meshgrid in zip(keys, meshgrids_interp)}

        values_interp = []
        x_interp_plot = []
        y_interp_plot = []
        for i in meshgrid_interp_dict['in'][:,0,0,0]:
            for j in meshgrid_interp_dict['tail'][0,0,0,:]:
                x_interp_plot.append(i)
                y_interp_plot.append(j)
                values_interp.append(fn_table[plt_key][0].__call__([i,opt_x['nfb'], opt_x['pfb'],j]))

        # # Create a scatter plot
        # plt.figure(figsize=(8, 6))

        # # Create a scatter plot with color mapping
        # sc = plt.scatter(x_plot, y_plot, c=values, cmap='viridis', s=100, edgecolor='k')

        # # Add a colorbar to show the values
        # plt.colorbar(sc, label='Values')

        # Create a figure and axes
        ## fig, ax = plt.subplots(figsize=(8, 6))
#
        ## # Create scatter plots
        ## # sc3 = ax.scatter(x=[opt_x['cap_n']], y=[opt_x['off0']],
        ## #          c=[opt_y], cmap='viridis', edgecolor='red', marker='D')
#
        ## sc2 = ax.scatter(x=x_interp_plot, y=y_interp_plot, c=values_interp, cmap='viridis', edgecolor='white', label='Scatter 2')
        ## sc1 = ax.scatter(x=x_plot, y=y_plot, c=values, cmap='viridis', edgecolor='k', label='Scatter 1')
        #
#
        ## # Create a single colorbar
        ## # Combine all scatter plot color data for a consistent color mapping
        sel_point = fn_table[plt_key][0].__call__([opt_x['in'],opt_x['nfb'], opt_x['pfb'],opt_x['tail']])
        combined_values = np.concatenate([values, values_interp, [sel_point]])
        ## vmin = np.min(combined_values)
        ## vmax = np.max(combined_values)
        ## print(values)
        ## print(values_interp)
        ## print(vmin, vmax, sel_point)
        ## # Create a scatter plot to use for the colorbar
        ## sc_colorbar = ax.scatter(x=[opt_x['in']], y=[opt_x['tail']],
        ##          c=[sel_point],
        ##              cmap='viridis', edgecolor='red', marker='D', vmin=vmin, vmax=vmax)
#
        ## # Add colorbar to the figure
        ## cbar = plt.colorbar(sc_colorbar, ax=ax, orientation='vertical', pad=0.02)
        ## cbar.set_label(plt_key)
        ## # Add labels and title
        ## plt.xlabel('diff pair fingers')
        ## plt.ylabel('tail fingers')
        ## plt.title('Noise Constraint: {:#.3g}V - Delay Constraint: {:#.3g}s'.format(
        ##     self._constraints['noise'][1], self._constraints['delay'][1]) )
#
        ## # Show the plot
        ## plt.show()

        # Create a colormap
        import matplotlib.cm as cm
        import matplotlib.colors as colors


        norm = colors.Normalize(vmin=combined_values.min(), vmax=combined_values.max())
        cmap = cm.ScalarMappable(norm=norm, cmap='cool') 

        # Create the figure and 3D axes
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        #--------
        x, y = np.meshgrid(meshgrid_interp_dict['in'][:,0,0,0], meshgrid_interp_dict['tail'][0,0,0,:])
        z = np.zeros_like(x)
        dx = dy = 0.5
        dz = np.array(values_interp).reshape(x.shape) #np.random.randint(1, 10, size=x.shape)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                zbar = fn_table[plt_key][0].__call__([x[i,j],opt_x['nfb'], opt_x['pfb'],y[i,j]])
                color = cmap.to_rgba(zbar)
                ax.bar3d(x[i, j], y[i, j], z[i, j], dx, dy, zbar, color=color)

        
        # ------------
        x, y = np.meshgrid(meshgrid_dict['in'][:,0,0,0], meshgrid_dict['tail'][0,0,0,:])
        z = np.zeros_like(x)
  
        dz = np.array(values).reshape(x.shape) #np.random.randint(1, 10, size=x.shape)

        # Plot bars
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                zbar = fn_table[plt_key][0].__call__([x[i,j],opt_x['nfb'], opt_x['pfb'],y[i,j]])
                color = [0,1, 0] #cmap.to_rgba(zbar)
                ax.bar3d(x[i, j], y[i, j], z[i, j], dx, dy, zbar, color=color)
        
        dx = dy = 1
        color = [1,0,0] #cmap.to_rgba(sel_point)
        ax.bar3d(opt_x['in'], opt_x['tail'],
                        0, dx, dy, sel_point, color=color)
        
        # Add a color bar which maps values to colors
        #cax = fig.add_axes([0.05, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(cmap, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(plt_key + "(W)")

        # Set labels and show the plot
        ax.set_xlabel('diff pair fingers')
        ax.set_ylabel('tail fingers')
        ax.set_ylim(4, 20)
        ax.set_zlabel(plt_key + " (W)")
        ax.set_title('Noise Constraint: {:#.3g}V - Delay Constraint: {:#.3g}s'.format(
            self._constraints['noise'][1], self._constraints['delay'][1]) )
#
        plt.savefig('comparator_space.png')
        plt.close()
        #plt.show()

        ####

        # Write optimized circuit specs and corresponding performance to two files
        self.write_specs_to_yaml(opt_x, self.dsn_specs['dest_file'], swp_var, swp_vals)
        write_yaml(self.dsn_specs['result_dest_file'], spec_vals)

        if not success_idx_list:
            raise OptimizationError("All optimization points failed")

    async def verify_design(self, dut: DesignInstance, dsn_params: Dict[str, Any],
                            sim_swp_params: Dict[str, Any]) -> Dict[str, Any]:
        dsn_name = self.get_design_name(dsn_params)
        gatherer = GatherHelper()
        gatherer.append(self.run_sim('delay', ComparatorMM, dut, dsn_name, dsn_params,
                                     sim_swp_params,
                                     self.setup_delay, self.postproc_delay))
        res_list = await gatherer.gather_err()
        res = self.aggregate_results(res_list)
        return res

    async def run_sim(self, meas_name: str, mm_cls: Type[MeasurementManager], dut: DesignInstance, dsn_name: str,
                      dsn_params, sim_swp_params: Dict[str, Any], 
                      setup_fn: Callable, postproc_fn: Callable) -> Dict[str, Any]:
       sim_dir = self.get_meas_dir(dsn_name)
       out_dir = self.get_data_dir(dsn_name)

       res_fpath = out_dir / f'{meas_name}.hdf5'
       run_meas = self.check_run_meas(res_fpath)
       if not run_meas:
           prev_res = load_sim_file(str(res_fpath))
           self.reorder_data_swp(prev_res, self.sim_swp_order)
           #return prev_res

       mm_specs = deepcopy(self.get_shared_meas_specs(sim_swp_params))
       mm_specs.update({k: deepcopy(v) for k, v in self._meas_params.get(meas_name, {}).items()})

       mm_specs['dest_file'] = self.result_dest
       
       mm_specs = setup_fn(mm_specs)

       mm = self.make_mm(mm_cls, mm_specs)
       data = (await self._sim_db.async_simulate_mm_obj(meas_name, sim_dir / meas_name, dut, mm)).data
       res = postproc_fn(data, dsn_params)
       res['sweep_params'] = {k: self.sim_swp_order[1:] for k in res}
       res['corner'] = np.array(mm_specs['tbm_specs']['sim_envs']) #FIXME np.array(data['sim_env'])
       res.update({k: np.array(sim_swp_params[k]) for k in sim_swp_params})

       save_sim_results(res, str(res_fpath))
       return res

    
    def get_shared_meas_specs(self, sim_swp_params: Mapping[str, Any]):
        # corner is handled separately from other sweep variables
        env_list: List[str] = sim_swp_params['corner']
        shared_meas_specs = todict(deepcopy(self._meas_params).to_dict())
        # shared_meas_specs['tbm_specs']['sim_envs'] = env_list FIXME
        shared_meas_specs['swp_order'] = self.sim_swp_order[1:]
        for k in self.sim_swp_order[1:]:
            shared_meas_specs['swp_info'][k] =  dict(type='LIST', values=sim_swp_params[k]) 
        return shared_meas_specs

    def postproc_delay(self, data, dsn_params):
        violate = False
        
        if data['delay']['td'][0] < self._constraints['delay'][1] and \
           data['delay']['td'][0] > self._constraints['delay'][0]:
            d_rew = 1
        else:
            violate = True
            d_rew=1
        #n_rew = 0
        if data['noise']['Input Ref noise'][0] < self._constraints['noise'][1] and \
            data['noise']['Input Ref noise'][0] > self._constraints['noise'][0]:
            n_rew = 1 
        else: 
            n_rew=(data['noise']['Input Ref noise'][0])
            violate = True

        v_list = [v for k,v in dsn_params.items()]
        #print(v_list)
        dsn_rew = 1/data['delay']['avg_power'] #1/sum(v_list)
        reward = n_rew * d_rew *(-1) if violate else dsn_rew
        # print(dsn_rew)
        # print(n_rew+d_rew*dsn_rew)
        return {'delay': data['delay']['td'],
                'noise': data['noise']['Input Ref noise'],
                'reward': np.array([reward]),
                'avg_power': data['delay']['avg_power']} #data['outn'][0,0,0]} #minimize transistor size
    

    def setup_delay(self, mm_specs):
        return deepcopy(mm_specs)
    
    @staticmethod
    def aggregate_results(res_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        ans = {}
        for res in res_list:
            for k, v in res.items():
                if k == 'sweep_params':
                    if k not in ans:
                        ans[k] = {}
                    ans[k].update(v)
                elif k not in ans:
                    ans[k] = v
                elif isinstance(v, np.ndarray):
                    assert np.all(ans[k] == v)
                else:
                    assert ans[k] == v
        return ans

    def write_specs_to_yaml(self, opt_x: Dict[str, Any], dest_file: str, 
                            swp_var: str, swp_vals: Union[List[float], np.ndarray]) -> None:
        if swp_var == '':
            opt_specs = self.get_dut_gen_specs(False, self.base_gen_specs, opt_x)
            write_params = parse_params_file(self.dsn_specs['gen_specs']).to_dict()
            write_params['params'] = opt_specs
            write_yaml(dest_file, write_params)
        else:
            for idx, val in enumerate(swp_vals):  
                opt_dict = dict()
                for k, v in opt_x.items():
                    opt_dict[k] = v[idx]
                opt_specs = self.get_dut_gen_specs(False, self.base_gen_specs, opt_dict)
                write_params = parse_params_file(self.dsn_specs['gen_specs']).to_dict()
                write_params['params'] = opt_specs
                write_yaml(dest_file.replace('.yaml','')+'swp_var'+'_'+str(idx)+'.yaml', write_params)
from typing import Union, List, Any, Mapping, Type, Optional, Sequence
import functools

import numpy as np

from bag.math import si_string_to_float, float_to_si_string
from bag.design.module import Module
from bag.io import read_yaml
from bag.layout.template import TemplateBase
from bag.util.importlib import import_class
from bag.util.immutable import Param

def get_dut_cls(gen_specs: Mapping[str, Any], sch_cls: Optional[Type[Module]] = None,
                lay_cls: Optional[Type[TemplateBase]] = None):
    """ Copied from bag3_serdes_rx
    """

    if 'dut_class' in gen_specs:
        dut_cls = import_class(gen_specs['dut_class'])
        if not issubclass(dut_cls, (Module, TemplateBase)):
            raise ValueError(f"Invalid generator class {dut_cls.get_qualified_name()}")
    elif 'lay_class' in gen_specs:
        dut_cls = import_class(gen_specs['lay_class'])
        if not issubclass(dut_cls, TemplateBase):
            raise ValueError(f"Invalid layout generator class {dut_cls}")
    elif 'sch_class' in gen_specs:
        dut_cls = gen_specs['sch_class']
        if not issubclass(dut_cls, Module):
            raise ValueError(f"Incorrect schematic generator class {dut_cls}")
    elif lay_cls is not None:
        dut_cls = lay_cls
    elif sch_cls is not None:
        dut_cls = sch_cls
    else:
        raise ValueError("Either schematic or layout class must be specified")
    return dut_cls

def parse_params_file(params: Union[Mapping[str, Any], str]) -> Param:
    """ Copied from bag3_serdes_rx
    """
    if isinstance(params, str):
        params = read_yaml(params)
    return Param(params)

def get_param(key: str, params: Mapping[str, Any], default_params: Optional[Mapping[str, Any]] = None,
              default_key: Optional[Sequence[Any]] = None, default_val: Optional[Any] = None,
              dtype: Optional[Any] = None):
    """ Copied from bag3_serdes_rx
    """
    if default_params:
        if key in params:
            val = params[key]
        else:
            default_key = default_key or []
            try:
                val = functools.reduce(lambda sub_dict, k: sub_dict[k], [default_params] + default_key)
            except KeyError:
                val = default_val
    else:
        val = params[key]

    if int(val)%2:
        if abs(int(val)+1 - val) > abs(int(val)-1 -val):
            val = int(val)-1
        else:
            val = int(val)+1
    else:
        val = int(val)
        
    return val if dtype is None else dtype(val)

def todict(obj):
        if isinstance(obj, dict):
            data = {}
            for (k, v) in obj.items():
                try: 
                    data[k] = todict(v.to_dict())
                except Exception:
                    data[k] = v
            return data
        # elif hasattr(obj, "_ast"):
        #     return todict(obj._ast())
        # elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        #     return [todict(v) for v in obj]
        # elif hasattr(obj, "__dict__"):
        #     data = dict([(key, todict(value)) 
        #         for key, value in obj.__dict__.items() 
        #         if not callable(value) and not key.startswith('_')])
        #     return data
        else:
            return obj
        
def rec_str(obj):
        if isinstance(obj, dict):
            data = {}
            for (k, v) in obj.items():
                print(k)
                try: 
                    data[k] = rec_str(v.to_dict())
                    print('success')
                except Exception:
                    print('not heppi ....', v)
                    if not isinstance(v, dict):
                        data[k] = str(v)
                    else:
                        data[k] = rec_str(v)
            return data
        # elif hasattr(obj, "_ast"):
        #     return todict(obj._ast())
        # elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        #     return [todict(v) for v in obj]
        # elif hasattr(obj, "__dict__"):
        #     data = dict([(key, todict(value)) 
        #         for key, value in obj.__dict__.items() 
        #         if not callable(value) and not key.startswith('_')])
        #     return data
        else:
            return obj
        
# def opt_specs_to_yaml():
#     """ Write optimal design specs to a yaml file that can be used later"""
#     pass

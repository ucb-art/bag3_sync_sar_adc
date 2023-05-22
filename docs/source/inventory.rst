
Inventory 
==========================

Provided is a list of generator files with descriptions:

* Top level ADC
	- Generator: `sar_sync_bootstrap.py <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/sar_sync_bootstrap.py>`_ 
	- Specs (8 bit): `specs_slice_sync_bootstrap.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_slice_sync_bootstrap.yaml>`_
	- Specs (4 bit): `specs_slice_sync_bootstrap_small.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_slice_sync_bootstrap_small.yaml>`_
* Bootstrapped Sampler
	- Generates sampling switch
	- Generator: `sampler_top.py <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/sampler_top.py>`_
	- Specs:  `specs_lay_sample_top.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/bootstrap/specs_lay_sample_top.yaml>`_
* Synchronous Logic
	- This block is hierarchically put together in units
	- Generator (for all blocks): `sar_logic_sync.py <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/sar_logic_sync.py>`_ 
	- Specs: 
		+ For top level: `specs_logic_sync.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_logic_sync.yaml>`_
		+ For array: `specs_logic_array_sync.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_logic_array_sync.yaml>`_
		+ For unit:  `specs_logic_unit_sync.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_logic_unit_sync.yaml>`_
		+ For OAI logic gate in unit: 
* MIM Cap DAC
	- This block generates a capacitor dac of MIM caps
	- Generator (for all blocks): `sar_cdac.py <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/sar_cdac.py>`_ 
	- Specs: 
		+ For cap DAC: `specs_cdac_mim.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_cdac_mim.yaml>`_
		+ For switch bank: `specs_capdrv_unit.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_capdrv_unit.yaml>`_
		+ For single MIM capacitor:  `specs_cap_mim.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_cap_mim.yaml>`_
* Comparator
	- This block contains a wrapper around a comparator. A strongARM is used.
	- Generator: `sar_comp.py <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/sar_comp.py>`_ 
	- Specs:  `specs_comp_sa.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_comp.yaml>`_ 
* Clock generator
	- Divides input clock for a reset/sampling signal and provides buffering for clock signals
	- Generator: `clk_sync_sar.py <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/clk_sync_sar.py>`_ 
	- Specs:  `specs_clkgen_sync_sar.yaml <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/blob/main/specs_gen/sar_lay/specs_clkgen_sync_sar.yaml>`_ 
* Digital blocks
	- Contains BAG generator code for some standard logic gates (inv, nand, flip flop, latch, etc.)
	- Generator: `digital.py <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/digital.py>`_ 
	- Does not have any dedicated yaml files. Specified when used in other files
* Util Folder
	- Contains layout helper functions
	- `folder <https://github.com/ucb-art/bag3_sync_sar_adc/blob/main/src/bag3_sync_sar_adc/layout/util>`_

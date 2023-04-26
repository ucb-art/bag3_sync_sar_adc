Initial Setup
==============

#. If you have not already, read the BAG3++ documentation, found `here <https://bag3-readthedocs.readthedocs.io/en/latest/>`_. Follow the instructions for initial Server Setup. 

#. This synchronous SAR ADC generator has only been released with skywater configurations. You should therefore set up a ``bag3_skywater130_workspace``, following the instructions on the `page <https://github.com/ucb-art/bag3_skywater130_workspace>`_.

#. ``cd`` into your set-up workspace directory.

#. Clone the ``bag3_sync_sar_adc`` repo. Link to the github can be found `here <https://github.com/ucb-art/bag3_sync_sar_adc>`_.
   
#. Go into the ``data`` directory by running ``cd data`` in the workspace directory.

#. Clone the yaml data folder. Currently only the Skywater 130 data is available. The repo can be found `here <https://github.com/ucb-art/bag3_sync_sar_adc_data_skywater130/tree/main>`_.

#. In ``.bashrc_pypath`` add the following to include the ``bag3_sync_sar_adc`` generator directory in the ``PYTHONPATH``:

    .. code-block:: bash

      export PYTHONPATH="${PYTHONPATH}:${BAG_WORK_DIR}/bag3_sync_sar_adc/src"

#. In ``cds.lib.bag`` add the following to include the ``bag3_sync_sar_adc`` ``OA`` views in your virtuoso setup:

    .. code-block:: bash

      DEFINE bag3_sync_sar_adc $BAG_WORK_DIR/bag3_sync_sar_adc/OA/bag3_sync_sar_adc
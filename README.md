# Multidimensional_dopamine_RL

# Introduction  

This code was used for studying the **A multidimensional distributional map of future reward in dopamine neurons.** 
Raw data is available in a Figshare public [repository](https://doi.org/10.6084/m9.figshare.28390151.v1).

# Usage

This code was developed in Python 3.7. All required packages are in the _requirements.txt_ file. 

To generate each figure, update the `directory` and the `type_neurons` (DA or Putative_DA) and run the corresponding script. For example, to replicate Figures 3A, B and C run script _Fig3a_b_c.py_ with `type_neurons="DA"`.

The script _Parse_data_neurons.py_ processes raw data of the public repository (saved in `directory`) and outputs the estimated tuning parameters for reward magnitude and time, as well as the mean responses over the selected time windows aligned to cue (time_init_cue and time_end_cue) and reward (time_init_reward and time_end_reward) for all neurons of the chosen type (DA or Putative_DA) and saves it in `directory_save`.

All scripts except Extended_data_Fig10 take less than  10 seconds on a MacBook Pro M2 with 8GB of RAM (macOS 13.0). Extended_data_Fig10 scripts take 20 minutes to run. 






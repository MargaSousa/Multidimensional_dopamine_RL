# Multidimensional_dopamine_RL

# Introduction  

This code was used for studying the **A multidimensional distributional map of future reward in dopamine neurons.** 
Raw data is available in a Figshare public [repository](https://doi.org/10.6084/m9.figshare.28390151.v1).

# Usage

This code was developed in Python 3. All required packages are in the _requirements.txt_ file. 

The script _Parse_data_neurons.py_ processes raw data saved in `directory`, including PSTHs aligned to cue and reward delivery, as well as alignment times for each trial and trial information.
It outputs the estimated tuning parameters for reward magnitude and time, as well as the mean responses within the selected time windows (time_init_reward, time_end_reward, time_init_cue and time_end_cue) for all neurons of the chosen type (Photo_ided or Putative_DA) and saves it in `directory_save`.

To generate each figure, run the corresponding script.




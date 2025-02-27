import pdb
import time as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
start_time = timer.time()
import matplotlib as mpl
from aux_Fig6 import *
from aux_functions import *
import os

# Parameters for plots
length_ticks = 2
linewidth = 1.2
scatter_size = 4
font_size = 11

labelpad_x = 10
labelpad_y = -10
labelsize = font_size
legendsize = font_size
horizontal_size = 0.8
vertical_size = 1.3

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size-3
mpl.rcParams['ytick.labelsize'] = font_size-3
mpl.rcParams['lines.linewidth'] = linewidth
mpl.use("TkAgg")

color_tmrl = "darkblue"
color_sr = "peru"
color_value = "mediumvioletred"
color_value_time = "saddlebrown"
colors=[color_value,color_sr,color_tmrl]

# All algorithms
algos=["value","SR","TMRL"] #"value","SR",

# Task/enviornment
task="sated_hungry"#"sated_hungry", "dusk_dawn", "hungry_dusk_dawn"

# Plot probability of selecting optimal action after state change
fig_p_optimal, ax_p_optimal = plt.subplots(figsize=(horizontal_size, vertical_size))  #
ax_p_optimal.spines['left'].set_linewidth(linewidth)
ax_p_optimal.spines['bottom'].set_linewidth(linewidth)
ax_p_optimal.tick_params(width=linewidth, length=length_ticks)
ax_p_optimal.set_ylabel("P(optimal action at\n hungry first trial)")
ax_p_optimal.set_xlabel("")
ax_p_optimal.set_yticks([0,1],["0","1"])
ax_p_optimal.set_ylim(-0.2,1.2)
ax_p_optimal.set_xlim(-0.25,2.25)
ax_p_optimal.set_xticks([0,1,2],["TDRL","SR","TMRL"],rotation="vertical")
ax_p_optimal.set_box_aspect(1)

# Plot discounted utility after state change
fig_utility, ax_utility = plt.subplots(figsize=(horizontal_size, vertical_size))  #
ax_utility.spines['left'].set_linewidth(linewidth)
ax_utility.spines['bottom'].set_linewidth(linewidth)
ax_utility.tick_params(width=linewidth, length=length_ticks)
ax_utility.set_ylabel("Utility at first trial after hungry")
ax_utility.set_xlabel("")
#ax_utility.set_yticks([])
ax_utility.set_box_aspect(1)

# Plot value over time
fig_value, ax_value = plt.subplots(figsize=(horizontal_size, vertical_size))  #
ax_value.spines['left'].set_linewidth(linewidth)
ax_value.spines['bottom'].set_linewidth(linewidth)
ax_value.tick_params(width=linewidth, length=length_ticks)
ax_value.set_ylabel("Patch selection\nprobability")
ax_value.set_xlabel("Trials since hungry")
ax_value.set_yticks([0],["0"])
ax_value.set_xticks([0],["0"])
ax_value.set_box_aspect(1)

# Trials relative to state change to save
left=-200
right=2999

colors_patches=["#7570b3","#d95f02","firebrick"]

# Where simulations results are saved
main_dir=r"/Users/margaridasousa/Learning Lab Dropbox/Learning Lab Team Folder/Patlab protocols/Data/MS/Data_paper_organized/Simulations"


action_after_change_state_algo={}
for i_algo,algo in enumerate(algos):

    action_after_change_state=np.load(os.path.join(main_dir,"p_optimal_action_"+algo+"_"+task+".npy"))
    utility_after_change_state=np.load(os.path.join(main_dir,"utility_"+algo+"_"+task+".npy"))
    value=np.load(os.path.join(main_dir,"foraging_value_runs_"+algo+"_"+task+".npy"))
    trials_reference=np.load(os.path.join(main_dir,"trials_reference_"+algo+"_"+task+".npy"))
    action_after_change_state_algo[algo]=action_after_change_state

    first_trial=np.where(trials_reference>=0)[0][0]
    action_after_change_state=np.argmax(value[:,first_trial,:],axis=1)

    # Check if the optimal action was selected for each task
    if task=="sated_hungry" or task=="hungry_dusk_dawn":
        ax_p_optimal.scatter(i_algo, np.sum(action_after_change_state == 1)/action_after_change_state.shape[0], color=colors[i_algo])  # for dusk-dawn
    if task=="dusk_dawn":
        ax_p_optimal.scatter(i_algo, (np.sum(action_after_change_state==0)+np.sum(action_after_change_state==1))/action_after_change_state.shape[0],color=colors[i_algo]) #for dusk-dawn

    ax_utility.errorbar(i_algo, np.mean(utility_after_change_state), yerr=np.std(utility_after_change_state), fmt="-o", capsize=2, color=colors[i_algo])


    left_trial=np.where(trials_reference>=left)[0][0]
    right_trial=np.where(trials_reference>=right)[0][0]


    # Plot value over time
    fig_dynamics, ax_dynamics = plt.subplots(figsize=(horizontal_size, vertical_size))  #
    ax_dynamics.spines['left'].set_linewidth(linewidth)
    ax_dynamics.spines['bottom'].set_linewidth(linewidth)
    ax_dynamics.tick_params(width=linewidth, length=length_ticks)
    ax_dynamics.set_ylabel("Value per patch")
    if task=="sated_hungry":
        ax_dynamics.set_xlabel("Trials since hungry")
    else:
        ax_dynamics.set_xlabel("Trials since dawn")
    ax_dynamics.set_ylim([-1,9])
    ax_dynamics.set_yticks([0,8])
    ax_dynamics.set_xticks([0,2999],["0","3000"])
    value_algo={}

    # Compute mean and std over runs
    for patch in range(3):

        mean=np.mean(value[:,:,patch],axis=0)[left_trial:right_trial]
        std=np.std(value[:,:,patch],axis=0)[left_trial:right_trial]

        ax_dynamics.plot(trials_reference[left_trial:right_trial],mean,color=colors_patches[patch],lw=linewidth*0.5)
        ax_dynamics.fill_between(trials_reference[left_trial:right_trial], mean-std,mean+std, color=colors_patches[patch],alpha=0.5)


plt.show()
import numpy as np
import pdb
from functools import reduce
import matplotlib.pyplot as plt
from aux_functions import *
import matplotlib as mpl
import scipy
from scipy.stats import pearsonr
import os
import pandas as pd
from ast import literal_eval
from ast import literal_eval

length_ticks = 2
linewidth = 1.2
scatter_size = 4
horizontal_size = 1.7
vertical_size = 1.7
font_size = 11
labelsize = 8

# Parameters for presentation plots
# length_ticks=10
# font_size=30
# linewidth=4
# scatter_size=50
# horizontal_size=6
# vertical_size=6
# labelpad_x=10
# labelpad_y=10
# labelsize=font_size
# legendsize=font_size
# scatter_size_fr=15
# capsize=8

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rc('xtick', labelsize=labelsize)
mpl.rc('ytick', labelsize=labelsize)
mpl.use('TkAgg')

# colors for plots
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay_lead = summer(np.linspace(0.4, 1, 4))
winter = mpl.cm.get_cmap('Blues', 12)
colors_delay_lag = winter(np.linspace(0.4, 1, 4))

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "Putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get responses aligned to cue delivery
responses_cue = np.load(os.path.join(directory_parsed_data, "responses_cue_different_delays_constant_magnitude.npy"))

# Get estimated tuning for reward time
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))
all_estimates_gamma = data_frame_neurons_info['Gamma'].values
all_estimates_gain = data_frame_neurons_info['Gain'].values
n_neurons = len(all_estimates_gamma)

# Get data frame with trial-to-trial licking information
dataframe_licking = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_licking_trials.csv"),
                                converters={'Licking PSTH aligned to cue': pd.eval})

# Behavior information data
directory_raw_data = os.path.join(directory, "Raw")
dataframe_behavior_times = pd.read_csv(os.path.join(directory_raw_data, "Neurons_behavior_trials.csv"))

# Get neuron id for the selected type of neurons
if type_neurons == "DA":
    selected_neuron_ids = dataframe_behavior_times.loc[
        dataframe_behavior_times['Is photo ided'] == 1, 'Neuron id'].drop_duplicates().values
else:
    selected_neuron_ids = dataframe_behavior_times.loc[
        dataframe_behavior_times['Type of neuron'] == 'Putative_DA', 'Neuron id'].drop_duplicates().values


# Bins for which licking PSTH was computed
time_min = 0
time_max = 6.8
width_bins = 0.13
n_bins = int(time_max / width_bins)
bins_licking = np.linspace(time_min, time_max, n_bins + 1)
n_bins = len(bins_licking)
time_max = bins_licking[-1]

# Window used to compute licking slope
delays = np.unique(dataframe_licking['Delay reward'].values)
bins_delays = []
for d in delays:
    time_d = d
    bin = np.where(bins_licking >= time_d)[0][0]
    bins_delays.append(bin)

# Decoding time using Laplace transform
n_time = int(time_max / 0.13)
time_decoder = np.linspace(0, time_max, n_time - 10)  # Same time-axis as licking
alpha_delay = np.array([2.5, 2.5, 2.5, 4.2])  # Smooting parameter for each delay

F = np.zeros((n_neurons, len(time_decoder)))
for i_t, t in enumerate(time_decoder):
    F[:, i_t] = (all_estimates_gamma ** t)
U, s, vh = np.linalg.svd(F, full_matrices=True)
L = np.min([vh.shape[0], np.shape(U)[1]])

n_delays = len(delays)
mean_licking_early_all = np.zeros((n_delays, len(bins_licking) - 1))  # PSTH when mice started licking earlier
mean_licking_late_all = np.zeros((n_delays, len(bins_licking) - 1))  # PSTH when mice started licking later
mean_decoding_early_all = np.zeros(
    (n_delays, len(time_decoder)))  # Decoded reward time for trials for which mice started licking earlier
mean_decoding_late_all = np.zeros(
    (n_delays, len(time_decoder)))  # Decoded reward time for trials for which mice started licking later

fig, ax = plt.subplots(2, 3, figsize=(3 * horizontal_size, 2 * vertical_size), sharex='col', sharey='row')
delay_legend = ["0s", "1.5s", "3s", "6s"]

for i_d, d in enumerate(delays[1:]):

    i_d += 1
    decoded_early = np.zeros(n_time)
    decoded_late = np.zeros(n_time)
    decoded_med = np.zeros(n_time)

    slope_early = 0
    slope_late = 0

    alpha = alpha_delay[i_d]

    concatenation_late = []
    concatenation_early = []
    concatenation_licking_late = []
    concatenation_licking_early = []

    prev_session = -1
    prev_animal = -1

    for i_neuron, neuron in enumerate(selected_neuron_ids):

        dataframe_neuron = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron]
        animal = dataframe_neuron['Animal'].values[0]
        session = dataframe_neuron['Session'].values[0]

        trials_delay = reduce(np.intersect1d, (
        np.where(dataframe_neuron['Distribution reward ID'] == 0)[0], np.where(dataframe_neuron['Is rewarded'] == 1)[0],
        np.where(dataframe_neuron['Delay reward'] == d)[0]))
        trial_switch = np.where(dataframe_neuron['Is time manipulation trial'] == 1)[0][0]
        trials_delay = trials_delay[trials_delay < trial_switch]

        responses_delay = dataframe_neuron['Responses cue'].values[trials_delay]
        dataframe_licking_session = dataframe_licking[
            np.logical_and(dataframe_licking['Session'] == session, dataframe_licking['Animal'] == animal)]
        slopes_licking = dataframe_licking_session['Slopes licking aligned to cue'].values[trials_delay]
        psth_licking = dataframe_licking_session['Licking PSTH aligned to cue'].values[trials_delay]

        # Compute median slopes
        median_slope = np.nanquantile(slopes_licking, 0.5)
        early_slope = np.where(slopes_licking > median_slope)[0]
        late_slope = np.where(slopes_licking < median_slope)[0]

        # Create sudo trials for late and early licking trials
        concatenation_late.append(np.nanmean(responses_delay[late_slope] / all_estimates_gain[i_neuron]))
        concatenation_early.append(np.nanmean(responses_delay[early_slope] / all_estimates_gain[i_neuron]))

        if session != prev_session or animal != prev_animal:
            concatenation_licking_late.append(psth_licking[late_slope])
            concatenation_licking_early.append(psth_licking[early_slope])
            prev_session = session
            prev_animal = animal

    concatenation_late = np.array(concatenation_late)
    concatenation_early = np.array(concatenation_early)
    concatenation_licking_early = np.concatenate(concatenation_licking_early, axis=0)
    concatenation_licking_early = np.array(concatenation_licking_early.tolist(), dtype=float)
    concatenation_licking_late = np.concatenate(concatenation_licking_late, axis=0)
    concatenation_licking_late = np.array(concatenation_licking_late.tolist(), dtype=float)

    # Decoding future reward time
    prob_time_late = np.zeros(len(time_decoder))
    prob_time_early = np.zeros(len(time_decoder))
    pos_non_nan_late = np.argwhere(~np.isnan(concatenation_late))
    pos_non_nan_early = np.argwhere(~np.isnan(concatenation_early))
    for i in range(L):
        prob_time_late += (s[i] ** 2 / (s[i] ** 2 + alpha ** 2)) * np.dot(U[pos_non_nan_late, i][:, 0],
                                                                          concatenation_late[pos_non_nan_late][:,
                                                                          0]) * vh[i, :] / s[i]
        prob_time_early += (s[i] ** 2 / (s[i] ** 2 + alpha ** 2)) * np.dot(U[pos_non_nan_early, i][:, 0],
                                                                           concatenation_early[pos_non_nan_early][:,
                                                                           0]) * vh[i, :] / s[i]

    decoded_late = prob_time_late
    decoded_early = prob_time_early
    decoded_early[decoded_early < 0] = 0
    decoded_late[decoded_late < 0] = 0
    decoded_late = decoded_late / np.sum(decoded_late)
    decoded_early = decoded_early / np.sum(decoded_early)

    # Mean licking PSTHs
    mean_licking_early = np.nanmedian(concatenation_licking_early, axis=0)
    mean_licking_late = np.nanmedian(concatenation_licking_late, axis=0)
    sem_licking_early = scipy.stats.sem(concatenation_licking_early.astype(float), axis=0, nan_policy='omit').astype(
        float)
    sem_licking_late = scipy.stats.sem(concatenation_licking_late.astype(float), axis=0, nan_policy='omit').astype(
        float)

    ax[0, i_d - 1].plot(bins_licking[1:], mean_licking_early, color=colors_delay_lead[2], label="early licking")
    ax[0, i_d - 1].fill_between(bins_licking[1:], mean_licking_early - sem_licking_early,
                                mean_licking_early + sem_licking_early, alpha=0.2, color=colors_delay_lead[2])
    ax[0, i_d - 1].plot(bins_licking[1:], mean_licking_late, color=colors_delay_lag[2], label="late licking")
    ax[0, i_d - 1].fill_between(bins_licking[1:], mean_licking_late - sem_licking_late,
                                mean_licking_late + sem_licking_late, alpha=0.2, color=colors_delay_lag[2])
    ax[0, i_d - 1].set_xticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
    ax[0, i_d - 1].set_yticks([0, 6], ["0", "6"])
    ax[0, i_d - 1].spines['left'].set_linewidth(linewidth)
    ax[0, i_d - 1].spines['bottom'].set_linewidth(linewidth)
    ax[0, i_d - 1].tick_params(width=linewidth, length=length_ticks)
    ax[0, i_d - 1].set_title("Reward time=" + delay_legend[i_d])
    ax[0, i_d - 1].plot([0, bins_licking[bins_delays[i_d]]], [-0.05, -0.05], linewidth=linewidth * 2, color="black")

    if i_d == 1:
        ax[0, i_d - 1].legend()

    mean_decoding_late_all[i_d, :] = decoded_late
    mean_decoding_early_all[i_d, :] = decoded_early
    mean_licking_late_all[i_d, :] = mean_licking_late
    mean_licking_early_all[i_d, :] = mean_licking_early

    ax[1, i_d - 1].plot(time_decoder, decoded_early, color=colors_delay_lead[2], label="early licking")
    ax[1, i_d - 1].plot(time_decoder, decoded_late, color=colors_delay_lag[2], label="late licking")

    ax[0, i_d - 1].set_xticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
    ax[1, i_d - 1].set_xticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
    ax[1, i_d - 1].set_yticks([0, 0.08], ["0", "0.08"])
    ax[1, i_d - 1].set_ylim(-0.001, 0.08)
    ax[1, i_d - 1].set_xlabel("Time since cue (s)")
    ax[1, i_d - 1].spines['left'].set_linewidth(linewidth)
    ax[1, i_d - 1].spines['bottom'].set_linewidth(linewidth)
    ax[1, i_d - 1].tick_params(width=linewidth, length=length_ticks)

    median_late = np.where(np.cumsum(prob_time_late) >= 0.5)[0][0]
    median_early = np.where(np.cumsum(prob_time_early) >= 0.5)[0][0]

    max_late = np.argmax(prob_time_late)
    max_early = np.argmax(prob_time_early)

    mean_late_slope = np.sum(prob_time_late * time_decoder)
    mean_early_slope = np.sum(prob_time_early * time_decoder)

    if i_d != 3:
        ax[1, i_d - 1].axvline(x=time_decoder[max_late], ymin=0, ymax=decoded_late[max_late] / 0.08,
                               color=colors_delay_lag[2], ls="--")
        ax[1, i_d - 1].axvline(x=time_decoder[max_early], ymin=0, ymax=decoded_early[max_early] / 0.08,
                               color=colors_delay_lead[2], ls="--")

ax[0, 0].set_ylabel("Lick rate (licks/s)")
ax[1, 0].set_ylabel("Decoded density")
plt.show()

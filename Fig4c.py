import pdb
from aux_functions import *
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

mpl.use('TkAgg')

# Plot parameters
# length_ticks=10
# font_size=30
# linewidth=4
# scatter_size=500

length_ticks = 2
linewidth = 1.2
scatter_size = 4
font_size = 11

labelpad_x = 10
labelpad_y = -10
labelsize = font_size
legendsize = font_size


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size-3
mpl.rcParams['ytick.labelsize'] = font_size-3
mpl.rcParams['lines.linewidth'] = linewidth

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"
directory_raw_data = os.path.join(directory, "Raw")

# Get pupil diameter PSTH
data_frame_pupil_diameter = pd.read_csv(os.path.join(directory_raw_data, "Pupil_diameter.csv"))
data_frame_pupil_diameter.Animal = data_frame_pupil_diameter.Animal.astype(str)
data_frame_pupil_diameter.Session = data_frame_pupil_diameter.Session.astype(str)
data_frame_pupil_diameter["PSTH_pupil_diameter_aligned_to_cue"] = data_frame_pupil_diameter.loc[:,
                                                              'PSTH pupil diameter bin 0':'PSTH pupil diameter bin 1559'].values.tolist()
animals = np.unique(data_frame_pupil_diameter['Animal'])
data_frame_pupil_diameter.Animal = data_frame_pupil_diameter.Animal.astype(str)
data_frame_pupil_diameter.Session = data_frame_pupil_diameter.Session.astype(str)

n_quantiles = 3
taus_quantiles = np.arange(0, n_quantiles + 1) / n_quantiles
cmap = mpl.cm.get_cmap('cool', 12)
colors = cmap(np.arange(0, 1, 1.0 / n_quantiles))

FPS = 120
time_left = -2
time_right = 11
n_frames_pupil = (time_right - time_left) * FPS
time_pupil = np.arange(time_left, time_right, (time_right - time_left) / n_frames_pupil)
bin_0 = np.where(time_pupil >= 0)[0][0]
bin_1 = np.where(time_pupil >= 1)[0][0]
bin_1_5 = np.where(time_pupil >= 1.5)[0][0]

bin_3 = np.where(time_pupil >= 3)[0][0]
bin_10 = np.where(time_pupil >= 10)[0][0]

kerneltype = "exponential"

psth_pupil = np.empty((n_quantiles, 6, n_frames_pupil))
psth_pupil[:, :, :] = np.nan

psth_all_animals = np.empty((6, n_quantiles, n_frames_pupil))
psth_all_animals[:, :] = np.nan

# Figure with pupil diameter PSTH for each mouse
horizontal_size = 1
vertical_size = 4.3#vertical_size * 2.75
fig_psth_all_animals, ax_psth_all_animals = plt.subplots(figsize=(horizontal_size, vertical_size), nrows=6,
                                                         ncols=1, sharex=True)
ax_psth_all_animals[0].set_ylabel("Pupil diameter change (" + r"$\frac{\Delta P}{P_0}$" + ")")
ax_psth_all_animals[0].set_yticks([0, 0.05, 0.1])

# Reward history vs mean pupil diameter plot
fig_summary, ax_summary = plt.subplots(figsize=(horizontal_size * 0.66, vertical_size * 0.66), nrows=1, ncols=1)
ax_summary.set_xticks([1, 4.5, 8], ["1", "4.5", "8"], fontsize=int(font_size * 0.7))
ax_summary.set_yticks([0, 0.1], ["0", "0.1"], fontsize=int(font_size * 0.7))
ax_summary.set_xlabel("Reward history (" + r"$\mu$" + "l)", fontsize=int(font_size * 0.8))
ax_summary.set_ylabel("Pupil  diameter\nchange (" + r"$\frac{\Delta P}{P_0}$" + ")", fontsize=int(font_size * 0.8))
ax_summary.spines['left'].set_linewidth(linewidth)
ax_summary.spines['bottom'].set_linewidth(linewidth)
label_quantile = ["1st", "2nd", "3rd"]

sum_max_psth = 0
ticks_axis_position = [0]
ticks_axis_value = [0]



for i_animal, animal in enumerate(["4096", "4418", "4099", "3353", "4098", "4140"]):


    # Get individual animal time-scale to integrate reward
    m = data_frame_pupil_diameter[data_frame_pupil_diameter['Animal'] == animal]['Time scale (trials)'].values[0]

    prev_session = -1
    i_session = 0

    psth_pupil_animal = np.empty((10000, n_frames_pupil))
    psth_pupil_animal[:, :] = np.nan

    reward_history_animal = np.empty(10000)
    reward_history_animal[:] = np.nan

    counts_trials = 0

    ax_psth_all_animals[i_animal].axvline(x=0, color="k", ls="--")
    #ax_psth_all_animals[i_animal].axvline(x=3, color="k", ls="--")
    ax_psth_all_animals[i_animal].tick_params(width=linewidth, length=length_ticks)
    ax_psth_all_animals[i_animal].spines['left'].set_linewidth(linewidth)
    ax_psth_all_animals[i_animal].spines['bottom'].set_linewidth(linewidth)

    if i_animal == 5:
        ax_psth_all_animals[i_animal].set_xlabel("Time since cue (s)")

    if i_animal < 5:
        ax_psth_all_animals[i_animal].spines['bottom'].set_linewidth(0)
        ax_psth_all_animals[i_animal].tick_params(axis='x', width=0, length=0)

    # Go through sessions
    for date in np.unique(data_frame_pupil_diameter[(data_frame_pupil_diameter.Animal == animal)]["Session"].values):

        if len(data_frame_pupil_diameter[
                   (data_frame_pupil_diameter.Animal == animal) & (data_frame_pupil_diameter.Session == date)][
                   "PSTH_pupil_diameter_aligned_to_cue"].values) == 0:
            continue

        data_frame_pupil_diameter_session = data_frame_pupil_diameter[
            (data_frame_pupil_diameter.Animal == animal) & (data_frame_pupil_diameter.Session == date)]

        # Variable reward magnitude trials
        variable_amount_trials = np.intersect1d(np.where(data_frame_pupil_diameter_session["Is rewarded"] == 1)[0],
                                                np.where(data_frame_pupil_diameter_session["Distribution reward ID"] == 2)[
                                                    0])

        reward_amounts = data_frame_pupil_diameter_session["Amount reward"].values
        reward_amounts = reward_amounts[variable_amount_trials]
        unique_reward_amounts = np.unique(reward_amounts)

        # Reward history
        moving_average_amounts = moving_average(reward_amounts, m, 1, kerneltype)
        psth_pupil_session = data_frame_pupil_diameter_session["PSTH_pupil_diameter_aligned_to_cue"].tolist()
        psth_pupil_session = np.array(psth_pupil_session)

        max_trial_number = len(variable_amount_trials) - 1  # Take out last trial

        # Use reward history at the previous trial, and look at the pupil area at current trial
        psth_pupil_animal[counts_trials:counts_trials + max_trial_number, :] = psth_pupil_session[
                                                                               variable_amount_trials[1:], :]
        reward_history_animal[counts_trials:counts_trials + max_trial_number] = moving_average_amounts[:-1]

        counts_trials += max_trial_number
        i_session += 1

    # Compute the peak pupil diameter using the half-peak window
    mean_over_all = np.nanmean(psth_pupil_animal[:, :], axis=0)
    peak = np.argmax(mean_over_all[bin_3:]) + bin_3
    half_peak = np.where(np.round(mean_over_all, 3) == np.round(mean_over_all[peak] * 0.5, 3))[0]
    dif = (half_peak - peak)
    order = np.argsort(dif)
    window = [np.min(half_peak), np.max(half_peak)]
    number_trials = np.sum(~np.isnan(reward_history_animal))

    mean_pupil_window = np.nanmean(psth_pupil_animal[:number_trials, window[0]:window[1]], axis=1)
    reward_history_animal = reward_history_animal[:number_trials]

    # Plot pupil diameter PSTH change conditioned on the reward history for each animal
    quantiles = np.quantile(reward_history_animal, taus_quantiles)
    max_quantile_animal = -1
    for i_quantile, quantile in enumerate(quantiles[:-1]):
        trials_quantile = reduce(np.intersect1d, (np.where(reward_history_animal >= quantiles[i_quantile])[0],
                                                  np.where(reward_history_animal <= quantiles[i_quantile + 1])[0]))
        trials_quantile = trials_quantile[trials_quantile < len(reward_history_animal)]

        median = np.nanmedian(mean_pupil_window[trials_quantile])
        quant = np.quantile(mean_pupil_window[trials_quantile], [0.4, 0.6])
        asym_error = [[median - quant[0]], [quant[1] - median]]
        quantiles_summary_plot = np.quantile(reward_history_animal,
                                             (taus_quantiles[i_quantile] + taus_quantiles[i_quantile + 1]) / 2)
        ax_summary.errorbar(quantile, median, yerr=asym_error, capsize=0, fmt="o", color=colors[i_quantile],
                            markersize=5, elinewidth=1.2)  # scatter_size-10

        mean = np.nanmean(psth_pupil_animal[trials_quantile, :], axis=0)
        sem = scipy.stats.sem(psth_pupil_animal[trials_quantile, :].astype(float), axis=0, nan_policy='omit').astype(
            float)
        psth_all_animals[i_animal, i_quantile, :] = mean

        ax_psth_all_animals[i_animal].plot(time_pupil, mean, color=colors[i_quantile], label=str(i_quantile),
                                           lw=linewidth * 0.5)
        ax_psth_all_animals[i_animal].fill_between(time_pupil, mean - sem, mean + sem, alpha=0.2,
                                                   color=colors[i_quantile])

        max_quantile_animal = np.maximum(max_quantile_animal, np.max(mean))
        psth_all_animals[i_animal, i_quantile, :] = mean

    sum_max_psth += max_quantile_animal
    ticks_axis_position.append(sum_max_psth)
    ticks_axis_value.append(max_quantile_animal)

    ax_psth_all_animals[i_animal].set_ylim(-0.009, max_quantile_animal + 0.005)
    ax_psth_all_animals[i_animal].set_yticks([0, max_quantile_animal], ["0", np.round(max_quantile_animal, 2)])

    # Per animal regression of reward history and mean pupil diameter
    ax_psth_all_animals[i_animal].plot([time_pupil[window[0]], time_pupil[window[1]]], [0, 0], color="k", lw=2)
    reg = LinearRegression().fit((reward_history_animal).reshape(-1, 1), mean_pupil_window)
    x_example = np.linspace(quantiles[0], quantiles[-1], 10)
    y_example = reg.predict(x_example.reshape(-1, 1))
    ax_summary.plot(x_example, y_example, color="k")

    #ax_psth_all_animals[i_animal].set_title(animal)

#fig_psth_all_animals.savefig("psth_all_animals.svg")
plt.show()

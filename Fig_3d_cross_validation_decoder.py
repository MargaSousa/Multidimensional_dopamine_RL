import matplotlib as mpl
from aux_functions import *
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pdb
import pandas
import os

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import pandas as pd

mpl.use('TkAgg')

# Parameters for paper plots
length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 1.5  # 2.2
vertical_size = 1.5  # 2.2
# horizontal_size=5
# vertical_size=5
font_size = 11
labelsize = 8
legendsize = font_size

# Parameters for presentation plots
# length_ticks=10
# font_size=30
# linewidth=4
# scatter_size=500
# horizontal_size=10
# vertical_size=10
# labelpad_x=10
# labelpad_y=-10
# labelsize=font_size
# legendsize=font_size


mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rc('xtick', labelsize=labelsize)
mpl.rc('ytick', labelsize=labelsize)
mpl.rc('legend', fontsize=legendsize)

# Colors for plots
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get estimated tuning for reward time
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))

# Get responses aligned to cue delivery
responses_cue = np.load(os.path.join(directory_parsed_data, "responses_cue_different_delays_constant_magnitude.npy"))

# Get estimated tuning for reward time
all_estimates_gamma = data_frame_neurons_info['Gamma'].values
all_estimates_gain = data_frame_neurons_info['Gain'].values
n_neurons = len(all_estimates_gamma)

# Mean responses across delays for individual neurons
mean_responses_per_delay = np.nanmean(responses_cue, axis=2)  # n_neurons * delays
mean_responses_per_delay = mean_responses_per_delay / all_estimates_gain[:, None]  # Normalize by gain
n_neurons_less_delays = np.sum(np.isnan(mean_responses_per_delay))

# Discretize time
n_time = 50  # number bins for time #30
time = np.linspace(0, 7.5, n_time)

# Inverse La Place decoder for animal 3353 neurons (wasn't exposed to delay=0s)
n_neurons = mean_responses_per_delay.shape[0]
F = np.zeros((n_neurons, n_time))
for i_t, t in enumerate(time):
    F[:, i_t] = (all_estimates_gamma ** t)

# Inverse La Place for reminding animals
n_neurons_new = n_neurons - n_neurons_less_delays
F_new = np.zeros((n_neurons_new, n_time))
for i_t, t in enumerate(time):
    F_new[:, i_t] = (all_estimates_gamma[n_neurons_less_delays:] ** t)

# Smoothing parameter for decoding
alpha = 1
# alpha=0.1 # for animal 3353
# alpha=0.5 # for animal 4098

# Cross validate decoding
n_resamples = 10

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
mean_prob_time = np.zeros((4, n_time))
runs_prob_time = np.zeros((4, n_resamples, n_time))

delays = [0, 1.5, 3, 6]
label_delays = ["0s", "1.5s", "3s", "6s"]
for r in range(n_resamples):
    mean_responses = np.zeros((n_neurons, 4))
    variance_responses = np.zeros((n_neurons, 4))
    for i_d in range(4):
        for neu in range(n_neurons):
            n_trials = np.sum(~np.isnan(responses_cue[neu, i_d, :]))
            n_trials_resample = int(n_trials * 0.7)
            if n_trials_resample > 0:
                resample_responses = resample(responses_cue[neu, i_d, :n_trials], replace=False,
                                              n_samples=n_trials_resample)
                mean_responses[neu, i_d] = np.nanmean(resample_responses) / all_estimates_gain[neu]
                variance_responses[neu, i_d] = np.nanvar(resample_responses)

        if i_d == 0:
            inverse_variance = 1.0 / variance_responses[n_neurons_less_delays:, i_d]
            F_resample = F_new * inverse_variance[:, None]
            F_resample = F_resample
            U_new, s_new, vh_new = np.linalg.svd(F_resample, full_matrices=True)
            L_new = np.min([vh_new.shape[0], np.shape(U_new)[1]])
            prob_time = np.zeros(n_time)
            for i in range(L_new):
                prob_time += (s_new[i] ** 2 / (s_new[i] ** 2 + alpha ** 2)) * np.dot(U_new[:, i], mean_responses[
                                                                                                  n_neurons_less_delays:,
                                                                                                  i_d] * inverse_variance) * vh_new[
                                                                                                                             i,
                                                                                                                             :] / \
                             s_new[i]

            prob_time[prob_time < 0] = 0
            prob_time = prob_time / np.sum(prob_time)
            ax.plot(time, prob_time, color=colors_delay[0], linewidth=linewidth * 0.1)
            mean_prob_time[i_d, :] += prob_time


        else:
            inverse_variance = 1.0 / variance_responses[:, i_d]
            inverse_variance = inverse_variance
            F_d = F * inverse_variance[:, None]
            U, s, vh = np.linalg.svd(F_d, full_matrices=True)
            L = np.min([vh.shape[0], np.shape(U)[1]])
            prob_time = np.zeros(n_time)
            for i in range(L):
                prob_time += (s[i] ** 2 / (s[i] ** 2 + alpha ** 2)) * np.dot(U[:, i], mean_responses[:,
                                                                                      i_d] * inverse_variance) * vh[i,
                                                                                                                 :] / s[
                                 i]

            prob_time[prob_time < 0] = 0
            prob_time = prob_time / np.sum(prob_time)
            ax.plot(time, prob_time, color=colors_delay[i_d], linewidth=linewidth * 0.1)
            mean_prob_time[i_d, :] += prob_time

        runs_prob_time[i_d, r, :] = prob_time

# Mean decoded density for each delay
mean_prob_time = mean_prob_time / n_resamples
norm = np.sum(mean_prob_time, axis=1)
mean_prob_time = mean_prob_time / norm[:, None]
sum_prob = mean_prob_time[0, :] * 0

for i_d in range(4):
    ax.plot(time, mean_prob_time[i_d, :], color=colors_delay[i_d], label=label_delays[i_d])
    ax.axvline(x=delays[i_d], color=colors_delay[i_d], linestyle="--")

plt.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.15, 1))
plt.xlabel("Time since cue (s)")
plt.ylabel("Decoded density")
plt.xticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
plt.yticks([0, 0.1], ["0", "0.1"])
plt.show()




pdb.set_trace()






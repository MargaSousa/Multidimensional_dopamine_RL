import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

mpl.use('TkAgg')
from aux_functions import *
import pandas as pd
from sklearn.utils import resample
import os

# Parameters for paper plots
length_ticks = 2
linewidth = 1.2
scatter_size = 10
horizontal_size = 1.8
vertical_size = 1.8
font_size = 11
labelsize = 8
legendsize = font_size

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rc('xtick', labelsize=labelsize)
mpl.rc('ytick', labelsize=labelsize)
mpl.rc('legend', fontsize=legendsize)

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "Putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get estimated tuning for each neuron
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))

# Select neurons from given animals
selected_animal = 3353
selected_neurons = data_frame_neurons_info.index[data_frame_neurons_info.Animal == selected_animal].tolist()

# Choose instead neurons from all animals
selected_neurons = np.arange(len(data_frame_neurons_info))

# Get estimated tuning for reward time and magnitude
discount = data_frame_neurons_info['Gamma'].values[selected_neurons]
estimated_reversals = data_frame_neurons_info['Reversals'].values[selected_neurons]
estimated_taus = data_frame_neurons_info['Taus'].values[selected_neurons]
gains = data_frame_neurons_info['Gain'].values[selected_neurons]
n_neurons = len(discount)
bias_taus = data_frame_neurons_info['Bias in the estimation of tau'].values[
    selected_neurons]  # Generated from Fig4_get_bias_variance_estimation_tau
variance_tau = data_frame_neurons_info['Variance in the estimation of tau'].values[
    selected_neurons]  # Generated from Fig4_get_bias_variance_estimation_tau
animals = data_frame_neurons_info['Animal'].values[selected_neurons]

responses_cue_second_phase = np.load(
    os.path.join(directory_parsed_data, "responses_cue_different_delays_constant_magnitude.npy"))[selected_neurons, :,
                             :]
responses_cue_bimodal_second_phase = np.loadtxt(
    os.path.join(directory_parsed_data, "responses_cue_different_magnitudes_constant_delay.csv"))[selected_neurons, :]

directory_parsed_data_first_phase = os.path.join(directory, "Parsed_data_" + type_neurons + "_first_phase")
responses_cue_first_phase = np.load(
    os.path.join(directory_parsed_data_first_phase, "responses_cue_different_delays_constant_magnitude.npy"))[
                            selected_neurons, :, :]
responses_cue_bimodal_first_phase = np.loadtxt(
    os.path.join(directory_parsed_data_first_phase, "responses_cue_different_magnitudes_constant_delay.csv"))[
                                    selected_neurons, :]

# Regression between gain in first and second phase (we don't have the responses for the 0s for some neurons)
gains_first = np.nanmean(responses_cue_first_phase[:, 0, :], axis=1)
gains_second = gains
non_nan_positions = np.argwhere(~np.isnan(gains_first))
reg = LinearRegression(fit_intercept=1).fit(gains[non_nan_positions].reshape(-1, 1),
                                            gains_first[non_nan_positions])  # fit_intercept=0
gain_first_fit = reg.predict(gains.reshape(-1, 1))[:, 0]

# Colors for plots
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))[:, :3]

# Discretize time
n_time = 50  # number bins for time
time = np.linspace(0, 7.5, n_time)

# Smoothing parameter for decoding
alpha = 0.1

# Cross validate decoding
n_resamples = 10
fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)

mean_prob_time = np.zeros((10, n_time))

# For saving results
label_cues = ["0s", "1.5s cue", "3s cue", "6s cue", "Variable 3s cue"]
delays = [0, 1.5, 3, 6]

for r in range(n_resamples):
    mean_responses = np.zeros((n_neurons, 10))

    for i_d in range(1, 7):

        # Compute variance over all of responses
        if i_d == 1:
            variance = []

        for neu in range(n_neurons):

            if i_d == 1:
                all_trials = np.concatenate((np.ndarray.flatten(responses_cue_bimodal_first_phase[neu, :]),
                                             np.ndarray.flatten(responses_cue_first_phase[neu, :, :])))
                n_trials = np.sum(~np.isnan(all_trials))
                variance.append(np.var(all_trials[~np.isnan(all_trials)]))

            if i_d <= 3:
                # Decode time for certain magnitude cues
                n_trials = np.sum(~np.isnan(responses_cue_first_phase[neu, i_d, :]))
                n_trials_resample = int(n_trials * 0.7)
                resample_responses = resample(responses_cue_first_phase[neu, i_d, :n_trials], replace=False,
                                              n_samples=n_trials_resample)
                mean_responses[neu, i_d] = np.nanmean(resample_responses) / gain_first_fit[neu]

            if i_d == 4:
                # Decode time for variable magnitude cue
                n_trials = np.sum(~np.isnan(responses_cue_bimodal_first_phase[neu, :]))
                n_trials_resample = int(n_trials * 0.7)
                resample_responses = resample(responses_cue_bimodal_first_phase[neu, :n_trials], replace=False,
                                              n_samples=n_trials_resample)
                mean_responses[neu, i_d] = np.nanmean(resample_responses) / gain_first_fit[neu]

            if i_d >= 5:
                # Decode time for mean over all cues
                all_trials = np.concatenate((np.ndarray.flatten(responses_cue_bimodal_first_phase[neu, :]),
                                             np.ndarray.flatten(responses_cue_first_phase[neu, :, :])))
                n_trials = np.sum(~np.isnan(all_trials))
                n_trials_resample = int(n_trials * 0.7)
                resample_responses = resample(all_trials[:n_trials], replace=False, n_samples=n_trials_resample)
                mean_responses[neu, i_d] = np.nanmean(resample_responses) / gain_first_fit[neu]

        if i_d == 1:
            variance = np.array(variance)

        # Decode time
        if i_d <= 5:
            pdf_time = run_decoding_time(time, discount, variance, mean_responses[:, i_d], alpha)


        else:
            # Shuffle tuning towards reward time
            neurons_shuffled = np.arange(len(discount))
            np.random.shuffle(neurons_shuffled)
            pdf_time = run_decoding_time(time, discount[neurons_shuffled], variance, mean_responses[:, i_d], alpha)

        mean_prob_time[i_d, :] += pdf_time

        if i_d <= 3:
            plt.plot(time, pdf_time, color=colors_delay[i_d], lw=linewidth * 0.1)
        if i_d == 4:
            plt.plot(time, pdf_time, color="blue", lw=linewidth * 0.1)
        if i_d == 6:
            plt.plot(time, pdf_time, color="gray", lw=linewidth * 0.1)

# Mean decoded density for each delay
mean_prob_time = mean_prob_time / n_resamples
norm = np.sum(mean_prob_time, axis=1)
mean_prob_time = mean_prob_time / norm[:, None]
labels = ["0s", "1.5s", "3s", "6s", "Variable 3s", "All", "Shuffled"]
for i_d in range(7):

    if i_d <= 3:
        ax.plot(time, mean_prob_time[i_d, :], color=colors_delay[i_d])  # ,label=labels[i_d]
    if i_d == 4:
        ax.plot(time, mean_prob_time[i_d, :], color="blue", label=labels[i_d])
    if i_d == 6:
        ax.plot(time, mean_prob_time[i_d, :], color="gray", label=labels[i_d])

plt.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.15, 1))
plt.xlabel("Time since cue (s)")  # ,labelpad=labelpad_x
plt.ylabel("Decoded density")  # ,labelpad=labelpad_y
plt.xticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
plt.yticks([0, 0.1], ["0", "0.1"])

plt.show()
pdb.set_trace()

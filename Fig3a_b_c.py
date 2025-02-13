import pdb
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from aux_functions import *
import cmasher as cmr
import os
from sklearn.utils import resample
from scipy.stats import pearsonr
import pandas as pd

# # Parameters for paper plots
length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 2.2
vertical_size = 2.2
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
mpl.use('TkAgg')

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get estimated tuning for reward time
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))

# Get estimated tuning for reward time
all_estimates_gamma = data_frame_neurons_info['Gamma'].values
all_estimates_gain = data_frame_neurons_info['Gain'].values
animals = data_frame_neurons_info['Animal'].values

# Get responses at the cue
responses_cue = np.load(os.path.join(directory_parsed_data, "responses_cue_different_delays_constant_magnitude.npy"))
mean_responses_cue = np.nanmean(responses_cue, axis=2)  # mean responses aligned to cue for each neuron
idx_sorted_gammas = np.argsort(all_estimates_gamma)[::-1]
sorted_gammas = all_estimates_gamma[idx_sorted_gammas]
n_neurons = len(sorted_gammas)

# Plot normalized firing rate at cue
fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
ax.set_box_aspect(1)
discritization = (sorted_gammas - sorted_gammas[0]) / (sorted_gammas[-1] - sorted_gammas[0]) * 0.8
cmap = cmr.cm.apple(discritization)
delays_example = np.linspace(0, 6, 100)
count = 0
for neu in idx_sorted_gammas:

    n_delays = np.sum(~np.isnan(mean_responses_cue[neu, :]))

    # Plot estimated temporal discount function
    if neu != idx_sorted_gammas[0]:  # Inverted neuron
        plt.plot(delays_example, (all_estimates_gamma[neu] ** delays_example), c=cmap[-count - 1])
    else:
        plt.plot(delays_example,
                 (all_estimates_gamma[neu] ** delays_example) * all_estimates_gain[neu] / mean_responses_cue[neu, 0],
                 c=cmap[-count - 1])

    # Plot mean responses
    if n_delays == 3:
        mean_responses_norm = mean_responses_cue[neu, 1:4] / all_estimates_gain[neu]
        delays_unique = np.array([1.5, 3, 6])
        plt.scatter(delays_unique, mean_responses_norm, s=scatter_size, color=cmap[-count - 1], zorder=1000)

    else:
        mean_responses_norm = mean_responses_cue[neu, :] / mean_responses_cue[neu, 0]
        delays_unique = np.array([0, 1.5, 3, 6])
        plt.scatter(delays_unique, mean_responses_norm, s=scatter_size, color=cmap[-count - 1], zorder=1000)
    count += 1

# Mean over neurons temporal discount function
mean_responses_all = np.nanmean(mean_responses_cue, axis=0)
popt, pcov_before = curve_fit(exponential, delays_unique, mean_responses_all / mean_responses_all[0], gtol=1e-20,
                              maxfev=1000000)
plt.plot(delays_example, np.exp(-popt[-1]) ** delays_example, color="royalblue",
         zorder=2000)  # Plot mean responses for the population of DANs

plt.ylabel("Normalized firing rate at cue")
plt.xlabel(r"Reward time (s)")
plt.xticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
plt.yticks([1], ["1"])
plt.show()

# Variability: estimate temporal discount factors in different sets of trials (cross-validation)
n_resamples = 10000
median_neurons = []
quantile_lower = []
quantile_upper = []
level_lower = 0.005
lever_upper = 0.995
delays = np.array([0, 1.5, 3, 6])
for i_neuron in range(n_neurons):
    all_estimates_gammas_resamples_neuron = []

    for r in range(n_resamples):
        delays_neuron = delays[~np.isnan(np.nanmean(responses_cue[i_neuron, :], axis=1))]
        pos_delays_neuron = np.arange(len(delays))[~np.isnan(np.nanmean(responses_cue[i_neuron, :], axis=1))]
        delta_fr_resample = []
        for pos_delay in pos_delays_neuron:
            n_trials = int(np.sum(~np.isnan(responses_cue[i_neuron, pos_delay, :])))
            new_responses = resample(responses_cue[i_neuron, pos_delay, :n_trials], replace=False,
                                     n_samples=int(0.5 * n_trials))  # Take randomly 50% of the trials of each delay
            delta_fr = np.nanmean(new_responses)
            delta_fr_resample.append(delta_fr)

        # Estimate temporal discount factor
        popt, pcov = curve_fit(exponential, delays_neuron, delta_fr_resample, gtol=1e-20,
                               maxfev=1000000)  # Estimate gamma
        estimated_gamma_resample = np.exp(-popt[-1])
        all_estimates_gammas_resamples_neuron.append(estimated_gamma_resample)

    all_estimates_gammas_resamples_neuron = np.array(all_estimates_gammas_resamples_neuron)
    median_neurons.append(np.nanmedian(all_estimates_gammas_resamples_neuron))
    quantile_lower.append(np.quantile(all_estimates_gammas_resamples_neuron, level_lower))
    quantile_upper.append(np.quantile(all_estimates_gammas_resamples_neuron, lever_upper))

# Plot cross validated temporal discounts
fig, ax = plt.subplots(1, 1, figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
for i_neuron, neuron in enumerate(np.argsort(median_neurons)):
    asym_error = [[median_neurons[neuron] - quantile_lower[neuron]], [quantile_upper[neuron] - median_neurons[neuron]]]
    ax.errorbar(i_neuron + 1, median_neurons[neuron], yerr=asym_error, capsize=0.4, elinewidth=linewidth * 0.5,
                color=cmap[i_neuron])
    plt.scatter(i_neuron + 1, median_neurons[neuron], color=cmap[i_neuron], s=0.1)
ax.set_xlabel("Neurons", labelpad=0)
ax.set_ylabel(r"$\gamma$", labelpad=0)
ax.set_xticks([1, n_neurons], ["1", str(n_neurons)])
ax.set_yticks([0.6, 1], ["0.6", "1"])
mean_neurons = np.mean(median_neurons)
ax.axhline(y=mean_neurons, color="royalblue", ls="--")
ax.axhline(y=1, color="gray", ls="--")
plt.show()

# Consistency: estimate temporal discount factors in disjoint sets of trials
samples_all_neurons = np.zeros((n_resamples, n_neurons, 2))
for i_neuron in range(n_neurons):
    all_estimates_gammas_resamples_neuron = []
    for r in range(n_resamples):
        delays_neuron = delays[~np.isnan(np.nanmean(responses_cue[i_neuron, :], axis=1))]
        pos_delays_neuron = np.arange(len(delays))[~np.isnan(np.nanmean(responses_cue[i_neuron, :], axis=1))]

        delta_fr_first_half = []
        delta_fr_second_half = []

        for pos_delay in pos_delays_neuron:
            n_trials = int(np.sum(~np.isnan(responses_cue[i_neuron, pos_delay, :])))
            trials_delay = np.arange(n_trials)
            np.random.shuffle(trials_delay)

            delta_fr_first_half.append(
                np.nanmean(responses_cue[i_neuron, pos_delay, trials_delay[:int(n_trials * 0.5)]]))
            delta_fr_second_half.append(
                np.nanmean(responses_cue[i_neuron, pos_delay, trials_delay[int(n_trials * 0.5):]]))

        popt_first_half, pcov_first_half = curve_fit(exponential, delays_neuron, delta_fr_first_half, gtol=1e-20,
                                                     maxfev=1000000)
        popt_second_half, pcov_second_half = curve_fit(exponential, delays_neuron, delta_fr_second_half, gtol=1e-20,
                                                       maxfev=1000000)
        estimated_gamma_first_half = np.exp(-popt_first_half[-1])
        estimated_gamma_second_half = np.exp(-popt_second_half[-1])
        samples_all_neurons[r, i_neuron, 0] = estimated_gamma_first_half
        samples_all_neurons[r, i_neuron, 1] = estimated_gamma_second_half


# Compute pearson correlation between estimated temporal discount in disjoint sets of trials
r_values = []
p_values = []
slope_values = []
for r in range(n_resamples):
    res = pearsonr(samples_all_neurons[r, :, 0], samples_all_neurons[r, :, 1])
    r_values.append(res[0])
    p_values.append(res[1])
    reg = LinearRegression(fit_intercept=False).fit(np.expand_dims(samples_all_neurons[r, :, 0], axis=1),
                                                    samples_all_neurons[r, :, 1])
    slope_values.append(reg.coef_[0])

slope_values = np.array(slope_values)
r_values = np.array(r_values)
p_values = np.array(p_values)
log_geometric_mean = (1.0 / len(p_values)) * np.sum(np.log(p_values))
geometric_p_value = np.exp(log_geometric_mean)
mean_slope = np.nanmean(slope_values)
mean_r_values = np.nanmean(r_values)

# Regression for the plot
reg = LinearRegression(fit_intercept=False).fit(np.expand_dims(samples_all_neurons[0, :, 0], axis=1),
                                                samples_all_neurons[0, :, 1])
regression_x_plot = np.expand_dims(samples_all_neurons[0, :, 0], axis=1)
regression_y_plot = reg.predict(regression_x_plot)

# Order neurons with respect to temporal discount factor
samples_all_neurons = samples_all_neurons[:, idx_sorted_gammas[::-1], :]

# Plot consistency temporal discount factors
fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.scatter(samples_all_neurons[0, :, 0], samples_all_neurons[0, :, 1], c=cmap, s=scatter_size)
ax.plot(regression_x_plot, regression_y_plot, color="black", ls="--")
ax.set_xlabel(r"$\gamma$ one half of data", labelpad=0.00001)
ax.set_ylabel(r"$\gamma$ other half of data", labelpad=0.00001)
ax.set_xticks([0.6, 1.1], ["0.6", "1.1"])
ax.set_yticks([0.6, 1.1], ["0.6", "1.1"])
ax.set_box_aspect(1)
plt.text(0.65, 0.95, "r=" + str(np.round(mean_r_values, 2)) + "\n" + "p<1e-15", fontsize=font_size - 1)

# Plot histogram of regression slopes
axins = inset_axes(ax, width="40%", height="40%", loc='lower right', borderpad=3)
axins.spines['left'].set_linewidth(linewidth)
axins.spines['bottom'].set_linewidth(linewidth)
axins.hist(slope_values, color="lightgrey")
axins.tick_params(width=linewidth, length=length_ticks)
axins.set_xlabel("Regression \n slope", labelpad=0.00002)
axins.set_ylabel("Probability", labelpad=0.00001)
axins.set_xticks([mean_slope], [str(np.round(mean_slope, 2))], fontsize=int(font_size * 0.7))
axins.set_yticks([])
quantile_25 = np.quantile(slope_values, 0.025)
quantile_975 = np.quantile(slope_values, 0.975)
axins.set_xticks([quantile_25, mean_slope, quantile_975],
                 [str(np.round(quantile_25, 2)), "1", str(np.round(quantile_975, 2))], fontsize=int(font_size * 0.7))

plt.show()

pdb.set_trace()

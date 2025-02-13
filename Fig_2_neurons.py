import pdb
from aux_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import os

mpl.use('TkAgg')

# Parameters for plots
horizontal_size = 2.4
vertical_size = 2.4
length_ticks = 2
linewidth = 1.2
scatter_size = 4
font_size = 11
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth

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


# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
directory_parsed_data = os.path.join(directory, "Parsed_data_DA")
psth_reward = np.load(os.path.join(directory_parsed_data,
                                   "psth_reward.npy"))  # PSTH aligned to reward delivery for different amounts (neurons  x amounts x bins x trials)
psth_cue = np.load(os.path.join(directory_parsed_data,
                                "psth_cue.npy"))  # Liking PSTH aligned to cue delivery for different delays (neurons  x amounts x bins x trials)
responses_cue = np.load(os.path.join(directory_parsed_data,
                                     "responses_cue_different_delays_constant_magnitude.npy"))  # Responses aligned to cue delivery computed in the window 0.2-0.65s (neurons  x delays x trials)
responses_reward = np.load(os.path.join(directory_parsed_data,
                                        "responses_reward_different_magnitudes_constant_delay.npy"))  # Responses aligned to reward delivery computed in the window 0.2-0.65s (neurons  x amounts x trials)

# Tuning parameters for reward magnitude and time estimated for each neuron
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))
gammas = data_frame_neurons_info['Gamma'].values
gains = data_frame_neurons_info['Gain'].values
taus = data_frame_neurons_info['Taus'].values
reversal_points = data_frame_neurons_info['Reversals'].values
slopes_positive = data_frame_neurons_info['Positive slope'].values
slopes_negative = data_frame_neurons_info['Negative slope'].values

# Spike times data
directory_raw_data = os.path.join(directory, "Raw")
dataframe_spike_times = pd.read_csv(os.path.join(directory_raw_data, "Spike_times.csv"))

# Behavior information data
dataframe_behavior_times = pd.read_csv(os.path.join(directory_raw_data, "Neurons_behavior_trials.csv"))

# Plot colors
winter = mpl.cm.get_cmap('winter', 12)
colors_amount = winter(np.linspace(0, 1, 5))
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))
labels_delay = ["0", "1.5", "3", "6"]
labels_amount = ["1", "2.75", "4.5", "6.25", "8"]

# Select time range to plot
axes_correct = np.linspace(-5, 8, 650)
time_left = -0.5
time_right = 1.2
bin_left = np.where(axes_correct >= time_left)[0][0]
bin_right = np.where(axes_correct >= time_right)[0][0]
axes = axes_correct[bin_left:bin_right]

# Get 'Neuron id' for photo ided-neurons
photo_ided_neuron_ids = dataframe_behavior_times.loc[dataframe_behavior_times['Type of neuron'] == 'Photo_ided', 'Neuron id'].drop_duplicates().values


fig, ax = plt.subplots(2, 4, figsize=(horizontal_size * 4, vertical_size * 2), sharex='col', sharey='row')  #
fig.subplots_adjust(wspace=0.1, hspace=0.1)

# First and second columns: responses to different reward delays for two example neurons
idx_neurons_photo_ided = [19, 20]  # index of neurons
neuron_ids = photo_ided_neuron_ids[idx_neurons_photo_ided]
name_neurons = ["shortsighted", "longsighted"]

for i in range(2):

    neuron_id = neuron_ids[i]
    neuron_idx = idx_neurons_photo_ided[i]

    # Get spike times
    spike_times = dataframe_spike_times[dataframe_spike_times['Neuron id'] == neuron_id]['Spike times (s)'].values

    # Get give cue times for this session
    give_cue_times = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Give cue times'].values

    # Distribution reward ID: 0 is certain reward of 4.5uL, 2 is variable bimodal reward
    distribution_id_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Distribution reward ID'].values

    # Delay to reward for each trial
    delay_reward_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Delay reward'].values

    # If the mice was rewarded in each trial
    isRewarded_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Is rewarded'].values

    # If the trial was a time manipulation trial (one of the delays was removed), or not
    is_switch_trial = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Is time manipulation trial'].values
    trial_switch = np.where(is_switch_trial == 1)[0]

    # Get trial for which variable reward delays and certain reward amounts were given
    flag_delay_all = np.intersect1d(np.where(distribution_id_trials == 0)[0], np.where(isRewarded_trials == 1)[0])
    flag_delay_all = flag_delay_all[flag_delay_all < trial_switch]  # don't consider trials after time manipulation

    # Sort trials with respect to reward delays
    delays_filtered = delay_reward_trials[flag_delay_all]
    delays_unique = np.unique(delays_filtered)
    idx_sort_delay = np.argsort(delays_filtered)
    n_trials_delays = len(flag_delay_all)

    # Raster ordered by reward delay
    for i_trial, trial in enumerate(flag_delay_all[idx_sort_delay]):
        after = spike_times - give_cue_times[trial] > axes[0]
        before = spike_times - give_cue_times[trial] < axes[-1]
        pos = np.logical_and(after, before)
        spike_times_trial = spike_times[pos] - give_cue_times[trial]
        ax[0, i].scatter(spike_times_trial, i_trial * np.ones(len(spike_times_trial)), marker='.', s=scatter_size,
                         c="black")

    if i == 0:
        ax[0, i].set_ylabel("Trials", labelpad=0.1)

    ax[0, i].spines['left'].set_linewidth(linewidth)
    ax[0, i].spines['bottom'].set_linewidth(linewidth)
    ax[1, i].tick_params(width=linewidth, length=length_ticks)
    ax[0, i].axvline(x=0, ls="--", color="black")
    ax[0, i].set_yticks([0, n_trials_delays - 1], ["1", str(n_trials_delays)])
    ax[0, i].set_ylim(0, n_trials_delays)
    ax[0, i].set_box_aspect(1)
    ax[1, i].set_xlabel("Time since cue (s)")

    # Inset: Scatter of mean responses for trials where variable reward delays and certain reward amounts were given
    axins_delay = inset_axes(ax[1, i], width="40%", height="40%")
    axins_delay.spines['left'].set_linewidth(linewidth)
    axins_delay.spines['bottom'].set_linewidth(linewidth)
    axins_delay.tick_params(width=linewidth, length=length_ticks)
    axins_delay.set_ylabel("FR (sp/s)", labelpad=-1, fontsize=font_size - 2)
    axins_delay.set_xlabel("Time (s)", labelpad=-1, fontsize=font_size - 2)
    axins_delay.set_xticks([0, 1.5, 3, 6], ["0", "", "", "6"])
    axins_delay.set_xlim((-0.5, delays_unique[-1] + 0.5))
    axins_delay.set_yticks([2, 8])
    axins_delay.set_ylim(1.8, 8.6)
    axins_delay.set_box_aspect(1)

    count_delays = 0
    low = 0
    # PSTH for trials where variable reward delays and certain reward amounts were given
    for i_d, d in enumerate(delays_unique):
        if len(delays_unique) < 4:
            i_d = i_d + 1

        mean_psth = np.nanmean(psth_cue[neuron_idx, i_d, :, :], axis=1)  # Mean over trials
        sem_psth = scipy.stats.sem(psth_cue[neuron_idx, i_d, :, :], axis=1,
                                   nan_policy='omit')  # Standard error of the mean over trials
        ax[1, i].plot(axes, mean_psth, color=colors_delay[i_d], label=labels_delay[i_d] + "s")
        ax[1, i].fill_between(axes, mean_psth - sem_psth, mean_psth + sem_psth, alpha=0.2, color=colors_delay[i_d])

        mean_delta_fr = np.nanmean(responses_cue[neuron_idx, i_d, :])
        sem_delta_fr = scipy.stats.sem(responses_cue[neuron_idx, i_d, :], nan_policy='omit')

        if len(delays_unique) < 4:
            axins_delay.errorbar((delays_unique[i_d - 1]), mean_delta_fr, yerr=sem_delta_fr, capsize=scatter_size,
                                 fmt="o", color=colors_delay[i_d], markersize=scatter_size)
        else:
            axins_delay.errorbar((delays_unique[i_d]), mean_delta_fr, yerr=sem_delta_fr, capsize=scatter_size, fmt="o",
                                 color=colors_delay[i_d], markersize=scatter_size)

        n_trials = np.sum(~np.isnan(responses_cue[neuron_idx, i_d, :]))
        ax[0, i].axvline(x=time_right, ymin=(low) / n_trials_delays, ymax=(low + n_trials) / n_trials_delays,
                         color=colors_delay[i_d], linewidth=linewidth * 5)  # colored bar on the right of the raster
        low += n_trials

    ax[1, i].set_xticks([-0.5, 0, 1.2], ["-0.5", "0", "1.2"])
    ax[1, i].set_box_aspect(1)
    ax[1, i].axvline(x=0, color="black", ls="--")
    ax[1, i].plot([0.2, 0.65], [-0.5, -0.5], linewidth=linewidth * 2, color="black")
    if i == 0:
        ax[1, i].set_ylabel("Firing rate (sp/s)", labelpad=0.1)

    # Get tuning with respect to reward time for each neuron
    gamma_neuron = gammas[neuron_idx]
    gain_neuron = gains[neuron_idx]

    # Plot fitted temporal discounting curve in inset
    time_plot = np.linspace(0, (delays_unique[-1]), 100)
    axins_delay.plot(time_plot, gain_neuron * gamma_neuron ** time_plot, color="grey")
    axins_delay.set_title(r"$\gamma=$" + str(np.round(gamma_neuron, 2)), fontsize=font_size - 2)

# Third and forth column: responses to different reward magnitudes for two example neurons
idx_neurons_photo_ided = [22, 35]  # index of neurons (with respect to photo-ided neurons)
neuron_ids = photo_ided_neuron_ids[idx_neurons_photo_ided]
name_neurons = ["optimistic", "pessimistic"]

for i in range(2):

    neuron_id = neuron_ids[i]
    neuron_idx = idx_neurons_photo_ided[i]

    # Get spike times
    spike_times = dataframe_spike_times[dataframe_spike_times['Neuron id'] == neuron_id]['Spike times (s)'].values

    # Get give reward times for this session
    give_reward_times = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Give reward times'].values

    # Distribution reward ID: 0 is certain reward of 4.5uL, 2 is variable bimodal reward
    distribution_id_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Distribution reward ID'].values

    # Amount of reward for each trial
    amount_reward_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Amount reward'].values

    # If the mice was rewarded in each trial
    isRewarded_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Is rewarded'].values

    # If the trial was a time manipulation trial (one of the delays was removed), or not
    is_switch_trial = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
        'Is time manipulation trial'].values
    trial_switch = np.where(is_switch_trial == 1)[0]

    # Get trial for which variable reward amount and certain reward delay was given
    flag_amount_all = np.intersect1d(np.where(distribution_id_trials == 2)[0], np.where(isRewarded_trials == 1)[0])
    flag_amount_all = flag_amount_all[flag_amount_all < trial_switch]  # don't consider trials after time manipulation

    # Sort trials with respect to reward amounts
    amounts_filtered = amount_reward_trials[flag_amount_all]
    amounts_unique = np.unique(amounts_filtered)
    idx_sort_amount = np.argsort(amounts_filtered)
    n_trials_amounts = len(flag_amount_all)

    # Raster sorted on reward amount aligned on reward delivery
    for i_trial, trial in enumerate(flag_amount_all[idx_sort_amount]):
        after = spike_times - give_reward_times[trial] > axes[0]
        before = spike_times - give_reward_times[trial] < axes[-1]
        pos = np.logical_and(after, before)
        spike_times_trial = spike_times[pos] - give_reward_times[trial]
        ax[0, 2 + i].scatter(spike_times_trial, i_trial * np.ones(len(spike_times_trial)), marker='.', s=scatter_size,
                             c="black")

    ax[0, 2 + i].axvline(x=0, ls="--", color="black")
    ax[0, 2 + i].set_yticks([0, n_trials_amounts - 1])
    ax[0, 2 + i].set_yticklabels([1, n_trials_amounts])
    ax[0, 2 + i].set_ylim(0, n_trials_amounts)
    ax[0, 2 + i].spines['left'].set_linewidth(linewidth)
    ax[0, 2 + i].spines['bottom'].set_linewidth(linewidth)
    ax[0, 2 + i].tick_params(width=linewidth, length=length_ticks)
    ax[0, 2 + i].set_xticks([-0.5, 0, 1.2], ["-0.5", "0", "1.2"])
    ax[0, 2 + i].set_box_aspect(1)

    # Inset: Scatter of mean responses for trials where variable reward magnitudes were given
    axins_amount = inset_axes(ax[1, 2 + i], width="40%", height="40%")
    axins_amount.spines['left'].set_linewidth(linewidth)
    axins_amount.spines['bottom'].set_linewidth(linewidth)
    axins_amount.tick_params(width=linewidth, length=length_ticks)
    axins_amount.set_ylabel(r"$\Delta$" + " FR (sp/s)", fontsize=font_size - 2)
    axins_amount.set_xlabel("Magnitude " + r"($\mu l$)", fontsize=font_size - 2)
    axins_amount.set_xlim((-0.5, amounts_unique[-1] / 10 + 0.5))
    axins_amount.set_yticks([-2.5, 0, 2.5], ["-2.5", "0", "2.5"])
    axins_amount.set_xticks([1, 2.75, 4.5, 6.25, 8], ["1", "", "", "", "8"])
    axins_amount.axhline(y=0, ls="--", color="black")

    # PSTH for trials where variable reward amounts were given
    low = 0  # For bar on the right of raster plot
    for i_a, a in enumerate(amounts_unique):
        mean_psth = np.nanmean(psth_reward[neuron_idx, i_a, :, :], axis=1)
        sem_psth = scipy.stats.sem(psth_reward[neuron_idx, i_a, :, :], axis=1, nan_policy='omit')
        ax[1, 2 + i].plot(axes, mean_psth, color=colors_amount[i_a], label=labels_amount[i_a] + "uL")
        ax[1, 2 + i].fill_between(axes, mean_psth - sem_psth, mean_psth + sem_psth, alpha=0.2, color=colors_amount[i_a])

        mean_delta_fr = np.nanmean(responses_reward[neuron_idx, i_a, :])
        sem_delta_fr = scipy.stats.sem(responses_reward[neuron_idx, i_a, :], nan_policy='omit')
        axins_amount.errorbar(amounts_unique[i_a], mean_delta_fr, yerr=sem_delta_fr, capsize=scatter_size, fmt="o",
                              color=colors_amount[i_a], markersize=scatter_size)

        n_trials = np.sum(~np.isnan(responses_reward[neuron_idx, i_a, :]))
        ax[0, 2 + i].axvline(x=time_right, ymin=(low) / n_trials_amounts, ymax=(low + n_trials) / n_trials_amounts,
                             color=colors_amount[i_a],
                             linewidth=linewidth * 5)  # colored bar on the right of the raster
        low += n_trials

    ax[1, 2 + i].axvline(x=0, color="black", ls="--")
    ax[1, 2 + i].plot([0.2, 0.65], [-0.5, -0.5], linewidth=linewidth * 2, color="black")
    ax[1, 2 + i].spines['left'].set_linewidth(linewidth)
    ax[1, 2 + i].spines['bottom'].set_linewidth(linewidth)
    ax[1, 2 + i].tick_params(width=linewidth, length=length_ticks)

    ax[1, 2 + i].set_xticks([-0.5, 0, 1.2], ["-0.5", "0", "1.2"])
    ax[1, 2 + i].set_box_aspect(1)
    ax[1, 2 + i].set_xlabel("Time since reward (s)")

    # Get tuning with respect to reward time for each neuron
    reversal = reversal_points[neuron_idx]
    x_neg_plt = np.linspace(0, reversal, 100)
    x_pos_plt = np.linspace(reversal, amounts_unique[-1], 100)  # /10
    slope_neg = slopes_negative[neuron_idx]
    slope_pos = slopes_positive[neuron_idx]

    # Plot positive and negative slope
    if not np.isnan(slope_neg):
        y_neg_plt = slope_neg * (x_neg_plt - reversal)
        axins_amount.plot(x_neg_plt, y_neg_plt, color="blue")
    if not np.isnan(slope_pos):
        y_pos_plt = slope_pos * (x_pos_plt - reversal)
        axins_amount.plot(x_pos_plt, y_pos_plt, color="red")
    axins_amount.legend().remove()
    axins_amount.set_title(r"$V=$" + str(np.round(reversal, 2)) + R"$\mu$" + "l",
                           fontsize=font_size - 2)  # + " " + r"$\tau=$" + str(np.round(estimated_tau, 2)),
    axins_amount.set_box_aspect(1)

plt.show()

# Mean PSTH for the population of photo-ided neurons for each reward delay
fig, ax = plt.subplots(1, 2, figsize=(horizontal_size * 2, vertical_size), sharex='col', sharey='row')
ax[0].spines['left'].set_linewidth(linewidth)
ax[0].spines['bottom'].set_linewidth(linewidth)
ax[0].tick_params(width=linewidth, length=length_ticks)
ax[0].set_xticks([-0.5, 0, 1.2], ["-0.5", "0", "1.2"])
ax[0].set_xlabel("Time since cue (s)")
ax[0].set_ylabel("Firing rate (Hz)")
ax[0].set_box_aspect(1)
for i_d, d in enumerate(delays_unique):
    mean_across_neurons = np.nanmean(psth_cue[:, i_d, :, :], axis=2)
    mean = np.nanmean(mean_across_neurons, axis=0).astype(float)
    sem = scipy.stats.sem(mean_across_neurons, axis=0, nan_policy='omit').astype(float)
    ax[0].plot(axes, mean, color=colors_delay[i_d, :], label=str(d / 1000) + "s")
    ax[0].fill_between(axes, mean - sem, mean + sem, alpha=0.2, color=colors_delay[i_d, :])
ax[0].legend()
ax[0].axvline(x=0, color="black", ls="--")

# Mean PSTH for the population of photo-ided neurons for each reward magnitude
ax[1].spines['left'].set_linewidth(linewidth)
ax[1].spines['bottom'].set_linewidth(linewidth)
ax[1].tick_params(width=linewidth, length=length_ticks)
ax[1].set_xticks([-0.5, 0, 1.2], ["-0.5", "0", "1.2"])
ax[1].set_xlabel("Time since reward (s)")
ax[1].set_box_aspect(1)
for i_a, amount in enumerate(amounts_unique):
    mean_across_neurons = np.nanmean(psth_reward[:, i_a, :, :], axis=2).astype(float)
    mean = np.nanmean(mean_across_neurons, axis=0)
    sem = scipy.stats.sem(mean_across_neurons, axis=0, nan_policy='omit').astype(float)
    ax[1].plot(axes, mean, color=colors_amount[i_a, :], label=str(amount) + "uL")
    ax[1].fill_between(axes, mean - sem, mean + sem, alpha=0.2, color=colors_amount[i_a, :])
ax[1].legend()
ax[1].axvline(x=0, color="black", ls="--")
ax[1].plot([0.2, 0.65], [2, 2], linewidth=linewidth * 2, color="black")
ax[0].plot([0.1, 0.25], [2, 2], linewidth=linewidth * 2, color="grey")
ax[0].plot([0.5, 0.65], [2, 2], linewidth=linewidth * 2, color="black")
plt.show()

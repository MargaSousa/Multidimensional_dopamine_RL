import numpy as np
from aux_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl

length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 2.2
vertical_size = 2.2
font_size = 11
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 10
from scipy.optimize import curve_fit
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from sklearn.utils import resample
import pandas

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Spike times data
directory_raw_data = os.path.join(directory, "Raw")

# Behavior information data
dataframe_behavior_times = pd.read_csv(os.path.join(directory_raw_data, "Neurons_behavior_trials_with_PSTH.csv"))
dataframe_behavior_times['PSTH cue'] = dataframe_behavior_times.loc[:,
                                       'PSTH aligned to cue bin 0':'PSTH aligned to cue bin 649'].values.tolist()
dataframe_behavior_times['PSTH reward'] = dataframe_behavior_times.loc[:,
                                          'PSTH aligned to reward bin 0':'PSTH aligned to reward bin 649'].values.tolist()

# An upper estimate
n_neurons = 100

def get_kernel(mean_kernel, bin_width):
    """Outputs an exponential decaying kernel with mean
        mean_kernel and bin width bin_width."""
    n_bins = (mean / bin_width) * 30
    bins = np.arange(n_bins / 2)
    x = bins * bin_width
    decay = 1 / mean_kernel
    kernel = np.exp(-x * decay)
    kernel = kernel / np.sum(kernel)
    kernel = np.concatenate((np.zeros(int(n_bins / 2) - 1), kernel))
    return np.arange(-n_bins / 2 + 1, n_bins / 2), kernel


def do_bootstrap(x, y, n_resamples, round):
    """Bootstrap to test if the samples in y are smaller than the ones in x.
        Repeat n_resamples time. And return the p-value with round number of decimal places.  """
    n = 0
    all_dif = []
    for r in range(n_resamples):
        new_x = resample(x, replace=True, n_samples=len(x))
        new_y = resample(y, replace=True, n_samples=len(y))
        if np.round(np.mean(new_x), round) > np.round(np.mean(new_y), round):
            n += 1
        all_dif.append(np.round(np.mean(new_y), round) - np.round(np.mean(new_x), round))
    return n / n_resamples



color_remove_short = "limegreen"
color_remove_long = "mediumvioletred"

# Get neuron ids that are photo-ided
photo_ided_neuron_ids = dataframe_behavior_times.loc[
    dataframe_behavior_times['Is photo ided'] == 1, 'Neuron id'].drop_duplicates().values

# Time-scales
means = np.arange(10, 300)

p_values_gamma = []
p_values_gain = []
for mean in means:

    i_neuron = 0
    responses_cue_low = np.empty((n_neurons, 4, 150))
    responses_cue_low[:, :, :] = np.nan
    responses_cue_high = np.empty((n_neurons, 4, 150))
    responses_cue_high[:, :, :] = np.nan

    all_gammas_high = []
    all_gammas_low = []

    all_gains_high = []
    all_gains_low = []

    for neuron_id in photo_ided_neuron_ids:

        # Get give cue times for this session
        give_cue_times = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Give cue times'].values

        # Get give reward times for this session
        give_reward_times = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Give reward times'].values

        # Distribution reward ID: 0 is certain reward of 4.5uL, 2 is variable bimodal reward
        distribution_id_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Distribution reward ID'].values

        # If the trial was a time manipulation trial (one of the delays was removed), or not
        is_switch_trial = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Is time manipulation trial'].values
        trial_switch = np.where(is_switch_trial == 1)[0]

        fix_amount_trials = np.where(distribution_id_trials == 0)[0]
        fix_amount_trials = fix_amount_trials[fix_amount_trials < trial_switch]

        # Delay to reward for each trial
        all_delays = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Delay reward'].values
        # all_delays=all_delays[fix_amount_trials]

        delays_unique = np.unique(all_delays)
        n_delays = len(delays_unique)

        reward_times = []

        # Bins to compute rate of reward occurrence
        bin_width = 1
        bins = np.arange(np.min(give_cue_times), np.max(give_reward_times) + 100, bin_width)

        bins_reward_time = []
        for trial in range(len(give_reward_times)):
            reward_times = reward_times + [give_reward_times[trial]] * 1  # reward occurence #int(reward_amounts[trial])
            bins_reward_time.append(np.where(bins >= give_reward_times[trial])[0][0])

        bins_reward_time = np.array(bins_reward_time)

        # Convolve with an exponential kernel
        reward_times = np.array(reward_times)
        rwdrate_counts, _ = np.histogram(reward_times, bins=bins)
        _, kernel = get_kernel(mean, bin_width)
        rwdrate_continuous = np.convolve(rwdrate_counts, kernel, 'same')

        # Get the rate of reward occurence at the cues
        give_cue_times_rwdrate = []
        bins_give_cue = np.zeros(len(fix_amount_trials))
        for i_t, t in enumerate(fix_amount_trials):
            bins_give_cue[i_t] = np.where(bins >= give_cue_times[i_t])[0][0] + 1
            give_cue_times_rwdrate.append(rwdrate_continuous[int(bins_give_cue[i_t])])

        give_cue_times_rwdrate = np.array(give_cue_times_rwdrate)
        bins_give_cue = bins_give_cue.astype(int)

        # Compute low and high quantiles
        ma_low = np.quantile(give_cue_times_rwdrate, 0.33)
        ma_high = np.quantile(give_cue_times_rwdrate, 0.66)

        # Get trials for low and high quantiles
        trials_low = np.where(give_cue_times_rwdrate <= ma_low)[0]
        trials_low = fix_amount_trials[trials_low]
        trials_high = np.where(give_cue_times_rwdrate >= ma_high)[0]
        trials_high = fix_amount_trials[trials_high]

        psth_cue = np.array(
            dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id]['PSTH cue'].tolist())
        responses_cue = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Responses cue'].values

        # Compute temporal discount factors for high and low reward occurence rates
        for i_d, d in enumerate(delays_unique):
            trials_delay_low = np.where(all_delays[trials_low] == d)[0]
            n_trials_delay_low = len(trials_delay_low)
            responses_cue_low[i_neuron, i_d, :n_trials_delay_low] = responses_cue[trials_low[trials_delay_low]]

            trials_delay_high = np.where(all_delays[trials_high] == d)[0]
            n_trials_delay_high = len(trials_delay_high)
            responses_cue_high[i_neuron, i_d, :n_trials_delay_high] = responses_cue[trials_high[trials_delay_high]]

        mean_responses_low = np.nanmean(responses_cue_low[i_neuron, :n_delays, :], axis=1)
        mean_responses_high = np.nanmean(responses_cue_high[i_neuron, :n_delays, :], axis=1)

        responses_overall = np.concatenate(
            (responses_cue_high[i_neuron, :n_delays, :], responses_cue_low[i_neuron, :n_delays, :]), axis=1)
        mean_responses = np.nanmean(responses_overall, axis=1)

        if np.sum(np.isnan(mean_responses_low)) > 0 or np.sum(np.isnan(mean_responses_high)) > 0:
            continue

        popt_low, pcov_low = curve_fit(exponential, (delays_unique[~np.isnan(mean_responses_low)]),
                                       mean_responses_low[~np.isnan(mean_responses_low)], gtol=1e-20, maxfev=1000000)
        popt_high, pcov_high = curve_fit(exponential, (delays_unique[~np.isnan(mean_responses_high)]),
                                         mean_responses_high[~np.isnan(mean_responses_high)], gtol=1e-20,
                                         maxfev=1000000)

        gamma_low = np.round(np.exp(-popt_low[-1]), 2)
        a_low = popt_low[0]

        gamma_high = np.round(np.exp(-popt_high[-1]), 2)
        a_high = popt_high[0]

        all_gains_high.append(a_high)
        all_gains_low.append(a_low)

        all_gammas_low.append(gamma_low)
        all_gammas_high.append(gamma_high)
        i_neuron += 1

    all_gammas_high = np.array(all_gammas_high)
    all_gammas_low = np.array(all_gammas_low)
    all_gains_high = np.array(all_gains_high)
    all_gains_low = np.array(all_gains_low)

    ttest_relative_take_short_long = do_bootstrap(all_gammas_high, all_gammas_low, 10000, 2)  # 10000
    p_gamma = ttest_relative_take_short_long
    p_values_gamma.append(p_gamma)
    t_test_relative_take_short_long = do_bootstrap(all_gains_high, all_gains_low, 10000, 0)
    p_gain = t_test_relative_take_short_long
    p_values_gain.append(p_gain)

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
ax.plot(means, p_values_gain, color="k")
plt.xlabel("Time scale (s)")
plt.ylabel("Boostrapped p-value")
plt.ylim(0, 0.65)
plt.show()

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
ax.plot(means, p_values_gamma, color="k")
plt.xlabel("Time scale (s)")
plt.ylabel("Bootstrapped log p-value")
plt.ylim(0, 0.7)
plt.yticks([0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"])
plt.axhline(y=np.log(0.1), ls="--")
plt.show()

pdb.set_trace()

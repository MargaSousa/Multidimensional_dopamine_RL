import pdb
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.isotonic import IsotonicRegression
from aux_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import os
from scipy.stats import gaussian_kde
import pandas

mpl.use('TkAgg')

# Parameters for paper plots
length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 1.75
vertical_size = 1.75
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
directory_parsed_data = os.path.join(directory, "Parsed_data_DA")
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))

# Get neuron's tuning features
gammas = data_frame_neurons_info['Gamma'].values
estimated_reversals = data_frame_neurons_info['Reversals'].values
estimated_taus = data_frame_neurons_info['Taus'].values
gains = data_frame_neurons_info['Gain'].values
bias_taus = data_frame_neurons_info['Bias in the estimation of tau'].values
variance_taus = data_frame_neurons_info['Variance in the estimation of tau'].values
n_neurons = len(gammas)

# Responses for variable reward magnitudes
responses_cue_bimodal = np.loadtxt(os.path.join(directory_parsed_data,
                                                "responses_cue_different_magnitudes_constant_delay.csv"))  # Responses at the cue for delay of 3s and variable magnitudes (neurons x trials)

# Behavior information data
directory_raw_data = os.path.join(directory, "Raw")
dataframe_behavior_times = pd.read_csv(os.path.join(directory_raw_data, "Neurons_behavior_trials.csv"))

# Get 'Neuron id' for photo ided-neurons
photo_ided_neuron_ids = dataframe_behavior_times.loc[
    dataframe_behavior_times['Type of neuron'] == 'Photo_ided', 'Neuron id'].drop_duplicates().values
animals = dataframe_behavior_times.loc[
    dataframe_behavior_times['Type of neuron'] == 'Photo_ided', 'Animal'].drop_duplicates().values

# Get pupil diameter PSTH
data_frame_pupil_area = pd.read_csv(os.path.join(directory_raw_data, "Pupil_diameter.csv"))

# For the reward time Laplace decoder
n_time = 100
time = np.linspace(0, 6.5, n_time)
n_runs_decoder = 10
alpha_time = 1

# Decode time for variable cue
population_responses_variable = np.nanmean(responses_cue_bimodal, axis=1)
mean_responses_variable_cue_discount = population_responses_variable / (
        estimated_reversals * gains)  # Correct for diversity in reward magnitude tuning
variance_variable = np.nanvar(responses_cue_bimodal, axis=1)
pdf_time_variable = run_decoding_time(time, gammas, variance_variable, mean_responses_variable_cue_discount, alpha_time)
estimate_time_variable = np.sum(time * pdf_time_variable)

# For decoding reward magnitude
# Regression between firing rate at the cue and reversal points
population_responses_variable_corrected = population_responses_variable / (gains * gammas ** estimate_time_variable)
well_estimated_reversals = np.intersect1d(np.where(estimated_reversals > 1.1)[0],
                                          np.where(estimated_reversals < 7.9)[0])
n_neurons_well_estimated_reverals = len(well_estimated_reversals)
reg_variable = HuberRegressor().fit((population_responses_variable_corrected[well_estimated_reversals]).reshape(-1, 1),
                                    estimated_reversals[well_estimated_reversals])

# Mapping from reversals to taus at cue
corrected_taus = estimated_taus[well_estimated_reversals] - bias_taus[well_estimated_reversals]
corrected_taus[corrected_taus < 0] = 0
corrected_taus[corrected_taus > 1] = 1
w_reversals = 1.0 / (1 + variance_taus[well_estimated_reversals])
reversal_estimated_at_cue = reg_variable.predict(
    population_responses_variable_corrected[well_estimated_reversals].reshape(-1, 1))
iso_reg_cue = IsotonicRegression(increasing=True).fit(reversal_estimated_at_cue, corrected_taus,
                                                      sample_weight=w_reversals)
pred_cue = iso_reg_cue.predict(reversal_estimated_at_cue)

# For smoothing distribution in reward amount
n_amount = 80
smooth = 0.35
amount = np.linspace(0, 9, n_amount)

# Number of quantiles of reward history
n_quantiles = 5
cmap = mpl.cm.get_cmap('cool', 12)
colors = cmap(np.arange(0, 1, 1.0 / n_quantiles))

# Kernel type for computing reward history
kerneltype = "exponential"

legend_quantiles = ["1st", "2nd", "3rd", "4th", "5th"]

# Get responses of individual neurons for different reward history quantiles
responses_cue_variable = np.empty((n_quantiles, n_neurons, 150))
responses_cue_variable[:, :, :] = np.nan
i_neuron = 0
i_session_overall = 0

for i_animal, animal in enumerate(animals):

    prev_session = -1
    i_session = 0

    # Get individual animal time-scale to intergrate reward
    m = data_frame_pupil_area[data_frame_pupil_area['Animal'] == animal]['Time scale (trials)'].values[0]

    photo_ided_animal = dataframe_behavior_times.loc[
        dataframe_behavior_times['Type of neuron'] == 'Photo_ided', 'Neuron id'].drop_duplicates().values

    for i_neuron, neuron_id in enumerate(photo_ided_animal):

        session = dataframe_behavior_times.loc[dataframe_behavior_times['Neuron id'] == neuron_id, 'Session'].iat[0]

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
        variable_amount_trials = np.intersect1d(np.where(distribution_id_trials == 2)[0],
                                                np.where(isRewarded_trials == 1)[0])
        variable_amount_trials = variable_amount_trials[
            variable_amount_trials < trial_switch]  # don't consider trials after time manipulation

        reward_amounts = amount_reward_trials[variable_amount_trials]
        unique_reward_amounts = np.unique(reward_amounts)

        moving_average_amounts = moving_average(reward_amounts, m, 1, kerneltype)
        moving_average_amounts = moving_average_amounts[:len(reward_amounts)]

        quantiles = np.quantile(moving_average_amounts, np.arange(0, 1, 1.0 / n_quantiles))
        quantiles = np.insert(quantiles, [n_quantiles], [np.max(moving_average_amounts)])
        max_trial_number = len(variable_amount_trials)

        responses_neuron = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Responses cue'].values
        responses_neuron = responses_neuron[variable_amount_trials]

        for i_quantile in range(n_quantiles):
            trials_pos_quantile = np.intersect1d(np.where(moving_average_amounts >= quantiles[i_quantile])[0],
                                                 np.where(moving_average_amounts <= quantiles[i_quantile + 1])[0]) + 1
            trials_pos_quantile = trials_pos_quantile[trials_pos_quantile < max_trial_number]
            n_trials_quantile = len(trials_pos_quantile)
            responses_cue_variable[i_quantile, i_neuron, :n_trials_quantile] = responses_neuron[trials_pos_quantile]

            if i_quantile == n_quantiles - 1:
                i_session_overall += 1
                i_session += 1

        i_neuron += 1
        prev_session = int(session)

fig_dist, ax_dist = plt.subplots(figsize=(horizontal_size * 2, vertical_size), nrows=1, ncols=2)

mean_time_all_runs = np.empty((n_runs_decoder, n_quantiles))
mean_magnitude_all_runs = np.empty((n_runs_decoder, n_quantiles))

for i_quantile in range(n_quantiles):

    mean_pdf_amount = np.zeros(n_amount)
    mean_pdf_time = np.zeros(n_time)

    mean_amount = 0
    mean_time = 0

    for run in range(n_runs_decoder):
        number_trials = np.sum(~np.isnan(responses_cue_variable[i_quantile, :, :]), axis=1)

        # Take randomly 70% of the trials
        random_selection = np.random.choice(np.arange(np.min(number_trials)), int(0.7 * np.min(number_trials)))
        mean_responses = np.nanmean(responses_cue_variable[i_quantile, :, random_selection], axis=0)

        # Correct for the diversity in tuning to reward magnitude and gain
        mean_responses_corrected_diverse_tuning_magnitude = reg_variable.coef_[0] * mean_responses / (
                    gains * (estimated_reversals - reg_variable.intercept_))  #
        # mean_responses_corrected_diverse_tuning_magnitude = mean_responses / (gains * estimated_reversals)

        # Decoding time
        pdf_time = run_decoding_time(time, gammas, np.ones(len(gammas)),
                                     mean_responses_corrected_diverse_tuning_magnitude, 1)
        mean_pdf_time += pdf_time
        mean = np.sum(time * pdf_time)
        mean_time += mean
        mean_time_all_runs[run, i_quantile] = mean
        ax_dist[1].plot(time, pdf_time, color=colors[i_quantile], linewidth=linewidth * 0.1)

        mean_responses_corrected_diversity_time = mean_responses / (gains * gammas ** 3)
        estimated_reversals_cue = reg_variable.predict(
            (mean_responses_corrected_diversity_time[well_estimated_reversals]).reshape(-1, 1))

        # Decoding magnitude
        samples, _ = run_decoding_magnitude(estimated_reversals_cue, pred_cue, np.ones(len(estimated_reversals_cue)),
                                            minv=1, maxv=8, N=20, max_samples=2000, max_epochs=15, method='TNC')
        kde = gaussian_kde(samples, bw_method=smooth)
        pdf_amount = kde.pdf(amount)
        pdf_amount = pdf_amount / np.sum(pdf_amount)
        mean_pdf_amount += pdf_amount
        mean = np.sum(amount * pdf_amount)
        mean_amount += mean
        mean_magnitude_all_runs[run, i_quantile] = mean

        ax_dist[0].plot(amount, pdf_amount, color=colors[i_quantile], linewidth=linewidth * 0.1)

    mean_pdf_amount = mean_pdf_amount / n_runs_decoder
    mean_amount = mean_amount / n_runs_decoder
    ax_dist[0].plot(amount, mean_pdf_amount, color=colors[i_quantile])

    mean_pdf_time = mean_pdf_time / n_runs_decoder
    mean_time = mean_time / n_runs_decoder
    ax_dist[1].plot(time, mean_pdf_time, color=colors[i_quantile], label=legend_quantiles[i_quantile])

plt.show()

# Plot mean decoded time and magnitude
fig_mean_time, ax_mean_time = plt.subplots(figsize=(horizontal_size, vertical_size))  #
ax_mean_time.spines['left'].set_linewidth(linewidth)
ax_mean_time.spines['bottom'].set_linewidth(linewidth)
ax_mean_time.tick_params(axis="y", labelcolor="gray")

all_points_x_plot = []
all_points_y_plot_time = []
all_points_y_plot_magnitude = []
all_errors_plot_time = []
all_errors_plot_magnitude = []

for i_quantile in range(n_quantiles):
    median_magnitude = np.nanmedian(mean_magnitude_all_runs[:, i_quantile])
    sem_amount = scipy.stats.sem(mean_magnitude_all_runs[:, i_quantile], nan_policy='omit')
    quantiles = np.quantile(mean_magnitude_all_runs[:, i_quantile], [0.25, 0.75])
    asym_error_magnitude = [[median_magnitude - quantiles[0]], [quantiles[1] - median_magnitude]]

    median_time = np.nanmedian(mean_time_all_runs[:, i_quantile])
    sem_time = scipy.stats.sem(mean_time_all_runs[:, i_quantile], nan_policy='omit')
    quantiles = np.quantile(mean_time_all_runs[:, i_quantile], [0.25, 0.75])
    asym_error_time = [[median_time - quantiles[0]], [quantiles[1] - median_time]]

    all_points_x_plot.append(i_quantile)
    all_points_y_plot_time.append(median_time)
    all_points_y_plot_magnitude.append(median_magnitude)

    all_errors_plot_time.append(asym_error_time)
    all_errors_plot_magnitude.append(asym_error_magnitude)

all_errors_plot_magnitude = np.asarray(all_errors_plot_magnitude)[:, :, 0]
all_errors_plot_time = np.asarray(all_errors_plot_time)[:, :, 0]
ax_mean_time.errorbar(all_points_x_plot, all_points_y_plot_time, yerr=np.transpose(all_errors_plot_time), ls='none',
                      capsize=3, color="gray", zorder=2)  #
ax_mean_time.scatter(all_points_x_plot, all_points_y_plot_time, s=15, color="gray")

# Regression time
x = np.repeat(np.arange(n_quantiles).reshape(1, -1), n_runs_decoder, axis=0)
reg_time = LinearRegression().fit((x.flatten()).reshape(-1, 1), mean_time_all_runs.flatten())
x_example = np.arange(n_quantiles)
y_example = reg_time.predict(x_example.reshape(-1, 1))
ax_mean_time.plot(x_example, y_example, color="gray")

ax_mean_magnitude = ax_mean_time.twinx()
ax_mean_magnitude.set_box_aspect(1)
ax_mean_magnitude.spines['right'].set_visible(True)
ax_mean_magnitude.spines['right'].set_linewidth(linewidth)

ax_mean_magnitude.errorbar(all_points_x_plot, all_points_y_plot_magnitude, yerr=np.transpose(all_errors_plot_magnitude),
                           capsize=3, color="k", ls='none', zorder=1)  # fmt="o",
ax_mean_magnitude.scatter(all_points_x_plot, all_points_y_plot_magnitude, s=15, color="k", zorder=1)

# Regression magnitude
reg_amount = LinearRegression().fit((x.flatten()).reshape(-1, 1), mean_magnitude_all_runs.flatten())
x_example = np.arange(n_quantiles)
y_example = reg_amount.predict(x_example.reshape(-1, 1))
ax_mean_magnitude.plot(x_example, y_example, color="k")
ax_mean_magnitude.set_xticks(np.arange(5), np.arange(1, 6))

ax_mean_magnitude.set_ylabel("Magnitude " + "(" + r"$\mu$l)")
ax_mean_time.set_ylabel("Time (s)", color="gray")
ax_mean_time.set_xlabel("Reward history quintiles")
ax_mean_magnitude.set_yticks([4.1, 4.45, 4.8])
ax_mean_time.set_yticks([2.8 - 0.35, 2.8, 2.8 + 0.35])
ax_mean_time.set_title("Decoded mean")

# Regression time
means_time = np.mean(mean_time_all_runs, axis=0)
reg = LinearRegression().fit((mean_magnitude_all_runs.flatten()).reshape(-1, 1), mean_time_all_runs.flatten())
x_example = np.linspace(np.min(mean_magnitude_all_runs), np.max(mean_magnitude_all_runs), 100)
y_example = reg.predict(x_example.reshape(-1, 1))
plt.show()

pdb.set_trace()

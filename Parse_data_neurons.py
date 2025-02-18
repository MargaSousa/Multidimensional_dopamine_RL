import ast
import os
import pdb
from functools import reduce
from aux_functions import *

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"
directory_raw_data = os.path.join(directory, "Raw")

# Behavior information data
dataframe_behavior_times = pd.read_csv(os.path.join(directory_raw_data, "Neurons_behavior_trials_with_PSTH.csv"))

type_neurons = "Putative_DA" #"Photo_ided" or "Putative_DA"

if type_neurons=="Photo_ided":
    dataframe_behavior_times = dataframe_behavior_times[dataframe_behavior_times['Is photo ided'] == 1]

if type_neurons=="Putative_DA":
    dataframe_behavior_times = dataframe_behavior_times[dataframe_behavior_times['Type of neuron'] == type_neurons]

dataframe_behavior_times['PSTH cue'] = dataframe_behavior_times['PSTH cue'].apply(ast.literal_eval)
dataframe_behavior_times['PSTH reward'] = dataframe_behavior_times['PSTH reward'].apply(ast.literal_eval)

# Time limits for PSTH
axes_correct = np.linspace(-5, 8, 650)
time_left = axes_correct[0]
time_right = axes_correct[-1]
bin_left = 0  # np.where(axes_correct >= time_left)[0][0]
bin_right = len(axes_correct)  # np.where(axes_correct >= time_right)[0][0]
axes = axes_correct[bin_left:bin_right]

# Window aligned to reward delivery for computing responses
time_init_reward = 0.2
time_end_reward = 0.65
time_window_integration_reward = [time_init_reward, time_end_reward]
window_integration_reward = [np.where(axes_correct >= time_init_reward)[0][0],
                             np.where(axes_correct >= time_end_reward)[0][0]]

# Window aligned to cue delivery for computing responses
time_init_cue = 0.2
time_end_cue = 0.65
time_window_integration_cue = [time_init_cue, time_end_cue]
window_integration_cue = [np.where(axes_correct >= time_init_cue)[0][0], np.where(axes_correct >= time_end_cue)[0][0]]

# Window baseline for computing responses
time_init_baseline = -4
time_end_baseline = -1
window_baseline = [np.where(axes_correct >= time_init_baseline)[0][0],
                   np.where(axes_correct >= time_end_baseline)[0][0]]

# An upper estimate
n_neurons = 230

# Arrays to store responses at cue for delay trials before ctx switch
responses_cue = np.empty((n_neurons, 4, 50))
responses_cue[:, :, :] = np.nan

# Arrays to store responses at cue for delay trials after ctx switch
responses_cue_after = np.empty((n_neurons, 3, 50))
responses_cue_after[:, :, :] = np.nan

# Arrays to store responses certain cue for variable delays (baseline corrected)
responses_reward_all = np.empty((n_neurons, 4, 70))
responses_reward_all[:, :, :] = np.nan

# Arrays to store responses certain cue for variable delays (baseline corrected)
responses_reward_all_not_baseline_corrected = np.empty((n_neurons, 4, 70))
responses_reward_all_not_baseline_corrected[:, :, :] = np.nan

# Arrays to store responses at reward for variable cue
responses_reward = np.empty((n_neurons, 5, 50))
responses_reward[:, :, :] = np.nan

# Arrays to store responses at reward for certain cue
responses_reward_certain = np.empty((n_neurons, 100))
responses_reward_certain[:, :] = np.nan

# Arrays to store responses at cue for bimodal cue
responses_cue_bimodal = np.empty((n_neurons, 150))
responses_cue_bimodal[:, :] = np.nan

responses_cue_bimodal_mat = np.empty((n_neurons, 5, 60))
responses_cue_bimodal_mat[:, :, :] = np.nan

# Arrays to store PSTH aligned to cue and reward
psth_cue = np.empty((n_neurons, 4, bin_right - bin_left, 50))
psth_cue[:, :, :, :] = np.nan
psth_reward = np.empty((n_neurons, 5, bin_right - bin_left, 50))
psth_reward[:, :, :, :] = np.nan

# What we want to estimate
all_estimated_gamma = []
all_estimated_gamma_not_zero = []  # Estimated gamma without considering the delay of 0s
gain_not_zero = []  # Estimated gain without considering the delay of 0s
all_estimated_gain = []
all_estimated_taus = []
all_estimated_reversals = []
is_take_long = []  # 1: longest delay was removed in the end of the session, 0: shortest delay was removed
all_estimated_gamma_after = []  # Estimated gamma after one of the delays was removed
all_estimated_gamma_before = []  # Estimated gamma before one of the delays was removed (estimated using the same delays as after the manipulation)
all_estimated_gamma_before_1st_half = []  # Estimated gamma before one of the delays was removed, 1st half
all_estimated_gamma_before_2nd_half = []  # Estimated gamma before one of the delays was removed, 2nd half
all_estimated_gain_before = []  # Estimated gain before one of the delays was removed
all_estimated_gain_after = []  # Estimated gain after one of the delays was removed
all_neurons_n_delays = []
slope_positive = []  # Slope for positive RPEs
slope_negative = []  # Slope for negative RPEs
constant_slope = []

# Save extra information
all_neurons_id = []
all_animals = []
all_sessions = []
is_take_long = []
animals = np.unique(dataframe_behavior_times['Animal'])
total_neurons = 0

for animal in [3353, 4098, 4099, 4096, 4140, 4418]:

    if type_neurons=="Putative_DA":
        neurons_ids = dataframe_behavior_times.loc[ (dataframe_behavior_times['Type of neuron'] == "Putative_DA") &(dataframe_behavior_times['Animal'] == animal),'Neuron id'].drop_duplicates().values

    if type_neurons=="Photo_ided":
        neurons_ids = dataframe_behavior_times.loc[ (dataframe_behavior_times['Is photo ided'] == 1) &(dataframe_behavior_times['Animal'] == animal),'Neuron id'].drop_duplicates().values


    for i_neuron, neuron_id in enumerate(neurons_ids):

        dataframe_behavior_neuron = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id]

        session = dataframe_behavior_neuron['Session'].iat[0]

        # Distribution reward ID: 0 is certain reward of 4.5uL, 2 is variable bimodal reward
        distribution_id_trials = dataframe_behavior_neuron['Distribution reward ID'].values

        # If the mice was rewarded in each trial
        isRewarded_trials = dataframe_behavior_neuron['Is rewarded'].values

        # Amount of reward for each trial
        amount_reward_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Amount reward'].values

        # Amount of reward for each trial
        delay_reward_trials = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron_id][
            'Delay reward'].values

        # If the trial was a time manipulation trial (one of the delays was removed), or not
        is_switch_trial = dataframe_behavior_neuron['Is time manipulation trial'].values
        trial_switch = np.where(is_switch_trial == 1)[0]

        # PSTH aligned to cue and reward delivery
        psth_cue_neuron = np.array(dataframe_behavior_neuron['PSTH cue'].tolist())
        psth_reward_neuron = np.array(dataframe_behavior_neuron['PSTH reward'].tolist())

        # Compute responses aligned to cue and reward delivery
        delta_fr_cue = np.nanmean(psth_cue_neuron[:, window_integration_cue[0]:window_integration_cue[1]], axis=1)
        delta_fr_reward = np.nanmean(psth_reward_neuron[:, window_integration_reward[0]:window_integration_reward[1]],
                                     axis=1)
        baseline = np.nanmean(psth_cue_neuron[:, window_baseline[0]:window_baseline[1]], axis=1)

        n_trials = len(delta_fr_cue)

        # Which delay was removed?
        is_take_long_neuron = dataframe_behavior_neuron['Is take long'].unique()[0]
        is_take_long.append(is_take_long_neuron)

        # Variable delay trials
        delays_unique =np.sort(dataframe_behavior_neuron.loc[dataframe_behavior_neuron['Distribution reward ID'] == 0, 'Delay reward'].unique())
        n_delays = len(delays_unique)
        count_delays_after = 0
        delays_after = []
        for i_d, d in enumerate(delays_unique):

            flag = reduce(np.intersect1d, (
            np.where(distribution_id_trials == 0)[0], np.where(isRewarded_trials == 1)[0],
            np.where(delay_reward_trials == d)[0]))
            flag_before = flag[flag < trial_switch]  # Before removing one of the delays
            flag_after = flag[flag > trial_switch][5:]  # After removing one of the delays

            if len(flag_after) != 0:
                delta_fr_cue_after = delta_fr_cue[flag_after]
                responses_cue_after[total_neurons, count_delays_after, :len(flag_after)] = delta_fr_cue_after
                count_delays_after += 1
                delays_after.append(d)

            delta_fr_cue_before = delta_fr_cue[flag_before]
            n_samples = flag_before.shape[0]
            responses_cue[total_neurons, i_d, :n_samples] = delta_fr_cue_before

            delta_fr_reward_before = delta_fr_reward[flag_before]
            baseline_before = baseline[flag_before]
            responses_reward_all[total_neurons, i_d, :n_samples] = delta_fr_reward_before - baseline_before
            responses_reward_all_not_baseline_corrected[total_neurons, i_d, :n_samples] = delta_fr_reward_before
            psth_cue[total_neurons, i_d, :, :n_samples] = psth_cue_neuron[flag_before, :].T

            # Responses to mean reward magnitude (4.5 ul)
            if d == 3:
                responses_reward_certain[total_neurons, :n_samples] = delta_fr_reward_before - baseline_before

        mean_responses_delay = np.nanmean(responses_cue[total_neurons, :n_delays, :], axis=1)

        # Estimate temporal discount factor and gain before context switch
        popt, pcov = curve_fit(exponential,delays_unique, mean_responses_delay, gtol=1e-20, maxfev=1000000)
        gamma = np.round(np.exp(-popt[-1]), 2)
        gain = popt[0]
        all_estimated_gamma.append(np.exp(-popt[-1]))
        all_estimated_gain.append(gain)

        # Estimate temporal discount factor and gain before context switch not considering the responses to the 0s cue
        popt_zero, _ = curve_fit(exponential,delays_unique[delays_unique != 0],
                                 mean_responses_delay[delays_unique != 0], gtol=1e-20, maxfev=1000000)
        gamma_zero = np.round(np.exp(-popt_zero[-1]), 2)
        gain_zero = popt_zero[0]
        all_estimated_gamma_not_zero.append(gamma_zero)
        gain_not_zero.append(gain_zero)

        # Consider the same delays before and after context switch
        if is_take_long_neuron:
            mean_responses_delay_before = np.nanmean(responses_cue[total_neurons, :n_delays - 1, :], axis=1)
            min_trials = np.min(np.sum(1 - np.isnan(responses_cue[total_neurons, 1:n_delays, :]), axis=1))
            min_half_trials = int(min_trials * 0.2)
            mean_responses_delay_before_1st_half = np.nanmean(
                responses_cue[total_neurons, :n_delays - 1, :min_half_trials], axis=1)
            mean_responses_delay_before_2nd_half = np.nanmean(
                responses_cue[total_neurons, :n_delays - 1, min_half_trials:min_trials], axis=1)
        else:
            mean_responses_delay_before = np.nanmean(responses_cue[total_neurons, 1:n_delays, :], axis=1)
            min_trials = np.min(np.sum(1 - np.isnan(responses_cue[total_neurons, 1:n_delays, :]), axis=1))
            min_half_trials = int(min_trials * 0.2)
            mean_responses_delay_before_1st_half = np.nanmean(
                responses_cue[total_neurons, 1:n_delays, :min_half_trials], axis=1)
            mean_responses_delay_before_2nd_half = np.nanmean(
                responses_cue[total_neurons, 1:n_delays, min_half_trials:min_trials], axis=1)

        delays_after = np.array(delays_after)
        popt_before, pcov_before = curve_fit(exponential, delays_after, mean_responses_delay_before, gtol=1e-20,
                                             maxfev=1000000)
        gain_before = popt_before[0]
        all_estimated_gain_before.append(gain_before)
        all_estimated_gamma_before.append(np.exp(-popt_before[-1]))

        popt_before_1st_half, pcov_before_1st_half = curve_fit(exponential, delays_after,
                                                               mean_responses_delay_before_1st_half, gtol=1e-20,
                                                               maxfev=1000000)
        gain_before_1st_half = popt_before_1st_half[0]
        all_estimated_gamma_before_1st_half.append(np.exp(-popt_before_1st_half[-1]))

        popt_before_2nd_half, pcov_before_2nd_half = curve_fit(exponential, delays_after,
                                                               mean_responses_delay_before_2nd_half, gtol=1e-20,
                                                               maxfev=1000000)
        gain_before_2nd_half = popt_before_2nd_half[0]
        all_estimated_gamma_before_2nd_half.append(np.exp(-popt_before_2nd_half[-1]))

        mean_responses_delay_after = np.nanmean(responses_cue_after[total_neurons, :n_delays - 1, :], axis=1)
        popt_after, pcov_after = curve_fit(exponential, delays_after, mean_responses_delay_after, gtol=1e-20,
                                           maxfev=1000000)
        gain_after = popt_after[0]
        all_estimated_gain_after.append(gain_after)
        all_estimated_gamma_after.append(np.exp(-popt_after[-1]))

        # Variable amount trials
        amounts_unique = np.sort(dataframe_behavior_neuron.loc[dataframe_behavior_neuron[
                                                                   'Distribution reward ID'] == 2, 'Amount reward'].unique())
        sum_samples = 0
        for i_a, a in enumerate(amounts_unique):
            flag = reduce(np.intersect1d, (
            np.where(distribution_id_trials == 2)[0], np.where(isRewarded_trials == 1)[0],
            np.where(amount_reward_trials == a)[0]))
            delta_fr_reward_a = delta_fr_reward[flag]
            delta_fr_reward_a = delta_fr_reward_a - baseline[flag]
            n_samples = len(flag)
            responses_reward[total_neurons, i_a, :n_samples] = delta_fr_reward_a

            delta_fr_cue_a = delta_fr_cue[flag]
            responses_cue_bimodal[total_neurons, sum_samples:sum_samples + n_samples] = delta_fr_cue_a
            responses_cue_bimodal_mat[total_neurons, i_a, :n_samples] = delta_fr_cue_a
            sum_samples += n_samples

            psth_reward[total_neurons, i_a, :, :n_samples] = psth_reward_neuron[flag, :].T

        # Estimate reversal points, taus, slope positive and slope negative
        estimated_expectile, estimated_tau, popt_neg_rew, popt_pos_rew, con = get_estimated_expectile(amounts_unique,
                                                                                                      responses_reward[
                                                                                                      total_neurons, :,
                                                                                                      :])
        all_estimated_taus.append(estimated_tau)
        all_estimated_reversals.append(estimated_expectile)
        slope_positive.append(popt_pos_rew)
        slope_negative.append(popt_neg_rew)
        constant_slope.append(con)

        total_neurons += 1
        all_animals.append(animal)
        all_sessions.append(session)
        all_neurons_id.append(neuron_id)


# Data frame with estimated tuning functions for each neuron
column_names = ["Neuron id", "Animal", "Session", "Gamma", "Gamma estimated excluding delay=0", "Is take long",
                "Gamma 1st half before time manipulation", "Gamma 2nd half before time manipulation",
                "Gamma before time manipulation", "Gamma after time manipulation", "Gain",
                "Gain estimated excluding delay=0", "Reversals", "Taus", "Positive slope", "Negative slope"]
info_neurons = np.column_stack((all_neurons_id, all_animals, all_sessions, all_estimated_gamma,
                                all_estimated_gamma_not_zero, is_take_long, all_estimated_gamma_before_1st_half,
                                all_estimated_gamma_before_2nd_half, all_estimated_gamma_before,
                                all_estimated_gamma_after, all_estimated_gain, gain_not_zero, all_estimated_reversals,
                                all_estimated_taus, slope_positive, slope_negative))
df = pd.DataFrame(info_neurons, columns=column_names)

# Directory where to solve parsed data
dir_save="/Users/margaridasousa/Desktop/Data_repository_paper/Parsed_data_putative_DA"
#df.to_csv(dir_save+r'/dataframe_neurons_info.csv',index=False,header=True, sep=',')

# Tables where each line is a single neuron
responses_cue = responses_cue[:total_neurons, :, :]
responses_reward = responses_reward[:total_neurons, :, :]
responses_reward_all = responses_reward_all[:total_neurons, :, :]
responses_reward_all_not_baseline_corrected = responses_reward_all_not_baseline_corrected[:total_neurons, :, :]
responses_cue_bimodal = responses_cue_bimodal[:total_neurons, :]
responses_reward_certain = responses_reward_certain[:total_neurons, :]
responses_cue_bimodal_mat = responses_cue_bimodal_mat[:total_neurons, :, :]

psth_cue = psth_cue[:total_neurons, :, :, :]
psth_reward = psth_reward[:total_neurons, :, :, :]

all_neurons_n_delays = np.array(all_neurons_n_delays)
n_neurons_less_delays = len(np.where(all_neurons_n_delays < 4)[0])

# Correct for neurons that only had 3 instead of 4 delays
responses_cue[:n_neurons_less_delays, 3, :] = responses_cue[:n_neurons_less_delays, 2, :]
responses_cue[:n_neurons_less_delays, 2, :] = responses_cue[:n_neurons_less_delays, 1, :]
responses_cue[:n_neurons_less_delays, 1, :] = responses_cue[:n_neurons_less_delays, 0, :]
responses_cue[:n_neurons_less_delays, 0, :] = 'nan'

responses_reward_all[:n_neurons_less_delays, 3, :] = responses_reward_all[:n_neurons_less_delays, 2, :]
responses_reward_all[:n_neurons_less_delays, 2, :] = responses_reward_all[:n_neurons_less_delays, 1, :]
responses_reward_all[:n_neurons_less_delays, 1, :] = responses_reward_all[:n_neurons_less_delays, 0, :]
responses_reward_all[:n_neurons_less_delays, 0, :] = 'nan'

responses_reward_all_not_baseline_corrected[:n_neurons_less_delays, 3, :] = responses_reward_all_not_baseline_corrected[
                                                                            :n_neurons_less_delays, 2, :]
responses_reward_all_not_baseline_corrected[:n_neurons_less_delays, 2, :] = responses_reward_all_not_baseline_corrected[
                                                                            :n_neurons_less_delays, 1, :]
responses_reward_all_not_baseline_corrected[:n_neurons_less_delays, 1, :] = responses_reward_all_not_baseline_corrected[
                                                                            :n_neurons_less_delays, 0, :]
responses_reward_all_not_baseline_corrected[:n_neurons_less_delays, 0, :] = 'nan'

psth_cue[:n_neurons_less_delays, 3, :] = psth_cue[:n_neurons_less_delays, 2, :]
psth_cue[:n_neurons_less_delays, 2, :] = psth_cue[:n_neurons_less_delays, 1, :]
psth_cue[:n_neurons_less_delays, 1, :] = psth_cue[:n_neurons_less_delays, 0, :]
psth_cue[:n_neurons_less_delays, 0, :] = 'nan'


# np.save(dir_save + r"/responses_cue_different_delays_constant_magnitude.npy", responses_cue)
# np.save(dir_save + r"/responses_reward_different_magnitudes_constant_delay.npy", responses_reward)
# np.save(dir_save + r"/responses_reward_different_delays_constant_magnitude.npy", responses_reward_all)
# np.save(dir_save + r"/responses_reward_different_delays_constant_magnitude_not_baseline_corrected.npy", responses_reward_all_not_baseline_corrected)
# np.savetxt(dir_save + r"/responses_cue_different_magnitudes_constant_delay.csv", responses_cue_bimodal)
# np.save(dir_save + r"/responses_cue_different_magnitudes_constant_delay_per_amount.npy", responses_cue_bimodal_mat)
# np.save(dir_save + r"/responses_cue_after_time_manipulation.npy", responses_cue_after)
# np.savetxt(dir_save + r"/responses_reward_certain_magnitude_3s_delay.csv", responses_reward_certain)
# np.save(dir_save + r"/psth_reward.npy", psth_reward) # PSTH aligned to reward delivery for constant delay (3s) and different reward magnitudes
# np.save(dir_save + r"/psth_cue.npy", psth_cue) # PSTH aligned to cue delivery for constant magnitude (4.5ul) and different reward delays (0s, 1.5s, 3s, 6s)


pdb.set_trace()

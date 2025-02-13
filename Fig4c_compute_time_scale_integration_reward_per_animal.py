from aux_functions import *
import matplotlib as mpl
import pandas as pd
import os

nan = float('nan')

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"
directory_raw_data = os.path.join(directory, "Raw")

# Get pupil diameter PSTH
data_frame_pupil_area = pd.read_csv(os.path.join(directory_raw_data, "Pupil_diameter.csv"))
data_frame_pupil_area.Animal = data_frame_pupil_area.Animal.astype(str)
data_frame_pupil_area.Session = data_frame_pupil_area.Session.astype(str)
data_frame_pupil_area["PSTH_pupil_diameter_aligned_to_cue"] = data_frame_pupil_area.loc[:,
                                                              'PSTH pupil diameter bin 0':'PSTH pupil diameter bin 1559'].values.tolist()
animals = np.unique(data_frame_pupil_area['Animal'])
data_frame_pupil_area.Animal = data_frame_pupil_area.Animal.astype(str)
data_frame_pupil_area.Session = data_frame_pupil_area.Session.astype(str)


# Time bins for PSTH of pupil diameter
FPS = 120
time_left = -2
time_right = 11
n_frames_pupil = (time_right - time_left) * FPS
time_pupil = np.arange(time_left, time_right, (time_right - time_left) / n_frames_pupil)
bin_0 = np.where(time_pupil >= 0)[0][0]
bin_1 = np.where(time_pupil >= 1)[0][0]
bin_1_5 = np.where(time_pupil >= 1.5)[0][0]
bin_3 = np.where(time_pupil >= 3)[0][0]
bin_5 = np.where(time_pupil >= 5)[0][0]
bin_10 = np.where(time_pupil >= 10)[0][0]

# What is the time-scale of integration of reward magnitudes?
means = np.arange(1, 8, step=0.1)  # Range of values considered
kerneltype = "exponential"

n_quantiles = 3  # Number of quantiles of pupil diameter to estimate
taus_quantiles = np.arange(0, n_quantiles + 1) / n_quantiles
cmap = mpl.cm.get_cmap('cool', 12)
colors = cmap(np.arange(0, 1, 1.0 / n_quantiles))

# Difference between quantiles of pupil diameter for different time-scales to integrate rewards
difference_pupil_diameter_1st_2nd = np.zeros((len(animals), len(means)))
difference_pupil_diameter_2nd_3rd = np.zeros((len(animals), len(means)))
difference_pupil_diameter_1st_3rd = np.zeros((len(animals), len(means)))
difference_pupil_diameter = np.empty((len(animals), len(means)))
difference_pupil_diameter[:, :] = 0

for i_m, m in enumerate(means):

    psth_pupil = np.empty((n_quantiles, 6, n_frames_pupil))
    psth_pupil[:, :, :] = np.nan

    i_session_overall = 0

    for i_animal, animal in enumerate(animals):

        prev_session = -1
        i_session = 0

        psth_pupil_animal = np.empty((10000, n_frames_pupil))
        psth_pupil_animal[:, :] = np.nan

        reward_history_animal = np.empty(10000)
        reward_history_animal[:] = np.nan

        counts_trials_quantile = np.zeros(n_quantiles)
        counts_trials = 0

        # Go through all sessions
        for date in np.unique(data_frame_pupil_area[(data_frame_pupil_area.Animal == animal)]["Session"]):

            if len(data_frame_pupil_area[
                       (data_frame_pupil_area.Animal == animal) & (data_frame_pupil_area.Session == date)][
                       "PSTH_pupil_diameter_aligned_to_cue"].values) == 0:
                continue

            data_frame_pupil_area_session = data_frame_pupil_area[
                (data_frame_pupil_area.Animal == animal) & (data_frame_pupil_area.Session == date)]

            # Variable reward magnitudes trials
            variable_amount_trials = np.intersect1d(np.where(data_frame_pupil_area_session["Is rewarded"] == 1)[0],
                                                    np.where(
                                                        data_frame_pupil_area_session["Distribution reward ID"] == 2)[
                                                        0])

            reward_amounts = data_frame_pupil_area_session["Amount reward"].values
            reward_amounts = reward_amounts[variable_amount_trials]
            unique_reward_amounts = np.unique(reward_amounts)

            moving_average_amounts = moving_average(reward_amounts, m, 1, kerneltype)

            psth_pupil_session = data_frame_pupil_area_session["PSTH_pupil_diameter_aligned_to_cue"].values
            psth_pupil_session = np.array(psth_pupil_session.tolist())

            max_trial_number = len(variable_amount_trials) - 1  # Take out last trial

            # Use reward history at the previous trial, to condition on current trial
            psth_pupil_animal[counts_trials:counts_trials + max_trial_number, :] = psth_pupil_session[
                                                                                   variable_amount_trials[1:], :]
            reward_history_animal[counts_trials:counts_trials + max_trial_number] = moving_average_amounts[:-1]

            counts_trials += max_trial_number
            i_session += 1

        # Compute the peak pupil diameter using the half-peak window
        mean_over_all = np.nanmean(psth_pupil_animal[:, :], axis=0)
        peak = np.argmax(mean_over_all)
        half_peak = np.where(np.round(mean_over_all, 3) == np.round(mean_over_all[peak] * 0.5, 3))[0]
        dif = (half_peak - peak)
        order = np.argsort(dif)
        window = [np.min(half_peak), np.max(half_peak)]
        number_trials = np.sum(~np.isnan(reward_history_animal))
        mean_pupil_window = np.nanmean(psth_pupil_animal[:number_trials, window[0]:window[1]], axis=1)
        reward_history_animal = reward_history_animal[:number_trials]

        # Pupil diameter conditioned on quantiles of reward history
        mean_quantiles = []
        quantiles = np.quantile(reward_history_animal, taus_quantiles)
        for i_quantile, quantile in enumerate(quantiles[:-1]):
            trials_quantile = np.intersect1d(np.where(reward_history_animal >= quantiles[i_quantile])[0],
                                             np.where(reward_history_animal <= quantiles[i_quantile + 1])[0])
            trials_quantile = trials_quantile[trials_quantile < len(reward_history_animal)]
            mean = np.nanmean(psth_pupil_animal[trials_quantile, :], axis=0)

            if i_quantile == 0:
                difference_pupil_diameter[i_animal, i_m] += -np.nanmean(mean[window[0]:window[1]])
                difference_pupil_diameter_1st_2nd[i_animal, i_m] += -np.nanmean(mean[window[0]:window[1]])
            if i_quantile == 1:
                difference_pupil_diameter_1st_2nd[i_animal, i_m] += np.nanmean(mean[window[0]:window[1]])
                difference_pupil_diameter_2nd_3rd[i_animal, i_m] += -np.nanmean(mean[window[0]:window[1]])
            if i_quantile == n_quantiles - 1:
                difference_pupil_diameter[i_animal, i_m] += np.nanmean(mean[window[0]:window[1]])
                difference_pupil_diameter_2nd_3rd[i_animal, i_m] += np.nanmean(mean[window[0]:window[1]])

difference_pupil_diameter_1st_2nd[difference_pupil_diameter_1st_2nd < 0] = 0
difference_pupil_diameter_2nd_3rd[difference_pupil_diameter_2nd_3rd < 0] = 0

# Compute time-scale for each animal
mean_animals = []
for i_animal, animal in enumerate(animals):
    pos = np.argmax(difference_pupil_diameter_1st_2nd[i_animal, :] / np.max(
        difference_pupil_diameter_1st_2nd[i_animal, :]) + difference_pupil_diameter_2nd_3rd[i_animal, :] / np.max(
        difference_pupil_diameter_2nd_3rd[i_animal, :]))
    # pos_old=np.argmax(difference_pupil_diameter_1st_2nd[i_animal,:]+difference_pupil_diameter_2nd_3rd[i_animal,:])
    mean_animals.append(means[pos])
print(mean_animals)
pdb.set_trace()

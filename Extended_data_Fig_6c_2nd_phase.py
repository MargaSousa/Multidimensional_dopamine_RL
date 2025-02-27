import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special

mpl.use('TkAgg')
import numpy as np
from aux_functions import *
from scipy import stats
from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.stats import gaussian_kde
import seaborn as sns

length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 1.75  # 2
vertical_size = 1.75  # 2
font_size = 11
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 10
import matplotlib.colors as mcol
import os

flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [reward_cmap[1], reward_cmap[-1]])
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from scipy.stats import shapiro, kstest
import itertools
import pandas
import pandas as pd

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

# For the Laplace decoder for future reward time
n_time = 50
time = np.linspace(0, 7.5, n_time)
cue_times = np.array([0, 1.5, 3, 6])
alpha_time = 0.2  # Smoothing parameter

# Relationship between firing rate at the cue and reversal points
population_responses_variable_second_phase = np.nanmean(responses_cue_bimodal_second_phase, axis=1)
population_responses_variable_corrected_second_phase = population_responses_variable_second_phase / (
            gains * discount ** 3)
well_estimated_reversals = np.intersect1d(np.where(estimated_reversals > 1.1)[0],
                                          np.where(estimated_reversals < 7.9)[0])
reg_variable = HuberRegressor().fit(
    (population_responses_variable_corrected_second_phase[well_estimated_reversals]).reshape(-1, 1),
    estimated_reversals[well_estimated_reversals])

# Mapping from reversals to taus at cue
corrected_taus = estimated_taus[well_estimated_reversals] - bias_taus[well_estimated_reversals]
corrected_taus[corrected_taus < 0] = 0
corrected_taus[corrected_taus > 1] = 1
w_reversals = 1.0 / (1 + variance_tau[well_estimated_reversals])
reversal_estimated_at_cue = reg_variable.predict(
    population_responses_variable_corrected_second_phase[well_estimated_reversals].reshape(-1, 1))
iso_reg_cue = IsotonicRegression(increasing=True).fit(reversal_estimated_at_cue, corrected_taus,
                                                      sample_weight=w_reversals)
pred_cue = iso_reg_cue.predict(reversal_estimated_at_cue)

# Compute variance over all cues
variance = []
for neu in range(n_neurons):
    all_trials = np.concatenate((np.ndarray.flatten(responses_cue_bimodal_second_phase[neu, :]),
                                 np.ndarray.flatten(responses_cue_second_phase[neu, :, :])))
    n_trials = np.sum(~np.isnan(all_trials))
    variance.append(np.var(all_trials[~np.isnan(all_trials)]))
variance = np.array(variance)

# Colors for plots
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))[:, :3]

# True pdf in amount and time for variable cue
n_amount = 80
smooth_magnitude = 0.35
amount = np.linspace(0, 9, n_amount)

# Save joint pdf for 4 certain cues
joint_pdf_certain_reward = np.empty((n_time, n_amount, 4))
const = 0.01  # So we can compute the KL divergence
n_runs_2d_decoder = 1
is_shuffle = False

for run in range(n_runs_2d_decoder):

    # Decode reward time  for all cues that predict a certain reward magnitude at different delays
    for cue in range(1, 4):

        population_responses = np.nanmean(responses_cue_second_phase[:, cue, :], axis=1)
        n_neurons = np.sum(np.isnan(population_responses))  # Some neurons were not recorded in all delays

        # Decode reward times
        if is_shuffle:
            neurons_shuffled = np.arange(len(discount[n_neurons:]))
            np.random.shuffle(neurons_shuffled)
            pdf_time = run_decoding_time(time, discount[n_neurons:][neurons_shuffled], variance,
                                         population_responses[n_neurons:] / (gains[n_neurons:]),
                                         alpha_time)  # estimated_reversals[n_neurons:]*
        else:
            pdf_time = run_decoding_time(time, discount[n_neurons:], variance,
                                         population_responses[n_neurons:] / (gains[n_neurons:]),
                                         alpha_time)  # estimated_reversals[n_neurons:]*

        estimate_reward_time = np.sum(time * pdf_time)
        # Decode reward amount
        population_responses_corrected = population_responses / (
                    gains * discount ** estimate_reward_time)  # **estimate_reward_time)
        estimated_reversals_certain = reg_variable.predict(
            (population_responses_corrected[well_estimated_reversals][n_neurons:]).reshape(-1, 1))

        if is_shuffle:
            neurons_shuffled = np.arange(len(estimated_reversals_certain))
            np.random.shuffle(neurons_shuffled)  # [neurons_shuffled]
            samples, _ = run_decoding_magnitude(estimated_reversals_certain[neurons_shuffled], pred_cue[n_neurons:],
                                                np.ones(len(estimated_reversals)), minv=0, maxv=9, N=20,
                                                max_samples=2000, max_epochs=15, method='TNC')
        else:
            samples, _ = run_decoding_magnitude(estimated_reversals_certain, pred_cue[n_neurons:],
                                                np.ones(len(estimated_reversals)), minv=0, maxv=9, N=20,
                                                max_samples=2000, max_epochs=15, method='TNC')

        # Smooth
        kde = gaussian_kde(samples, bw_method=smooth_magnitude)
        pdf_amount = kde.pdf(amount)
        pdf_amount = pdf_amount / np.sum(pdf_amount)

        # Compute joint pdf over reward amount and time
        for i_t, i_a in itertools.product(np.arange(n_time), np.arange(n_amount)):
            joint_pdf_certain_reward[i_t, i_a, cue] = pdf_time[n_time - i_t - 1] * pdf_amount[i_a]
        joint_pdf_certain_reward[:, :, cue] = joint_pdf_certain_reward[:, :, cue] / np.sum(
            joint_pdf_certain_reward[:, :, cue])
        joint_pdf_certain_reward[:, :, cue] += np.nanmax(joint_pdf_certain_reward[:, :, cue]) * const
        joint_pdf_certain_reward[:, :, cue] = joint_pdf_certain_reward[:, :, cue] / np.sum(
            joint_pdf_certain_reward[:, :, cue])

    # Decode time for variable cue
    population_responses_variable = np.nanmean(responses_cue_bimodal_second_phase, axis=1)
    mean_responses_variable_cue_discount = population_responses_variable / gains
    variance_variable = np.nanvar(responses_cue_bimodal_second_phase, axis=1)

    if is_shuffle:
        neurons_shuffled = np.arange(len(discount[n_neurons:]))
        np.random.shuffle(neurons_shuffled)
        pdf_time_variable = run_decoding_time(time, discount[neurons_shuffled], variance_variable,
                                              mean_responses_variable_cue_discount, alpha_time)
    else:
        pdf_time_variable = run_decoding_time(time, discount, variance, mean_responses_variable_cue_discount,
                                              alpha_time)

    estimate_reward_time = np.sum(time * pdf_time_variable)  # time[np.argmax(pdf_time)]
    population_responses_variable_corrected = population_responses_variable / (
                gains * discount ** estimate_reward_time)  # estimate_reward_time
    estimated_reversals_variable = reg_variable.predict(
        (population_responses_variable_corrected[well_estimated_reversals][n_neurons:]).reshape(-1, 1))

    # Decode magnitude for variable cue
    if is_shuffle:
        neurons_shuffled = np.arange(len(estimated_reversals_variable))
        np.random.shuffle(neurons_shuffled)  # [neurons_shuffled],
        samples, _ = run_decoding_magnitude(estimated_reversals_variable[neurons_shuffled], pred_cue,
                                            np.ones(len(reversal_estimated_at_cue)), minv=1, maxv=8, N=20,
                                            max_samples=2000, max_epochs=15, method='TNC')
    else:
        samples, _ = run_decoding_magnitude(estimated_reversals_variable, pred_cue,
                                            np.ones(len(reversal_estimated_at_cue)), minv=1, maxv=8, N=20,
                                            max_samples=2000, max_epochs=15, method='TNC')

    # Smooth
    kde = gaussian_kde(samples, bw_method=smooth_magnitude)
    pdf_amount = kde.pdf(amount)
    pdf_amount_cue = pdf_amount / np.sum(pdf_amount)

    # Save joint pdf for variable cue
    joint_pdf_variable_reward = np.zeros((n_time, n_amount))
    for i_t, i_a in itertools.product(np.arange(n_time), np.arange(n_amount)):
        joint_pdf_variable_reward[i_t, i_a] = pdf_time_variable[n_time - i_t - 1] * pdf_amount_cue[i_a]
    joint_pdf_variable_reward = joint_pdf_variable_reward / np.sum(joint_pdf_variable_reward)
    joint_pdf_variable_reward += np.max(joint_pdf_variable_reward) * const
    joint_pdf_variable_reward = joint_pdf_variable_reward / np.sum(joint_pdf_variable_reward)

# Plot stacked heat map
color_map = sns.color_palette("coolwarm", as_cmap=True)
mesh = np.meshgrid(amount, time[::-1])
fig, ax = plt.subplots(figsize=(horizontal_size * 2, vertical_size * 2.5), subplot_kw={"projection": "3d"})
fig.set_facecolor('w')
ax.set_facecolor('w')
ax.view_init(elev=-150, azim=50)
ax.set_box_aspect((1, 1, (2.3)))  # *(3.0/4)

# Certain cues
for i_d in np.arange(1, 4):
    map = joint_pdf_certain_reward[:, :, i_d]
    scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(map), np.max(map)), cmap=color_map)
    ax.plot_surface(mesh[0], mesh[1], -0.01 * i_d + 0 * map, facecolors=scam.to_rgba(map), antialiased=True, rstride=1,
                    cstride=1, alpha=None, shade=False)

# Variable cue
scam = plt.cm.ScalarMappable(
    norm=mpl.colors.Normalize(np.min(joint_pdf_variable_reward), np.max(joint_pdf_variable_reward)), cmap=color_map)
ax.plot_surface(mesh[0], mesh[1], -0.04 + 0 * joint_pdf_variable_reward,
                facecolors=scam.to_rgba(joint_pdf_variable_reward), antialiased=True, rstride=1, cstride=1, alpha=None,
                shade=False)

ax.set_ylabel("Time\n since cue (s)")
ax.set_xlabel("Magnitude (" + r"$\mu$" + "l)")
ax.set_zticks([])
ax.set_yticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
ax.set_xticks([1, 4.5, 8], ["1", "4.5", "8"])  #
fig.tight_layout()
plt.show()
pdb.set_trace()

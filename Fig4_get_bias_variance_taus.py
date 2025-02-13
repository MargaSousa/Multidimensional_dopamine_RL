from aux_functions import *
from scipy.stats import gaussian_kde
import random
import os
import pandas as pd

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get estimated tuning for each neuron
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))
estimated_reversals = data_frame_neurons_info['Reversals'].values
estimated_taus = data_frame_neurons_info['Taus'].values
responses_reward = np.load(
    os.path.join(directory_parsed_data, "responses_reward_different_magnitudes_constant_delay.npy"))
max_std = np.nanstd(responses_reward, axis=2)

N_neurons = len(estimated_reversals)
n_trials = 100
n_bootstraps = 1000
n_trials_bootstrap = 20
estimated_tau_all_neurons = np.empty((N_neurons, n_bootstraps))
estimated_tau_all_neurons[:, :] = np.nan
all_taus_neurons = []

# True probability distribution given in the experiment
amount = np.array([1, 2.75, 4.5, 6.25, 8])
n_amounts = len(amount)
probability = np.array([0.25, 0.167, 0.167, 0.167, 0.25])
probability = probability / np.sum(probability)
n_bins = 80
smooth = 0.3
x = np.linspace(0, 9, n_bins)
all_samples = np.random.choice(amount, 10000, p=probability)
kde = gaussian_kde(all_samples, bw_method=smooth)
y = kde.pdf(x)
y = y / np.sum(y)

# Get expectiles
N = 1000
taus = np.linspace(1.0 / N, 1 - 1.0 / N, N)
_, expectiles = get_expectiles(x, y, taus)

# Monte Carlo simulations to estimate the bias of the estimation of taus
for i_neuron in range(N_neurons):
    responses_neuron = np.empty(shape=(n_amounts, n_trials))
    responses_neuron[:, :] = np.nan

    pos_neuron = np.where(expectiles >= estimated_reversals[i_neuron])[0][0]
    tau_neuron = taus[pos_neuron]
    all_taus_neurons.append(tau_neuron)

    for i_a, a in enumerate(amount):
        scale = max_std[i_neuron, i_a]
        if a >= expectiles[i_neuron]:
            responses_neuron[i_a, :] = np.random.normal(loc=(a - estimated_reversals[i_neuron]) * tau_neuron,
                                                        scale=scale, size=100)
        else:
            responses_neuron[i_a, :] = np.random.normal(loc=(a - estimated_reversals[i_neuron]) * (1 - tau_neuron),
                                                        scale=scale, size=100)

    # Estimate reversal
    for b in range(n_bootstraps):
        selected_trials = random.sample(range(n_trials), 20)
        estimated_expectile, estimated_tau, slope_pos, slope_neg, _ = get_estimated_expectile(amount,
                                                                                              responses_neuron[:,
                                                                                              selected_trials])
        estimated_tau_all_neurons[i_neuron, b] = estimated_tau

all_taus_neurons = np.array(all_taus_neurons)
var_tau = np.nanvar(estimated_tau_all_neurons, axis=1)
bias_taus = np.nanmean(estimated_tau_all_neurons, axis=1) - all_taus_neurons

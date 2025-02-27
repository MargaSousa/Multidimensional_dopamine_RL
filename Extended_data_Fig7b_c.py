import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

from aux_functions import *

length_ticks = 3
font_size = 11
linewidth = 1.2
scatter_size = 2
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
horizontal_size = 2
vertical_size = 2
import matplotlib.colors as mcol

flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [reward_cmap[1], reward_cmap[-1]])
from scipy.optimize import curve_fit
import random

# Reward magnitudes and probabilities given in the experiment
amount = np.array([1, 2.75, 4.5, 6.25, 8])
n_amounts = len(amount)
probability = np.array([0.25, 0.167, 0.167, 0.167, 0.25])
probability = probability / np.sum(probability)

# Smooth distribution
n_bins = 80
smooth = 0.3
x = np.linspace(0, 9, n_bins)
all_samples = np.random.choice(amount, 10000, p=probability)
kde = gaussian_kde(all_samples, bw_method=smooth)
y = kde.pdf(x)
y = y / np.sum(y)
plt.plot(x, y)
plt.show()

N = 50
taus = np.linspace(1.0 / N, 1 - 1.0 / N, N)
_, expectiles = get_expectiles(x, y, taus)

n_trials = 100
n_bootstraps = 1000
n_trials_bootstrap = 20

estimated_expectile_all_neurons = np.empty((N, n_bootstraps))
estimated_expectile_all_neurons[:, :] = np.nan

estimated_slope_pos_all_neurons = np.empty((N, n_bootstraps))
estimated_slope_pos_all_neurons[:, :] = np.nan

estimated_slope_neg_all_neurons = np.empty((N, n_bootstraps))
estimated_slope_neg_all_neurons[:, :] = np.nan

estimated_tau_all_neurons = np.empty((N, n_bootstraps))
estimated_tau_all_neurons[:, :] = np.nan

for i_neuron in range(N):

    # Generate responses for each reward magnitude
    responses_neuron = np.empty(shape=(n_amounts, n_trials))
    responses_neuron[:, :] = np.nan
    for i_a, a in enumerate(amount):
        if a >= expectiles[i_neuron]:
            responses_neuron[i_a, :] = np.random.normal(loc=(a - expectiles[i_neuron]) * taus[i_neuron] * 12, scale=50,
                                                        size=100)
        else:
            responses_neuron[i_a, :] = np.random.normal(loc=(a - expectiles[i_neuron]) * (1 - taus[i_neuron]) * 12,
                                                        scale=50, size=100)

    # Estimate reversal
    for b in range(n_bootstraps):
        selected_trials = random.sample(range(n_trials), 20)
        estimated_expectile, estimated_tau, slope_pos, slope_neg, _ = get_estimated_expectile(amount,
                                                                                              responses_neuron[:,
                                                                                              selected_trials])
        estimated_expectile_all_neurons[i_neuron, b] = estimated_expectile
        estimated_tau_all_neurons[i_neuron, b] = estimated_tau
        estimated_slope_pos_all_neurons[i_neuron, b] = slope_pos
        estimated_slope_neg_all_neurons[i_neuron, b] = slope_neg

var_expectile = np.nanvar(estimated_expectile_all_neurons, axis=1)
var_tau = np.nanvar(estimated_tau_all_neurons, axis=1)
bias_expectile = np.nanmean(estimated_expectile_all_neurons, axis=1) - expectiles
bias_taus = np.nanmean(estimated_tau_all_neurons, axis=1) - taus
bias_pos_slope = np.nanmean(estimated_slope_pos_all_neurons, axis=1) - taus
bias_neg_slope = np.nanmean(estimated_slope_pos_all_neurons, axis=1) - (1 - taus)


def quadratic_function(x, a, b, c):
    return a + b * (x - c) ** 2


fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
plt.plot(expectiles, var_tau)
popt, pcov = curve_fit(quadratic_function, expectiles, var_tau)
plt.plot(expectiles, quadratic_function(expectiles, *popt), color="black")
plt.xlabel("Reversal")
plt.ylabel("Variance in the\nestimation of " + r"$\tau$")
plt.show()

reg_reversal_bias = LinearRegression().fit(expectiles.reshape(-1, 1), bias_expectile)
pred_reversal_bias = reg_reversal_bias.predict(expectiles.reshape(-1, 1))
reg_reversal_bias_tau = LinearRegression().fit(expectiles.reshape(-1, 1), bias_taus)
pred_reversal_bias_tau = reg_reversal_bias_tau.predict(expectiles.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))  #
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
plt.plot(expectiles, bias_taus)
plt.plot(expectiles, pred_reversal_bias_tau, color="black")
plt.xlabel("Reversal")
plt.ylabel("Bias in the estimation of " + r"$\tau$")
plt.show()

pdb.set_trace()

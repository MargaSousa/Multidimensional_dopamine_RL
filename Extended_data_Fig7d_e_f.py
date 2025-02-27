import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor

from aux_functions import *

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
from sklearn.isotonic import IsotonicRegression

# Select neurons (either "DA" or "putative_DA")
# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "Putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))

# Get estimated tuning for reward time and magnitude
discount = data_frame_neurons_info['Gamma'].values
estimated_reversals = data_frame_neurons_info['Reversals'].values
estimated_taus = data_frame_neurons_info['Taus'].values
gains = data_frame_neurons_info['Gain'].values
n_neurons = len(discount)
bias_taus = data_frame_neurons_info[
    'Bias in the estimation of tau'].values  # Generated from Fig4_get_bias_variance_estimation_tau
variance_tau = data_frame_neurons_info[
    'Variance in the estimation of tau'].values  # Generated from Fig4_get_bias_variance_estimation_tau

# Get responses for certain and variable cues
responses_cue = np.load(os.path.join(directory_parsed_data,
                                     "responses_cue_different_delays_constant_magnitude.npy"))  # Responses at the cue for variable reward delays
responses_cue_bimodal = np.loadtxt(os.path.join(directory_parsed_data,
                                                "responses_cue_different_magnitudes_constant_delay.csv"))  # Responses at the cue for delay of 3s and variable magnitudes (neurons x trials)

cue_times = np.array([0, 1.5, 3, 6])

# Regression between firing rate at the cue and reversal points
population_responses_variable = np.nanmean(responses_cue_bimodal, axis=1)
population_responses_variable_corrected = population_responses_variable / (gains * discount ** cue_times[2])
well_estimated_reversals = np.intersect1d(np.where(estimated_reversals > 1.1)[0],
                                          np.where(estimated_reversals < 7.9)[0])
well_estimated_reversals = np.intersect1d(well_estimated_reversals, np.where(discount < 1.2)[0])
reg_variable = HuberRegressor().fit((population_responses_variable_corrected[well_estimated_reversals]).reshape(-1, 1),
                                    estimated_reversals[well_estimated_reversals])
y_example = reg_variable.predict(population_responses_variable_corrected[well_estimated_reversals].reshape(-1, 1))
plt.scatter(population_responses_variable_corrected[well_estimated_reversals],
            estimated_reversals[well_estimated_reversals])
plt.plot(population_responses_variable_corrected[well_estimated_reversals], y_example)
plt.show()

# Mapping from reversals to taus at cue
corrected_taus = estimated_taus[well_estimated_reversals] - bias_taus[well_estimated_reversals]
corrected_taus[corrected_taus < 0] = 0
corrected_taus[corrected_taus > 1] = 1
w_reversals = 1.0 / (1 + variance_tau[well_estimated_reversals])
reversal_estimated_at_cue = reg_variable.predict(
    population_responses_variable_corrected[well_estimated_reversals].reshape(-1, 1))
iso_reg_cue = IsotonicRegression(increasing=True).fit(reversal_estimated_at_cue, corrected_taus,
                                                      sample_weight=w_reversals)
pred_cue = iso_reg_cue.predict(reversal_estimated_at_cue)

# Mapping from reversals to taus at reward
iso_reg_reward = IsotonicRegression(increasing=True).fit(estimated_reversals[well_estimated_reversals], corrected_taus,
                                                         sample_weight=w_reversals)
pred_reward = iso_reg_reward.predict(estimated_reversals[well_estimated_reversals])

# Extended Data Figure 7

# Reversal point and tau at cue
fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
scatter = plt.scatter(reversal_estimated_at_cue, corrected_taus, c=w_reversals, cmap="Purples")
order = np.argsort(reversal_estimated_at_cue)
# plt.plot(selected_reversals_variable_cue[order],fit_taus_cue_sigmoid[order],ls="--")
plt.plot(reversal_estimated_at_cue[order], pred_cue[order], color="k", ls="--", label="Isotonic regression")
plt.colorbar(scatter, label="Confidence", ticks=[])
plt.xlabel("Reversal point at cue")
plt.ylabel("Corrected " + r"$\tau$")
plt.legend()
# fig.savefig(save_dir+r"\reversal_tau_cue.svg")
plt.show()

# Reversal point and tau at reward
sort = np.argsort(estimated_reversals[well_estimated_reversals])
w_reversals = 1.0 / (1 + variance_tau[well_estimated_reversals])
fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
scatter = plt.scatter(estimated_reversals[well_estimated_reversals], corrected_taus, c=w_reversals, cmap="Blues")
plt.plot(estimated_reversals[well_estimated_reversals][sort], pred_reward[sort], color="k", ls="--",
         label="Isotonic regression")
plt.legend()
plt.colorbar(scatter, label="Confidence", ticks=[])
plt.xlabel("Reversal point at reward")
plt.ylabel("Corrected " + r"$\tau$")
# fig.savefig(save_dir+r"\reversal_tau_reward.svg")
plt.show()

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
from aux_functions import *
from sklearn.linear_model import LinearRegression
import cmasher as cmr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import os

# Parameters for paper plots
length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 1.8
vertical_size = 1.8
font_size = 11
labelsize = 8
legendsize = font_size

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size - 3
mpl.rcParams['ytick.labelsize'] = font_size - 3
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rc('xtick', labelsize=labelsize)
mpl.rc('ytick', labelsize=labelsize)
mpl.rc('legend', fontsize=legendsize)

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

# 1st phase responses
directory_parsed_data_first_phase = os.path.join(directory, "Parsed_data_" + type_neurons + "_first_phase")
responses_cue_first_phase = np.load(
    os.path.join(directory_parsed_data_first_phase, "responses_cue_different_delays_constant_magnitude.npy"))[
                            selected_neurons, :, :]
responses_cue_bimodal_first_phase = np.loadtxt(
    os.path.join(directory_parsed_data_first_phase, "responses_cue_different_magnitudes_constant_delay.csv"))[
                                    selected_neurons, :]

# Second phase responses considering a shorter window
directory_parsed_data_second_phase_shorter_window = os.path.join(directory,
                                                                 "Parsed_data_" + type_neurons + "_second_phase_shorter_window")
responses_cue_second_phase = np.load(os.path.join(directory_parsed_data_second_phase_shorter_window,
                                                  "responses_cue_different_delays_constant_magnitude.npy"))[
                             selected_neurons, :, :]
responses_cue_bimodal_second_phase = np.loadtxt(os.path.join(directory_parsed_data_second_phase_shorter_window,
                                                             "responses_cue_different_magnitudes_constant_delay.csv"))[
                                     selected_neurons, :]

n_neurons = responses_cue_second_phase.shape[0]
sorted_gammas = np.sort(discount)
discritization = (sorted_gammas - sorted_gammas[0]) / (sorted_gammas[-1] - sorted_gammas[0]) * 0.8
cmap = cmr.cm.apple(discritization)

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
axins = inset_axes(ax, width="30%", height="30%", loc='upper right')  # ,borderpad=3
axins.spines['left'].set_linewidth(linewidth)
axins.spines['bottom'].set_linewidth(linewidth)
axins.set_box_aspect(1)
axins.tick_params(width=linewidth, length=length_ticks)

all_delays = [0, 1.5, 3, 6]
slopes = []
for i_neuron, neuron in enumerate(np.argsort(discount)):

    # Mean responses over neurons
    mean_responses_first_phase = np.nanmean(responses_cue_first_phase[neuron, 1:, :], axis=1)
    mean_responses_second_phase = np.nanmean(responses_cue_second_phase[neuron, 1:, :], axis=1)

    non_nan_positions = np.argwhere(~np.isnan(mean_responses_first_phase))

    # Regression between second and first phase responses
    reg = LinearRegression().fit((mean_responses_second_phase[non_nan_positions]).reshape(-1, 1),
                                 mean_responses_first_phase[non_nan_positions])
    x = np.linspace(np.nanmin(mean_responses_second_phase[non_nan_positions]),
                    np.nanmax(mean_responses_second_phase[non_nan_positions]), 10)
    y = reg.predict(x.reshape(-1, 1))

    # So the plot is understandable
    if mean_responses_first_phase[-1] > 30:
        continue
    else:
        ax.scatter(mean_responses_second_phase, mean_responses_first_phase, color=cmap[i_neuron], alpha=0.25,
                   edgecolor="white")
        ax.plot(x, y, color=cmap[i_neuron])

    slopes.append(reg.coef_[0][0])

axins.hist(slopes, bins=np.linspace(-3, 3, 15), color="gray", alpha=0.7)
axins.set_xticks([-1, 0, 1])
axins.set_yticks([])
axins.set_xlabel("Slope")
axins.set_ylabel("Probability")
axins.axvline(x=np.mean(slopes), ls="--", color="k")
ax.set_xlabel("Mean firing rate\n2nd phase (sp/s)")
ax.set_ylabel("Mean firing rate\n1st phase (sp/s)")
plt.show()

pdb.set_trace()

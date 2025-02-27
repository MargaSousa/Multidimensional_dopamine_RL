import os
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt

from aux_functions import *

length_ticks = 3
font_size = 11
linewidth = 1.2
scatter_size = 2
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
horizontal_size = 2
vertical_size = 2

# Parameters for presentation plots
# length_ticks=10
# font_size=30
# linewidth=4
# scatter_size=100
# horizontal_size=10
# vertical_size=10
# labelpad_x=10
# labelpad_y=10
# labelsize=font_size
# legendsize=font_size
# scatter_size_fr=15
# capsize=8

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "Putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

responses_reward = np.load(
    os.path.join(directory_parsed_data, "responses_reward_different_magnitudes_constant_delay.npy"))
pdb.set_trace()
# Get estimated tuning for each neuron
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))
all_estimated_reversals = data_frame_neurons_info['Reversals'].values

# Neurons with well estimated reversal points
chosen_neurons = np.intersect1d(np.where(all_estimated_reversals < 7.9)[0], np.where(all_estimated_reversals > 1.1)[0])
mean_responses_reward = np.nanmean(responses_reward[chosen_neurons, :, :], axis=2)
sorted_reversals = np.argsort(all_estimated_reversals[chosen_neurons])
n_neurons = len(sorted_reversals)
winter = mpl.cm.get_cmap('winter', 12)
colors_amount = winter(np.linspace(0, 1, 5))
amounts_unique = np.array([1, 2.75, 4.5, 6.25, 8])

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
for n in range(n_neurons):
    plt.scatter(mean_responses_reward[sorted_reversals[n], :], n * np.ones(len(amounts_unique)), c=colors_amount,
                s=scatter_size)  # scatter_size

# Down weight endpoints for the interpolating spline
weights = np.ones(n_neurons)
weights[-1] = 0.5
weights[0] = 0.5
counts_neurons = np.arange(n_neurons)
smooth = 145.
xs = np.linspace(0, n_neurons - 1, 100)

save_responses = []
save_magnitudes = []
save_reversals = []
for i_amount, amount in enumerate(amounts_unique):
    cs = scipy.interpolate.UnivariateSpline(counts_neurons, mean_responses_reward[:, i_amount][sorted_reversals],
                                            w=weights)
    cs.set_smoothing_factor(smooth)
    plt.plot(cs(xs), xs, c=colors_amount[i_amount], zorder=3, alpha=1.)
    save_responses.extend(mean_responses_reward[:, i_amount][sorted_reversals])
    save_reversals.extend(all_estimated_reversals[sorted_reversals])
    save_magnitudes.extend([amount] * len(sorted_reversals))

plt.axvline(x=0, color="black", ls=":")
plt.ylabel("Neuron \n (sorted by reversal point)")
plt.xlabel(r"$\Delta$" + " Firing Rate (sp/s)")
# plt.yticks([0,n_neurons],[1,n_neurons])
plt.yticks([])
plt.xticks([-2.5, 0, 2.5, 5, 7.5], ["-2.5", "0", "2.5", "5", "7.5"])
plt.ylim([-1, n_neurons])
plt.show()

pdb.set_trace()

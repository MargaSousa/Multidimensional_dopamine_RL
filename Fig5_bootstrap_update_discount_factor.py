import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from aux_functions import *
mpl.use('TkAgg')

length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 2
vertical_size = 2
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

flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [reward_cmap[1], reward_cmap[-1]])

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get estimated tuning for reward time and magnitude
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))
gamma_before = data_frame_neurons_info['Gamma before time manipulation'].values
gamma_after = data_frame_neurons_info['Gamma after time manipulation'].values
is_take_long = data_frame_neurons_info['Is take long'].values
n_neurons = len(gamma_before)

# Adaptation in reward delay statistics
gamma_before_take_short = []
gamma_after_take_short = []
gamma_before_take_long = []
gamma_after_take_long = []

gamma_1st_half_take_short = []
gamma_2nd_half_take_short = []

gamma_1st_half_take_long = []
gamma_2nd_half_take_long = []

colors_take_short = []
colors_take_long = []
for i_neuron in range(n_neurons):
    take = is_take_long[i_neuron]
    if take == 0:
        gamma_before_take_short.append(gamma_before[i_neuron])
        gamma_after_take_short.append(gamma_after[i_neuron])


    else:
        gamma_before_take_long.append(gamma_before[i_neuron])
        gamma_after_take_long.append(gamma_after[i_neuron])

gamma_before_take_short = np.array(gamma_before_take_short)
gamma_before_take_long = np.array(gamma_before_take_long)
gamma_after_take_short = np.array(gamma_after_take_short)
gamma_after_take_long = np.array(gamma_after_take_long)

plt.hist(gamma_after_take_short - gamma_before_take_short)
plt.hist(gamma_after_take_long - gamma_before_take_long)
plt.show()

neurons_id = np.arange(n_neurons)
resamples = 10000
var_certain = []
var_variable = []
n_take_short = 0
n_take_long = 0
n_take_short_long = 0
dif_means = []
means_take_short = []
means_take_long = []
for r in range(resamples):
    new_gamma_before_take_short = resample(gamma_before_take_short, replace=True,
                                           n_samples=len(gamma_before_take_short))
    new_gamma_after_take_short = resample(gamma_after_take_short, replace=True, n_samples=len(gamma_after_take_short))
    new_gamma_before_take_long = resample(gamma_before_take_long, replace=True, n_samples=len(gamma_before_take_long))
    new_gamma_after_take_long = resample(gamma_after_take_long, replace=True, n_samples=len(gamma_after_take_long))

    mean_take_short = np.nanmean(new_gamma_after_take_short - new_gamma_before_take_short)
    mean_take_long = np.nanmean(new_gamma_after_take_long - new_gamma_before_take_long)
    means_take_short.append(mean_take_short)
    means_take_long.append(mean_take_long)
    if mean_take_short < 0:
        n_take_short += 1
    if mean_take_long > 0:
        n_take_long += 1
    if np.round(mean_take_short, 2) == np.round(mean_take_long, 2):
        n_take_short_long += 1

    dif_means.append(np.mean(mean_take_short - mean_take_long))

dif_means = np.array(dif_means)
means_take_long = np.array(means_take_long)
means_take_short = np.array(means_take_short)


plt.hist(means_take_short)
plt.hist(means_take_long)
plt.show()

quantile_25 = np.quantile(np.abs(dif_means), 0.025)
quantile_975 = np.quantile(np.abs(dif_means), 0.975)

print("mean abs", np.mean(np.abs(dif_means)))
print("2.5-quantile absolute difference in update: ", quantile_25)
print("97.5-quantile absolute difference in update: ", quantile_975)

quantile_5_take_short = np.quantile(means_take_short, 0.05)
quantile_1_take_short = np.max(means_take_short)

quantile_0_take_long = np.min(means_take_long)
quantile_95_take_long = np.quantile(means_take_long, 0.95)

print("5-quantile update take short: ", quantile_5_take_short)
print("1-quantile update take short: ", quantile_1_take_short)

print("0-quantile update take long: ", quantile_0_take_long)
print("0.95-quantile update take long: ", quantile_95_take_long)

plt.hist(dif_means)
plt.axvline(x=quantile_25)
plt.axvline(x=quantile_975)
plt.show()

print("p-value take short= " + str(n_take_short / resamples))
print("p-value take long= " + str(n_take_long / resamples))
print("p-value take short/long= " + str(n_take_short_long / resamples))

pdb.set_trace()

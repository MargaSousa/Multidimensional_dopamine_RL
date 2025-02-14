import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from aux_functions import *
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.colors as mcol
import seaborn as sns

flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [reward_cmap[1], reward_cmap[-1]])
import os

mpl.use('TkAgg')

# Parameters for paper plots
length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 1.5
vertical_size = 1.5
font_size = 8
labelsize = 8
legendsize = font_size

# Parameters for presentation plots
# length_ticks=10
# font_size=30
# linewidth=4
# scatter_size=500
# horizontal_size=10
# vertical_size=10
# labelpad_x=8
# labelpad_y=--8
# labelsize=font_size
# legendsize=font_size

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

# Select neurons (either "DA" or "putative_DA")
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
animals = data_frame_neurons_info['Animal'].values
n_neurons = len(gamma_before)

# Adaptation in reward delay tuning
gamma_before_take_short = []
gamma_after_take_short = []
gamma_before_take_long = []
gamma_after_take_long = []

gain_before_take_short = []
gain_after_take_short = []
gain_before_take_long = []
gain_after_take_long = []

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

color_remove_short = "limegreen"
color_remove_long = "mediumvioletred"

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
ax.set_box_aspect(1)
bins = np.linspace(-0.45, 0.45, 10)

ax.scatter(gamma_before_take_long, gamma_after_take_long, color=color_remove_long, s=20, label="longest",
           edgecolor="white")  # scatter_size+30
ax.scatter(gamma_before_take_short, gamma_after_take_short, color=color_remove_short, s=20, label="shortest",
           edgecolor="white")  # catter_size+30

# Histogram of adaptation in temporal discounts when taking the longest or shortest delay

axins_distribution = inset_axes(ax, width="45%", height="45%", loc=2, borderpad=1.8)
ip = InsetPosition(ax, [0.1, 0.82, 0.3, 0.3])
axins_distribution.set_axes_locator(ip)
axins_distribution.spines['left'].set_linewidth(linewidth)
axins_distribution.spines['bottom'].set_linewidth(linewidth)
axins_distribution.hist(gamma_after_take_short - gamma_before_take_short, density=True, bins=bins,
                        color=color_remove_short, alpha=0.5)
axins_distribution.hist(gamma_after_take_long - gamma_before_take_long, density=True, bins=bins,
                        color=color_remove_long, alpha=0.5)
axins_distribution.set_box_aspect(1)
axins_distribution.tick_params(axis='both', which='major')
axins_distribution.set_xlabel(r"$\Delta \gamma$", labelpad=0, fontsize=font_size - 2)
axins_distribution.set_ylabel("Probability", labelpad=0, fontsize=font_size - 2)
axins_distribution.set_xticks([0])
axins_distribution.set_xticklabels([0], fontsize=font_size - 5)
axins_distribution.set_yticks([])
axins_distribution.axvline(x=np.mean(gamma_after_take_short - gamma_before_take_short), color=color_remove_short,
                           linewidth=linewidth - 0.4)
axins_distribution.axvline(x=np.mean(gamma_after_take_long - gamma_before_take_long), color=color_remove_long,
                           linewidth=linewidth - 0.4)

ax.plot([0.36, 1.25], [0.36, 1.25], color="black", ls="--")
ax.set_xlabel(r"$\gamma$" + " before")
ax.set_ylabel(r"$\gamma$" + " after")
ax.set_xticks([0.5, 1], ["0.5", "1"])
ax.set_yticks([0.5, 1], ["0.5", "1"])
plt.show()

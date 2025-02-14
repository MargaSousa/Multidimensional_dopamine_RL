import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
from aux_functions import *
from scipy import stats
from sklearn.linear_model import LinearRegression, HuberRegressor
import matplotlib.colors as mcol
import seaborn as sns

flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [reward_cmap[1], reward_cmap[-1]])
import pandas
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

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get parsed data
responses_reward_certain = np.loadtxt(
    os.path.join(directory_parsed_data, "responses_reward_certain_magnitude_3s_delay.csv"))
responses_reward_variable = np.load(
    os.path.join(directory_parsed_data, "responses_reward_different_magnitudes_constant_delay.npy"))
responses_cue_variable = np.loadtxt(
    os.path.join(directory_parsed_data, "responses_cue_different_magnitudes_constant_delay.csv"))
responses_cue_certain = np.load(
    os.path.join(directory_parsed_data, "responses_cue_different_delays_constant_magnitude.npy"))[:, 2, :]

# Get estimated tuning for reward time and magnitude
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))
discount = data_frame_neurons_info['Gamma'].values
estimated_reversals = data_frame_neurons_info['Reversals'].values
estimated_taus = data_frame_neurons_info['Taus'].values
gains = data_frame_neurons_info['Gain'].values
animals = data_frame_neurons_info['Animal'].values
n_neurons = len(discount)

# Select neurons from given animals
selected_animal = 3353
selected_neurons = data_frame_neurons_info.index[data_frame_neurons_info.Animal == selected_animal].tolist()

# Choose instead neurons from all animals
selected_neurons = np.arange(len(data_frame_neurons_info))

discount = discount[selected_neurons]
responses_reward_certain = responses_reward_certain[selected_neurons]
responses_reward_variable = responses_reward_variable[selected_neurons]
estimated_reversals = estimated_reversals[selected_neurons]
gains = gains[selected_neurons]

# Linear regression between temporal dicount factors and gains
reg = LinearRegression().fit((discount).reshape(-1, 1), gains)
x_example = np.linspace(np.min(discount) - 0.1, np.max(discount) + 0.1, 100)
y_example = reg.predict(x_example.reshape(-1, 1))
out_variable = stats.pearsonr(discount, gains)

print("Pearson correlation discount factor and gain: ", out_variable[0])
print("p-value Pearson correlation discount factor and gain: ", out_variable[1])
# print("Confidence interval discount factor and gain: ",out_variable.confidence_interval(confidence_level=0.95))

# Figure 5H
fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
plt.scatter(discount, gains, color="k", s=scatter_size + 30, edgecolor="white")  # ,
plt.plot(x_example, y_example, ls="--", color="k")
plt.xlabel(r"$\gamma$")
plt.ylabel("Gain (sp/s)")
plt.text(1, 15, "r=" + str(np.round(out_variable[0], 2)) + "\n" + "p=" + str(np.round(out_variable[1], 3)),
         fontsize=font_size - 1)  # "p<1e-7"
plt.ylim(0, 20)
plt.show()

colors_optimism = asym_cmap(np.linspace(0, 0.85, n_neurons))
mean_responses_certain = np.nanmean(responses_reward_certain, axis=1)
mean_responses_variable = np.nanmean(responses_reward_variable[:, 2, :], axis=1)
sorted_reversals = np.argsort(estimated_reversals)

# Correct for  diversity in temporal discount and gain
fr_variable = np.nanmean(responses_cue_variable, axis=1)
fr_certain_corrected = np.nanmean(responses_cue_certain, axis=1) / (gains * discount ** 3)
fr_variable_corrected = fr_variable / (gains * discount ** 3)

# Regression of reversal as a function of firing rate at the cue
well_estimated_reversals = np.intersect1d(np.where(estimated_reversals > 1.1)[0],
                                          np.where(estimated_reversals < 7.9)[0])
n_neurons_well_estimated_reversals = len(well_estimated_reversals)
out_variable = stats.pearsonr(fr_variable_corrected[well_estimated_reversals],
                              estimated_reversals[well_estimated_reversals])
reg_variable = HuberRegressor(fit_intercept=True).fit((fr_variable_corrected[well_estimated_reversals]).reshape(-1, 1),
                                                      estimated_reversals[well_estimated_reversals])
x_example_uncertain = np.linspace(np.min(fr_variable_corrected[well_estimated_reversals]) - 0.1,
                                  np.max(fr_variable_corrected[well_estimated_reversals]) + 0.1, 100)
y_example_uncertain = reg_variable.predict(x_example_uncertain.reshape(-1, 1))
reversal_estimated_at_cue = reg_variable.predict(fr_variable_corrected.reshape(-1, 1))
colors_optimism = asym_cmap(np.linspace(0, 0.85, n_neurons_well_estimated_reversals))
reversal_certain_cue = reg_variable.predict(fr_certain_corrected.reshape(-1, 1))
reversal_variable_cue = reg_variable.predict(fr_variable_corrected.reshape(-1, 1))

# Figure 5D: Adaptation to reward amount distribution plot
fr_certain_corrected = np.array(fr_certain_corrected)
fr_variable_corrected = np.array(fr_variable_corrected)
# fig,ax=plt.subplots(figsize=(horizontal_size,vertical_size))

fig, ax = plt.subplots(figsize=(1, 1))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
colors_optimism_well_estimated = asym_cmap((estimated_reversals[sorted_reversals] - 1.1) / (7.9 - 1.1))
# ax.scatter(fr_certain_corrected[sorted_reversals],fr_variable_corrected[sorted_reversals],color="black",s=scatter_size+30,edgecolor="white")

# Or
# ax.scatter(reversal_certain_cue[sorted_reversals],reversal_variable_cue[sorted_reversals],color="black",s=scatter_size+30,edgecolor="white")
ax.scatter(reversal_certain_cue[sorted_reversals], reversal_variable_cue[sorted_reversals], color="black", s=20,
           edgecolor="white")
# ax.set_xlabel("Reversal point certain cue")
# ax.set_ylabel("Reversal point variable cue")

ax.set_xlabel("Value certain")
ax.set_ylabel("Value variable")

ax.set_xlim(0.5, 8.5)
ax.set_ylim(0.5, 8.5)
ax.set_yticks([1, 4.5, 8])
ax.set_xticks([1, 4.5, 8])
plt.show()

# Check if the order of reversals is maintained across cues
sorted_well_estimated_reversals = np.argsort(estimated_reversals[well_estimated_reversals])
out_1 = stats.pearsonr(estimated_reversals[well_estimated_reversals], mean_responses_certain[well_estimated_reversals])
reg = HuberRegressor().fit((estimated_reversals[well_estimated_reversals]).reshape(-1, 1),
                           mean_responses_certain[well_estimated_reversals])
x_example = np.linspace(np.min(estimated_reversals[well_estimated_reversals]) - 0.1,
                        np.max(estimated_reversals[well_estimated_reversals]) + 0.1, 100)
y_example = reg.predict(x_example.reshape(-1, 1))

print("Pearson correlation reversal firing rate certain: ", out_1[0])
print("p-value Pearson correlation reversal firing rate certain: ", out_1[1])
# print("Confidence interval discount reversal firing rate certain: ",out_1.confidence_interval(confidence_level=0.95))
print("n=" + str(len(well_estimated_reversals)))

# Figure 5C:  Reversals point order seems consistent across different cues
fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=length_ticks)
ax.set_box_aspect(1)
colors_optimism_well_estimated = asym_cmap(
    (estimated_reversals[well_estimated_reversals][sorted_well_estimated_reversals] - 1.1) / (7.9 - 1.1))
plt.scatter(estimated_reversals[well_estimated_reversals][sorted_well_estimated_reversals],
            mean_responses_certain[well_estimated_reversals][sorted_well_estimated_reversals], s=scatter_size,
            c=colors_optimism_well_estimated)
plt.xlabel("Reversal point (" + r"$\mu$" + "l)")  # ,fontsize=labelsize
plt.ylabel(r"$\Delta$" + " FR certain reward (sp/s)")
plt.xticks([0, 2, 4, 6, 8], ["0", "2", "4", "6", "8"])
plt.yticks([-4, 0, 4], ["-4", "0", "4"])
plt.plot(x_example, y_example, ls="--", color="black")
plt.text(6, 1, "r=" + str(np.round(out_1[0], 2)) + "\n" + "p=" + str(np.round(out_1[1], 2)), fontsize=font_size - 1)
plt.xlim(0, 9)
plt.show()

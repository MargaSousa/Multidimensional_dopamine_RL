import os
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
mpl.use('TkAgg')

# Parameters for paper plots
# length_ticks=2
# font_size=11
# linewidth=1.2
# scatter_size=4
# horizontal_size=2
# vertical_size=2.4

# Parameters for presentation plots
length_ticks = 10
font_size = 30
linewidth = 4
scatter_size = 50
horizontal_size = 9
vertical_size = 9
labelpad_x = 10
labelpad_y = 10
labelsize = font_size
legendsize = font_size
scatter_size_fr = 15
capsize = 8

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
directory_parsed_data = os.path.join(directory, "Parsed_data_DA")
psth_licking_amounts = np.load(os.path.join(directory_parsed_data,
                                            "psth_licking_amounts.npy"))  # Liking PSTH aligned to reward delivery for different amounts (animal x amounts x bins)
psth_licking_delays = np.load(os.path.join(directory_parsed_data,
                                           "psth_licking_delays.npy"))  # Liking PSTH aligned to cue delivery for different delays (animal x delays x bins)
bins = np.load(os.path.join(directory_parsed_data, "bins_licking.npy"))  # bins in time (seconds)
bin_end_amount = np.where(bins >= 6.5)[0][0]

# colors for plotting delays
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))

# colors for plotting amounts
winter = mpl.cm.get_cmap('winter', 12)
colors_amount = winter(np.linspace(0, 1, 5))

# Plot mean PSTH across animals for different reward delays
fig, ax = plt.subplots(1, 2, figsize=(horizontal_size * 2, vertical_size))  # ,sharex='col',sharey='row'
ax[0].tick_params(width=linewidth, length=length_ticks)
ax[0].spines['left'].set_linewidth(linewidth)
ax[0].spines['bottom'].set_linewidth(linewidth)
ax[0].set_box_aspect(1)
delays_unique = np.array([0, 1.5, 3, 6])
label_delay = ["0", "1.5", "3", "6"]
for i_d, d in enumerate(delays_unique):
    mean = np.nanmean(psth_licking_delays[:, i_d, :], axis=0).astype(float)
    sem = scipy.stats.sem(psth_licking_delays[:, i_d, :].astype(float), axis=0, nan_policy='omit').astype(float)
    ax[0].plot(bins[:-1], mean, color=colors_delay[i_d, :], label=label_delay[i_d] + " s")
    ax[0].fill_between(bins[:-1], mean - sem, mean + sem, alpha=0.2, color=colors_delay[i_d, :])
    ax[0].axvline(x=d, color=colors_delay[i_d, :], ls="--")
ax[0].legend(handlelength=0, frameon=False)
ax[0].set_ylabel("Lick rate (licks/s)")
ax[0].set_xlabel("Time since cue (s)")
ax[0].set_xticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])

# Plot mean PSTH across animals for different reward amount
ax[1].tick_params(width=linewidth, length=length_ticks)
ax[1].spines['left'].set_linewidth(linewidth)
ax[1].spines['bottom'].set_linewidth(linewidth)
ax[1].set_box_aspect(1)
label_amount = ["1", "2.75", "4.5", "6.25", "8"]
amounts_unique = np.array([1, 2.75, 4.5, 6.25, 8])
for i_a, a in enumerate(amounts_unique):
    mean = np.nanmean(psth_licking_amounts[:, i_a, :], axis=0).astype(float)
    sem = scipy.stats.sem(psth_licking_amounts[:, i_a, :].astype(float), axis=0, nan_policy='omit').astype(float)
    ax[1].plot(bins[:bin_end_amount - 1], mean[:bin_end_amount - 1], color=colors_amount[i_a, :],
               label=label_amount[i_a] + " " + r"$\mu$" + "l")
    ax[1].fill_between(bins[:bin_end_amount - 1], mean[:bin_end_amount - 1] - sem[:bin_end_amount - 1],
                       mean[:bin_end_amount - 1] + sem[:bin_end_amount - 1], alpha=0.2, color=colors_amount[i_a, :])
ax[1].axvline(x=0, color="black", ls="--")
ax[1].legend(handlelength=0, frameon=False)
ax[1].set_yticks([0, 6], ["0", "6"])
ax[0].set_yticks([0, 6], ["0", "6"])
ax[1].set_xticks([0, 2.5, 5], ["0", "2.5", "5"])
ax[1].set_xlabel("Time since reward (s)", labelpad=labelpad_x)
ax[1].set_ylabel("Lick rate (licks/s)", labelpad=labelpad_y)
plt.show()

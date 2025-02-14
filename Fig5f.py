import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
from aux_functions import *

length_ticks = 3
font_size = 8
linewidth = 1.2
scatter_size = 2
horizontal_size = 1.8
vertical_size = 1.8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth
mpl.use('TkAgg')

color_take_short = "mediumvioletred"
color_take_long = "limegreen"

N = 20  # Number of neurons for each context switch
N_exp = 10000  # Number of samples to estimate expectiles
time_max = 8  # reward time range
time_exp = np.linspace(0, time_max, N_exp)

scale = 0.25
mean1 = 0
mean2 = 1.5
mean3 = 3
mean4 = 6

# Distribution before taking one of the delays
probability_exp_before = scipy.stats.norm(loc=mean1, scale=scale).pdf(time_exp) + scipy.stats.norm(loc=mean2,
                                                                                                   scale=scale).pdf(
    time_exp) + scipy.stats.norm(loc=mean3, scale=scale).pdf(time_exp) + scipy.stats.norm(loc=mean4, scale=scale).pdf(
    time_exp)
probability_exp_before = probability_exp_before / np.sum(probability_exp_before)

# Expectiles and temporal discount gammas before taking one of the delays
taus = np.linspace(1.0 / N, 1.0 - 1 / N, N)
_, expectiles_before = get_expectiles(time_exp, probability_exp_before, taus)
gamma_before = np.exp(-1.0 / expectiles_before)

# Distribution after taking the shortest delay
probability_exp = scipy.stats.norm(loc=mean2, scale=scale).pdf(time_exp) + scipy.stats.norm(loc=mean3, scale=scale).pdf(
    time_exp) + scipy.stats.norm(loc=mean4, scale=scale).pdf(time_exp)
probability_exp = probability_exp / np.sum(probability_exp)

# Expectiles and temporal discount factors after taking the shortest delay
_, expectiles_after_take_long = get_expectiles(time_exp, probability_exp, taus)
gamma_after_take_long = np.exp(-1.0 / expectiles_after_take_long)

# Distribution after taking the longest delay
probability_exp = scipy.stats.norm(loc=mean1, scale=scale).pdf(time_exp) + scipy.stats.norm(loc=mean2, scale=scale).pdf(
    time_exp) + scipy.stats.norm(loc=mean3, scale=scale).pdf(time_exp)
probability_exp = probability_exp / np.sum(probability_exp)

# Expectiles and temporal discount factors after taking the longest delay
taus, expectiles_after_take_short = get_expectiles(time_exp, probability_exp, taus)
gamma_after_take_short = np.exp(-1.0 / expectiles_after_take_short)

noise_1 = np.random.normal(loc=0, scale=0.01, size=gamma_before.shape)
noise_2 = np.random.normal(loc=0, scale=0.01, size=gamma_before.shape)

# Update in temporal discount factors plot
fig, ax = plt.subplots(figsize=(1, 1))  # ,tight_layout=True
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
plt.scatter(gamma_before + noise_1, gamma_after_take_short + noise_2, color=color_take_short, s=10,
            label="shortest")  # scatter_size
plt.scatter(gamma_before + noise_1, gamma_after_take_long + noise_2, color=color_take_long, s=10,
            label="longest")  # scatter_size
plt.xlabel(r"$\gamma$ before", labelpad=0.01)
plt.ylabel(r"$\gamma$ after", labelpad=0.01)
plt.plot([0.35, 0.9], [0.35, 0.9], color="black", ls="--")
plt.xticks([])
plt.yticks([])
plt.show()

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size), tight_layout=True)
ax.tick_params(width=linewidth, length=15)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)

for i in range(N):
    plt.plot(time_exp, gamma_after_take_short[i] ** time_exp, color=color_take_short)
    plt.plot(time_exp, gamma_after_take_long[i] ** time_exp, color=color_take_long)
plt.plot(time_exp, probability_exp_before * 2500, color="k")
plt.xlabel("Reward time (s)")
plt.ylabel("Density," + r"$V_{\gamma},$" + r"$V_{\gamma}$")
plt.xticks([])
plt.yticks([])
plt.show()

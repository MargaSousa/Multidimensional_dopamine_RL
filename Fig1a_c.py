import time as timer
start_time = timer.time()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

# Parameters for plots
length_ticks = 5
font_size = 11
linewidth = 1.2
scatter_size = 2
horizontal_size = 1.5
vertical_size = 1.5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rc('xtick', labelsize=8)
mpl.rc('ytick', labelsize=8)

# Number of units
N = 1000
gamma = np.linspace(0.01, 1, N)

# Define reward times
time_reward = np.array([2, 15])

# Define color of each cue
colors_plot = ["#7fc97f", "#beaed4", "#fdc086"]  # ss,ll,variable

# Discretizing time
N_time = 120
times_disc = np.linspace(0, 15, N_time)

# Value at cue sooner smaller (ss) reward = later larger (ll) reward
gamma_sel = 0.7
r_soon = 0.5
value_soon = r_soon * gamma_sel ** (time_reward[0])
r_late = value_soon / (gamma_sel ** time_reward[1])
bin_time_soon = np.where(times_disc >= time_reward[0])[0][0]
bin_time_late = np.where(times_disc >= time_reward[1])[0][0]
time_soon = times_disc[bin_time_soon] - times_disc[:bin_time_soon]
time_late = times_disc[bin_time_late] - times_disc[:bin_time_late]
value_soon_time = r_soon * gamma_sel ** (time_soon)
value_late_time = r_late * gamma_sel ** (time_late)

fig, ax = plt.subplots(figsize=(vertical_size, horizontal_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
plt.plot(time_late[::-1], value_late_time, color=colors_plot[1])
plt.plot(time_soon[::-1], value_soon_time, color=colors_plot[0])
plt.vlines(x=times_disc[bin_time_late], ymin=0, ymax=r_late, color=colors_plot[1])
plt.vlines(x=times_disc[bin_time_soon], ymin=0, ymax=r_soon, color=colors_plot[0])
plt.xlabel("Time")
plt.ylabel("Value")
plt.xticks([0], ["0"], fontsize=font_size)
plt.yticks([])
plt.show()

# Decode future reward using inverse Laplace transform (Tano et al 2020, Yable et al 2005)
F = np.zeros((N, N_time))
Value_stationary = np.zeros((N))
for i_t, t in enumerate(times_disc):
    F[:, i_t] = gamma ** t
U, s, vh = np.linalg.svd(F, full_matrices=False)

# Define value for stationary case
for i_g, g in enumerate(gamma):
    Value_stationary[i_g] = np.sum(g ** times_disc)

alpha = 20  # smoothing parameter
fig, ax = plt.subplots(figsize=(vertical_size, horizontal_size))
L = np.shape(U)[1]
for i_time, time in enumerate(time_reward):
    Value = gamma ** time
    p = np.zeros(N_time)
    p_stationary = np.zeros(N_time)
    for i in range(L):
        p += (s[i] ** 2) / ((s[i] ** 2) + (alpha ** 2)) * (np.dot(U[:, i], Value) * vh[i, :] / s[i])
        p_stationary += (s[i] ** 2) / ((s[i] ** 2) + (alpha ** 2)) * (
                np.dot(U[:, i], Value_stationary) * vh[i, :] / s[i])
    p[p < 0] = 0
    p = p / np.sum(p)
    plt.plot(times_disc, p, color=colors_plot[i_time])
    plt.xlabel("Reward delay")
    np.save("pdf_time_" + str(i_time) + ".npy", p)

plt.ylabel("Decoded density\nat cue")
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
plt.xticks([])
plt.yticks([])
p_stationary = p_stationary / np.sum(p_stationary)
plt.show()

np.save("time_disc.npy", times_disc)
np.save("pdf_stationary.npy", p_stationary)

# Multiple temporal discounts value plot
N_neurons = 6
cmap = cmr.cm.apple(np.linspace(0.1, 1 - 1.0 / N_neurons, N_neurons))
gammas = np.linspace(0.1, 0.9, N_neurons)
fig, ax = plt.subplots(figsize=(vertical_size, horizontal_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
time = np.linspace(0, 7, 100)
for i_neu in range(N_neurons):
    plt.plot(time, gammas[i_neu] ** time, color=cmap[i_neu])
plt.yticks([])
plt.xlabel("Reward delay")
plt.ylabel("Responses at cue")
plt.ylim(0, 1.1)
plt.show()

# Responses at the cue as a function of neuron id
y_ss = []
y_ll = []
for i_neuron in range(N_neurons):
    y_ss.append(gammas[i_neuron] ** 1.5)
    y_ll.append(gammas[i_neuron] ** 7)

fig, ax = plt.subplots(figsize=(vertical_size, horizontal_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
plt.scatter(np.arange(N_neurons), y_ll, c=cmap)
plt.plot(np.arange(N_neurons), y_ll, color=colors_plot[1])
plt.scatter(np.arange(N_neurons), y_ss, c=cmap)
plt.plot(np.arange(N_neurons), y_ss, color=colors_plot[0])
plt.plot(np.arange(N_neurons), y_ss, color=colors_plot[2])
plt.xlabel("Neurons")
plt.ylabel("Responses at cue")
plt.xticks(np.arange(N_neurons), np.arange(1, N_neurons + 1))
plt.yticks([])
plt.show()

# Heat map value for reward amounts and times
n_time_steps = 100
n_amounts = 100
time = np.linspace(1, 10, n_time_steps)
amount = np.linspace(1, 10, n_amounts)
Value = np.zeros((n_amounts, n_time_steps))
gamma = 0.9
fig, ax = plt.subplots(figsize=(vertical_size, horizontal_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
for i_t, t in enumerate(time):
    for i_a, a in enumerate(amount):
        Value[n_time_steps - 1 - i_t, i_a] = (gamma ** t) * a
im = plt.imshow(Value, cmap='Blues')
plt.ylabel("Reward time")
plt.xlabel("Reward magnitude")
plt.xticks([])
plt.yticks([])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax,ticks=[],label="Value at cue")#r"$V$"+"(cue)"
# fig.savefig(save_dir+r"/heat_map_value_time_amount.svg")
plt.show()

print("time elapsed: {:.2f}s".format(timer.time() - start_time))

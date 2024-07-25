import time as timer
start_time = timer.time()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as mcol
import pandas as pd

# Parameters for plots
length_ticks = 3
font_size = 11
linewidth = 1.2
scatter_size = 20
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth
horizontal_size = 1.5
vertical_size = 1.5

# Directory to save intermediary data
dir_save_for_plot=r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\Figure_1"

# Optimism colors (from Dabney et al, 2020)
flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [reward_cmap[1], reward_cmap[-1]])

n_states = 5
n_unique = 10

n_trials = 50000
unique_taus = np.random.uniform(1.0 / n_unique, 1.0 - 1.0 / n_unique, n_unique)  #
unique_gammas = np.linspace(0.6, 0.999, n_unique)[::-1]
mesh = np.meshgrid(unique_taus, unique_gammas)

for j in range(mesh[0].shape[0]):
    mesh[0][j, :] = np.random.uniform(1.0 / n_unique, 1.0 - 1.0 / n_unique, n_unique)

taus = np.ndarray.flatten(mesh[0])
gammas = np.ndarray.flatten(mesh[1])

plt.scatter(taus, gammas)
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\gamma$")
plt.show()

n_neurons = len(taus)

Value = np.zeros((n_neurons, n_states))
Value_save = np.zeros((n_trials, n_neurons, n_states))

alpha = 0.01  # learning rate
for trial in range(n_trials):
    for time in range(n_states):
        if time == n_states - 1:
            reward = np.random.normal(loc=10, scale=30)
            imputation = reward
        else:
            reward = 0

            ## A possible way to implement imputation
            sample_future = np.random.choice(np.round(Value[:, time + 1] / ((gammas) ** (n_states - time - 1)), 20),
                                             size=n_neurons)
            sample_future_neurons = np.round(sample_future * (gammas) ** (n_states - time - 1), 20)

            imputation = reward + gammas * sample_future_neurons

        error = imputation - Value[:, time]
        Value[:, time] += alpha * (taus * error * (error > 0) + (1 - taus) * error * (error < 0))

    if trial % 10000 == 0:
        alpha = alpha * 0.9
plt.imshow(Value, aspect="auto")
plt.ylabel("Neurons")
plt.xlabel("Time")
plt.xticks([0, n_states - 1], ["cue", "reward"])
plt.colorbar(label="Value")
plt.show()

fig, ax = plt.subplots(figsize=(horizontal_size, vertical_size))
ax.tick_params(width=linewidth, length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
colors_optimism = asym_cmap(np.linspace(0, 1, n_neurons))
corrected_values_at_cue=Value[:, 0] / gammas ** n_states + np.random.normal(loc=0, scale=1, size=n_neurons) * 0.05
plt.scatter(corrected_values_at_cue, Value[:, -1],
            c=taus, s=scatter_size, cmap=asym_cmap)  #
plt.xlabel(r"$V_i$" + "(reward)")
plt.ylabel("Values at reward")
plt.xlabel("Values at cue\n" + r"corrected for $\gamma$")
plt.xticks([])
plt.yticks([])
plt.show()


values_cue_reward={"Values at cue corrected for diversity in temporal discount": corrected_values_at_cue, "Values at reward":Value[:, -1]}
df = pd.DataFrame(values_cue_reward)
df.to_csv(dir_save_for_plot+r'\Fig1d_values_cue_reward.csv',index=False,header=True, sep=',')

print("time elapsed: {:.2f}s".format(timer.time() - start_time))

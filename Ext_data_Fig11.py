import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import pdb
from scipy.stats import gaussian_kde
import matplotlib as mpl
from aux_functions import run_decoding, get_expectiles
import seaborn as sns

# Parameters for plots
length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 1.5
vertical_size = 1.5
font_size = 11
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 10


# Define probability distributions over amounts
min_reward = 0
max_reward = 9
n_reward = 100
reward = np.linspace(min_reward, max_reward, n_reward)

amount = np.array([1, 2.75, 4.5, 6.25, 8])
n_amounts = len(amount)
probability = np.array([0.25, 0.167, 0.167, 0.167, 0.25])
probability = probability / np.sum(probability)
n_bins = 50
smooth = 0.3
all_samples = np.random.choice(amount, 10000, p=probability)
kde = gaussian_kde(all_samples, bw_method=smooth)
y = kde.pdf(reward)
probability_bi = y / np.sum(y)
probability_uni = scipy.stats.norm(loc=4.5, scale=1).pdf(reward)
probability_uni = probability_uni / np.sum(probability_uni)

n_neurons = 100
taus = np.linspace(1.0 / n_neurons, 1 - 1.0 / n_neurons, n_neurons)
_, expectiles = get_expectiles(reward, probability_bi, taus)
_, expectiles_uni = get_expectiles(reward, probability_uni, taus)
sampled_bi, loss_synthetic = run_decoding(expectiles, taus, np.ones(n_neurons), N=20, minv=min_reward, maxv=max_reward,
                                          max_samples=100, max_epochs=5, method='TNC')
sampled_uni, loss_synthetic = run_decoding(expectiles_uni, taus, np.ones(n_neurons), N=20, minv=min_reward,
                                           maxv=max_reward, max_samples=100, max_epochs=5, method='TNC')

gammas = np.linspace(0, 1, n_neurons)
time_reward = [0, 1.5, 3, 6]

X = np.zeros((n_neurons, n_neurons))
Y = np.zeros((n_neurons, n_neurons))
Y_uni = np.zeros((n_neurons, n_neurons))

# Compute synthetic FR at cue
FR_synthetic = np.zeros((n_neurons, n_neurons))  # Lines correspond to reversal points, columns to temporal discounts
FR_uni = np.zeros((4, n_neurons, n_neurons))
for cue in range(4):
    for i_g, gamma in enumerate(gammas):
        for i_r, r in enumerate(expectiles):
            FR_uni[cue, i_r, i_g] = (gamma ** time_reward[cue]) * expectiles_uni[i_r]
            X[i_r, i_g] = gamma
            Y[i_r, i_g] = expectiles[i_r]
            Y_uni[i_r, i_g] = expectiles_uni[i_r]
            if cue == 2:
                FR_synthetic[i_r, i_g] = (gamma ** time_reward[cue]) * expectiles[i_r]

fig, ax = plt.subplots(1, 2, figsize=(2 * horizontal_size, 1 * vertical_size))
ax[0].imshow(FR_synthetic, extent=[gammas[0], gammas[-1], expectiles[-1], expectiles[0]], aspect="auto", cmap="Blues")
ax[0].set_xlabel(r"$\gamma$")
ax[0].set_ylabel("Reversal point")
ax[0].set_title("Firing Rate at cue")

flat_FR = np.ndarray.flatten(FR_synthetic)
gammas_flat = np.ndarray.flatten(X)
expectiles_flat = np.ndarray.flatten(Y)
expectiles_uni_flat = np.ndarray.flatten(Y_uni)

n_time = 5
time = np.linspace(0, 6, n_time)
synthetic_delta_fr_time = np.zeros((n_neurons, n_time))
synthetic_delta_fr_time_uni = np.zeros((4, n_neurons, n_time))
alpha = np.array([0.2, 0.001, 0.000001, 0.00001])
max_fr = 0


# Construct matrix F for Laplace decoder
F = np.zeros((n_neurons, n_time))
for i_t, t in enumerate(time):
    F[:, i_t] = gammas ** t
# Find eigenvectors
U, s, vh = np.linalg.svd(F, full_matrices=True)
L = np.min([vh.shape[0], np.shape(U)[1]])

# Laplace decoder applied to each line
for cue in range(4):
    for neuron in range(n_neurons):
        for i in range(L):
            synthetic_delta_fr_time_uni[cue, neuron, :] += (s[i] ** 2 / (s[i] ** 2 + alpha[cue] ** 2)) * (
                    np.dot(U[:, i], FR_uni[cue, neuron, :]) * vh[i, :]) / s[i]
            if cue == 2:
                synthetic_delta_fr_time[neuron, :] += (s[i] ** 2 / (s[i] ** 2 + alpha[cue] ** 2)) * (
                        np.dot(U[:, i], FR_synthetic[neuron, :]) * vh[i, :]) / s[i]
        synthetic_delta_fr_time[neuron, :][synthetic_delta_fr_time[neuron, :] < 0] = 0
        synthetic_delta_fr_time_uni[cue, neuron, :][synthetic_delta_fr_time_uni[cue, neuron, :] < 0] = 0
    max_time = np.argmax(synthetic_delta_fr_time_uni[cue, neuron, :])
    # Correct the range
    synthetic_delta_fr_time_uni[cue, :, max_time] = (synthetic_delta_fr_time_uni[cue, :, max_time] - np.min(
        synthetic_delta_fr_time_uni[cue, :, max_time])) / (np.max(
        synthetic_delta_fr_time_uni[cue, :, max_time]) - np.min(synthetic_delta_fr_time_uni[cue, :, max_time]))
    synthetic_delta_fr_time_uni[cue, :, max_time] = synthetic_delta_fr_time_uni[cue, :, max_time] * (
            np.max(expectiles_uni) - np.min(expectiles_uni)) + np.min(expectiles_uni)
    if cue == 2:
        max_time = np.argmax(synthetic_delta_fr_time[neuron, :])
        synthetic_delta_fr_time[:, max_time] = (synthetic_delta_fr_time[:, max_time] - np.min(
            synthetic_delta_fr_time[:, max_time])) / (np.max(synthetic_delta_fr_time[:, max_time]) - np.min(
            synthetic_delta_fr_time[:, max_time]))
        synthetic_delta_fr_time[:, max_time] = synthetic_delta_fr_time[:, max_time] * (
                np.max(expectiles) - np.min(expectiles)) + np.min(
            expectiles)  # for neuron in range(n_neurons):  # ax[cue].plot(time, synthetic_delta_fr_time[neuron, :], color="red")  # ax[cue].scatter(time, synthetic_delta_fr_time[0, :], color="yellow")

    # for neuron in range(n_neurons):  # ax[cue].plot(time, synthetic_delta_fr_time_uni[cue, neuron, :], color="blue")  # ax[cue].scatter(time, synthetic_delta_fr_time_uni[cue, 0, :], color="yellow")
# plt.show()

ax[1].imshow(synthetic_delta_fr_time, extent=[time[0], time[-1], expectiles[-1], expectiles[0]], aspect="auto",
             cmap="Blues")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Reversal point")
# plt.savefig(save_dir+"\Laplace_decoder.svg", bbox_inches=0)
plt.show()

# Check if we can recover the distribution
# bin=np.where(time>=2.8)[0][0]
# sampled_bi, loss_synthetic = run_decoding(synthetic_delta_fr_time[:,bin], taus, np.ones(n_neurons),N=20, minv=min_reward, maxv=max_reward, max_samples=100,max_epochs=5, method='TNC')
# sampled_uni, loss_synthetic = run_decoding(synthetic_delta_fr_time_uni[2,:,bin], taus, np.ones(n_neurons),N=20, minv=min_reward, maxv=max_reward, max_samples=100,max_epochs=5, method='TNC')
# plt.hist(sampled_uni)
# plt.hist(sampled_bi)
# plt.show()


# Expectile decoder for each column
all_samples_synthetic = []
all_samples_synthetic_uni = []
all_samples_synthetic_uni = []
hist = np.zeros((n_reward, n_time))
hist_uni = np.zeros((4, n_reward, n_time))
n_samples = 100
bin_start = 15
for cue in range(4):
    for bin_time in range(n_time):
        sampled_dist_synthetic_uni, loss_synthetic = run_decoding(synthetic_delta_fr_time_uni[cue, :, bin_time], taus,
                                                                  np.ones(n_neurons), N=20, minv=min_reward,
                                                                  maxv=max_reward, max_samples=100, max_epochs=5,
                                                                  method='TNC')

        if cue == 2:
            sampled_dist_synthetic, loss_synthetic = run_decoding(synthetic_delta_fr_time[:, bin_time], taus,
                                                                  np.ones(n_neurons), N=20, minv=min_reward,
                                                                  maxv=max_reward, max_samples=100, max_epochs=5,
                                                                  method='TNC')
            if np.sum(sampled_dist_synthetic) == 0 or np.sum(sampled_dist_synthetic_uni) == max_reward * n_samples:
                hist[:, bin_time] = 0
            else:
                hist[:, bin_time] = gaussian_kde(sampled_dist_synthetic, bw_method=0.35).pdf(reward)
                hist[:, bin_time] = hist[:, bin_time] / np.sum(hist[:, bin_time])
                hist[bin_start:, bin_time] = hist[bin_start:, bin_time] * (
                        1 - np.sum(hist[:bin_start, bin_time]))  # Not consider reward =0

        if np.sum(sampled_dist_synthetic_uni) == 0 or np.sum(sampled_dist_synthetic_uni) == max_reward * n_samples:
            hist_uni[cue, :, bin_time] = 0
        else:
            hist_uni[cue, :, bin_time] = gaussian_kde(sampled_dist_synthetic_uni, bw_method=0.35).pdf(reward)
            hist_uni[cue, :, bin_time] = hist_uni[cue, :, bin_time] / np.sum(hist_uni[cue, :, bin_time])
            hist_uni[cue, bin_start:, bin_time] = hist_uni[cue, bin_start:, bin_time] * (
                    1 - np.sum(hist_uni[cue, :bin_start, bin_time]))  # Not consider reward =0

        # for i_s in range(n_samples):  # all_samples_synthetic.append([time[bin_time],sampled_dist_synthetic[i_s]])  # all_samples_synthetic_uni.append([time[bin_time],sampled_dist_synthetic_uni[i_s]])

# Smooth using gaussian kernel
# bins=np.linspace(min_reward,max_reward,n_neurons)
X, Y = np.meshgrid(time, reward[bin_start - 1:])
positions = np.vstack([X.ravel(), Y.ravel()])

# Stacked heat map
color_map = sns.color_palette("coolwarm", as_cmap=True)
fig, ax = plt.subplots(figsize=(horizontal_size * 2.5, vertical_size * 5), subplot_kw={"projection": "3d"})
scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 0.05), cmap=color_map)
ax.grid(False)
ax.set_box_aspect((1, 1, 3))
for cue in range(3):
    ax.plot_surface(Y, X, (cue) + 0 * hist_uni[0, bin_start - 1:, :],
                    facecolors=scam.to_rgba(hist_uni[cue, bin_start - 1:, :]), alpha=None, rstride=1,
                    cstride=1)  # ,rstride=1,cstride=1,

cue = 3
hist_3 = hist_uni[cue, bin_start - 1:, :]
# Upsample
hist_new = np.zeros((hist_3.shape[0], hist_3.shape[1] * 2 - 1))
X_new = np.zeros((X.shape[0], X.shape[1] * 2 - 1))
Y_new = np.zeros((Y.shape[0], Y.shape[1] * 2 - 1))
count = 0
for j in range(hist_3.shape[1] - 1):
    hist_new[:, count] = hist_3[:, j]
    hist_new[:, count + 1] = hist_3[:, j]
    X_new[:, count] = X[:, j]
    X_new[:, count + 1] = (X[:, j] + X[:, j + 1]) / 2
    Y_new[:, count] = Y[:, 0]
    Y_new[:, count + 1] = Y[:, 0]
    count += 2
Y_new[:, -1] = Y[:, 0]
Y_new[:, -2] = Y[:, 0]
X_new[:, -1] = X[:, -1]
X_new[:, -2] = (X[:, -1] + X[:, -2]) / 2
hist_new[:, -1] = hist_3[:, -1]
hist_new[:, -2] = hist_3[:, -1]
hist_new[:, -3] = hist_3[:, -1]
scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(hist_new), np.max(hist_new)), cmap=color_map)
ax.plot_surface(Y_new, X_new, cue + 0 * hist_new, facecolors=scam.to_rgba(hist_new))

# Variable cue
scam = plt.cm.ScalarMappable(
    norm=mpl.colors.Normalize(np.min(hist[bin_start - 1:, :]), np.max(hist[bin_start - 1:, :])), cmap=color_map)
ax.plot_surface(Y, X, (cue + 1) + 0 * hist[bin_start - 1:, :], facecolors=scam.to_rgba(hist[bin_start - 1:, :]),
                antialiased=True, rstride=1, cstride=1, alpha=None)
ax.set_ylabel("Time (s)")
ax.set_xlabel("Magnitude (" + r"$\mu$" + "l)")
ax.set_zticks([])
ax.set_yticks([0, 1.5, 3, 6], ["0", "1.5", "3", "6"])
ax.set_xticks([1, 4.5, 8], ["1", "4.5", "8"])
# plt.colorbar()
# fig.subplots_adjust(left=0, right=0, bottom=0, top=0)
fig.tight_layout()
# plt.savefig(save_dir+"\heat_map_simulations.eps", bbox_inches=0)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(1 * horizontal_size, 1 * vertical_size))
ax.imshow(hist[bin_start - 1:, :], extent=[time[0], time[-1], reward[-1], reward[0]], aspect="auto", cmap="coolwarm")
ax.set_xlabel("Time")
ax.set_ylabel("Magnitude")
ax.set_title("Decoded density")
# plt.savefig(save_dir+"\pdf_variable_cue_simulations.svg", bbox_inches=0)
plt.show()

pdb.set_trace()

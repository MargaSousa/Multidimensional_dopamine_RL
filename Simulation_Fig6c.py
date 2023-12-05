import pdb
import time as timer

start_time = timer.time()
import numpy as np
import matplotlib as mpl

# Parameters for plots
length_ticks = 5
font_size = 22
linewidth = 2.4
scatter_size = 2
length_ticks = 2
scatter_size = 20
horizontal_size = 2.5
vertical_size = 2.5
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size - 5
mpl.rcParams['ytick.labelsize'] = font_size - 5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = font_size - 2
mpl.rcParams['legend.fontsize'] = font_size - 2


def predict_time_reward(Value, smooth):
    """Given a value at the cue Value and a smoothing parameter alpha, predict the reward time."""
    p = np.zeros(N_time)
    for i in range(L):
        p += (s[i] ** 2) / ((s[i] ** 2) + (smooth ** 2)) * (np.dot(U[:, i], Value) * vh[i, :] / s[i])
    p[p < 0] = 0
    p = p / np.sum(p)
    pos_max = np.argmax(p)
    return time_disc[pos_max], p


def get_action(value, temperature_policy):
    """Output the softmax policy over value with temperature parameter temperature_policy."""
    value[value > 3429] = 3429  # So it doesn't overflow
    policy = np.round(np.exp(temperature_policy * value), 10)
    policy = policy / np.sum(policy)
    selected_action = np.random.choice(policy.shape[0], size=1, p=policy)[0]
    return policy, selected_action


def do_action(action, values, gammas, sum_reward):
    """ Exectute an action, and update values and cumulative rewards."""
    for t_action in range(end_time_reward[action]):
        rew = np.random.choice(a=reward[action, t_action, :], p=probability[action, t_action, :])
        if t_action < end_time_reward[action] - 1:
            if isinstance(gammas, np.ndarray):
                values[action, :, t_action] += alpha * (
                        rew + gammas * values[action, :, t_action + 1] - values[action, :, t_action])
            else:
                values[action, t_action] += alpha * (
                        rew + gammas * values[action, t_action + 1] - values[action, t_action])
        else:
            if isinstance(gammas, np.ndarray):
                values[action, :, t_action] += alpha * (rew - values[action, :, t_action])
            else:
                values[action, t_action] += alpha * (rew - values[action, t_action])
        sum_reward += rew
    return values, sum_reward


def do_action_sr(action, sr, gamma, sum_reward):
    """ Exectute an action, and update occupancy matrix and cumulative rewards."""
    for t_action in range(end_time_reward[action]):
        rew = np.random.choice(a=reward[action, t_action, :], p=probability[action, t_action, :])
        if t_action < end_time_reward[action] - 1:
            sr[action, t_action] += alpha * (1 + gamma * sr[action, t_action + 1] - sr[action, t_action])
        else:
            sr[action, t_action] += alpha * (1 - sr[action, t_action])
        sum_reward += rew
    return sr, sum_reward


# Simulate Planning example

# Define environment

scale_time = 0.25
time_disc = np.arange(0, 12, scale_time)
N_time = len(time_disc)

# Stationary environment (Figure 6c left)
environment_name = "stationary"
n_actions = 2  # number of patches
reward_magnitude = np.array([[0, 1], [0, 2]])
n_magnitudes = reward_magnitude.shape[1]
probability_magnitude = np.array([[0, 1], [0, 1]])
init_time_reward = [0, 0]
end_time_reward = [N_time, N_time]

# Non-stationary environment (Figure 6c right)
# environment_name = "non-stationary"
# n_actions = 3  # number of patches
# reward_magnitude = np.array([[0, 1], [0, 2], [0, 2]])
# n_magnitudes = reward_magnitude.shape[1]
# probability_magnitude = np.array([[0, 1], [0.5, 0.5], [0, 1]])
# bin_3 = np.where(time_disc >= 3)[0][0]
# bin_6 = np.where(time_disc >= 6)[0][0]
# bin_12 = N_time
# init_time_reward = [0, 0, bin_6]
# end_time_reward = [bin_3, bin_3, bin_12]

# General
reward = np.zeros((n_actions, N_time, n_magnitudes))
probability = np.zeros((n_actions, N_time, n_magnitudes))
for patch in range(n_actions):
    reward[patch, init_time_reward[patch]:end_time_reward[patch], :] = reward_magnitude[patch, :]
    probability[patch, init_time_reward[patch]:end_time_reward[patch], :] = probability_magnitude[patch, :]
    probability[patch, :init_time_reward[patch], :] = [1, 0]
    probability[patch, end_time_reward[patch]:, :] = [1, 0]

temperature_policy = 0.2  # temperature parameter for softmax
n_neurons = 200
gammas = np.linspace(1.0 / n_neurons, 1, n_neurons)

# To compute Laplace inverse transform and predict reward time for each patch
F = np.zeros((n_neurons, N_time))
for i_t, t in enumerate(time_disc):
    F[:, i_t] = gammas ** t
U, s, vh = np.linalg.svd(F, full_matrices=False)
L = np.shape(U)[1]

T_early = 1000
T = 30000
alpha = 0.1  # Learning rate
gamma = 0.99  # Reference temporal discount factor
n_runs = 10  # Number of runs

# Standard value RL
all_runs_value_early = []
all_runs_value_late = []
for r in range(n_runs):
    sum_reward_value = 0
    t = 0
    past_early = False
    value = np.zeros((n_actions, N_time))  # Initialize value for each option
    while t < T:
        _, action = get_action(value[:, 0], temperature_policy)
        t += end_time_reward[action]
        value, sum_reward_value = do_action(action, value, gamma, sum_reward_value)
        if not past_early and t > T_early:
            all_runs_value_early.append(sum_reward_value)
            past_early = True
    all_runs_value_late.append(sum_reward_value)

# SR
temperature_policy = 0.1
all_runs_sr_early = []
all_runs_sr_late = []
n_runs = 10
cum_expected_value_reward = np.sum(reward * probability, axis=(1, 2))
for r in range(n_runs):
    sum_reward_sr = 0
    t = 0
    past_early = False
    sr = np.ones((n_actions, N_time)) * 0.25
    while t < T:
        value = sr[:,0] * cum_expected_value_reward
        _, action = get_action(value, temperature_policy)
        t += end_time_reward[action]
        sr, sum_reward_sr = do_action_sr(action, sr, gamma, sum_reward_sr)
        if not past_early and t > T_early:
            all_runs_sr_early.append(sum_reward_sr)
            past_early = True
    all_runs_sr_late.append(sum_reward_sr)


def predicted_time_all_actions(values, smooth):
    for action in range(n_actions):
        predicted_time[action], prob = predict_time_reward(values[action, :, 0], smooth)
    return predicted_time


# Distributional RL in time
reference_gamma = np.where(gammas == gamma)[0][0]
actions = np.arange(n_actions)
all_runs_dist_rl_early = []
all_runs_dist_rl_late = []
predicted_time = np.zeros(n_actions)
for r in range(n_runs):
    sum_reward = 0
    t = 0
    past_early = False
    values = np.zeros((n_actions, n_neurons, N_time))
    while t < T:
        predicted_time = predicted_time_all_actions(values, 25)
        earlier_patches = np.intersect1d(np.where(predicted_time < predicted_time[-1])[0],
                                         np.where(values[:, reference_gamma, 0] > 0))
        if len(earlier_patches) > 0:
            _, action = get_action(values[earlier_patches, reference_gamma, 0], temperature_policy)
            # Go to earlier patched and then to later
            values, sum_reward = do_action(action, values, gammas, sum_reward)
            if end_time_reward[action] < end_time_reward[-1]:
                values, sum_reward = do_action(-1, values, gammas, sum_reward)
                t += end_time_reward[-1]
            else:
                t += end_time_reward[action]

        else:
            _, action = get_action(values[:, reference_gamma, 0], temperature_policy)
            values, sum_reward = do_action(action, values, gammas, sum_reward)
            t += end_time_reward[action]
        if not past_early and t > T_early:
            all_runs_dist_rl_early.append(sum_reward)
            past_early = True
    all_runs_dist_rl_late.append(sum_reward)

print(all_runs_sr_late)
print(all_runs_value_late)
print(all_runs_dist_rl_late)

pdb.set_trace()

np.save("foraging_runs_dist_rl_early_" + environment_name + ".npy", all_runs_dist_rl_early)
np.save("foraging_runs_dist_rl_late_" + environment_name + ".npy", all_runs_dist_rl_late)
np.save("foraging_runs_value_early_" + environment_name + ".npy", all_runs_value_early)
np.save("foraging_runs_value_late_" + environment_name + ".npy", all_runs_value_late)
np.save("foraging_runs_sr_early_" + environment_name + ".npy", all_runs_sr_early)
np.save("foraging_runs_sr_late_" + environment_name + ".npy", all_runs_sr_late)

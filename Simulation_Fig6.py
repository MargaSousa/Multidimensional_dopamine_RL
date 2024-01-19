import pdb
import time as timer
start_time = timer.time()
import matplotlib as mpl
from aux_Fig6 import *

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

# Define environment
scale_time = 0.25
time = np.arange(0, 12, scale_time)
N_time = len(time)
gamma_hungry = 0.6
environment_name = "hungry-sated"  # "hungry-sated" (Figure 6e), "stationary" (Figure 6c left), "non-stationary" (Figure 6c right)
dict_environment = np.load(environment_name + "-environment.npy", allow_pickle=True)
dict_environment = dict_environment.item()
n_actions = dict_environment["n_actions"]  # number of patches
reward_magnitude = dict_environment["reward_magnitude"]
n_magnitudes = reward_magnitude.shape[1]
probability_magnitude = dict_environment["probability_magnitude"]
init_time_reward = dict_environment["init_time_reward"]
end_time_reward = dict_environment["end_time_reward"]

# Convert from time to bin number
for action in range(n_actions):
    init_time_reward[action] = np.where(time >= init_time_reward[action])[0][0]
    end_time_reward[action] = np.where(time >= end_time_reward[action])[0][0]

# General
reward = np.zeros((n_actions, N_time, n_magnitudes))
probability = np.zeros((n_actions, N_time, n_magnitudes))
for patch in range(n_actions):
    reward[patch, init_time_reward[patch]:end_time_reward[patch], :] = reward_magnitude[patch, :]
    probability[patch, init_time_reward[patch]:end_time_reward[patch], :] = probability_magnitude[patch, :]
    probability[patch, :init_time_reward[patch], :] = [1, 0]
    probability[patch, end_time_reward[patch]:, :] = [1, 0]

temperature_policy_init = 0.1  # initial temperature parameter for softmax
temperature_policy = 0.3
n_neurons = 200
gammas = np.linspace(1.0 / n_neurons, 1, n_neurons)

# To compute Laplace inverse transform and predict reward time for each patch
F = np.zeros((n_neurons, N_time))
for i_t, t in enumerate(time):
    F[:, i_t] = gammas ** t
U, s, vh = np.linalg.svd(F, full_matrices=False)

T_early = 20000  # 20000
T = 50000

alpha = 0.2  # Learning rate
gamma = 0.99  # Reference temporal discount factor
n_runs = 10  # Number of runs

# Standard value RL
all_runs_value_early = []
all_runs_value_late = []
all_runs_value_policy_early = np.empty((0, 3))
all_runs_value_policy_late = np.empty((0, 3))

for r in range(n_runs):
    sum_reward_value = 0
    t = 0
    past_early = False
    value = np.zeros((n_actions, N_time))  # Initialize value for each option
    if environment_name == "hungry-sated":
        value = initalize_hungry_state_value(value, gamma_hungry, time, reward, probability)
    while t < T:
        policy, action = get_action(value[:, 0], t, temperature_policy_init, temperature_policy, T)
        t += end_time_reward[action]
        value, sum_reward_value = do_action(action, value, gamma, sum_reward_value, end_time_reward, reward,
                                            probability, alpha)
        if not past_early and t > T_early:
            all_runs_value_early.append(sum_reward_value)
            all_runs_value_policy_early = np.concatenate((all_runs_value_policy_early, np.expand_dims(policy, axis=0)))
            past_early = True
    all_runs_value_late.append(sum_reward_value)
    all_runs_value_policy_late = np.concatenate((all_runs_value_policy_late, np.expand_dims(policy, axis=0)))

# SR
all_runs_sr_early = []
all_runs_sr_late = []
all_runs_sr_policy_early = np.empty((0, 3))
all_runs_sr_policy_late = np.empty((0, 3))

n_runs = 10
for r in range(n_runs):
    sum_reward_sr = 0
    t = 0
    past_early = False
    sr = np.zeros((n_actions, N_time, N_time))  # matrix M(chosen patch, state t)
    expected_reward = np.zeros((n_actions, N_time))  # average estimate of reward function

    # For the hungry-sated case
    if environment_name == "hungry-sated":
        sr = initalize_hungry_state_sr(sr, gamma, time)
        expected_reward = np.sum((reward ** 4) * probability, axis=2)

    while t < T:
        value = np.sum(sr[:, 0, :] * expected_reward[:, :], axis=1)
        policy, action = get_action(value, t, temperature_policy_init, temperature_policy, T)
        t += end_time_reward[action]
        sr, expected_reward, sum_reward_sr = do_action_sr(action, sr, expected_reward, gamma, sum_reward_sr,
                                                          end_time_reward, reward, probability, alpha)
        if not past_early and t > T_early:
            all_runs_sr_early.append(sum_reward_sr)
            all_runs_sr_policy_early = np.concatenate((all_runs_sr_policy_early, np.expand_dims(policy, axis=0)))
            past_early = True
    all_runs_sr_late.append(sum_reward_sr)
    all_runs_sr_policy_late = np.concatenate((all_runs_sr_policy_late, np.expand_dims(policy, axis=0)))

# Distributional RL in time
reference_gamma = np.where(gammas == gamma)[0][0]
actions = np.arange(n_actions)
all_runs_dist_rl_early = []
all_runs_dist_rl_late = []

all_runs_dist_rl_policy_early = np.empty((0, 3))
all_runs_dist_rl_policy_late = np.empty((0, 3))

values = np.zeros((n_actions, n_neurons, N_time))
for r in range(n_runs):
    sum_reward = 0
    t = 0
    past_early = False

    # For the hungry-sated case
    if environment_name == "hungry-sated":
        values = initalize_hungry_state_values(gammas, values, time, reward, probability)
        predicted_times, probability_time = predict_time_all_actions(values, time, U, s, vh, 25)
        values = recompute_values_sated(gammas, values, time, probability_time)

    while t < T:
        predicted_time, prob_time = predict_time_all_actions(values, time, U, s, vh, 25)
        earlier_patches = np.intersect1d(np.where(predicted_time < predicted_time[-1])[0],
                                         np.where(values[:, reference_gamma, 0] > 0))

        if len(earlier_patches) > 0 and environment_name != "hungry-sated":
            _, action = get_action(values[earlier_patches, reference_gamma, 0], t, temperature_policy_init,
                                   temperature_policy, T)

            # Go to earlier patches and then to later
            values, sum_reward = do_action(action, values, gammas, sum_reward, end_time_reward, reward, probability,
                                           alpha)
            if end_time_reward[action] < end_time_reward[-1]:
                values, sum_reward = do_action(-1, values, gammas, sum_reward, end_time_reward, reward, probability,
                                               alpha)
                t += end_time_reward[-1]
            else:
                t += end_time_reward[action]

        else:
            policy, action = get_action(values[:, reference_gamma, 0], t, temperature_policy_init, temperature_policy,
                                        T)
            values, sum_reward = do_action(action, values, gammas, sum_reward, end_time_reward, reward, probability,
                                           alpha)
            t += end_time_reward[action]
        if not past_early and t > T_early:
            all_runs_dist_rl_early.append(sum_reward)
            all_runs_dist_rl_policy_early = np.concatenate(
                (all_runs_dist_rl_policy_early, np.expand_dims(policy, axis=0)))
            past_early = True
    all_runs_dist_rl_late.append(sum_reward)
    all_runs_dist_rl_policy_late = np.concatenate((all_runs_dist_rl_policy_late, np.expand_dims(policy, axis=0)))

print("time elapsed: {:.2f}s".format(timer.time() - start_time))

if environment_name == "hungry-sated":
    # Save policy
    np.save("foraging_runs_dist_rl_policy_early_" + environment_name + ".npy", all_runs_dist_rl_policy_early)
    np.save("foraging_runs_dist_rl_policy_late_" + environment_name + ".npy", all_runs_dist_rl_policy_late)
    np.save("foraging_runs_value_policy_early_" + environment_name + ".npy", all_runs_value_policy_early)
    np.save("foraging_runs_value_policy_late_" + environment_name + ".npy", all_runs_value_policy_late)
    np.save("foraging_runs_sr_policy_early_" + environment_name + ".npy", all_runs_sr_policy_early)
    np.save("foraging_runs_sr_policy_late_" + environment_name + ".npy", all_runs_sr_policy_late)

else:
    # Save cumulative rewards
    np.save("foraging_runs_dist_rl_early_" + environment_name + ".npy", all_runs_dist_rl_early)
    np.save("foraging_runs_dist_rl_late_" + environment_name + ".npy", all_runs_dist_rl_late)
    np.save("foraging_runs_value_early_" + environment_name + ".npy", all_runs_value_early)
    np.save("foraging_runs_value_late_" + environment_name + ".npy", all_runs_value_late)
    np.save("foraging_runs_sr_early_" + environment_name + ".npy", all_runs_sr_early)
    np.save("foraging_runs_sr_late_" + environment_name + ".npy", all_runs_sr_late)

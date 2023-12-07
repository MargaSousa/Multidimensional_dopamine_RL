import matplotlib.pyplot as plt
import numpy as np


def get_action(value, t, temperature_policy_init, temperature_policy, T):
    """Output the softmax policy over value with temperature parameter temperature_policy."""
    value[value > 3429] = 3429  # So it doesn't overflow
    if t < T * 0.1:
        policy = np.exp(temperature_policy_init * value)  # To guarantee the agent explores
    else:
        policy = np.exp(temperature_policy * value)
    policy = policy / np.sum(policy)
    selected_action = np.random.choice(policy.shape[0], size=1, p=policy)[0]
    return policy, selected_action


def do_action(action, values, gammas, sum_reward, end_time_reward, reward, probability, alpha):
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


def do_action_sr(action, sr, expected_reward, gamma, sum_reward, end_time_reward, reward, probability, alpha):
    """ Exectute an action, and update occupancy matrix and average estimate of reward for each state."""
    N_time = sr.shape[1]
    for t_action in range(end_time_reward[action]):
        rew = np.random.choice(a=reward[action, t_action, :], p=probability[action, t_action, :])
        I = np.zeros(N_time)
        I[t_action] = 1
        if t_action < end_time_reward[action] - 1:
            sr[action, t_action, :] += alpha * (I + gamma * sr[action, t_action + 1, :] - sr[action, t_action, :])
        else:
            sr[action, t_action, :] += alpha * (I - sr[action, t_action, :])
        sum_reward += rew
        expected_reward[action, t_action] += alpha * (rew - expected_reward[action, t_action])
    return sr, expected_reward, sum_reward


def initalize_hungry_state_value(value, gamma, time, reward, probability):
    """ Initialize value for a hungry state."""
    n_actions = value.shape[0]
    for action in range(n_actions):
        expected_rewards = np.sum((reward[action, :, :] ** 4) * probability[action, :, :],
                                  axis=1)  # convex utility function
        for i_t, t in enumerate(time):
            value[action, i_t] = np.sum(expected_rewards[i_t:] * gamma ** (time)[i_t:])
    return value


def initalize_hungry_state_sr(sr, gamma, time):
    """ Initialize sr for a hungry state."""
    n_actions = sr.shape[0]
    for action in range(n_actions):
        for i_t, t in enumerate(time):
            sr[action, i_t, i_t:] = gamma ** (time - t)[i_t:]
    return sr


def predict_time_reward(Value, time, U, s, vh, smooth):
    """Given a value at the cue Value and a smoothing parameter alpha, predict the expected rewards over time."""
    N_time = time.shape[0]
    p = np.zeros(N_time)
    L = np.shape(U)[1]
    for i in range(L):
        p += (s[i] ** 2) / ((s[i] ** 2) + (smooth ** 2)) * (np.dot(U[:, i], Value) * vh[i, :] / s[i])
    pos_max = np.argmax(p)
    return time[pos_max], p


def predict_time_all_actions(values, time, U, s, vh, smooth):
    n_actions = values.shape[0]
    N_time = values.shape[2]
    predicted_time = np.zeros(n_actions)
    probability_time = np.zeros((n_actions, N_time))
    for action in range(n_actions):
        predicted_time[action], probability_time[action, :] = predict_time_reward(values[action, :, 0], time, U, s, vh,
                                                                                  smooth)
    return predicted_time, probability_time


def initalize_hungry_state_values(gammas, values, time, reward, probability):
    """ Predict reward time probability and recompute value for each patch."""
    n_actions = values.shape[0]
    # Predict reward time
    for action in range(n_actions):
        expected_rewards = np.sum((reward[action, :, :] ** 4) * probability[action, :, :],
                                  axis=1)  # convex utility function
        for i_t, t in enumerate(time):
            for i_neuron, gamma_neuron in enumerate(gammas):
                values[action, i_neuron, i_t] = np.sum(expected_rewards[i_t:] * gamma_neuron ** (time - t)[i_t:])
    return values


def recompute_values_sated(gammas, values, time, probability_time):
    # Recompute value
    n_actions = probability_time.shape[0]
    for action in range(n_actions):
        plt.plot(probability_time[action])
        for i_t, t in enumerate(time):
            for i_neuron, gamma_neuron in enumerate(gammas):
                values[action, i_neuron, i_t] = np.sum(probability_time[action, :] * gamma_neuron ** (time))
    return values

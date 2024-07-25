from aux_functions import *


def get_action(value,  temperature_policy):
    """Output the softmax policy over value with temperature parameter temperature_policy."""
    policy = np.exp(temperature_policy * value)
    policy = policy / np.sum(policy)
    selected_action = np.random.choice(policy.shape[0], size=1, p=policy)[0]
    return policy, selected_action


def do_action(action, values, gammas, end_time_reward, reward, probability, alpha, scale_time):
    """ Exectute an action, and update values and cumulative rewards."""
    reward_trial=0
    for t_action in range(end_time_reward[action]):
        rew = np.random.choice(a=reward[action, t_action, :], p=probability[action, t_action, :])
        if t_action < end_time_reward[action] - 1:
            if isinstance(gammas, np.ndarray):
                values[action, :, t_action] += alpha * ( rew + (gammas**scale_time) * values[action, :, t_action + 1] - values[action, :, t_action])
            else:
                values[action, t_action] += alpha * (rew + (gammas**scale_time) * values[action, t_action + 1] - values[action, t_action])
        else:
            if isinstance(gammas, np.ndarray):
                values[action, :, t_action] += alpha * (rew - values[action, :, t_action])
            else:
                values[action, t_action] += alpha * (rew - values[action, t_action])

        reward_trial += rew*gammas**(scale_time*t_action)

    return values, reward_trial



def do_action_TMRL(action, Value, gammas,taus, end_time_reward, reward, probability, alpha, scale_time, batch_size,gamma,is_discounted):
    """ Exectute an action, and update values and cumulative rewards."""
    reward_trial=0

    for t_action in range(end_time_reward[action]):
        print(reward[action, t_action, :],probability[action, t_action, :])
        rew = np.random.choice(a=reward[action, t_action, :],p=probability[action, t_action, :])
        if t_action < end_time_reward[action] - 1:

            # Next time-step value resorted
            #sorted_value=np.reshape(Value[action,:,t_action+1],(n_unique,n_unique))

            ## A possible way to implement imputation
            #future_samples=np.apply_along_axis(np.random.choice, axis=1, arr=sorted_value, size=50)
            #future_samples=np.nanmean(future_samples, axis=1).reshape(-1,1)
            #future_samples=np.repeat(future_samples,n_unique,axis=1)
            #future_samples.flatten()

            # Batch size x number of values
            imputation = rew + (gammas**scale_time)*Value[action,:,t_action+1]
        else:
            imputation = rew

        error = imputation - Value[action,:, t_action]
        Value[action,:, t_action] += alpha * (taus * error * (error > 0) + (1 - taus) * error * (error < 0))

        if is_discounted:
            reward_trial += rew*gamma**(scale_time*t_action)
        else:
            reward_trial+=rew
    return Value, reward_trial

def do_action_sr(action, sr, expected_reward, gamma,end_time_reward, reward, probability, alpha, scale_time):
    """ Exectute an action, and update occupancy matrix and average estimate of reward for each state."""
    N_time = sr.shape[1]
    reward_episode=0
    for t_action in range(end_time_reward[action]):
        rew = np.random.choice(a=reward[action, t_action, :], p=probability[action, t_action, :])
        I = np.zeros(N_time)
        I[t_action] = 1
        if t_action < end_time_reward[action] - 1:
            sr[action, t_action, :] += alpha * (I + (gamma**scale_time) * sr[action, t_action + 1, :] - sr[action, t_action, :])
        else:
            sr[action, t_action, :] += alpha * (I - sr[action, t_action, :])
        reward_episode += rew*gamma**(scale_time*t_action)
        expected_reward[action, t_action] += alpha * (rew - expected_reward[action, t_action])
    return sr, expected_reward, reward_episode


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
        predicted_time[action], probability_time[action, :] = predict_time_reward(values[action, :, 0], time, U, s, vh, smooth)
    return predicted_time, probability_time
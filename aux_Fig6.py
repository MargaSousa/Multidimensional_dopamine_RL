import matplotlib.pyplot as plt

from aux_functions import *
from scipy.stats import gaussian_kde

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


def run_decoder_magnitude_time(values,gammas,taus,time,reward,alpha):
    bin_start=15
    min_reward=np.min(reward)
    max_reward=np.max(reward)
    n_time=len(time)
    n_reward=len(reward)
    n_neurons=values.shape[0]

    # Construct matrix F for Laplace decoder
    F = np.zeros((values.shape[1], n_time))
    for i_t, t in enumerate(time):
        F[:, i_t] = gammas ** t
    # Find eigenvectors
    U, s, vh = np.linalg.svd(F, full_matrices=True)
    L = np.min([vh.shape[0], np.shape(U)[1]])


    # Laplace decoder applied to each column
    synthetic_delta_fr_time = np.zeros((values.shape[1], n_time))

    for neuron_tau in range(values.shape[1]):
        for i in range(L):
            synthetic_delta_fr_time[neuron_tau, :] += (s[i] ** 2 / (s[i] ** 2 + alpha ** 2)) * (np.dot(U[:, i], values[:,neuron_tau]) * vh[i, :]) / s[i]

        negative_pos=np.where(synthetic_delta_fr_time[neuron_tau,:]<0)[0]
        synthetic_delta_fr_time[neuron_tau,negative_pos]=0


    # Correct scale of reward magnitude
    max_mean_neuron=np.max(synthetic_delta_fr_time)
    min_mean_neuron=np.min(synthetic_delta_fr_time)
    synthetic_delta_fr_time=max_reward*(synthetic_delta_fr_time-min_mean_neuron)/(max_mean_neuron-min_mean_neuron)


    # Expectile decoder for each column
    hist = np.zeros((n_reward, n_time))
    all_samples=[]
    for bin_time in range(n_time):
        sampled_dist_synthetic, loss_synthetic = run_decoding_magnitude(synthetic_delta_fr_time[:, bin_time], taus,np.ones(n_neurons), N=20, minv=np.min(synthetic_delta_fr_time),maxv=np.max(synthetic_delta_fr_time), max_samples=100, max_epochs=5,method='TNC')
        all_samples.append(list(sampled_dist_synthetic))
        if np.sum(sampled_dist_synthetic) == 0:
            hist[:, bin_time] = 0

        else:
            kde = gaussian_kde(sampled_dist_synthetic, bw_method=1)
            smoothed_pdf= kde.pdf(reward)
            smoothed_pdf=smoothed_pdf/np.sum(smoothed_pdf)
            hist[:, bin_time] = smoothed_pdf
    return hist

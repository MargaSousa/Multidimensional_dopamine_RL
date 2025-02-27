import pdb
import time as timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
start_time = timer.time()
import matplotlib as mpl
from aux_Fig6 import *
from aux_functions import *
import os

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
mpl.use("TkAgg")

# Environment
dict_environment = np.load("environment.npy", allow_pickle=True)
dict_environment = dict_environment.item()
n_actions = dict_environment["n_actions"]  # number of patches
reward_magnitude = dict_environment["reward_magnitude"]
n_magnitudes = reward_magnitude.shape[1]
probability_magnitude = dict_environment["probability_magnitude"]
init_time_reward = dict_environment["init_time_reward"]
end_time_reward = dict_environment["end_time_reward"]


n_unique = 150 # Number of unique values for taus and gammas
unique_taus = np.linspace(1.0 / n_unique, 1.0 , n_unique) # taus (optimism level)
unique_gammas = np.linspace(1.0/n_unique, 1, n_unique) # temporal discount factor
mesh = np.meshgrid(unique_taus, unique_gammas)

taus = np.ndarray.flatten(mesh[0])
gammas = np.ndarray.flatten(mesh[1])
n_neurons = taus.shape[0]


alpha = 0.02 # Learning rate
n_runs = 3 # Number of runs
scale_time=0.5
number_states=11.0/scale_time
time = np.arange(0, 12, scale_time)
N_time = len(time)



init_bin_reward=np.copy(init_time_reward)
end_bin_reward=np.copy(end_time_reward)

# Convert from time to bin number
for action in range(n_actions):
    init_bin_reward[action] = np.where(time >= init_time_reward[action])[0][0]
    end_bin_reward[action] = np.where(time >= end_time_reward[action])[0][0]


# Construct arrays with reward magnitude and probability over time
reward = np.zeros((n_actions, N_time, n_magnitudes))
probability = np.zeros((n_actions, N_time, n_magnitudes))
for patch in range(n_actions):
    reward[patch, init_bin_reward[patch]:end_bin_reward[patch], :] = reward_magnitude[patch, :] * scale_time
    probability[patch, init_bin_reward[patch]:end_bin_reward[patch], :] = probability_magnitude[patch, :]
    probability[patch, :init_bin_reward[patch], :] = [1, 0]
    probability[patch, end_bin_reward[patch]:, :] = [1, 0]



algo="TMRL"
task="sated_hungry" #"sated_hungry", "hungry_dusk_dawn" or "dusk_dawn"
task_dictionary=np.load(algo+"_"+task+".npy",allow_pickle=True)
task_dictionary = task_dictionary.item()
n_trials=task_dictionary["n_trials"]
trial_state_change=task_dictionary["trial_state_change"]
temperature_policy = 0.5 #this parameter is not really being used, we are using a random policy
utility_power_before=task_dictionary["utility_power_before"]
utility_power_after=task_dictionary["utility_power_after"]
gamma_before=task_dictionary["gamma_before"]
gamma_after=task_dictionary["gamma_after"]
alpha_decoder=0.00000001 # smoothing parameter for decoding future reward time


# Neuron that is used to select action
reference_neuron_before = np.intersect1d(np.where(np.round(gammas,2)==gamma_before)[0],np.where(np.round(taus,3)==0.5)[0])[0]
reference_neuron_after = np.intersect1d(np.where(np.round(gammas,2)>=gamma_after)[0],np.where(np.round(taus,3)==0.5)[0])[0]


trials_reference = np.arange(n_trials)
tmrl_prob_actions= np.empty((n_runs,len(trials_reference),n_actions))
tmrl_prob_actions[:,:,:]=np.nan
sum_rewards=np.zeros((n_runs,500))
actions = np.arange(n_actions)
action_after_change_state=np.empty(n_runs)
action_after_change_state[:]=np.nan
utility_after_change_state=np.empty(n_runs)
utility_after_change_state[:]=np.nan
value_save= np.empty((n_runs,n_trials,n_actions))
value_save[:,:]=np.nan



for r in range(n_runs):

    values = np.zeros((n_actions, n_neurons, N_time))
    sum_reward_tmrl = 0
    track_reward=[]
    track_policy = []
    trial_number=0
    is_recompute = task_dictionary["is_recompute"]

    while trial_number < n_trials:

        if trial_number==trial_state_change+1:
            sum_reward_tmrl=0

        if trial_number<trial_state_change:
            policy, action = get_action(values[:, reference_neuron_before, 0],temperature_policy)
            action = np.random.randint(3)
            values, reward_trial = do_action_TMRL(action, values, gammas, taus, end_bin_reward, reward**utility_power_before, probability, alpha, scale_time,200,gamma_before,True)
            value_save[r,trial_number,:]=values[:, reference_neuron_before, 0]
        else:
            if is_recompute:
                for patch in range(n_actions):

                    rew_decoder = np.linspace(np.min(reward[patch, :]**utility_power_before), np.max(reward[patch, :]**utility_power_before),120)

                    values_cue_sorted = np.reshape(values[patch, :, 0], (n_unique, n_unique))

                    # Decode joint probability distribution over time and magnitude
                    joint_pdf = run_decoder_magnitude_time(values_cue_sorted, unique_gammas, unique_taus, time,rew_decoder, alpha_decoder)  # np.max(reward_magnitude)

                    # Normalize for each time
                    joint_pdf = joint_pdf / np.sum(joint_pdf, axis=0).reshape(1, -1)


                    # Mean of bins
                    amount_decode=np.copy(rew_decoder**(utility_power_after/utility_power_before))
                    amount_decode[1:-1]=(amount_decode[1:-1]+amount_decode[2:])/2
                    amount = np.repeat((amount_decode).reshape(-1, 1), joint_pdf.shape[1], axis=1)

                    # Compute expected value over time
                    expected_over_time = np.sum(amount * joint_pdf, axis=0)

                    # Recompute reference value
                    for i_t, t in enumerate(time):
                        values[patch, reference_neuron_after, i_t] = np.sum(expected_over_time[i_t:] * gamma_after ** (time - t)[i_t:])
                is_recompute=False

            policy, action = get_action(values[:, reference_neuron_after, 0], temperature_policy)
            if trial_number==trial_state_change:
                action=np.argmax(values[:, reference_neuron_after, 0])
            else:
                action = np.random.randint(3)

            values, reward_trial = do_action_TMRL(action, values, gammas, taus, end_bin_reward, reward**utility_power_after, probability, alpha, scale_time, n_unique,gamma_after,True)
            value_save[r,trial_number,:]=values[:, reference_neuron_after, 0]

        sum_reward_tmrl+= reward_trial
        track_reward.append(sum_reward_tmrl)
        track_policy.append(policy)
        trial_number+=1

    track_policy=np.asarray(track_policy)
    tmrl_prob_actions[r, :, :]=track_policy
    for patch in range(n_actions):
        plt.plot(trials_reference-trial_state_change,value_save[r,:,patch])
    plt.xlim(-200,3000)
    plt.show()


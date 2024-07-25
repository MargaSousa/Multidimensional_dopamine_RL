import pdb
import time as timer

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
start_time = timer.time()
import matplotlib as mpl
from aux_Fig6 import *
from aux_functions import *

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

# Where environment and task features are saved
main_dir=r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\Simulations"


# Environment
dict_environment = np.load(os.path.join(main_dir,"environment.npy"), allow_pickle=True)
dict_environment = dict_environment.item()
n_actions = dict_environment["n_actions"]  # number of patches
reward_magnitude = dict_environment["reward_magnitude"]
n_magnitudes = reward_magnitude.shape[1]
probability_magnitude = dict_environment["probability_magnitude"]
init_time_reward = dict_environment["init_time_reward"]
end_time_reward = dict_environment["end_time_reward"]


alpha = 0.02 # Learning rate
n_runs = 10  # Number of runs
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


# Standard value RL
algo="value"
task="sated_hungry"#"dusk_dawn"#"hungry_dusk_dawn"#"dusk_dawn"#"dusk_dawn"###
task_dictionary=np.load(os.path.join(main_dir,algo+"_"+task+".npy"),allow_pickle=True)
task_dictionary = task_dictionary.item()
n_trials=task_dictionary["n_trials"]+8000 # dusk to dawn
trial_state_change=task_dictionary["trial_state_change"]+5000 # dusk to dawn
temperature_policy = task_dictionary["temperature_policy"]
utility_power_before=task_dictionary["utility_power_before"]
utility_power_after=task_dictionary["utility_power_after"]
gamma_before=task_dictionary["gamma_before"] # For sated-hungry
gamma_after=task_dictionary["gamma_after"] # For sated-hungry




# Save info
trials_reference = np.arange(n_trials)
value_prob_actions= np.empty((n_runs,len(trials_reference),n_actions))
value_prob_actions[:,:,:]=np.nan
value_save= np.empty((n_runs,n_trials,n_actions))
value_save[:,:]=np.nan
sum_rewards=np.zeros((n_runs,500))
action_after_change_state=np.empty(n_runs)
action_after_change_state[:]=np.nan
utility_after_change_state=np.empty(n_runs)
utility_after_change_state[:]=np.nan


for r in range(n_runs):

    track_policy = []
    track_reward=[]
    sum_reward_value = 0
    value = np.zeros((n_actions, N_time))  # Initialize value for each option
    trial_number=0

    while trial_number < n_trials:

        policy, action = get_action(value[:, 0],temperature_policy)


        if trial_number==trial_state_change:
            sum_reward_value=0
            action=np.argmax(value[:, 0])
        else:
            action = np.random.randint(3)


        if trial_number<trial_state_change:
            value, reward_trial = do_action(action, value, gamma_before, end_bin_reward, reward**utility_power_before,probability, alpha, scale_time)
        else:
            value, reward_trial= do_action(action, value, gamma_after, end_bin_reward, reward**utility_power_after,probability, alpha, scale_time)

        sum_reward_value += reward_trial

        if trial_number==trial_state_change:
            print("reward trial ",reward_trial)
            print("value ", value[:,0])
            print("policy ",policy)
            print("action ",action)
            action_after_change_state[r] = action
            utility_after_change_state[r] = reward_trial



        if trial_number>trial_state_change and trial_number<trial_state_change+500:
            sum_rewards[r,trial_number-trial_state_change]=sum_reward_value

        value_save[r,trial_number,:]=value[:,0]
        track_reward.append(sum_reward_value)
        track_policy.append(policy)
        trial_number+=1


    track_policy=np.asarray(track_policy)
    value_prob_actions[r,:,:] = track_policy[:,:]

    # Plot update policy
    # for patch in range(n_actions):
    #     plt.plot(trials_reference-trial_state_change,value_save[r,:,patch])
    # plt.xlim(-200,3000)
    # plt.xlabel("Time since sated")
    # plt.ylabel("Patch selection probability")
    # plt.show()

# Save
# np.save("trials_reference_"+algo+"_"+task+".npy",trials_reference-trial_state_change)
# np.save("sum_rewards_"+algo+"_"+task+".npy",sum_rewards)
# np.save("foraging_policy_runs_"+algo+"_"+task+".npy",value_prob_actions)
# np.save("foraging_value_runs_"+algo+"_"+task+".npy",value_save)
# np.save("p_optimal_action_"+algo+"_"+task+".npy",action_after_change_state)
# np.save("utility_"+algo+"_"+task+".npy",utility_after_change_state)

pdb.set_trace()



import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
# Parameters for plots
length_ticks=5
font_size=22
linewidth=2.4
scatter_size=2
length_ticks=2
scatter_size=20
horizontal_size=2.5
vertical_size=2.5
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize']=font_size-5
mpl.rcParams['ytick.labelsize']=font_size-5
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = font_size-2
mpl.rcParams['legend.fontsize'] = font_size-2

scale_time=1
gamma=0.9
temperature_policy=0.2

# For patch 1 and 2
reward_amount=[1,2]
init_time_reward=[0,6]
end_time_reward=[3,12]

# For plotting predicted distribution over reward time
value_patch_1=np.sum(reward_amount[0]*gamma**np.arange(init_time_reward[0],end_time_reward[0]+scale_time,scale_time))
value_patch_2=np.sum(reward_amount[1]*gamma**np.arange(init_time_reward[1],end_time_reward[1]+scale_time,scale_time))

n_neurons=100
gammas=np.linspace(1.0/n_neurons,1,n_neurons)
values_patch_1=np.zeros(n_neurons)
values_patch_2=np.zeros(n_neurons)

for neuron in range(n_neurons):
    values_patch_1[neuron]=np.sum(reward_amount[0]*gammas[neuron]**np.arange(init_time_reward[0],end_time_reward[0]+scale_time,scale_time))
    values_patch_2[neuron]=np.sum(reward_amount[1]*gammas[neuron]**np.arange(init_time_reward[1],end_time_reward[1]+scale_time,scale_time))


time_disc=np.arange(0,12,scale_time)
N_time=len(time_disc)
F=np.zeros((n_neurons,N_time))
for i_t,t in enumerate(time_disc):
    F[:,i_t]=gammas**t
U, s, vh = np.linalg.svd(F, full_matrices=False)
L=np.shape(U)[1]


def predict_time_reward(Value,alpha):
    """Given a value at the cue Value and a smoothing parameter alpha, predict the reward time."""
    p = np.zeros(N_time)
    for i in range(L):
        p+=(s[i]**2)/((s[i]**2)+(alpha**2))*(np.dot(U[:,i],Value)*vh[i,:]/s[i])
    p[p<0]=0
    p=p/np.sum(p)
    pos_max=np.argmax(p)
    return time_disc[pos_max],p


_,p_patch_1=predict_time_reward(values_patch_1,1)
_,p_patch_2=predict_time_reward(values_patch_2,1)

fig,ax=plt.subplots(figsize=(horizontal_size,vertical_size))
ax.tick_params(width=linewidth,length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
plt.plot(time_disc,p_patch_1,color="k")
plt.plot(time_disc,p_patch_2,color="k")
plt.yticks([])
plt.xticks([])
plt.xlabel("time")
plt.ylabel("P(reward)")
plt.show()


# Simulate Planning example

n_actions=2
temperature_policy=0.2
reward=np.zeros((n_actions,12))

# Patch 1
reward[0,init_time_reward[0]:end_time_reward[0]]=reward_amount[0]

# Patch 2
reward[1,init_time_reward[1]:end_time_reward[1]]=reward_amount[1]

epoch_training="init" # "init" or "end"

# define time range for "init" and "end" epoch in training
if epoch_training=="end":
    T=30000
else:
    T=1000

alpha=0.01 # Learning rate
gamma=0.99 # Reference temporal discount factor


def get_action(value,temperature_policy):
    """Output the softmax policy over value with temperature parameter temperature_policy."""
    policy = np.exp(temperature_policy * value)
    policy=policy/np.sum(policy)
    selected_action=np.random.choice(policy.shape[0],size=1,p=policy)[0]
    return policy,selected_action


# Standard value RL
all_runs_value=[]
n_runs=10
for r in range(n_runs):
    sum_reward_value=0
    t=0
    value = np.zeros((n_actions, np.max(time_disc)+1))  # Initialize value for each option
    sr=np.zeros((n_actions, np.max(time_disc)+1))
    while t<T:
            _,action=get_action(value[:,0],temperature_policy)
            t+=end_time_reward[action]
            for t_action in range(end_time_reward[action]):
                if t_action< end_time_reward[action]-1:
                    value[action,t_action]+=alpha*(reward[action,t_action]+gamma*value[action,t_action+1]-value[action,t_action])
                    sum_reward_value+=reward[action,t_action]
                else:
                    value[action, t_action] += alpha * (reward[action, t_action]-value[action,t_action])
                    sum_reward_value+=reward[action, t_action]

    all_runs_value.append(sum_reward_value)


# SR
all_runs_sr=[]
n_runs=10
for r in range(n_runs):
    sum_reward_value=0
    t=0
    sr=np.zeros((n_actions, np.max(time_disc)+1))
    while t<T:
            value=np.sum(sr*reward,axis=1)
            _,action=get_action(value,temperature_policy)
            t+=end_time_reward[action]
            for t_action in range(end_time_reward[action]):
                if t_action< end_time_reward[action]-1:
                    sr[action,t_action]+=alpha*(1+gamma*sr[action,t_action+1]-sr[action,t_action])
                    sum_reward_value+=reward[action,t_action]
                else:
                    sr[action, t_action] += alpha * (1-sr[action,t_action])
                    sum_reward_value+=reward[action, t_action]

    all_runs_sr.append(sum_reward_value)


def do_action(action,values,sum_reward):
    """ Exectute an action, and update value and cumulative rewards."""
    for t_action in range(end_time_reward[action]):
        if t_action < end_time_reward[action]-1:
            values[action, :, t_action] += alpha * (reward[action, t_action] + gammas * values[action,:, t_action + 1] - values[action,:, t_action])
            sum_reward += reward[action, t_action]
        else:
            values[action, :, t_action] += alpha * (reward[action, t_action] - values[action,:, t_action])
            sum_reward += reward[action, t_action]
    return values,sum_reward


# Distributional RL in time
reference_gamma=np.where(gammas==gamma)[0][0]
actions=np.arange(n_actions)
all_runs_dist_rl=[]
# Dist in time RL
for r in range(n_runs):
    sum_reward=0
    t=0
    values = np.zeros((n_actions, n_neurons, np.max(time_disc)+1))
    while t<T:
        predicted_time_action_1,prob_action_1=predict_time_reward(values[0,:,0],20)
        predicted_time_action_2,prob_action_2 = predict_time_reward(values[1, :, 0], 20)
        if predicted_time_action_1 < predicted_time_action_2 and values[0,reference_gamma,0]>0:
            values,sum_reward=do_action(0,values,sum_reward)
            values,sum_reward=do_action(1,values,sum_reward)
            t+=end_time_reward[1]
        else:
            _, action = get_action(values[:, reference_gamma, 0], temperature_policy)
            values,sum_reward=do_action(action, values, sum_reward)
            t+=end_time_reward[action]
    all_runs_dist_rl.append(sum_reward)


np.save("mice_runs_dist_rl_"+epoch_training+".npy",all_runs_dist_rl)
np.save("mice_runs_value_"+epoch_training+".npy",all_runs_value)
np.save("mice_runs_sr_"+epoch_training+".npy",all_runs_value)


pdb.set_trace()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
# Parameters for plots
length_ticks=5
font_size=22
linewidth=1.2
scatter_size=2
horizontal_size=1
vertical_size=1
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth


rv=scipy.stats.multivariate_normal(mean=[0,0],cov=20*np.array([[1,0],[0,1]]))



# Plot hungry and sated utility function
amount=np.linspace(0,10,100)
time=np.linspace(0,10,100)
mesh=np.meshgrid(amount,time)
pos=np.dstack((mesh[0],mesh[1]))
pdf=np.zeros((100,100))
#pos_fast=np.where(time>5)[0][0]
pdf[:,:]=np.tile(amount**2,(100,1)).T


pdf=rv.pdf(pos)
pdf=pdf/np.sum(pdf)


#vmin=amount[0]**2
#vmax=amount[-1]**2
fig,ax=plt.subplots(figsize=(horizontal_size,vertical_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
plt.imshow(pdf.T,extent=[amount[0],amount[-1],time[0],time[-1]],origin="lower",cmap="hot")#vmin=vmin,vmax=vmax,
plt.ylabel("time")
plt.xlabel("amount")
#plt.colorbar(label="Reward\nutility",ticks=[])
plt.xticks([])
plt.yticks([])

#plt.savefig("hungry.svg")

pdb.set_trace()
# Sated bear
n_neurons=100
gammas=np.linspace(1.0/n_neurons,1,n_neurons)
consumption_time=1
n_actions=3
temperature_policy=0.05
amount=np.array([6,5,5])
time=np.array([6,3,3])
reward=np.zeros((n_actions,np.max(time)))
reward[0,time[0]-1]=amount[0]

T=10000#100
alpha=0.01
gamma=0.99


def get_action(value,temperature_policy):
    policy = np.exp(temperature_policy * value)
    policy=policy/np.sum(policy)
    selected_action=np.random.choice(policy.shape[0],size=1,p=policy)[0]
    return policy,selected_action

all_runs_value=[]
n_runs=10

all_runs_policy_value=np.zeros((n_runs,n_actions))
# Standard value RL
for r in range(n_runs):
    sum_reward_value=0
    t=0
    # Hungry value
    value = np.zeros((n_actions, np.max(time)))  # Initialize value for each option
    value[1,:time[0]-1]=gamma**np.arange(1,time[0])*amount[1]**2
    value[2, :time[0] - 1] = gamma**np.arange(1,time[0])*(0.5*4**2+0.5*6**2)


    while t<T:
            pol,action=get_action(value[:,0],temperature_policy)
            if r==0 and t==0:
                print("initial policy ",pol)
            for t_action in range(time[action]):
                if t_action< time[action]-1:
                    value[action,t_action]+=alpha*(reward[action,t_action]+gamma*value[action,t_action+1]-value[action,t_action])
                    #sum_reward_value+=reward[action,t_action]
                else:
                    value[action, t_action] += alpha * (reward[action, t_action]-value[action,t_action])
                    sum_reward_value+=reward[action, t_action]
            t += time[action]
            #sum_rewards_value_through_time.append(sum_reward_value)
            #value_through_time.append(t)

            #if t==T-1:
                # plt.bar([1,2,3],value[:,0])
                # plt.xticks([1,2,3],["patch 1","patch 2","patch 3"])
                # plt.ylabel("Value")
                # plt.title("Middle of training")
                # plt.show()
    all_runs_value.append(sum_reward_value)
    all_runs_policy_value[r,:]=pol



print("sum_reward value rl: "+str(all_runs_value))
print("policy: "+str(pol))


#policy,_=get_action(value[:,0],temperature_policy)
# plt.bar([1, 2], value[:, 0])
# plt.xticks([1, 2], ["option 1", "option 2"])
# plt.ylabel("Value")
# plt.title("End of training")
# plt.show()
# plt.plot(value[0,:])
# plt.plot(value[1,:])
# plt.show()


#pdb.set_trace()


reference_gamma=np.where(gammas==gamma)[0][0]

#For decoding future reward
N_time=100
time_disc=np.linspace(0,6,N_time)
F=np.zeros((n_neurons,N_time))
for i_t,t in enumerate(time_disc):
    F[:,i_t]=gammas**t
U, s, vh = np.linalg.svd(F, full_matrices=False)
L=np.shape(U)[1]
def predict_time_reward(Value,alpha):
    p = np.zeros(N_time)
    for i in range(L):
        p+=(s[i]**2)/((s[i]**2)+(alpha**2))*(np.dot(U[:,i],Value)*vh[i,:]/s[i])
    p[p<0]=0
    p=p/np.sum(p)
    pos_max=np.argmax(p)
    return time_disc[pos_max],p

def do_action(action,values,sum_reward):
    for t_action in range(time[action]):
        if t_action < time[action] - 1:
            values[action, :, t_action] += alpha * (reward[action, t_action] + gammas * values[action,:, t_action + 1] - values[action,:, t_action])
            sum_reward += reward[action, t_action]
        else:
            values[action, :, t_action] += alpha * (reward[action, t_action] - values[action,:, t_action])
            sum_reward += reward[action, t_action]
    return values,sum_reward

# Debug decoder
# predicted_time_action_1,prob_action_1=predict_time_reward(gammas**time[0],0.1)
# predicted_time_action_2,prob_action_2 = predict_time_reward(gammas**time[1], 0.1)
# plt.subplot(1,2,1)
# plt.plot(time_disc,prob_action_1)
# plt.subplot(1,2,2)
# plt.plot(time_disc,prob_action_2)
# plt.show()


actions=np.arange(n_actions)
all_runs_dist_rl=[]
all_runs_policy_dist_rl=np.zeros((n_runs,n_actions))
#sum_rewards_dist_rl_through_time=[]
#dist_rl_through_time=[]


# Dist in time RL
for r in range(n_runs):
    sum_reward=0
    t=0
    # Sated
    values = np.zeros((n_actions, n_neurons, np.max(time)))
    for i_neuron in range(n_neurons):
        values[0, i_neuron, :time[0] - 1] = gammas[i_neuron] ** np.arange(1, time[0]) * (amount[0])
    while t<T:
        pol, action = get_action(values[:, reference_gamma, 0], temperature_policy)
        values,sum_reward=do_action(action, values, sum_reward)
        t+=time[action]
        #sum_rewards_dist_rl_through_time.append(sum_reward)
        #dist_rl_through_time.append(t)
    all_runs_dist_rl.append(sum_reward)
    all_runs_policy_dist_rl[r,:]=pol


all_runs_dist_rl=np.array(all_runs_dist_rl)
all_runs_value=np.array(all_runs_value)
print("sum_reward dist rl: "+str(all_runs_dist_rl))
print("policy: "+str(pol))

pdb.set_trace()

#plt.plot(dist_rl_through_time,sum_rewards_dist_rl_through_time)
#plt.plot(value_through_time,sum_rewards_value_through_time)
#plt.show()

#pdb.set_trace()

np.save("sate_bear_policy_dist_rl_end.npy",all_runs_policy_dist_rl)
np.save("sate_bear_policy_value_rl_end.npy",all_runs_policy_value)



#np.save("sate_bear_runs_dist_rl_end.npy",all_runs_dist_rl)
#np.save("sate_bear_runs_value_rl_end.npy",all_runs_value)

pdb.set_trace()


print("sum_reward dist rl: "+str(sum_reward))
plt.subplot(1,2,1)
plt.plot(time_disc,prob_action_1)
plt.subplot(1,2,2)
plt.plot(time_disc,prob_action_2)
plt.show()

for neu in range(n_neurons):
    plt.subplot(1,2,1)
    plt.plot(values[0, neu, :])
    plt.subplot(1,2,2)
    plt.plot(values[1, neu, :])
plt.show()

fig,ax=plt.subplots(figsize=(2.4,2.4))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
plt.bar([1,2],[sum_reward_value,sum_reward])
plt.xticks([1,2],["Value RL","Time dist RL"])#,rotation=90
plt.yticks([])
plt.ylabel("Cumulative rewards")
plt.title("Initial training")
plt.ylim(0,3700)
plt.show()
#plt.savefig("planning_init_training.pdf")

#print(values[:, reference_gamma, 0])
#pdb.set_trace()

#init_training_cumulative=[119.0, 151.0]
#end_trainind_cumulative=[2479.0, 3637.0]


#plt.scatter([1,2,3,4],[119.0, 151.0,2479.0, 3637.0])
#plt.xticks([1,2,3,4],["Value RL","Time dist RL","Value RL","Time dist RL"])
#plt.ylabel("Cumulative rewards")
#plt.yticks([])
#plt.show()

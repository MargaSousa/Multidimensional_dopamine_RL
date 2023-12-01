import time as timer
start_time = timer.time()
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
length_ticks=3
font_size=11
linewidth=1.2
scatter_size=20
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize']=font_size
mpl.rcParams['ytick.labelsize']=font_size
mpl.rcParams['lines.linewidth']=linewidth
horizontal_size=1.5
vertical_size=1.5
import seaborn as sns
import matplotlib.colors as mcol
flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",[reward_cmap[1], reward_cmap[-1]])



n_states=10
n_neurons=40
Value=np.zeros((n_neurons,n_states))
n_trials=5000
Value_save=np.zeros((n_trials,n_neurons,n_states))
taus=np.linspace(1.0/n_neurons,1.0-1.0/n_neurons,n_neurons)
gammas=np.linspace(0.9,0.999,n_neurons)[::-1]
alpha=0.1
#print("Mean value ",5*gamma**(n_states-1))
#pdb.set_trace()
for trial in range(n_trials):
    for time in range(n_states):
        if time==n_states-1:
            reward=np.random.normal(loc=10,scale=10)
            imputation=reward
        else:
            reward=0
            idx_sample_from_future=np.random.choice(n_neurons,p=np.ones(n_neurons)*(1.0/n_neurons))
            sample_from_future=Value[idx_sample_from_future,time+1]
            imputation=reward+gammas*sample_from_future

        error=imputation-Value[:,time]
        Value[:,time]+=alpha*(taus*error*(error>0)+(1-taus)*error*(error<0))

    if trial%1000==0:
        alpha=alpha*0.1

plt.imshow(Value)
plt.ylabel("Neurons")
plt.xlabel("Time")
plt.xticks([0,n_states-1],["cue","reward"])
plt.colorbar(label="Value")
plt.show()

fig,ax=plt.subplots(figsize=(horizontal_size,vertical_size))
ax.tick_params(width=linewidth,length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
colors_optimism=asym_cmap(np.linspace(0,1,n_neurons))
plt.scatter(Value[:,0]/gammas**n_states+np.random.normal(loc=0,scale=1,size=n_neurons)*0.05,Value[:,-1],c=colors_optimism,s=scatter_size)#
plt.xlabel(r"$V_i$"+"(reward)")
plt.ylabel("Values at reward")
#plt.ylabel(r"$V_i$"+"(cue)"+"/"+r"$\gamma_{i}^{t_{reward}}$")#
plt.xlabel("Values at cue\n"+r"corrected for $\gamma$")
plt.xticks([])
plt.yticks([])
plt.savefig("Value_cue_reward.svg")
plt.show()

print("time elapsed: {:.2f}s".format(timer.time() - start_time))
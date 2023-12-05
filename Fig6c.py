import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
# Parameters for plots
length_ticks=2
font_size=11
linewidth=1.2
scatter_size=2
scatter_size=10
horizontal_size=0.75
vertical_size=1.5
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize']=font_size-2
mpl.rcParams['ytick.labelsize']=font_size-2
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = font_size-2
mpl.rcParams['legend.fontsize'] = font_size-2



x=np.array([1,4,7,10,13,16])

# Run Simulation_Fig6c before

# Non-stationary enviornment
all_runs_dist_rl_early=np.load("foraging_runs_dist_rl_early_non-stationary.npy")
all_runs_dist_rl_late=np.load("foraging_runs_dist_rl_late_non-stationary.npy")
all_runs_value_early=np.load("foraging_runs_value_early_non-stationary.npy")
all_runs_value_late=np.load("foraging_runs_value_late_non-stationary.npy")
all_runs_sr_early=np.load("foraging_runs_sr_early_non-stationary.npy")
all_runs_sr_late=np.load("foraging_runs_sr_late_non-stationary.npy")


color_tmrl="darkblue"
color_sr="peru"
color_value="mediumvioletred"


fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(horizontal_size,vertical_size))#
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth,length=length_ticks)
ax.scatter([0,1],[np.mean(all_runs_dist_rl_early),np.mean(all_runs_dist_rl_late)],color=color_tmrl,label="TMRL")
ax.scatter([-0.1,0.9],[np.mean(all_runs_value_early),np.mean(all_runs_value_late)],color=color_value,label="TDRL")
ax.scatter([0.1,1.1],[np.mean(all_runs_sr_early),np.mean(all_runs_sr_late)],color=color_sr,label="SR")
plt.xticks([0,1],["Early","Late"],rotation="vertical")
plt.xlim(-0.2,1.2)
#plt.ylim(4300,13000)
plt.ylabel("Cumulative rewards")
plt.yticks([])
plt.legend()
plt.show()


# Stationary enviornment

# Non-stationary enviornment
all_runs_dist_rl_late=np.load("foraging_runs_dist_rl_late_stationary.npy")
all_runs_value_late=np.load("foraging_runs_value_late_stationary.npy")
all_runs_sr_late=np.load("foraging_runs_sr_late_stationary.npy")


y=np.array([np.mean(all_runs_value_late),np.mean(all_runs_sr_late),np.mean(all_runs_dist_rl_late)])
error=np.array([np.std(all_runs_value_late),np.std(all_runs_sr_late),np.std(all_runs_dist_rl_late)])



fig,ax=plt.subplots(figsize=(horizontal_size/2,vertical_size))#
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth,length=length_ticks)
plt.errorbar([0.5],y[0],error[0],color=color_value,fmt="-o")
plt.errorbar([0.5],y[1],error[1],color=color_sr,fmt="-o")
plt.errorbar([0.5],y[2],error[2],color=color_tmrl,fmt="-o")
#plt.xticks([0,0.5,1],["TDRL","SR","TMRL"],rotation="vertical")
plt.ylabel("Cumulative rewards")
#plt.yticks([])
plt.xticks([0.5],["Late"],rotation="vertical")
plt.ylim(0,100000)
plt.xlim(0.1,1)
#plt.savefig("cumulative_rewards_stationary_bee.svg")
plt.show()
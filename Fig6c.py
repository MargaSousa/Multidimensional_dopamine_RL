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
all_runs_dist_rl_init=np.load("mice_runs_dist_rl_init.npy")
all_runs_value_init=np.load("mice_runs_value_init.npy")
all_runs_sr_init=np.load("mice_runs_sr_init.npy")
all_runs_dist_rl_end=np.load("mice_runs_dist_rl_end.npy")
all_runs_value_end=np.load("mice_runs_value_end.npy")
all_runs_sr_end=np.load("mice_runs_sr_end.npy")

color_tmrl="darkblue"
color_sr="peru"
color_value="mediumvioletred"


fig,ax=plt.subplots(ncols=1,nrows=1,figsize=(horizontal_size,vertical_size))#
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth,length=length_ticks)
ax.scatter([0,1],[np.mean(all_runs_dist_rl_init),np.mean(all_runs_dist_rl_end)],color=color_tmrl,label="TMRL")
ax.scatter([0,1],[np.mean(all_runs_value_init),np.mean(all_runs_value_end)],color=color_value,label="TDRL")
ax.scatter([0,1],[np.mean(all_runs_sr_init),np.mean(all_runs_sr_end)],color=color_sr,label="SR")
plt.xticks([0,1],["Early","Late"],rotation="vertical")
plt.xlim(-0.2,1.2)
#plt.ylim(4300,13000)
plt.ylabel("Cumulative rewards")
plt.yticks([])
#plt.savefig("sate_bee_early_late.svg")
plt.legend()
plt.show()
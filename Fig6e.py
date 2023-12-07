import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import pdb

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



policy_dist_rl_init=np.load("foraging_runs_dist_rl_policy_early_hungry-sated.npy")
policy_value_init=np.load("foraging_runs_value_policy_early_hungry-sated.npy")
policy_sr_init=np.load("foraging_runs_sr_policy_early_hungry-sated.npy")

policy_dist_rl_end=np.load("foraging_runs_dist_rl_policy_late_hungry-sated.npy")
policy_value_end=np.load("foraging_runs_value_policy_late_hungry-sated.npy")
policy_sr_end=np.load("foraging_runs_value_policy_late_hungry-sated.npy")

mean_policy_dist_rl_init=np.mean(policy_dist_rl_init,axis=0)
error_policy_dist_rl_init=np.std(policy_dist_rl_init,axis=0)

mean_policy_dist_rl_end=np.mean(policy_dist_rl_end,axis=0)
error_policy_dist_rl_end=np.std(policy_dist_rl_end,axis=0)


mean_policy_value_init=np.mean(policy_value_init,axis=0)
error_policy_value_init=np.std(policy_value_init,axis=0)

mean_policy_value_end=np.mean(policy_value_end,axis=0)
error_policy_value_end=np.std(policy_value_end,axis=0)

mean_policy_sr_init=np.mean(policy_sr_init,axis=0)
error_policy_sr_init=np.std(policy_sr_init,axis=0)

mean_policy_sr_end=np.mean(policy_sr_end,axis=0)
error_policy_sr_end=np.std(policy_sr_end,axis=0)

color_tmrl = "darkblue"
color_sr = "peru"
color_value = "mediumvioletred"

fig,ax=plt.subplots(figsize=(horizontal_size,vertical_size))#
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.tick_params(width=linewidth,length=length_ticks)
x=np.arange(1,7)
plt.errorbar(x[0:3],mean_policy_dist_rl_init,yerr=error_policy_dist_rl_init,fmt="-o",capsize=2,color=color_tmrl)
plt.errorbar(x[0:3],mean_policy_value_init,yerr=error_policy_value_init,fmt="-o",capsize=2,color=color_value)
plt.errorbar(x[0:3],mean_policy_sr_init,yerr=error_policy_sr_init,fmt="-o",capsize=2,color=color_sr)

plt.errorbar(x[3:6],mean_policy_dist_rl_end,yerr=error_policy_dist_rl_end,fmt="-o",capsize=2,color=color_tmrl,label="TMRL")
plt.errorbar(x[3:6],mean_policy_value_end,yerr=error_policy_value_end,fmt="-o",capsize=2,color=color_value,label="TDRL")
plt.errorbar(x[3:6],mean_policy_sr_end,yerr=error_policy_sr_end,fmt="-o",capsize=2,color=color_sr,label="SR")

plt.xticks(x,["Patch 1","Patch 2","Patch 3","Patch 1","Patch 2","Patch 3"],rotation="vertical")
plt.ylabel("Patch selection \n probability")
plt.yticks([])
plt.legend(fontsize=font_size-2)
plt.title("Sated",fontsize=font_size)
plt.show()

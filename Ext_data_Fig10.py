import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
length_ticks=2
linewidth=1.2
scatter_size=20
horizontal_size=2.2
vertical_size=2.2
font_size=11
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize']=8
mpl.rcParams['ytick.labelsize']=8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 10
import scipy.stats
from aux_functions import get_expectiles


N=10 # Number of neurons
N_exp=10000 # Number of samples to estimate expectiles
T=9 # Maximum time
time_init=np.linspace(0,T,N_exp) # Defining time


# Distribution before manipulation
scale=0.4 # Add uncertainty
mean1=0
mean2=1.5
mean3=3
mean4=6
probability=scipy.stats.norm(loc=mean1,scale=scale).pdf(time_init)+scipy.stats.norm(loc=mean2,scale=scale).pdf(time_init)+scipy.stats.norm(loc=mean3,scale=scale).pdf(time_init)+scipy.stats.norm(loc=mean4,scale=scale).pdf(time_init)
probability=probability/np.sum(probability)
taus_expectiles=np.linspace(1.0 / N, 1.0 - 1 / N, N) # Uniformly sampled
_,expectiles_before=get_expectiles(time_init,probability,taus_expectiles)
gamma_before=np.exp(-1.0/(expectiles_before))



fig,ax=plt.subplots(figsize=(horizontal_size,vertical_size))
ax.tick_params(width=linewidth,length=length_ticks)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
ax.set_box_aspect(1)
plt.plot(time_init,probability,color="black")
plt.xlabel("time (s)")
plt.ylabel("P(reward)")
plt.xticks([])
plt.yticks([])


# Colors for each context switch
color_remove_short="limegreen"
color_remove_long="mediumvioletred"


# After removing shortest delay
probability_remove_short=scipy.stats.norm(loc=mean2,scale=scale).pdf(time_init)+scipy.stats.norm(loc=mean3,scale=scale).pdf(time_init)+scipy.stats.norm(loc=mean4,scale=scale).pdf(time_init)
probability_remove_short=probability_remove_short/np.sum(probability_remove_short)
_,expectiles_remove_short=get_expectiles(time_init,probability_remove_short,taus_expectiles)
gammas_remove_short=np.exp(-1.0/(expectiles_remove_short))


# After removing longest delay
probability_remove_long=scipy.stats.norm(loc=mean1,scale=scale).pdf(time_init)+scipy.stats.norm(loc=mean2,scale=scale).pdf(time_init)+scipy.stats.norm(loc=mean3,scale=scale).pdf(time_init)
probability_remove_long=probability_remove_long/np.sum(probability_remove_long)
_,expectiles_remove_long=get_expectiles(time_init,probability_remove_long,taus_expectiles)
gammas_remove_long=np.exp(-1.0/(expectiles_remove_long))


R = 10 # Mean population firing rate

# Homogenous population
time=np.linspace(0.05,T,200)

probability_remove_short=scipy.stats.norm(loc=mean2,scale=scale).pdf(time)+scipy.stats.norm(loc=mean3,scale=scale).pdf(time)+scipy.stats.norm(loc=mean4,scale=scale).pdf(time)
probability_remove_short=probability_remove_short/np.sum(probability_remove_short)

probability_remove_long=scipy.stats.norm(loc=mean1,scale=scale).pdf(time)+scipy.stats.norm(loc=mean2,scale=scale).pdf(time)+scipy.stats.norm(loc=mean3,scale=scale).pdf(time)
probability_remove_long=probability_remove_long/np.sum(probability_remove_long)

tau_time_scale = 1
ns = np.arange(N).astype('float')
ns[0] = 1
taus = tau_time_scale*T*ns/N # define decay half-lives (linear)
lambdas =np.log(2)*N/(ns*tau_time_scale*T)


prob_at_time_scale_remove_short=scipy.stats.norm(loc=mean2,scale=scale).pdf(expectiles_remove_short)+scipy.stats.norm(loc=mean3,scale=scale).pdf(expectiles_remove_short)+scipy.stats.norm(loc=mean4,scale=scale).pdf(expectiles_remove_short)
prob_at_time_scale_remove_short=prob_at_time_scale_remove_short/np.sum(prob_at_time_scale_remove_short)
gains_after_remove_short=R/(N*np.cumsum(taus_expectiles))


prob_at_time_scale_remove_long=scipy.stats.norm(loc=mean1,scale=scale).pdf(expectiles_remove_long)+scipy.stats.norm(loc=mean2,scale=scale).pdf(expectiles_remove_long)+scipy.stats.norm(loc=mean3,scale=scale).pdf(expectiles_remove_long)
prob_at_time_scale_remove_long=prob_at_time_scale_remove_long/np.sum(prob_at_time_scale_remove_long)
gains_after_remove_long=R/(N*np.cumsum(taus_expectiles))


# Compute Fisher information and sum of firing rates
fisher_info_initial=0*time
sum_fr_initial=0*time
fisher_info_remove_short=0*time
fisher_info_remove_long=0*time


fig,ax=plt.subplots(1,4,figsize=(4*horizontal_size,1*vertical_size))#,sharex=True
ax[0].set_xlabel("Time (s)")
ax[1].set_xlabel("Time (s)")
ax[2].set_xlabel("Time (s)")

for i in range(4):
    ax[i].tick_params(width=linewidth,length=length_ticks)
    ax[i].spines['left'].set_linewidth(linewidth)
    ax[i].spines['bottom'].set_linewidth(linewidth)

ax[0].set_yticks([])
ax[1].set_yticks([])
ax[0].set_ylabel("Density")
ax[1].set_ylabel("Density")
ax[2].set_ylabel("Firing rate (A.U)")
#ax[1,1].set_ylabel("Firing rate (A.U)")
#ax[0,2].set_yticks([])
ax[0].set_ylim([0,0.022])
ax[1].set_ylim([0,0.022])

for n in range(N):
    ax[2].plot(time, gains_after_remove_short[n] * gammas_remove_short[n]**time, linewidth=linewidth,color=color_remove_short,alpha=0.8)
    ax[2].plot(time, gains_after_remove_long[n] * gammas_remove_long[n]**time, linewidth=linewidth,color=color_remove_long,alpha=0.8)
    sum_fr_initial+=R*lambdas[n]*np.exp(-lambdas[n]*time)
    fisher_info_initial+=R*((lambdas[n]**3)*np.exp(-lambdas[n]*time))
    fisher_info_remove_short+= gains_after_remove_short[n]*(gammas_remove_short[n] ** time) * np.log(gammas_remove_short[n])**2
    fisher_info_remove_long+= gains_after_remove_long[n]*(gammas_remove_long[n] ** time) * np.log(gammas_remove_long[n])**2


ax[0].plot(time,probability_remove_short,color=color_remove_short)
ax[1].plot(time,probability_remove_long,color=color_remove_long)

bin_2=np.where(time>=2)[0][0]

#ax[1,1].plot(time,fisher_info_initial,color=color_homogenous,label="Initial population")
ax[3].plot(time[bin_2:],fisher_info_remove_short[bin_2:],color=color_remove_short,label="Shortest removed")
ax[3].plot(time[bin_2:],fisher_info_remove_long[bin_2:],color=color_remove_long,label="Longest removed")
ax[3].set_ylabel("Fisher Information")
ax[3].legend()


#ax[1,2].set_xlabel("Time (s)")
#ax[1,2].set_yticks([0,R],["0","R"])
ax[2].set_yticks([0,R],["0","R"])
#ax[1,1].set_yticks([0,R],["0","R"])
#ax[1,2].set_xticks([0,T],["0","T"])
#ax[1,0].set_ylim([-0.2,R*1.25])
#ax[1,1].set_ylim([-0.2,R*1.25])
#ax[1,2].set_ylim([-0.2,R*1.25])
ax[3].set_yticks([])
ax[2].set_xticks([0,3,6],["0","3","6"])
ax[3].set_xticks([3,6],["3","6"])
ax[0].set_xticks([0,3,6],["0","3","6"])
ax[1].set_xticks([0,3,6],["0","3","6"])
#save_dir=r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Figures_paper_DA"
#plt.savefig(save_dir+"\efficient_coding_conditions.svg")

plt.show()



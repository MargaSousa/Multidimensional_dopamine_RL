import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from aux_functions import *
from scipy import stats
from sklearn.linear_model import LinearRegression,HuberRegressor
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.colors as mcol
import os
from sklearn.isotonic import IsotonicRegression
import itertools
import pandas as pd
mpl.use('TkAgg')

flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",[reward_cmap[1], reward_cmap[-1]])


# Parameters for presentation plots
length_ticks=2
linewidth=1.2
scatter_size=20
horizontal_size=1.5
vertical_size=1.5
font_size=11
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize']=8
mpl.rcParams['ytick.labelsize']=8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 10

# Color code for different animals
colors_animals=mpl.cm.get_cmap('Set2', 12)(np.linspace(0,1,6))[::-1]

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "Putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get estimated tuning for each neuron
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))

# Select neurons from given animals
selected_animal=3353
selected_neurons=data_frame_neurons_info.index[data_frame_neurons_info.Animal==selected_animal].tolist()

# Choose instead neurons from all animals
selected_neurons=np.arange(len(data_frame_neurons_info))

# Get estimated tuning for reward time and magnitude
discount=data_frame_neurons_info['Gamma'].values[selected_neurons]
estimated_reversals=data_frame_neurons_info['Reversals'].values[selected_neurons]
estimated_taus=data_frame_neurons_info['Taus'].values[selected_neurons]
gains=data_frame_neurons_info['Gain'].values[selected_neurons]
n_neurons=len(discount)
bias_taus=data_frame_neurons_info['Bias in the estimation of tau'].values[selected_neurons] # Generated from Fig4_get_bias_variance_estimation_tau
variance_tau=data_frame_neurons_info['Variance in the estimation of tau'].values[selected_neurons] # Generated from Fig4_get_bias_variance_estimation_tau
animals=data_frame_neurons_info['Animal'].values[selected_neurons]


# Get responses for certain and variable cues
responses_cue=np.load(os.path.join(directory_parsed_data,"responses_cue_different_delays_constant_magnitude.npy"))[selected_neurons,:,:] # Responses at the cue for variable reward delays
responses_cue_bimodal=np.loadtxt(os.path.join(directory_parsed_data,"responses_cue_different_magnitudes_constant_delay.csv"))[selected_neurons,:] # Responses at the cue for delay of 3s and variable magnitudes (neurons x trials)

# For smoothing distribution in reward amount
n_amount = 80
smooth = 0.35
amount= np.linspace(0, 9, n_amount)

# For the Laplace decoder for future reward time
n_time=50
time=np.linspace(0,7.5,n_time)
cue_times=np.array([0,1.5,3,6])
alpha_time=1 # Smoothing parameter

# Regression between firing rate at the cue and reversal points
population_responses_variable = np.nanmean(responses_cue_bimodal,axis=1)
population_responses_variable_corrected=population_responses_variable/(gains * discount**3) # Correct for diversity in temporal discouting functions
well_estimated_reversals=np.intersect1d(np.where(estimated_reversals>1.1)[0],np.where(estimated_reversals<7.9)[0]) # We only give rewards in the range 1-8 uL
reg_variable = HuberRegressor().fit((population_responses_variable_corrected[well_estimated_reversals]).reshape(-1, 1), estimated_reversals[well_estimated_reversals])
y_example=reg_variable.predict(population_responses_variable_corrected[well_estimated_reversals].reshape(-1, 1))

plt.figure()
plt.scatter(population_responses_variable_corrected[well_estimated_reversals],estimated_reversals[well_estimated_reversals],c=estimated_reversals[well_estimated_reversals],cmap=asym_cmap)
plt.plot(population_responses_variable_corrected[well_estimated_reversals],y_example,ls="--",color="k")
plt.xlabel("Firing rate at cue")
plt.ylabel("Reversal point")
plt.show()

# Pearson correlation: responses at cue and reversal points
out_variable = stats.pearsonr(population_responses_variable_corrected[well_estimated_reversals], estimated_reversals[well_estimated_reversals])


# Mapping from reversals to taus at cue
corrected_taus = estimated_taus[well_estimated_reversals] - bias_taus[well_estimated_reversals]
corrected_taus[corrected_taus < 0] = 0
corrected_taus[corrected_taus > 1] = 1
w_reversals = 1.0 / (1 + variance_tau[well_estimated_reversals]) # weight inversely proportional to the variance of the estimated tau
reversal_estimated_at_cue = reg_variable.predict(population_responses_variable_corrected[well_estimated_reversals].reshape(-1, 1))
iso_reg_cue = IsotonicRegression(increasing=True).fit(reversal_estimated_at_cue, corrected_taus,sample_weight=w_reversals)
pred_cue = iso_reg_cue.predict(reversal_estimated_at_cue)

# Save joint pdf for 4 certain cues
joint_pdf_certain_reward=np.empty((n_time,n_amount,4))

# Colors for plots
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))[:,:3]
reward_times=[0,1.5,3,6]

# Decode reward time and magnitude  for all cues that predict a certain reward magnitude at different delays
n_runs=10

for cue in range(4): # for animal 3353: range(1,4)

    pdf_time=0*time
    pdf_amount=0*amount


    for run in range(n_runs):

        population_responses = np.nanmean(responses_cue[:, cue], axis=1)
        n_neurons=np.sum(np.isnan(population_responses)) # Some neurons were not recorded in all delays

        # Decode reward times
        variance_certain=np.nanvar(responses_cue[:,cue],axis=1)[n_neurons:]
        pdf_time += run_decoding_time(time,discount[n_neurons:], variance_certain,population_responses[n_neurons:]/gains[n_neurons:], alpha_time)

        # Decode reward amount
        population_responses_corrected = population_responses/(gains * discount**reward_times[cue]) # Correct for diversity in temporal discouting functions
        estimated_reversals_certain = reg_variable.predict((population_responses_corrected[well_estimated_reversals][n_neurons:]).reshape(-1, 1))
        samples, _ = run_decoding_magnitude(estimated_reversals_certain, pred_cue,np.ones(len(estimated_reversals)), minv=0, maxv=9, N=20,max_samples=2000, max_epochs=15, method='TNC')

        # Smooth
        kde= gaussian_kde(samples, bw_method=smooth)
        pdf_amount_run= kde.pdf(amount)
        pdf_amount_run=pdf_amount_run/np.sum(pdf_amount_run)
        pdf_amount+=pdf_amount_run

        if cue==2 and run==0:
            pdf_amount_certain_cue=np.copy(pdf_amount_run)

    # Compute joint pdf over reward amount and time
    pdf_time=pdf_time/n_runs
    pdf_time=pdf_time/np.sum(pdf_time)

    pdf_amount=pdf_amount/n_runs
    pdf_amount=pdf_amount/np.sum(pdf_amount)

    for i_t,i_a in itertools.product(np.arange(n_time),np.arange(n_amount)):
        joint_pdf_certain_reward[i_t, i_a,cue] = pdf_time[n_time-i_t-1] * pdf_amount[i_a]

    joint_pdf_certain_reward[:, :, cue]=joint_pdf_certain_reward[:, :, cue]/np.sum(joint_pdf_certain_reward[:, :, cue])


# Decode amount for variable cue
color_reward="darkgreen"
color_cue="darkorange"

fig_variable,ax_variable=plt.subplots(figsize=(horizontal_size,vertical_size))
ax_variable.spines['left'].set_linewidth(linewidth)
ax_variable.spines['bottom'].set_linewidth(linewidth)
ax_variable.set_box_aspect(1)
mean_pdf_reward=np.zeros(n_amount)
mean_pdf_cue=np.zeros(n_amount)

# Mapping from reversals to taus at reward
iso_reg_reward = IsotonicRegression(increasing=True).fit(estimated_reversals[well_estimated_reversals], corrected_taus,sample_weight=w_reversals)
pred_reward = iso_reg_reward.predict(estimated_reversals[well_estimated_reversals])

# Run decoding over magnitude at the cue and reward
n_runs=1
for run in range(n_runs):

    samples, _ = run_decoding_magnitude(reversal_estimated_at_cue, pred_cue,np.ones(len(reversal_estimated_at_cue)), minv=1, maxv=8, N=20,max_samples=2000, max_epochs=15, method='TNC')
    kde= gaussian_kde(samples, bw_method=smooth)
    pdf_amount= kde.pdf(amount)
    pdf_amount=pdf_amount/np.sum(pdf_amount)
    mean_pdf_cue+=pdf_amount

    samples_at_reward, _ = run_decoding_magnitude(estimated_reversals[well_estimated_reversals], pred_reward,np.ones(len(reversal_estimated_at_cue)), minv=1, maxv=8, N=20,max_samples=2000, max_epochs=15, method='TNC')
    kde= gaussian_kde(samples_at_reward, bw_method=smooth)
    pdf_amount_at_reward= kde.pdf(amount)
    pdf_amount_at_reward=pdf_amount_at_reward/np.sum(pdf_amount_at_reward)
    mean_pdf_reward+=pdf_amount_at_reward
    plt.plot(amount,pdf_amount,color=color_cue,linewidth=linewidth*0.1)
    plt.plot(amount,pdf_amount_at_reward,color=color_reward,linewidth=linewidth*0.1)

    # Certain cue
    population_responses = np.nanmean(responses_cue[:, 2], axis=1)
    # Decode reward amount
    population_responses_corrected = population_responses/(gains * discount**3)
    estimated_reversals_certain = reg_variable.predict((population_responses_corrected[well_estimated_reversals]).reshape(-1, 1))
    samples, _ = run_decoding_magnitude(estimated_reversals_certain, pred_cue,np.ones(len(estimated_reversals)), minv=0, maxv=9, N=20,max_samples=2000, max_epochs=15, method='TNC')
    kde= gaussian_kde(samples, bw_method=smooth)
    pdf_amount_at_cue_certain= kde.pdf(amount)
    pdf_amount_at_cue_certain=pdf_amount_at_cue_certain/np.sum(pdf_amount_at_cue_certain)

# Mean over runs
mean_pdf_cue=mean_pdf_cue/n_runs
mean_pdf_reward=mean_pdf_reward/n_runs

# True pdf in amount or variable cue
amounts_experiment=np.array([1,2.75,4.5,6.25,8]) # Amounts of reward given in the experiment
n_amounts_exp=len(amounts_experiment)
probability=np.array([0.25,0.167,0.167,0.167,0.25]) # Probability of amounts of reward given in the experiment
probability=probability/np.sum(probability)
n_bins = 80

# A smoothed version of the distribution
all_samples=np.random.choice(amounts_experiment,10000,p=probability)
kde= gaussian_kde(all_samples, bw_method=smooth)
y= kde.pdf(amount)
pdf_amount_true=y/np.sum(y)

plt.plot(amount,pdf_amount,color=color_cue,label="at cue")
plt.plot(amount,pdf_amount_at_reward,color=color_reward,label="at reward")
plt.plot(amount,pdf_amount_true,color="k",label="true")
plt.plot(amount,pdf_amount_certain_cue,color="darkred",label="Certain")
ax_variable.set_ylabel("Decoded\ndensity (A.U)")
ax_variable.set_xlabel("Magnitude ("+r"$\mu$l)")
ax_variable.set_yticks([0,],["0"])
ax_variable.set_xticks([1,4.5,8],["1","4.5","8"])
ax_variable.legend(frameon=False)
plt.show()


# Decode time for variable cue
mean_responses_variable_cue_discount = reg_variable.coef_[0] * population_responses_variable/(gains* (estimated_reversals - reg_variable.intercept_)) # Correct for diversity in reversal points
#mean_responses_variable_cue_discount=population_responses_variable/(estimated_reversals*gains)
variance_variable=np.nanvar(responses_cue_bimodal,axis=1)
pdf_time_variable=run_decoding_time(time,discount,variance_variable,mean_responses_variable_cue_discount,0.5)
plt.plot(time,pdf_time_variable,label=str(alpha_time))
plt.legend()
plt.show()

# Save joint pdf for variable cue
joint_pdf_variable_reward=np.empty((n_time,n_amount))
for i_t, i_a in itertools.product(np.arange(n_time),np.arange(n_amount)):
    joint_pdf_variable_reward[i_t, i_a] = pdf_time_variable[n_time - i_t - 1] * mean_pdf_cue[i_a]
joint_pdf_variable_reward=joint_pdf_variable_reward/np.sum(joint_pdf_variable_reward)

# Plot stacked heat map
color_map=sns.color_palette("coolwarm", as_cmap=True)
mesh=np.meshgrid(amount,time[::-1])
fig, ax = plt.subplots(figsize=(horizontal_size*3,vertical_size*7),subplot_kw={"projection": "3d"})# 1.75 and 12 for animal 3353
fig.set_facecolor('w')
ax.set_facecolor('w')
ax.view_init(elev=-150, azim=50)
ax.set_box_aspect((1, 1, 2.25))
for i_d in np.arange(4):
    scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_certain_reward[:, :, i_d]), np.max(joint_pdf_certain_reward[:, :, i_d])),cmap=color_map)
    ax.plot_surface(mesh[0],mesh[1],-0.01*i_d + 0*joint_pdf_certain_reward[:, :, i_d], facecolors=scam.to_rgba(joint_pdf_certain_reward[:, :, i_d]),antialiased = True,rstride=1,cstride=1,alpha=None,shade=False)

scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_variable_reward), np.max(joint_pdf_variable_reward)),cmap=color_map)
ax.plot_surface(mesh[0],mesh[1],-0.04+0*joint_pdf_variable_reward, facecolors=scam.to_rgba(joint_pdf_variable_reward),antialiased = True,rstride=1,cstride=1,alpha=None,shade=False)
ax.set_ylabel("Time\n since cue (s)")
ax.set_xlabel("Magnitude ("+r"$\mu$"+"l)")
ax.set_zticks([])
ax.set_yticks([0,1.5,3,6],["0","1.5","3","6"])
ax.set_xticks([1,4.5,8],["1","4.5","8"])
fig.tight_layout()
ax.set_zlim([-0.04,0])
plt.savefig("heatmap.eps")
#plt.show()
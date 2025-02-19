import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special
mpl.use('TkAgg')
import numpy as np
from aux_functions import *
from scipy import stats
from sklearn.linear_model import LinearRegression,HuberRegressor
from scipy.stats import gaussian_kde
import seaborn as sns
length_ticks=2
linewidth=1.2
scatter_size=20
horizontal_size=1.75#2
vertical_size=1.75#2
font_size=11
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize']=8
mpl.rcParams['ytick.labelsize']=8
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 10
import matplotlib.colors as mcol
import os
flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",[reward_cmap[1], reward_cmap[-1]])
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from scipy.stats import shapiro,kstest
import itertools
import pandas
import pandas as pd




colors_animals=mpl.cm.get_cmap('Set2', 12)(np.linspace(0,1,6))[::-1]


# Select neurons (either "DA" or "putative_DA")
type_neurons="DA"

if type_neurons=="DA":
    data_frame_neurons_info=pandas.read_csv(r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\Figure_4\Fig4_dataframe_neurons_info.csv")
    bias_var = pd.read_csv(r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\Figure_4\Fig4b.csv")


# Select neurons from given animals
selected_animal=3353
selected_neurons=data_frame_neurons_info.index[data_frame_neurons_info.Animal==selected_animal].tolist()


# Choose instead neurons from all animals
selected_neurons=np.arange(len(data_frame_neurons_info))


# Tuning of each neuron
discount=data_frame_neurons_info['Gamma'].values[selected_neurons]
estimated_reversals=data_frame_neurons_info['Reversals'].values[selected_neurons]
estimated_taus=data_frame_neurons_info['Taus'].values[selected_neurons]
gains=data_frame_neurons_info['Gain'].values[selected_neurons]
n_neurons=len(discount)




bias_taus=bias_var['Bias in the estimation of tau'].values[selected_neurons] # Generated from Fig4_get_bias_variance_estimation_tau
variance_tau=bias_var['Variance in the estimation of tau'].values[selected_neurons] # Generated from Fig4_get_bias_variance_estimation_tau




parsed_data=r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\DA_data"
responses_cue_second_phase=np.load(parsed_data+r'\responses_cue_different_delays_constant_magnitude_shorter_window.npy')[selected_neurons,:,:]
responses_cue_bimodal_second_phase=np.loadtxt(parsed_data+r'\responses_cue_different_magnitudes_constant_delay_shorter_window.csv')[selected_neurons,:]




# For the Laplace decoder for future reward time
n_time = 50
time = np.linspace(0, 7.5, n_time)
cue_times=np.array([0,1.5,3,6])
alpha_time=0.2# Smoothing parameter




# Relationship between firing rate at the cue and reversal points
population_responses_variable_second_phase = np.nanmean(responses_cue_bimodal_second_phase,axis=1)
population_responses_variable_corrected_second_phase=population_responses_variable_second_phase/(gains * discount**3)
well_estimated_reversals=np.intersect1d(np.where(estimated_reversals>1.1)[0],np.where(estimated_reversals<7.9)[0])
reg_variable = HuberRegressor().fit((population_responses_variable_corrected_second_phase[well_estimated_reversals]).reshape(-1, 1), estimated_reversals[well_estimated_reversals])




# Mapping from reversals to taus at cue
corrected_taus = estimated_taus[well_estimated_reversals] - bias_taus[well_estimated_reversals]
corrected_taus[corrected_taus < 0] = 0
corrected_taus[corrected_taus > 1] = 1
w_reversals = 1.0 / (1 + variance_tau[well_estimated_reversals])
reversal_estimated_at_cue = reg_variable.predict(population_responses_variable_corrected_second_phase[well_estimated_reversals].reshape(-1, 1))
iso_reg_cue = IsotonicRegression(increasing=True).fit(reversal_estimated_at_cue, corrected_taus,sample_weight=w_reversals)
pred_cue = iso_reg_cue.predict(reversal_estimated_at_cue)




# Compute variance over all cues
variance=[]
for neu in range(n_neurons):
        all_trials = np.concatenate((np.ndarray.flatten(responses_cue_bimodal_second_phase[neu, :]),np.ndarray.flatten(responses_cue_second_phase[neu, :, :])))
        n_trials = np.sum(~np.isnan(all_trials))
        variance.append(np.var(all_trials[~np.isnan(all_trials)]))
variance=np.array(variance)


# Colors for plots
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))[:,:3]


# True pdf in amount and time for variable cue
n_amount = 80
smooth_magnitude = 0.35
amount= np.linspace(0, 9, n_amount)


amounts_exp=np.array([1,2.75,4.5,6.25,8])
n_amounts_exp=len(amounts_exp)
probability=np.array([0.25,0.167,0.167,0.167,0.25])
probability=probability/np.sum(probability)
n_bins = 80
all_samples=np.random.choice(amounts_exp,10000,p=probability)
kde= gaussian_kde(all_samples, bw_method=smooth_magnitude)
y= kde.pdf(amount)
std_time_prior=1
pdf_amount_true=y/np.sum(y)
pdf_time_true=scipy.stats.norm(loc=3,scale=std_time_prior).pdf(time)
pdf_time_true=pdf_time_true/np.sum(pdf_time_true)




# Compute prior joint
joint_pdf_prior=np.zeros((n_time,n_amount))


# Certain cues
std_magnitude_prior=smooth_magnitude
pdf_amount=scipy.stats.norm(loc=4.5, scale=std_magnitude_prior).pdf(amount)
pdf_amount=pdf_amount/np.sum(pdf_amount)
for delay in [0,1.5,3,6]:
    joint_pdf_cue=np.zeros((n_time,n_amount))
    pdf_time = scipy.stats.norm(loc=delay, scale=std_time_prior).pdf(time)
    pdf_time = pdf_time / np.sum(pdf_time)
    for i_t, i_a in itertools.product(np.arange(n_time),np.arange(n_amount)):
        joint_pdf_cue[i_t, i_a] += pdf_time[n_time - i_t - 1] * pdf_amount[n_amount-i_a-1]
    joint_pdf_cue=joint_pdf_cue/np.sum(joint_pdf_cue)
    joint_pdf_prior+=0.55*0.25*joint_pdf_cue


# Variable cue
pdf_time = scipy.stats.norm(loc=3, scale=std_time_prior).pdf(time)
pdf_time = pdf_time / np.sum(pdf_time)
joint_pdf_cue=np.zeros((n_time,n_amount))
for i_t, i_a in itertools.product(np.arange(n_time),np.arange(n_amount)):
    joint_pdf_cue[i_t, i_a]= pdf_time[n_time - i_t - 1] * pdf_amount_true[n_amount-i_a-1]


# Normalize by the probability of each cue
joint_pdf_cue=joint_pdf_cue/np.sum(joint_pdf_cue)
joint_pdf_prior+=0.45*joint_pdf_cue
joint_pdf_prior=joint_pdf_prior/np.sum(joint_pdf_prior)
const=0.01 # So we can compute the KL divergence
joint_pdf_prior+=np.max(joint_pdf_cue)*const
joint_pdf_prior=joint_pdf_prior/np.sum(joint_pdf_prior)




save_2d_decoder = {"Neuron":np.arange(n_neurons),"Estimated temporal discount factors":discount,"Estimated gain":gains}
label_cues=["1.5s cue","3s cue","6s cue","Variable 3s cue"]




# Save joint pdf for 4 certain cues
joint_pdf_certain_reward=np.empty((n_time,n_amount,4))
save_joint_magnitude=[]
save_joint_time=[]
save_joint_prob=[]
save_joint_cue=[]
label_cues=["0s","1.5s cue","3s cue","6s cue","Variable 3s cue"]

n_runs_2d_decoder=1
is_shuffle=False
kl_runs=np.zeros((n_runs_2d_decoder,5))
for run in range(n_runs_2d_decoder):




    # Decode reward time  for all cues that predict a certain reward magnitude at different delays
    for cue in range(1,4):


        population_responses = np.nanmean(responses_cue_second_phase[:, cue,:], axis=1)
        n_neurons=np.sum(np.isnan(population_responses)) # Some neurons were not recorded in all delays


        save_2d_decoder["Mean responses delay " + label_cues[cue-1] +" (spikes/s)"] = population_responses


        # Decode reward times
        if is_shuffle:
            neurons_shuffled=np.arange(len(discount[n_neurons:]))
            np.random.shuffle(neurons_shuffled)
            pdf_time = run_decoding_time(time,discount[n_neurons:][neurons_shuffled], variance,population_responses[n_neurons:]/(gains[n_neurons:]), alpha_time)#estimated_reversals[n_neurons:]*
        else:
            pdf_time = run_decoding_time(time,discount[n_neurons:], variance,population_responses[n_neurons:]/(gains[n_neurons:]), alpha_time)#estimated_reversals[n_neurons:]*


        estimate_reward_time=np.sum(time*pdf_time)
        # Decode reward amount
        population_responses_corrected = population_responses/(gains*discount**estimate_reward_time)#**estimate_reward_time)
        estimated_reversals_certain = reg_variable.predict((population_responses_corrected[well_estimated_reversals][n_neurons:]).reshape(-1, 1))


        if is_shuffle:
            neurons_shuffled=np.arange(len(estimated_reversals_certain))
            np.random.shuffle(neurons_shuffled)#[neurons_shuffled]
            samples, _ = run_decoding_magnitude(estimated_reversals_certain[neurons_shuffled], pred_cue[n_neurons:],np.ones(len(estimated_reversals)), minv=0, maxv=9, N=20,max_samples=2000, max_epochs=15, method='TNC')
        else:
            samples, _ = run_decoding_magnitude(estimated_reversals_certain, pred_cue[n_neurons:],np.ones(len(estimated_reversals)), minv=0, maxv=9, N=20,max_samples=2000, max_epochs=15, method='TNC')


        # Smooth
        kde= gaussian_kde(samples, bw_method=smooth_magnitude)
        pdf_amount= kde.pdf(amount)
        pdf_amount=pdf_amount/np.sum(pdf_amount)


        plt.plot(amount,pdf_amount)


        # Compute joint pdf over reward amount and time
        for i_t,i_a in itertools.product(np.arange(n_time),np.arange(n_amount)):
            joint_pdf_certain_reward[i_t, i_a,cue] = pdf_time[n_time-i_t-1] * pdf_amount[i_a]
        joint_pdf_certain_reward[:,:,cue]=joint_pdf_certain_reward[:,:,cue]/np.sum(joint_pdf_certain_reward[:,:,cue])
        joint_pdf_certain_reward[:,:,cue]+=np.nanmax(joint_pdf_certain_reward[:,:,cue])*const
        joint_pdf_certain_reward[:,:,cue]=joint_pdf_certain_reward[:,:,cue]/np.sum(joint_pdf_certain_reward[:,:,cue])

        # Save
        for i_t, i_a in itertools.product(np.arange(n_time), np.arange(n_amount)):
            save_joint_time.append(time[n_time-i_t-1])
            save_joint_magnitude.append(amount[i_a])
            save_joint_prob.append(joint_pdf_certain_reward[i_t,i_a,cue])
            save_joint_cue.append(label_cues[cue])




    plt.show()
    # Decode time for variable cue
    population_responses_variable = np.nanmean(responses_cue_bimodal_second_phase, axis=1)
    mean_responses_variable_cue_discount=population_responses_variable/gains
    variance_variable = np.nanvar(responses_cue_bimodal_second_phase, axis=1)


    save_2d_decoder["Mean responses " + label_cues[3] + " (spikes/s)"] = population_responses_variable


    if is_shuffle:
        neurons_shuffled = np.arange(len(discount[n_neurons:]))
        np.random.shuffle(neurons_shuffled)
        pdf_time_variable=run_decoding_time(time,discount[neurons_shuffled],variance_variable,mean_responses_variable_cue_discount,alpha_time)
    else:
        pdf_time_variable=run_decoding_time(time,discount,variance,mean_responses_variable_cue_discount,alpha_time)


    estimate_reward_time = np.sum(time * pdf_time_variable) #time[np.argmax(pdf_time)]  #
    print(estimate_reward_time)
    population_responses_variable_corrected = population_responses_variable/(gains*discount**estimate_reward_time)#estimate_reward_time
    estimated_reversals_variable = reg_variable.predict((population_responses_variable_corrected[well_estimated_reversals][n_neurons:]).reshape(-1, 1))




    # Decode magnitude for variable cue
    if is_shuffle:
        neurons_shuffled = np.arange(len(estimated_reversals_variable))
        np.random.shuffle(neurons_shuffled)  # [neurons_shuffled],
        samples, _ = run_decoding_magnitude(estimated_reversals_variable[neurons_shuffled], pred_cue,np.ones(len(reversal_estimated_at_cue)), minv=1, maxv=8, N=20,max_samples=2000, max_epochs=15, method='TNC')
    else:
        samples, _ = run_decoding_magnitude(estimated_reversals_variable, pred_cue,np.ones(len(reversal_estimated_at_cue)), minv=1, maxv=8, N=20,max_samples=2000, max_epochs=15, method='TNC')


    # Smooth
    kde= gaussian_kde(samples, bw_method=smooth_magnitude)
    pdf_amount= kde.pdf(amount)
    pdf_amount_cue=pdf_amount/np.sum(pdf_amount)


    plt.plot(amount,pdf_amount_cue)


    # Save joint pdf for variable cue
    joint_pdf_variable_reward=np.zeros((n_time,n_amount))
    for i_t, i_a in itertools.product(np.arange(n_time),np.arange(n_amount)):
        joint_pdf_variable_reward[i_t, i_a] = pdf_time_variable[n_time - i_t - 1] * pdf_amount_cue[i_a]
    joint_pdf_variable_reward=joint_pdf_variable_reward/np.sum(joint_pdf_variable_reward)
    joint_pdf_variable_reward+=np.max(joint_pdf_variable_reward)*const
    joint_pdf_variable_reward=joint_pdf_variable_reward/np.sum(joint_pdf_variable_reward)

    for i_t, i_a in itertools.product(np.arange(n_time), np.arange(n_amount)):
        save_joint_time.append(time[n_time - i_t - 1])
        save_joint_magnitude.append(amount[i_a])
        save_joint_prob.append(joint_pdf_variable_reward[i_t, i_a])
        save_joint_cue.append("Variable")


    # Compute KL between true prior joint distrution and the decoded from DA population
    for i_d in np.arange(4):
        kl_runs[run,i_d]=np.sum(scipy.special.rel_entr(np.ndarray.flatten(joint_pdf_prior),np.ndarray.flatten(joint_pdf_certain_reward[:, :, i_d])))
    kl_runs[run, 4] = np.sum(scipy.special.rel_entr(np.ndarray.flatten(joint_pdf_prior),np.ndarray.flatten(joint_pdf_variable_reward)))

df={}
df["Cue"]=save_joint_cue
df["Time"]=save_joint_time
df["Magnitude"]=save_joint_magnitude
df["Probability"]=save_joint_prob
df = pd.DataFrame(df)
df.to_csv(r'C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\MS\Source_data\Extended_data_Figure_7\Extended_data_Fig7c_late.csv', index=False, header=True,sep=',')



# Save
#df = pd.DataFrame(save_2d_decoder)
#df.to_csv(r'NEW_FIG_C.csv', index=False, header=True,sep=',')
#pdb.set_trace()


print(kl_runs)
# if is_shuffle:
#     np.save("kl_2d_map_shuffled.npy",kl_runs)
# else:
#     np.save("kl_2d_map.npy",kl_runs)


#np.save(save_dir+r"/joint_pdf_certain_reward.npy",joint_pdf_certain_reward)
#np.save(save_dir+r"/joint_pdf_variable_reward.npy",joint_pdf_variable_reward)




# Plot stacked heat map
color_map=sns.color_palette("coolwarm", as_cmap=True)
mesh=np.meshgrid(amount,time[::-1])
fig, ax = plt.subplots(figsize=(horizontal_size*2,vertical_size*2.5),subplot_kw={"projection": "3d"})
fig.set_facecolor('w')
ax.set_facecolor('w')
ax.view_init(elev=-150, azim=50)
ax.set_box_aspect((1, 1, (2.3)))#*(3.0/4)



# Certain cues
for i_d in np.arange(1,4):
    map=joint_pdf_certain_reward[:, :, i_d]
    scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(map), np.max(map)),cmap=color_map)
    ax.plot_surface(mesh[0],mesh[1],-0.01*i_d + 0*map, facecolors=scam.to_rgba(map),antialiased = True,rstride=1,cstride=1,alpha=None,shade=False)


# Variable cue
scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_variable_reward), np.max(joint_pdf_variable_reward)),cmap=color_map)
ax.plot_surface(mesh[0],mesh[1],-0.04+0*joint_pdf_variable_reward, facecolors=scam.to_rgba(joint_pdf_variable_reward),antialiased = True,rstride=1,cstride=1,alpha=None,shade=False)


# True prior
#scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_prior), np.max(joint_pdf_prior)),cmap=color_map)
#ax.plot_surface(mesh[0],mesh[1],-0.05+0*joint_pdf_prior, facecolors=scam.to_rgba(joint_pdf_prior),antialiased = True,rstride=1,cstride=1,alpha=None,shade=False)


ax.set_ylabel("Time\n since cue (s)")
ax.set_xlabel("Magnitude ("+r"$\mu$"+"l)")
ax.set_zticks([])
ax.set_yticks([0,1.5,3,6],["0","1.5","3","6"])
ax.set_xticks([1,4.5,8],["1","4.5","8"])#
fig.tight_layout()
#plt.show()
plt.savefig("heat_map_2nd_phase_shorter_window.svg")
plt.show()
pdb.set_trace()

from scipy.stats import gaussian_kde
from aux_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats
import matplotlib.colors as mcol
from aux_functions import get_expectiles

# Parameters for plots
length_ticks=3
font_size=11
linewidth=1.2
scatter_size=2
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize']=font_size
mpl.rcParams['ytick.labelsize']=font_size
mpl.rcParams['lines.linewidth']=linewidth
horizontal_size=1.5
vertical_size=1.5

# Optimism colors (from Dabney et al. (2019))
flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",[reward_cmap[1], reward_cmap[-1]])


N_exp=10000 # Number of samples to estimate expectiles
x_max=15
x_exp=np.linspace(0,x_max,N_exp)


# Define probabilities!
n_neurons=40
taus=np.linspace(1.0/n_neurons,1-1.0/n_neurons,n_neurons)
smooth=0.35
x_pdf=np.linspace(0,15,120)

# Long large cue
mean_ll=8
scale_ll=0.4
probability_exp=scipy.stats.norm(loc=mean_ll,scale=scale_ll).pdf(x_exp)
probability_exp=probability_exp/np.sum(probability_exp)
pos_expectiles,expectiles_ll=get_expectiles(x_exp,probability_exp,taus)

# Short small cue
mean_ss=3
scale_ss=0.4
probability_exp=scipy.stats.norm(loc=mean_ss,scale=scale_ss).pdf(x_exp)#+scipy.stats.norm(loc=mean_2,scale=scale_2).pdf(x_exp)
probability_exp=probability_exp/np.sum(probability_exp)
pos_expectiles,expectiles_ss=get_expectiles(x_exp,probability_exp,taus)

# Short variabe cue
mean_1=2
scale_1=0.2
mean_2=4
scale_2=0.2
probability_exp=scipy.stats.norm(loc=mean_1,scale=scale_1).pdf(x_exp)+scipy.stats.norm(loc=mean_2,scale=scale_2).pdf(x_exp)
probability_exp=probability_exp/np.sum(probability_exp)
pos_expectiles,expectiles_variable=get_expectiles(x_exp,probability_exp,taus)

# Save expectiles
np.save("expectiles_ss.npy",expectiles_ss)
np.save("expectiles_ll.npy",expectiles_ll)
np.save("expectiles_variable.npy",expectiles_variable)

# Decode distributions
samples_ss, _ = run_decoding(expectiles_ss, taus,np.ones(len(taus)), minv=1, maxv=10, N=20, max_samples=2000,max_epochs=15, method='TNC')
kde_ss = gaussian_kde(samples_ss, bw_method=smooth)
y_ss = kde_ss.pdf(x_pdf)
y_ss=y_ss/np.sum(y_ss)

samples_ll, _ = run_decoding(expectiles_ll, taus,np.ones(len(taus)), minv=1, maxv=10, N=20, max_samples=2000,max_epochs=15, method='TNC')
kde_ll = gaussian_kde(samples_ll, bw_method=smooth)
y_ll = kde_ll.pdf(x_pdf)
y_ll=y_ll/np.sum(y_ll)


samples_variable, _ = run_decoding(expectiles_variable, taus,np.ones(len(taus)), minv=1, maxv=10, N=20, max_samples=2000,max_epochs=15, method='TNC')
kde_variable = gaussian_kde(samples_variable, bw_method=smooth)
y_variable = kde_variable.pdf(x_pdf)
y_variable=y_variable/np.sum(y_variable)


colors=["#7fc97f","#beaed4","#fdc086"]

fig,ax=plt.subplots(figsize=(vertical_size,horizontal_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
plt.plot(x_pdf,y_ss,color=colors[0],label="SS")
plt.plot(x_pdf,y_ll,color=colors[1],label="LL")
plt.plot(x_pdf,y_variable,color=colors[2])
plt.xlabel("Reward magnitude")
plt.ylabel("Decoded density\nat reward")
plt.xticks([])
plt.yticks([])
plt.xlim(0,10)
plt.legend()
#plt.show()
#fig.savefig(save_dir+r"\decoded_amount_cartoon.svg")
plt.show()

np.save("amount_pdf.npy",x_pdf)
np.save("pdf_ss.npy",y_ss)
np.save("pdf_ll.npy",y_ll)
np.save("pdf_variable.npy",y_variable)
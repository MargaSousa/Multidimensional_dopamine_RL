import time as timer
start_time = timer.time()
import matplotlib as mpl
from aux_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcol
flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName",[reward_cmap[1], reward_cmap[-1]])

# Parameters for plots
length_ticks=5
font_size=11
linewidth=1.2
scatter_size=10
horizontal_size=1
vertical_size=1
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth']=linewidth
mpl.rcParams['xtick.labelsize']=8
mpl.rcParams['ytick.labelsize']=8


colors_plot=["#7fc97f","#beaed4","#fdc086"] # ss,ll,variable

# Time: run before Fig1a_c
pdf_time_short=np.load("pdf_time_0.npy")
pdf_time_long=np.load("pdf_time_1.npy")
pdf_time_stationary=np.load("pdf_stationary.npy")
time=np.load("time_disc.npy")
n_time=len(time)
gamma=np.random.uniform(0,1,40)

# Magnitude: run before Fig1b
expectiles_ss=np.load("expectiles_ss.npy")
expectiles_ll=np.load("expectiles_ll.npy")
expectiles_variable=np.load("expectiles_variable.npy")
pdf_amount_variable=np.load("pdf_variable.npy")
pdf_amount_ss=np.load("pdf_ss.npy")
pdf_amount_ll=np.load("pdf_ll.npy")
amount=np.load("amount_pdf.npy")
n_amount=len(amount)

# Time x Magnitude
mesh=np.meshgrid(amount,time)


# Compute joint distribution over magnitude and time
joint_pdf_variable=np.zeros((n_time,n_amount))
joint_pdf_ss=np.zeros((n_time,n_amount))
joint_pdf_ll=np.zeros((n_time,n_amount))
joint_pdf_stationary_small=np.zeros((n_time,n_amount))
joint_pdf_stationary_big=np.zeros((n_time,n_amount))
for i_t in range(n_time):
    for i_a in range(n_amount):
        joint_pdf_variable[i_t,i_a]=pdf_time_short[i_t]*pdf_amount_variable[i_a]
        joint_pdf_ss[i_t, i_a] = pdf_time_short[i_t] * pdf_amount_ss[i_a]
        joint_pdf_ll[i_t, i_a] = pdf_time_long[i_t] * pdf_amount_ll[i_a]
        joint_pdf_stationary_small[i_t,i_a] = pdf_time_stationary[i_t]*pdf_amount_ss[i_a]
        joint_pdf_stationary_big[i_t,i_a]=pdf_time_stationary[i_t]*pdf_amount_ll[i_a]

joint_pdf_ss=joint_pdf_ss/np.sum(joint_pdf_ss)
joint_pdf_ll=joint_pdf_ll/np.sum(joint_pdf_ll)
joint_pdf_variable=joint_pdf_variable/np.sum(joint_pdf_variable)
joint_pdf_stationary_small=joint_pdf_stationary_small/np.sum(joint_pdf_stationary_small)
joint_pdf_stationary_big=joint_pdf_stationary_big/np.sum(joint_pdf_stationary_big)

# Color map for decoded density
color_map=sns.color_palette("coolwarm", as_cmap=True)

# Stacked decoded distributions
fig, ax = plt.subplots(figsize=(horizontal_size*3,vertical_size*3),subplot_kw={"projection": "3d"})
fig.set_facecolor('w')
ax.set_facecolor('w')
#ax.set_box_aspect((1, 1, 1.1))

scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_ss), np.max(joint_pdf_ss)),cmap=color_map)
ax.plot_surface(mesh[1], mesh[0],0+0*joint_pdf_ss, facecolors=scam.to_rgba(joint_pdf_ss),antialiased = True,rstride=1,cstride=1,alpha=None)

scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_ll), np.max(joint_pdf_ll)),cmap=color_map)
ax.plot_surface(mesh[1], mesh[0],0.04+0*joint_pdf_ll, facecolors=scam.to_rgba(joint_pdf_ll),antialiased = True,rstride=1,cstride=1,alpha=None)

scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_variable), np.max(joint_pdf_variable)),cmap=color_map)
ax.plot_surface(mesh[1], mesh[0],0.02+0*joint_pdf_variable, facecolors=scam.to_rgba(joint_pdf_variable),antialiased = True,rstride=1,cstride=1,alpha=None)

ax.set_xlabel("Reward\ntime")
ax.set_ylabel("Reward\nmagnitude")
ax.set_zticks([])
ax.set_xticks([])
ax.set_yticks([])

#ax.set_xlim(0,10)
#plt.colorbar()
#fig.subplots_adjust(left=0, right=0, bottom=0, top=0)
#fig.tight_layout()
#plt.savefig(save_dir+"\simulations_heat_map.eps", bbox_inches=0)
plt.show()

print("time elapsed: {:.2f}s".format(timer.time() - start_time))
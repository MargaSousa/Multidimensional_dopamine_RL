import time as timer
start_time = timer.time()
import matplotlib as mpl
from aux_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcol
import pandas as pd
mpl.use("TKAgg")

flatui = ["#9B59B6", "#3498DB", "#95A5A6", "#E74C3C", "#34495E", "#2ECC71"]
reward_cmap = plt.cm.jet(np.linspace(0., 1., 8)[:-1])
animal_cmap = sns.color_palette(flatui)
raster_cmap = plt.cm.bone_r
asym_cmap = plt.cm.autumn_r
asym_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [reward_cmap[1], reward_cmap[-1]])

dir_save_for_plot=r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\Figure_1"


# Parameters for plots
length_ticks = 5
font_size = 11
linewidth = 1.2
scatter_size = 10
horizontal_size = 1
vertical_size = 1
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

# Directory to save intermediary data
dir_save_for_plot=r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\Figure_1"

# Time: run before Fig1a_c
table_pdf_time = pd.read_csv(dir_save_for_plot+r"\Fig1c_pdf_time_ss_ll.csv")

pdf_time_short = table_pdf_time["Decoded density SS"]
pdf_time_long =  table_pdf_time["Decoded density LL"]
time = table_pdf_time["Time"]
n_time = len(time)
gamma = np.random.uniform(0, 1, 40)

# Magnitude: run before Fig1b
table_pdf_magnitude= pd.read_csv(dir_save_for_plot+r"\Fig1b_pdf_amount_ss_ll_variable.csv")

pdf_amount_variable = table_pdf_magnitude["Decoded variable"]
pdf_amount_ss = table_pdf_magnitude["Decoded density SS"]
pdf_amount_ll = table_pdf_magnitude["Decoded density LL"]
amount = table_pdf_magnitude["Amount"]
n_amount = len(amount)

# Time x Magnitude
mesh = np.meshgrid(amount, time)

# Compute joint distribution over magnitude and time
joint_pdf_variable = np.zeros((n_time, n_amount))
joint_pdf_ss = np.zeros((n_time, n_amount))
joint_pdf_ll = np.zeros((n_time, n_amount))
for i_t in range(n_time):
    for i_a in range(n_amount):
        joint_pdf_variable[i_t, i_a] = pdf_time_short[i_t] * pdf_amount_variable[i_a]
        joint_pdf_ss[i_t, i_a] = pdf_time_short[i_t] * pdf_amount_ss[i_a]
        joint_pdf_ll[i_t, i_a] = pdf_time_long[i_t] * pdf_amount_ll[i_a]


joint_pdf_ss = joint_pdf_ss / np.sum(joint_pdf_ss)
joint_pdf_ll = joint_pdf_ll / np.sum(joint_pdf_ll)
joint_pdf_variable = joint_pdf_variable / np.sum(joint_pdf_variable)


# Color map for decoded density
color_map = sns.color_palette("coolwarm", as_cmap=True)

# Stacked decoded distributions
fig, ax = plt.subplots(figsize=(horizontal_size * 2, vertical_size * 6), subplot_kw={"projection": "3d"})
fig.set_facecolor('w')
ax.set_facecolor('w')
ax.view_init(elev=-150, azim=50)
ax.set_box_aspect((1,1,1.8))

joint_pdf_ss=np.transpose(joint_pdf_ss)
ax.set_zlim(-0.03,-0.01)
scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_ss), np.max(joint_pdf_ss)), cmap=color_map)
ax.plot_surface(mesh[1], mesh[0], -0.01 + 0 * joint_pdf_ss, facecolors=scam.to_rgba(joint_pdf_ss), antialiased=True,
                rstride=1, cstride=1, alpha=None, shade=False)

joint_pdf_ll=np.transpose(joint_pdf_ll)
scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_ll), np.max(joint_pdf_ll)), cmap=color_map)
ax.plot_surface(mesh[1], mesh[0], -0.03 + 0 * joint_pdf_ll, facecolors=scam.to_rgba(joint_pdf_ll), antialiased=True,
                rstride=1, cstride=1, alpha=None, shade=False)

joint_pdf_variable=np.transpose(joint_pdf_variable)
scam = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(np.min(joint_pdf_variable), np.max(joint_pdf_variable)),
                             cmap=color_map)
ax.plot_surface(mesh[1], mesh[0], -0.02 + 0 * joint_pdf_variable, facecolors=scam.to_rgba(joint_pdf_variable),
                antialiased=True, rstride=1, cstride=1, alpha=None, shade=False)

ax.set_ylabel("Time since\ncue")
ax.set_xlabel("Reward\nmagnitude")
ax.set_zticks([])
ax.set_xticks([])
ax.set_yticks([])
#fig.savefig("joint_simulations.eps")
plt.show()
print("time elapsed: {:.2f}s".format(timer.time() - start_time))

np.savetxt(dir_save_for_plot+r"\Fig1d_right_joint_pdf_ll.csv",joint_pdf_ll)
np.savetxt(dir_save_for_plot+r"\Fig1d_right_joint_pdf_ss.csv",joint_pdf_ss)
np.savetxt(dir_save_for_plot+r"\Fig1d_right_joint_pdf_variable.csv",joint_pdf_variable)


import time as timer
start_time = timer.time()
from scipy.stats import gaussian_kde
from aux_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
from aux_functions import get_expectiles
import pandas as pd


# Directory to save intermediary data
dir_save_for_plot=r"C:\Users\Margarida\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\MS\Data_paper_organized\Figure_1"


# Parameters for plots
length_ticks = 3
font_size = 11
linewidth = 1.2
scatter_size = 2
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth
horizontal_size = 1.5
vertical_size = 1.5

N_exp = 10000  # Number of samples to estimate expectiles
x_max = 15
x_exp = np.linspace(0, x_max, N_exp)

# Define probabilities
n_neurons = 40
taus = np.linspace(1.0 / n_neurons, 1 - 1.0 / n_neurons, n_neurons)
smooth = 0.35
x_pdf = np.linspace(0, 15, 120)

# Long large (ll) cue
mean_ll = 8
scale_ll = 0.4
probability_exp = scipy.stats.norm(loc=mean_ll, scale=scale_ll).pdf(x_exp)
probability_exp = probability_exp / np.sum(probability_exp)
pos_expectiles, expectiles_ll = get_expectiles(x_exp, probability_exp, taus)

# Short small (ss) cue
mean_ss = 3
scale_ss = 0.4
probability_exp = scipy.stats.norm(loc=mean_ss, scale=scale_ss).pdf(
    x_exp)  # +scipy.stats.norm(loc=mean_2,scale=scale_2).pdf(x_exp)
probability_exp = probability_exp / np.sum(probability_exp)
pos_expectiles, expectiles_ss = get_expectiles(x_exp, probability_exp, taus)

# Short variable cue
mean_1 = 2
scale_1 = 0.2
mean_2 = 4
scale_2 = 0.2
probability_exp = scipy.stats.norm(loc=mean_1, scale=scale_1).pdf(x_exp) + scipy.stats.norm(loc=mean_2,
                                                                                            scale=scale_2).pdf(x_exp)
probability_exp = probability_exp / np.sum(probability_exp)
pos_expectiles, expectiles_variable = get_expectiles(x_exp, probability_exp, taus)

# Save expectiles
np.savetxt(dir_save_for_plot+"\expectiles_ss.csv", expectiles_ss)
np.savetxt(dir_save_for_plot+"\expectiles_ll.csv", expectiles_ll)
np.savetxt(dir_save_for_plot+"\expectiles_variable.csv", expectiles_variable)

# Decode distributions
samples_ss, _ = run_decoding_magnitude(expectiles_ss, taus, np.ones(len(taus)), minv=1, maxv=10, N=20, max_samples=2000,
                             max_epochs=15, method='TNC')
kde_ss = gaussian_kde(samples_ss, bw_method=smooth)
y_ss = kde_ss.pdf(x_pdf)
y_ss = y_ss / np.sum(y_ss)

samples_ll, _ = run_decoding_magnitude(expectiles_ll, taus, np.ones(len(taus)), minv=1, maxv=10, N=20, max_samples=2000,
                             max_epochs=15, method='TNC')
kde_ll = gaussian_kde(samples_ll, bw_method=smooth)
y_ll = kde_ll.pdf(x_pdf)
y_ll = y_ll / np.sum(y_ll)

samples_variable, _ = run_decoding_magnitude(expectiles_variable, taus, np.ones(len(taus)), minv=1, maxv=10, N=20,
                                   max_samples=2000, max_epochs=15, method='TNC')
kde_variable = gaussian_kde(samples_variable, bw_method=smooth)
y_variable = kde_variable.pdf(x_pdf)
y_variable = y_variable / np.sum(y_variable)

colors = ["#7fc97f", "#beaed4", "#fdc086"]

fig, ax = plt.subplots(figsize=(vertical_size, horizontal_size))
ax.spines['left'].set_linewidth(linewidth)
ax.spines['bottom'].set_linewidth(linewidth)
plt.plot(x_pdf, y_ss, color=colors[0], label="SS")
plt.plot(x_pdf, y_ll, color=colors[1], label="LL")
plt.plot(x_pdf, y_variable, color=colors[2])
plt.xlabel("Reward magnitude")
plt.ylabel("Decoded density\nat reward")
plt.xticks([])
plt.yticks([])
plt.xlim(0, 10)
plt.legend()
# plt.show()
# fig.savefig(save_dir+r"\decoded_amount_cartoon.svg")
plt.show()

# np.savetxt(dir_save_for_plot+r"\amount_pdf.csv", x_pdf)
# np.savetxt(dir_save_for_plot+r"\pdf_ss.csv", y_ss)
# np.savetxt(dir_save_for_plot+r"\pdf_ll.csv", y_ll)
# np.savetxt(dir_save_for_plot+r"\pdf_variable.csv", y_variable)

decoded_info={"Amount": x_pdf, "Decoded density SS": y_ss, "Decoded density LL": y_ll, "Decoded variable": y_variable}
df = pd.DataFrame(decoded_info)
df.to_csv(dir_save_for_plot+r'\Fig1b_pdf_amount_ss_ll_variable.csv',index=False,header=True, sep=',')


print("time elapsed: {:.2f}s".format(timer.time() - start_time))

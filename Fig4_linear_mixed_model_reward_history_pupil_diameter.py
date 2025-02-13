import statsmodels.formula.api as smf
from aux_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

mpl.use('TkAgg')

# Parameters for paper plots
length_ticks = 2
linewidth = 1.2
scatter_size = 20
horizontal_size = 1.5 * 0.6
vertical_size = 1.5 * 0.6
font_size = 11
labelsize = 8
legendsize = font_size

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rc('xtick', labelsize=labelsize)
mpl.rc('ytick', labelsize=labelsize)
mpl.rc('legend', fontsize=legendsize)

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
directory_parsed_data = os.path.join(directory, "Parsed_data_DA")
df = pd.read_csv(os.path.join(directory_parsed_data, 'data_frame_pupil_diameter_reward_history.csv'))

md = smf.mixedlm("Pupil_diameter ~ Reward_history", df, groups=df["Animal"])
mdf = md.fit()
y_predict = np.array(mdf.fittedvalues)
print(mdf.summary())

# Bootstrap to get a p-value on the regression slope
n_bootstraps = 5
percentage = 0.5
id, counts = np.unique(df['Animal'], return_counts=True)
n_samples = int(np.min(counts))
slopes = []

# Bootstrap over trials
for boot in range(n_bootstraps):
    idx = df.groupby('Animal')['Animal'].sample(n=n_samples, replace=True).index
    new_dataframe = df.loc[idx, :]
    md = smf.mixedlm("Pupil_diameter ~ Reward_history", new_dataframe, groups=new_dataframe["Animal"])
    mdf = md.fit()
    slopes.append(mdf.params.Reward_history)

slopes = np.array(slopes)
print(np.quantile(slopes, 0.025))
print(np.quantile(slopes, 0.975))
p_value = np.sum(slopes <= 0) / len(slopes)
pdb.set_trace()

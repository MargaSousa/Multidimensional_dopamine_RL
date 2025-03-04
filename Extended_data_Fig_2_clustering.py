import os
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import cmasher as cmr

mpl.use('TkAgg')

length_ticks = 3
font_size = 10
linewidth = 1.2
scatter_size = 2
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
horizontal_size = 2.2
vertical_size = 2.2


# Get the auROC for each bin in the window around cue [window_roc_left_cue,window_roc_right_cue] and around reward [window_roc_left_reward, window_roc_right_reward]
# for all neurons in df_neurons
def get_auROC(df_neurons, window_roc_left_cue, window_roc_right_cue, window_roc_left_reward, window_roc_right_reward):
    # Get bins
    bin_left_reward = np.where(axes_correct >= window_roc_left_reward)[0][0]
    bin_right_reward = np.where(axes_correct >= window_roc_right_reward)[0][0]
    bin_left_cue = np.where(axes_correct >= window_roc_left_cue)[0][0]
    bin_right_cue = np.where(axes_correct >= window_roc_right_cue)[0][0]
    bin_0_ms = np.where(axes_correct >= 0)[0][0]

    n_bins_roc_cue = (bin_right_cue - bin_left_cue)
    n_bins_roc_reward = (bin_right_reward - bin_left_reward)
    X = np.zeros((n_bins_roc_cue + n_bins_roc_reward, 220))
    total_neurons = 0

    for neuron in dataframe_behavior_times['Neuron id'].drop_duplicates().values:

        dataframe_behavior_neuron = dataframe_behavior_times[dataframe_behavior_times['Neuron id'] == neuron]

        # Distribution reward ID: 0 is certain reward of 4.5uL, 2 is variable bimodal reward
        distribution_id_trials = dataframe_behavior_neuron['Distribution reward ID'].values

        # If the mice was rewarded in each trial
        isRewarded_trials = dataframe_behavior_neuron['Is rewarded'].values

        valid_trials = np.intersect1d(np.where(isRewarded_trials == 1)[0], np.where(distribution_id_trials == 2)[0])

        psth_cue_neuron = np.array(dataframe_behavior_neuron['PSTH cue'].tolist())
        psth_reward_neuron = np.array(dataframe_behavior_neuron['PSTH reward'].tolist())
        baseline = np.nanmedian(psth_cue_neuron[valid_trials, bin_left_cue:bin_0_ms], axis=1)
        n_trials = len(valid_trials)

        # Aligned to cue
        for i in range(n_bins_roc_cue):
            fr_after_cue = psth_cue_neuron[valid_trials, bin_left_cue + i]
            y_true_cue = np.concatenate((np.zeros(n_trials), np.ones(n_trials)), axis=0)
            y_score_cue = np.concatenate((baseline, fr_after_cue), axis=0)
            auc_cue = roc_auc_score(y_true_cue, y_score_cue)
            X[i, total_neurons] = auc_cue

        # Aligned to reward
        for i in range(n_bins_roc_reward):
            fr_after_rew = psth_reward_neuron[valid_trials, bin_left_reward + i]
            y_true_rew = np.concatenate((np.zeros(n_trials), np.ones(n_trials)), axis=0)
            y_score_rew = np.concatenate((baseline, fr_after_rew), axis=0)
            auc_rew = roc_auc_score(y_true_rew, y_score_rew)
            X[n_bins_roc_cue + i, total_neurons] = auc_rew
        total_neurons += 1

    X = 2 * X - 1  # Change the range to [-1,1]
    X = X[:, :total_neurons]
    bins_cue = axes_correct[bin_left_cue:bin_right_cue]
    bins_reward = axes_correct[bin_left_reward:bin_right_reward]
    return X, bins_cue, bins_reward


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dend = dendrogram(linkage_matrix, above_threshold_color='black', **kwargs)
    return dend


# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"
directory_raw_data = os.path.join(directory, "Raw")

# Behavior information data
dataframe_behavior_times = pd.read_csv(os.path.join(directory_raw_data, "Neurons_behavior_trials_with_PSTH.csv"))
dataframe_behavior_times['PSTH cue'] = dataframe_behavior_times.loc[:,
                                       'PSTH aligned to cue bin 0':'PSTH aligned to cue bin 649'].values.tolist()
dataframe_behavior_times['PSTH reward'] = dataframe_behavior_times.loc[:,
                                          'PSTH aligned to reward bin 0':'PSTH aligned to reward bin 649'].values.tolist()

# Axes of PSTH
axes_correct = np.linspace(-5, 8, 650)  # axis for PSTH

# auROC aligned in reward in the window [0,0.8]
window_roc_left_reward = 0
window_roc_right_reward = 0.8

# auROC aligned in cue in the window [-0.5,0.8]
window_roc_left_cue = -0.5
window_roc_right_cue = 0.8

# In bins
bin_left_cue = np.where(axes_correct >= window_roc_left_cue)[0][0]
bin_right_cue = np.where(axes_correct >= window_roc_right_cue)[0][0]
time_plot_cue = axes_correct[bin_left_cue:bin_right_cue]
n_bins_cue = bin_right_cue - bin_left_cue

X, bins_cue, bins_reward = get_auROC(dataframe_behavior_times, window_roc_left_cue, window_roc_right_cue,
                                     window_roc_left_reward, window_roc_right_reward)
X_auROC = np.transpose(X)

n_pcs = 3
pca = PCA(n_components=n_pcs)
pca.fit(X_auROC)
neurons_pc_space = pca.transform(X_auROC)
pcs = np.matmul(X, neurons_pc_space)
n_neurons = X_auROC.shape[0]

# Clustering
link = "complete"
metric = 'euclidean'
n_clusters = 3
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage=link, compute_distances=True)
cluster.fit_predict(neurons_pc_space[:, :3])
labels = cluster.labels_

# Plot dendogram
plt.subplot(1, 4, 4)
dend = plot_dendrogram(cluster, color_threshold=9, orientation="right", truncate_mode="level", no_labels=True)
leaves = np.flip(dend['leaves'])

# Plot auROC for cue
plt.subplot(1, 4, 1)
plt.imshow(X_auROC[leaves, :n_bins_cue], aspect="auto", cmap="cmr.watermelon",
           extent=[window_roc_left_cue, window_roc_right_cue, n_neurons, 1], vmin=-1, vmax=1)
plt.ylabel("Neurons", labelpad=-1)
plt.xlabel("Time since \n cue (s)")
plt.xticks([window_roc_left_cue, 0, window_roc_right_cue], ["-0.5", "0", "0.8"])
plt.yticks([1, n_neurons - 1], ["1", str(n_neurons)])
plt.axhline(y=np.sum(labels == 1), color="white")
plt.axhline(y=np.sum(labels == 1) + np.sum(labels == 2), color="white")


# auROC for each neuron
to_save_info = {"Neuron": np.arange(X_auROC.shape[0])}
for i_bin,bin in enumerate(bins_cue):
    to_save_info[str(np.round(bin,4))+"s relative to cue"]=X_auROC[leaves,i_bin]

n_bins_cue=len(bins_cue)
# for i_bin,bin in enumerate(bins_reward):
#     to_save_info[str(np.round(bin,4)) + "s relative to reward"] = X_auROC[leaves, n_bins_cue+i_bin]
#df = pd.DataFrame(to_save_info)
#df.to_csv(r'\Extended_data_Fig2a.csv', index=False, header=True,sep=',')

# Save PCA for each neuron
# to_save_info = {"Neuron": np.arange(neurons_pc_space.shape[0])}
# for i_pca in range(n_pcs):
#     to_save_info["PC "+str(i_pca+1)]=neurons_pc_space[leaves,i_pca]
# df = pd.DataFrame(to_save_info)
# df.to_csv('\Extended_data_Fig2b.csv', index=False, header=True,sep=',')


# Plot photo ided neurons
photo_ided_idx = np.unique(dataframe_behavior_times[dataframe_behavior_times['Is photo ided'] == 1]['Neuron id'].values)
for j, pos_neu in enumerate(photo_ided_idx):
    pos = np.where(leaves == pos_neu)[0][0]
    plt.scatter(0, pos, color="red", s=15)

# Plot auROC for reward
plt.subplot(1, 4, 2)
plt.imshow(X_auROC[leaves, n_bins_cue:], aspect="auto", cmap="cmr.watermelon",
           extent=[window_roc_left_reward, window_roc_right_reward, n_neurons, 1], vmin=-1, vmax=1)
plt.xlabel("Time since \n reward (s)")
plt.xticks([window_roc_left_reward, window_roc_right_reward], ["0", "0.8"])
plt.yticks([])
cbar = plt.colorbar(shrink=0.25, ticks=[-1, 0, 1])
cbar.set_label("auROC", labelpad=-1)
plt.axhline(y=np.sum(labels == 1), color="white")
plt.axhline(y=np.sum(labels == 1) + np.sum(labels == 2), color="white")

min_v = -6
max_v = 6
plt.subplot(1, 4, 3)
# plt.imshow(neurons_pc_space[leaves,:],extent=[0,1.5,n_neurons,1],aspect="auto",cmap="bwr",vmin=min_v,vmax=max_v,interpolation=None)
plt.matshow(neurons_pc_space[leaves, :], extent=[0, 1.5, n_neurons, 1], aspect="auto", cmap="bwr", vmin=min_v,
            vmax=max_v)
plt.xticks([0.25, 0.75, 1.25], ["1", "2", "3"])
plt.yticks([])
plt.xlabel("PC")
cbar = plt.colorbar(shrink=0.25, ticks=[min_v, 0, max_v])
cbar.set_label("PC", labelpad=-1)
plt.axhline(y=np.sum(labels == 1), color="k")
plt.axhline(y=np.sum(labels == 1) + np.sum(labels == 2), color="k")
plt.show()

# Plot neurons in PC space

# Color by cluster label
color = np.empty(n_neurons, dtype='<U16')
color[labels == 0] = "red"
color[labels == 1] = "orange"
color[labels == 2] = "forestgreen"

fig = plt.figure(figsize=(1.6, 1.6))
ax = plt.axes(projection='3d')
ax.scatter(neurons_pc_space[:, 0], neurons_pc_space[:, 1], neurons_pc_space[:, 2], c=color)
ax.set_xlabel("PC 1", labelpad=0.01)
ax.set_ylabel("PC 2", labelpad=0.01)
ax.set_zlabel("PC 3", labelpad=0.01)
ax.set_xticks([-6, 6])
ax.set_yticks([-6, 6])
ax.set_zticks([-6, 6])
plt.show()

# PSTH  for each cluster
neuron_types = ["Putative_DA", "Putative_GABA", "Type_1"]

winter = mpl.cm.get_cmap('winter', 12)
colors_amount = winter(np.linspace(0, 1, 5))
summer = mpl.cm.get_cmap('Reds', 12)
colors_delay = summer(np.linspace(0.4, 1, 4))

save_psth_neurons=[]
save_aligned_to=[]
save_bins_psth=[]
save_delay=[]
save_magnitude=[]


for neuron_type in neuron_types:

    # Filter dataframe for neurons with neuron type
    dataframe_neuron_type = dataframe_behavior_times[dataframe_behavior_times['Type of neuron'] == neuron_type]

    # Cue-aligned PSTH for Distribution ID == 0
    df_cue = dataframe_neuron_type[dataframe_neuron_type['Distribution reward ID'] == 0]

    delays = np.sort(df_cue['Delay reward'].unique())

    # PSTH aligned to cue for different delays
    for i_delay, delay in enumerate(delays):
        psth=(df_cue[df_cue['Delay reward'] == delay].groupby('Neuron id')['PSTH cue'].apply(lambda x: np.mean(np.vstack(x), axis=0)).to_numpy())
        psth=np.stack(psth)
        psth_mean = np.mean(psth, axis=0)
        psth_sem = scipy.stats.sem(psth, axis=0, nan_policy='omit').astype(float)
        plt.plot(axes_correct, psth_mean, color=colors_delay[i_delay])
        plt.fill_between(axes_correct, psth_mean - psth_sem, psth_mean + psth_sem, alpha=0.2,
                         color=colors_delay[i_delay])

    plt.xlabel('Time since cue delivery (s)')
    plt.ylabel('Firing rate (spikes/s)')
    plt.legend()
    plt.show()

    # Second plot: PSTH aligned to reward for trials with 'Distribution ID' == 1
    df_reward = dataframe_neuron_type[dataframe_neuron_type['Distribution reward ID'] == 2]
    amounts = np.sort(df_reward['Amount reward'].unique())

    plt.figure(figsize=(10, 5))

    # PSTH aligned to reward for different amounts
    for i_amount, amount in enumerate(amounts):
        psth=(df_reward[df_reward['Amount reward'] == amount].groupby('Neuron id')['PSTH reward'].apply(lambda x: np.mean(np.vstack(x), axis=0)).to_numpy())
        psth=np.stack(psth)
        psth_mean = np.mean(psth, axis=0)
        psth_sem = scipy.stats.sem(psth, axis=0, nan_policy='omit').astype(float)
        plt.plot(axes_correct, psth_mean, color=colors_amount[i_amount])
        plt.fill_between(axes_correct, psth_mean - psth_sem, psth_mean + psth_sem, alpha=0.2,
                         color=colors_amount[i_amount])

    plt.xlabel('Time since reward delivery (s)')
    plt.ylabel('Firing rate (spikes/s)')
    plt.legend()
    plt.show()

pdb.set_trace()

import os
from sklearn.linear_model import HuberRegressor
from sklearn.utils import resample
from aux_functions import *

# Where folder is saved
directory = "/Users/margaridasousa/Desktop/Data_repository_paper"

# Parsed data directory
type_neurons = "DA"  # either "DA" or "putative_DA"
directory_parsed_data = os.path.join(directory, "Parsed_data_" + type_neurons)

# Get estimated tuning for reward time and magnitude
data_frame_neurons_info = pd.read_csv(os.path.join(directory_parsed_data, "dataframe_neurons_info.csv"))
discount = data_frame_neurons_info['Gamma'].values
estimated_reversals = data_frame_neurons_info['Reversals'].values
estimated_taus = data_frame_neurons_info['Taus'].values
gains = data_frame_neurons_info['Gain'].values
n_neurons = len(discount)

responses_reward_certain = np.loadtxt(
    os.path.join(directory_parsed_data, "responses_reward_certain_magnitude_3s_delay.csv"))
responses_reward_variable = np.load(
    os.path.join(directory_parsed_data, "responses_reward_different_magnitudes_constant_delay.npy"))

responses_cue_variable = np.loadtxt(
    os.path.join(directory_parsed_data, "responses_cue_different_magnitudes_constant_delay.csv"))
responses_cue_certain = np.load(
    os.path.join(directory_parsed_data, "responses_cue_different_delays_constant_magnitude.npy"))[:, 2, :]

# Select neurons from given animals
selected_animal = 3353
selected_neurons = data_frame_neurons_info.index[data_frame_neurons_info.Animal == selected_animal].tolist()

# Choose instead neurons from all animals
selected_neurons = np.arange(len(data_frame_neurons_info))

# Regression FR at cue and reversal points
well_estimated_reversals = np.intersect1d(np.where(estimated_reversals > 1.1)[0],
                                          np.where(estimated_reversals < 7.9)[0])
population_responses_variable = np.nanmean(responses_cue_variable, axis=1)
population_responses_variable_corrected = population_responses_variable / (gains * discount ** 3)
reg_variable = HuberRegressor().fit((population_responses_variable_corrected[well_estimated_reversals]).reshape(-1, 1),
                                    estimated_reversals[well_estimated_reversals])

# Correct for  diversity in temporal discount and gain
fr_certain_corrected = np.nanmean(responses_cue_certain, axis=1) / (gains * discount ** 3)
fr_variable_corrected = np.nanmean(responses_cue_variable, axis=1) / (gains * discount ** 3)

reversals_certain = reg_variable.predict(fr_certain_corrected.reshape(-1, 1))
reversals_variable = reg_variable.predict(fr_variable_corrected.reshape(-1, 1))

neurons_id = np.arange(n_neurons)
resamples = 10000
var_certain = []
var_variable = []
n = 0
dif_var = []
for r in range(resamples):
    new_responses_variable = resample(reversals_variable, replace=True, n_samples=n_neurons)  # fr_variable_corrected
    new_responses_certain = resample(reversals_certain, replace=True, n_samples=n_neurons)  # fr_certain_corrected
    variance_variable = np.var(new_responses_variable)
    variance_certain = np.var(new_responses_certain)
    var_variable.append(variance_variable)
    var_certain.append(variance_certain)
    dif_var.append(variance_variable - variance_certain)
    if variance_certain >= variance_variable:
        n += 1

dif_var = np.array(dif_var)
alpha = 0.05
low_quantile = np.quantile(dif_var, alpha / 2)
high_quantile = np.quantile(dif_var, 1 - alpha / 2)

print("CI")
print(low_quantile, high_quantile)
print("Mean diff")
print(np.mean(dif_var))
print("p-value :", n / resamples)

pdb.set_trace()

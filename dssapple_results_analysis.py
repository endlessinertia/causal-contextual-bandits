import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

### IMPORT RESULTS ###

folder_name = './data/dssapple/results/3000inst-random-drop_sum-pca-sim-ctx/'
filenames = os.listdir(folder_name)
outputs_list = list()
for fn in filenames:
    with open(folder_name + fn, 'rb') as output_file:
        outputs_list.append(pickle.load(output_file))

### FUNCTIONS ###

def compute_metric_average(outputs_list, metric):
    list_metric_arrays = [np.array(out_dict[metric]) for out_dict in outputs_list]
    return sum(list_metric_arrays) / len(list_metric_arrays)

### RUN THE SCRIPT ###

causal_avg_rew = compute_metric_average(outputs_list, 'causal_reward')
ext_cmab_avg_rew = compute_metric_average(outputs_list, 'ext_cmab_reward')
cmab_avg_rew = compute_metric_average(outputs_list, 'cmab_reward')
obs_avg_rew = compute_metric_average(outputs_list, 'intuition_reward')

plt.plot(range(len(causal_avg_rew)), causal_avg_rew, 'r',
         range(len(ext_cmab_avg_rew)), ext_cmab_avg_rew, 'b',
         range(len(cmab_avg_rew)), cmab_avg_rew, 'g',
         range(len(obs_avg_rew)), obs_avg_rew, 'y')
plt.show()


### IMPORT RESULTS ###

folder_name = './data/dssapple/results/2000inst-random-drop_sum-pca-sim-ctx/'
filenames = os.listdir(folder_name)
outputs_list = list()
for fn in filenames:
    with open(folder_name + fn, 'rb') as output_file:
        outputs_list.append(pickle.load(output_file))

### RUN THE SCRIPT ###

bias_avg_rew = compute_metric_average(outputs_list, 'bias_reward')
causal_avg_rew = compute_metric_average(outputs_list, 'causal_reward')
ext_cmab_avg_rew = compute_metric_average(outputs_list, 'ext_cmab_reward')
cmab_avg_rew = compute_metric_average(outputs_list, 'cmab_reward')
obs_avg_rew = compute_metric_average(outputs_list, 'intuition_reward')

plt.plot(range(len(bias_avg_rew)), bias_avg_rew, 'k',
         range(len(causal_avg_rew)), causal_avg_rew, 'r',
         range(len(ext_cmab_avg_rew)), ext_cmab_avg_rew, 'b',
         range(len(cmab_avg_rew)), cmab_avg_rew, 'g',
         range(len(obs_avg_rew)), obs_avg_rew, 'y')
plt.show()
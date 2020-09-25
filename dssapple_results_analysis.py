import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import collections


### HELPER FUNCTIONS ###

def compute_metric_average(outputs_list, metric):
    list_metric_arrays = [np.array(out_dict[metric]) for out_dict in outputs_list]
    return sum(list_metric_arrays) / len(list_metric_arrays)


def plot_average_cumulative_reward(results_folder, plot_title):
    filenames = os.listdir(results_folder)
    outputs_list = list()
    for fn in filenames:
        with open(results_folder + fn, 'rb') as output_file:
            outputs_list.append(pickle.load(output_file))

    causal_avg_rew = compute_metric_average(outputs_list, 'causal_reward')
    ext_cmab_avg_rew = compute_metric_average(outputs_list, 'ext_cmab_reward')
    cmab_avg_rew = compute_metric_average(outputs_list, 'cmab_reward')
    obs_avg_rew = compute_metric_average(outputs_list, 'intuition_reward')

    plt.plot(range(len(causal_avg_rew)), causal_avg_rew, 'g',
             range(len(ext_cmab_avg_rew)), ext_cmab_avg_rew, 'b',
             range(len(cmab_avg_rew)), cmab_avg_rew, 'r',
             range(len(obs_avg_rew)), obs_avg_rew, 'k--')
    plt.legend(('Causal_TS', 'Extended_TS', 'Standard_TS', 'Observational'))
    plt.xlabel('# observation')
    plt.ylabel('cumulative reward')
    plt.title(plot_title)
    plt.show()


def compute_avg_arms_distr(outputs_list, target_arms, n_arms=5):
    list_arms_freq = list()
    for output in outputs_list:
        arms_count = collections.Counter(output[target_arms])
        ordered_freq = [freq for (arm, freq) in arms_count.most_common()]
        if len(ordered_freq) < n_arms:
            ordered_freq.extend([0]*(n_arms - len(ordered_freq)))
        list_arms_freq.append(ordered_freq)
    return np.sum(list_arms_freq, axis=0) / len(list_arms_freq)


def plot_arms_distribution(results_folder, plot_title):
    filenames = os.listdir(results_folder)
    outputs_list = list()
    for fn in filenames:
        with open(results_folder + fn, 'rb') as output_file:
            outputs_list.append(pickle.load(output_file))

    causal_arms_freq = compute_avg_arms_distr(outputs_list, 'causal_arms')
    ext_cmab_arms_freq = compute_avg_arms_distr(outputs_list, 'ext_cmab_arms')
    cmab_arms_freq = compute_avg_arms_distr(outputs_list, 'cmab_arms')
    obs_arms_freq = compute_avg_arms_distr(outputs_list, 'intuition_arms')
    true_arms_freq = compute_avg_arms_distr(outputs_list, 'true_labels')

    x = np.arange(len(causal_arms_freq))
    width = 0.13

    plt.bar(x - 2*width, causal_arms_freq, width)
    plt.bar(x - width, ext_cmab_arms_freq, width)
    plt.bar(x, cmab_arms_freq, width)
    plt.bar(x + width, obs_arms_freq, width)
    plt.bar(x + 2*width, true_arms_freq, width)

    plt.legend(('Causal_TS', 'Extended_TS', 'Standard_TS', 'Observational', 'True'))
    plt.xlabel('arms')
    plt.ylabel('frequency')
    plt.title(plot_title)
    plt.show()


### 1000 Observations - Full Random Data - Sum of Images PCA Context ###

folder_name = './data/dssapple/results/1000inst-random_sum-pca-ctx/'
plot_average_cumulative_reward(folder_name, '1000 Observations - Full Random Data - Sum of Images PCA Context')

### 1000 Observations - Full Random Data - Sum of Similarities PCA Context ###

folder_name = './data/dssapple/results/1000inst-random_sum-pca-sim-ctx/'
plot_average_cumulative_reward(folder_name, '1000 Observations - Full Random Data - Sum of Similarities PCA Context')

### 2000 Observations - Random Data Dropout - Sum of Images PCA Context ###

folder_name = './data/dssapple/results/2000inst-random-drop_sum-pca-ctx/'
plot_average_cumulative_reward(folder_name, '2000 Observations - Random Data Dropout - Sum of Images PCA Context')

### 2000 Observations - Random Data Dropout - Sum of Similarities PCA Context ###

folder_name = './data/dssapple/results/2000inst-random-drop_sum-pca-sim-ctx/'
plot_average_cumulative_reward(folder_name, '2000 Observations - Random Data Dropout - Sum of Similarities PCA Context')

### 2000 Observations - Test Images KB Context ###

folder_name = './data/dssapple/results/2000inst_test-kb-ctx/'
plot_average_cumulative_reward(folder_name, '2000 Observations - Test Images KB Context')

### 2000 Observations - Test Images KB Context - Negative Reward ###

folder_name = './data/dssapple/results/2000inst_test-kb-ctx_neg-1/'
plot_average_cumulative_reward(folder_name, '2000 Observations - Test Images KB Context - Negative Reward')


### 2000 Observations - Test Images KB PCA Context ###

folder_name = './data/dssapple/results/2000inst_pca-test-kb-ctx/'
plot_average_cumulative_reward(folder_name, '2000 Observations - Test Images KB PCA Context')

### 2000 Observations - Random Data Dropout - Hybrid Context: PCA Sum Sim + PCA KB Test ###

folder_name = './data/dssapple/results/2000inst-random-drop_hybrid-ctx-pca-test-kb+sum-pca-sim/'
plot_average_cumulative_reward(folder_name, '2000 Observations - Random Data Dropout - Hybrid Context: PCA Sum Sim + PCA KB Test')

### 3000 Observations - Random Data Dropout - Sum of Images PCA Context ###

folder_name = './data/dssapple/results/3000inst-random-drop_sum-pca-ctx/'
plot_average_cumulative_reward(folder_name, '3000 Observations - Random Data Dropout - Sum of Images PCA Context')
plot_arms_distribution(folder_name, '3000 Observations - Random Data Dropout - Sum of Images PCA Context')

### 3000 Observations - Random Data Dropout - Sum of Similarities PCA Context ###

folder_name = './data/dssapple/results/3000inst-random-drop_sum-pca-sim-ctx/'
plot_average_cumulative_reward(folder_name, '3000 Observations - Random Data Dropout - Sum of Similarities PCA Context')
plot_arms_distribution(folder_name, '3000 Observations - Random Data Dropout - Sum of Similarities PCA Context')
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./data/fashion/results/output_report.pickle', 'rb') as output_file:
    output_dict = pickle.load(output_file)

print(output_dict.keys())

plt.plot(range(len(output_dict['true_labels'])), output_dict['classifier_reward'], 'r',
         range(len(output_dict['true_labels'])), output_dict['causal_reward'], 'b',
         range(len(output_dict['true_labels'])), output_dict['cmab_reward'], 'g')
plt.show()

plt.plot(range(1000), output_dict['classifier_reward'][:1000], 'r',
         range(1000), output_dict['causal_reward'][:1000], 'b',
         range(1000), output_dict['cmab_reward'][:1000], 'g')
plt.show()

classifier_regret = np.arange(1, 10001) - output_dict['classifier_reward']
causal_regret = np.arange(1, 10001) - output_dict['causal_reward']
cmab_regret = np.arange(1, 10001) - output_dict['cmab_reward']

plt.plot(range(len(output_dict['true_labels'])), classifier_regret, 'r',
         range(len(output_dict['true_labels'])), causal_regret, 'b',
         range(len(output_dict['true_labels'])), cmab_regret, 'g')
plt.show()

plt.plot(range(1000), classifier_regret[:1000], 'r',
         range(1000), causal_regret[:1000], 'b',
         range(1000), cmab_regret[:1000], 'g')
plt.show()


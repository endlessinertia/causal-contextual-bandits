import pickle
import ast
import pandas as pd
import numpy as np
import random
from causal_contextual_bandit import CausalCMAB
from causal_contextual_bandit import CMAB


### PARAMETERS ###

MAX_ITERATIONS = 2000
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = 0.0
NUM_TRIALS = 100

### DATA IMPORT ###

logs_df = pd.read_csv('./data/dssapple/ChallengeData_08-04-19_filtered.csv', sep=';')
print(logs_df)

with open('./data/dssapple/img_list_150', 'rb') as list_f:
    img_list = pickle.load(list_f)
with open('./data/dssapple/context_sim_pca_matrix_150', 'rb') as context_f:
    context_matrix = pickle.load(context_f)
context_df = pd.DataFrame(context_matrix, index=img_list)
print(context_df)

### FUNCTIONS ###

def compute_challenge_context(list_selected_imgs, context_df, dropout):
    """
    Compute the context for a given challenge, as the combination (sum) of the context vector of each selected image.
    The context is compute with a dropout chance for each selected image.
    :param list_selected_imgs: The list of selected images in the challenge.
    :param context_df: The DataFrame that indexes the context vector of each candidate image.
    :param dropout: Dropout chance of not considering a given image in the context computation.
    :return: The np.array vector representing the context for the given challenge.
    """
    combined_context = np.zeros(context_df.shape[1])
    for img in list_selected_imgs:
        if random.random() >= dropout:
            combined_context += context_df.loc[img].tolist()
    return combined_context

def create_bandit_dataset(logs_df, context_df, replica=1, dropout=0.0):
    """
    Generate the dataset for the bandit simulation, starting from the logs of the DSSApple challenge.
    :param logs_df: The DataFrame collecting the user interactions with the DSSApple challenge.
    :param context_df: The DataFrame that indexes the context vector of each candidate image.
    :param replica: How many times the challenge logs are considered to generate the bandit dataset
    :param dropout: Dropout chance of not considering a given image in the context computation.
    :return: The list of dictionaries representing the bandit dataset
    """
    bandit_dataset = list()
    for i in range(replica):
        for _, row in logs_df.iterrows():
            challenge_data = dict()
            challenge_data['id'] = row['challenge_id']
            challenge_data['target'] = row['target_disease']
            challenge_data['selected'] = row['selected_disease']
            list_selected_imgs = ast.literal_eval(row['positive_feed'])
            challenge_data['context'] = compute_challenge_context(list_selected_imgs, context_df, dropout)
            bandit_dataset.append(challenge_data)

    random.shuffle(bandit_dataset)
    return bandit_dataset

def run_dssapple_bandit(bandit_dataset, verbose=False):

    #warnings.filterwarnings("ignore")
    output_report = {
        'true_labels': list(),
        'intuition_arms': list(),
        'intuition_reward': list(),
        'causal_arms': list(),
        'causal_reward': list(),
        'cmab_arms': list(),
        'cmab_reward': list(),
        'ext_cmab_arms': list(),
        'ext_cmab_reward': list()
    }

    arm_disease_map = {
        'Alternaria': 0,
        'Botrytis': 1,
        'Mucor': 2,
        'Neofabraea': 3,
        'Penicillium': 4,
    }

    num_arms = len(arm_disease_map)
    ctx_size = len(bandit_dataset[0]['context'])

    print('Total number of instances: {}, length of the context: {}'.format(len(bandit_dataset), ctx_size))

    intuition_cum_reward = 0.0
    causal_cum_reward = 0.0
    cmab_cum_reward = 0.0
    ext_cmab_cum_reward = 0.0
    iteration_count = 1

    cmab_model = CMAB(n_arms=num_arms, d=ctx_size)
    ext_cmab_model = CMAB(n_arms=num_arms, d=ctx_size+1)
    causal_model = CausalCMAB(n_arms=num_arms, d=ctx_size, bias=False)

    for log_record in bandit_dataset:

        if iteration_count > MAX_ITERATIONS:
            return output_report

        print('\n### ITERATION {} ###'.format(iteration_count))
        
        true_arm = arm_disease_map[log_record['target']]
        output_report['true_labels'].append(true_arm)

        ### USER SELECTION ###
        arm_intuition = arm_disease_map[log_record['selected']]
        reward = NEGATIVE_REWARD
        if arm_intuition == true_arm:
            reward = POSITIVE_REWARD
        intuition_cum_reward += reward        
        output_report['intuition_arms'].append(arm_intuition)
        output_report['intuition_reward'].append(intuition_cum_reward)

        ### CAUSAL MODEL ###
        exp_reward_list = causal_model.get_expected_reward(log_record['context'], arm_intuition)
        selected_arm = causal_model.select_best_arm(exp_reward_list)
        reward = NEGATIVE_REWARD
        if selected_arm == true_arm:
            reward = POSITIVE_REWARD
        causal_cum_reward += reward
        output_report['causal_arms'].append(selected_arm)
        output_report['causal_reward'].append(causal_cum_reward)
        causal_model.parameters_update(arm_intuition, selected_arm, log_record['context'], reward)
        if verbose:
            print('The arm intuition by the user is {}, the causal cmab selects arm {}, the actual label is {}.'
                .format(arm_intuition, selected_arm, true_arm))

        ### EXTENDED CMAB MODEL ###
        ext_context = np.append(log_record['context'], arm_intuition)
        exp_reward_list = ext_cmab_model.get_expected_reward(ext_context)
        selected_arm = ext_cmab_model.select_best_arm(exp_reward_list)
        reward = NEGATIVE_REWARD
        if selected_arm == true_arm:
            reward = POSITIVE_REWARD
        ext_cmab_cum_reward += reward
        output_report['ext_cmab_arms'].append(selected_arm)
        output_report['ext_cmab_reward'].append(ext_cmab_cum_reward)
        ext_cmab_model.parameters_update(selected_arm, ext_context, reward)
        if verbose:
            print('The extended cmab model selects arm {}, the actual label is {}.'
                  .format(selected_arm, true_arm))

        ### CMAB MODEL ###
        exp_reward_list = cmab_model.get_expected_reward(log_record['context'])
        selected_arm = cmab_model.select_best_arm(exp_reward_list)
        reward = NEGATIVE_REWARD
        if selected_arm == true_arm:
            reward = POSITIVE_REWARD
        cmab_cum_reward += reward
        output_report['cmab_arms'].append(selected_arm)
        output_report['cmab_reward'].append(cmab_cum_reward)
        cmab_model.parameters_update(selected_arm, log_record['context'], reward)
        if verbose:
            print('The cmab model selects arm {}, the actual label is {}.'
                  .format(selected_arm, true_arm))

        iteration_count += 1

    return output_report

### RUN THE SCRIPT ###

for t in range(0, NUM_TRIALS):
    print('\n****************** START OF {} TRIAL ******************\n'.format(t))
    bandit_dataset = create_bandit_dataset(logs_df, context_df, replica=MAX_ITERATIONS//500, dropout=0.2)
    output_report = run_dssapple_bandit(bandit_dataset, verbose=False)
    print(output_report)
    with open('./data/dssapple/results/output_report{}.pickle'.format(t), 'wb') as output_file:
        pickle.dump(output_report, output_file)
import pickle
import ast
import pandas as pd
import numpy as np
import random
from causal_contextual_bandit import CausalCMAB
from causal_contextual_bandit import CMAB
from sklearn.decomposition import PCA


### FUNCTIONS ###

def context_pca_transform(context_df, num_components):
    """
    Transform a binary context with PCA and retain num_components number of principal components.
    :param context_df: The DataFrame representing the context for each image/apple.
    :param num_components: The number of principal components to retain after PCA computation.
    :return: The transformed, dense context processed and reduced via PCA.
    """
    pca_model = PCA(n_components=num_components)
    pca_context = pca_model.fit_transform(context_df.values)
    return pd.DataFrame(pca_context, index=context_df.index)


# def compute_picture_context(list_selected_imgs, context_df, dropout):
#     """
#     Compute the context for a given challenge, as the combination (sum) of the context vector of each selected image.
#     The context is compute with a dropout chance for each selected image.
#     :param list_selected_imgs: The list of selected images in the challenge.
#     :param context_df: The DataFrame that indexes the context vector of each candidate image.
#     :param dropout: Dropout chance of not considering a given image in the context computation.
#     :return: The np.array vector representing the context for the given challenge.
#     """
#     combined_context = np.zeros(context_df.shape[1])
#     for img in list_selected_imgs:
#         if random.random() >= dropout:
#             combined_context += context_df.loc[img].tolist()
#     return combined_context

# def compute_knowledge_context(list_selected_imgs, kb_context_df, dropout):
#     """
#     Compute the context for a given challenge, as the combination (sum) of the context vector of each selected image.
#     The context is compute with a dropout chance for each selected image.
#     :param list_selected_imgs: The list of selected images in the challenge.
#     :param kb_context_df: The DataFrame that indexes the knowledge-based context vector for each candidate apple.
#     :param dropout: Dropout chance of not considering a given image in the context computation.
#     :return: The np.array vector representing the context for the given challenge.
#     """
#     combined_context = np.zeros(kb_context_df.shape[1])
#     for img in list_selected_imgs:
#         if random.random() >= dropout:
#             combined_context += kb_context_df.loc[img].tolist()
#     return combined_context


def compute_challenge_context(list_selected_imgs, img_context_df, kb_context_df, dropout, ctx_type):
    """
    Compute the context for a given challenge, as the combination (sum) of the context vector of each selected image/apple.
    The context could be selected among 3 types, namely, a fully picture-based context, a fully knowledge-based context,
    or an hybrid across these two types. The context is computed with a dropout chance for each selected image.
    :param list_selected_imgs: The list of selected images in the challenge.
    :param img_context_df: The DataFrame that indexes the context vector of each candidate image.
    :param kb_context_df: The DataFrame that indexes the knowledge-based context vector of each candidate apple.
    :param dropout: Dropout chance of not considering a given image/apple in the context computation.
    :param ctx_type: The type of aggregated context to be created:
                    'picture': for a context fully based on selected images context
                    'knowledge': for a context fully based on knowledge-based context for selected apples
                    'hybrid': for a context based as a combination of the two previous types
    :return: The np.array vector representing the context for the given challenge.
    """
    if ctx_type == 'knowledge':
        combined_context = np.zeros(kb_context_df.shape[1])
    elif ctx_type == 'picture':
        combined_context = np.zeros(img_context_df.shape[1])
    elif ctx_type == 'hybrid':
        combined_context = np.zeros(img_context_df.shape[1] + kb_context_df.shape[1])
    for img in list_selected_imgs:
        if random.random() >= dropout:
            img_id_tokens = img.split('_')
            app_id = img_id_tokens[0] + '_' + img_id_tokens[1] + '_' + img_id_tokens[2]
            try:
                kb_ctx = kb_context_df.loc[app_id].tolist()
            except:
                kb_ctx = np.zeros(kb_context_df.shape[1])
            img_ctx = img_context_df.loc[img].tolist()

            if ctx_type == 'knowledge':
                combined_context += kb_ctx
            elif ctx_type == 'picture':
                combined_context += img_ctx
            elif ctx_type == 'hybrid':
                img_ctx.extend(kb_ctx)
                combined_context += img_ctx
    return combined_context


def create_challenge_bandit_dataset(logs_df, img_context_df, kb_context_df, ctx_type, replica=1, dropout=0.0):
    """
    Generate the dataset for the bandit simulation, starting from the logs of the DSSApple challenge.
    :param logs_df: The DataFrame collecting the user interactions with the DSSApple challenge.
    :param img_context_df: The DataFrame that indexes the context vector of each candidate image.
    :param replica: How many times the challenge logs are considered to generate the bandit dataset
    :param dropout: Dropout chance of not considering a given image in the context computation.
    :return: The list of dictionaries representing the bandit dataset.
    """
    bandit_dataset = list()
    for i in range(replica):
        for _, row in logs_df.iterrows():
            challenge_data = dict()
            challenge_data['id'] = row['challenge_id']
            challenge_data['target'] = row['target_disease']
            challenge_data['selected'] = row['selected_disease']
            list_selected_imgs = ast.literal_eval(row['positive_feed'])
            challenge_data['context'] = compute_challenge_context(
                list_selected_imgs, img_context_df, kb_context_df, dropout, ctx_type)
            bandit_dataset.append(challenge_data)

    random.shuffle(bandit_dataset)
    return bandit_dataset


def create_test_knowledge_bandit_dataset(logs_df, kb_context_df, replica=1):
    """
    Generate the knowledge-based dataset for the bandit simulation as the knowledge-based vector of the target/test image.
    :param logs_df: The DataFrame collecting the user interactions with the DSSApple challenge.
    :param kb_context_df: The DataFrame that indexes the knowledge vector of each target/test image.
    :param replica: How many times the challenge logs are considered to generate the bandit dataset
    :return: The list of dictionaries representing the bandit dataset.
    """
    bandit_dataset = list()
    for i in range(replica):
        for _, row in logs_df.iterrows():
            challenge_data = dict()
            challenge_data['id'] = row['challenge_id']
            challenge_data['target'] = row['target_disease']
            challenge_data['selected'] = row['selected_disease']
            challenge_data['context'] = np.array(kb_context_df.loc[IMG_MAP[row['target_image']]].values)
            bandit_dataset.append(challenge_data)

    random.shuffle(bandit_dataset)
    return bandit_dataset


def create_hybrid_bandit_dataset(logs_df, img_context_df, kb_context_df, replica=1, dropout=0.0):
    """
    Generate the hybrid dataset for the bandit simulation - i.e. a combination of the image context and the test/target
    knowledge context - starting from the logs of the DSSApple challenge.
    :param logs_df: The DataFrame collecting the user interactions with the DSSApple challenge.
    :param img_context_df: The DataFrame that indexes the context vector of each candidate image.
    :param kb_context_df: The DataFrame that indexes the knowledge vector of each target/test image.
    :param replica: How many times the challenge logs are considered to generate the bandit dataset
    :param dropout: Dropout chance of not considering a given image in the context computation.
    :return: The list of dictionaries representing the bandit dataset.
    """
    bandit_dataset = list()
    for i in range(replica):
        for _, row in logs_df.iterrows():
            challenge_data = dict()
            challenge_data['id'] = row['challenge_id']
            challenge_data['target'] = row['target_disease']
            challenge_data['selected'] = row['selected_disease']
            list_selected_imgs = ast.literal_eval(row['positive_feed'])
            picture_ctx = compute_challenge_context(list_selected_imgs, img_context_df, None, dropout, 'picture')
            kb_ctx = np.array(kb_context_df.loc[IMG_MAP[row['target_image']]].values)
            hybrid_ctx = np.concatenate((picture_ctx, kb_ctx))
            challenge_data['context'] = hybrid_ctx
            bandit_dataset.append(challenge_data)

    random.shuffle(bandit_dataset)
    return bandit_dataset


def run_dssapple_bandit(bandit_dataset, verbose=False):
    # warnings.filterwarnings("ignore")
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
    ext_cmab_model = CMAB(n_arms=num_arms, d=ctx_size + num_arms)
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
        arm_context = np.zeros(num_arms)
        arm_context[arm_intuition] = 1
        ext_context = np.concatenate((log_record['context'], arm_context))
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



### PARAMETERS ###

MAX_ITERATIONS = 3000
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = 0.0
NUM_TRIALS = 100
IMG_MAP = {'a1': 'img1', 'a2': 'img2', 'a3': 'img3',
           'bt1': 'img4', 'bt2': 'img5', 'bt3': 'img6',
           'mr1': 'img7', 'mr2': 'img8', 'mr3': 'img9',
           'pc1': 'img10', 'pc2': 'img11', 'pc3': 'img12',
           'n1': 'img13', 'n2': 'img14', 'n3': 'img15'}

### DATA IMPORT ###

logs_df = pd.read_csv('./data/dssapple/ChallengeData_08-04-19_filtered.csv', sep=';')

with open('./data/dssapple/img_list_150', 'rb') as list_f:
    img_list = pickle.load(list_f)
with open('./data/dssapple/context_sim_pca_matrix_150', 'rb') as context_f:
    img_context_matrix = pickle.load(context_f)
img_context_df = pd.DataFrame(img_context_matrix, index=img_list)

with open('./data/dssapple/app_evidences_dataset', 'rb') as app_f:
    kb_app_df = pickle.load(app_f)
with open('./data/dssapple/frudistor-test_evidences_dataset', 'rb') as test_f:
    kb_test_df = pickle.load(test_f)

### RUN THE SCRIPT ###

for t in range(0, NUM_TRIALS):
    print('\n****************** START OF {} TRIAL ******************\n'.format(t))
    bandit_dataset = create_challenge_bandit_dataset(logs_df, img_context_df, context_pca_transform(kb_app_df, 8), 'picture',
                                                     replica=MAX_ITERATIONS // 500, dropout=0.2)
    output_report = run_dssapple_bandit(bandit_dataset, verbose=False)
    print(output_report)
    with open('./data/dssapple/results/output_report{}.pickle'.format(t), 'wb') as output_file:
        pickle.dump(output_report, output_file)

# bandit_dataset = create_challenge_bandit_dataset(logs_df, img_context_df, kb_app_df, 'picture', replica=MAX_ITERATIONS//500, dropout=0.2)
# output_report = run_dssapple_bandit(bandit_dataset, verbose=False)
# print(output_report)

# bandit_dataset = create_test_knowledge_bandit_dataset(logs_df, context_pca_transform(kb_test_df, 8), replica=MAX_ITERATIONS//500)
# output_report = run_dssapple_bandit(bandit_dataset, verbose=False)
# print(output_report)

# bandit_dataset = create_hybrid_bandit_dataset(logs_df, img_context_df, context_pca_transform(kb_test_df, 8), replica=MAX_ITERATIONS//500, dropout=0.2)
# output_report = run_dssapple_bandit(bandit_dataset, verbose=False)
# print(output_report)
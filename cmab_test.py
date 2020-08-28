import numpy as np
import mnist_reader
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from causal_contextual_bandit import CausalCMAB
from causal_contextual_bandit import CMAB
import warnings

######### PARAMETERS ##########

NUM_NEIGHS = 5
CTX_SIZE = 100
NUM_ARMS = 10
PRECOMPUTED_LABELS = 1000
MAX_ITERATIONS = 10000
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = 0.0

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

######### FUNCTIONS ##########

def init_classification(X_train, y_train):
    idx = np.random.permutation(len(y_train))
    X_train, y_train = X_train[idx], y_train[idx]

    pca_model = PCA(CTX_SIZE)
    X_train_pca = pca_model.fit_transform(X_train)
    print('Total explained variance by PCA = {}'.format(
        sum(pca_model.explained_variance_ratio_)))

    KNN = KNeighborsClassifier(NUM_NEIGHS)
    KNN.fit(X_train_pca[:PRECOMPUTED_LABELS], y_train[:PRECOMPUTED_LABELS])
    accu = accuracy_score(y_train[:PRECOMPUTED_LABELS], KNN.predict(X_train_pca[:PRECOMPUTED_LABELS]))
    print('Pre-computed model accuracy on first {} instances = {}\n'.format(PRECOMPUTED_LABELS, accu))

    return X_train_pca[PRECOMPUTED_LABELS:], y_train[PRECOMPUTED_LABELS:], KNN


def run_bandit_simulation(X_train, y_train, verbose=False):

    #warnings.filterwarnings("ignore")
    output_report = {
        'true_labels': list(),
        'classifier_arms': list(),
        'classifier_reward': list(),
        'causal_arms': list(),
        'causal_reward': list(),
        'cmab_arms': list(),
        'cmab_reward': list(),
    }

    data, labels, classifier = init_classification(X_train, y_train)
    print('Total number of instances and length of the context = {}'.format(data.shape))

    classifier_cum_reward = 0.0
    causal_cum_reward = 0.0
    cmab_cum_reward = 0.0
    iteration_count = 1

    cmab_model = CMAB(n_arms=NUM_ARMS, d=CTX_SIZE)
    causal_model = CausalCMAB(n_arms=NUM_ARMS, d=CTX_SIZE)

    for feat_vec, label in zip(data, labels):

        if iteration_count > MAX_ITERATIONS:
            return output_report

        print('\n### ITERATION {} ###'.format(iteration_count))

        arm_intuition = classifier.predict(feat_vec.reshape(1, -1))[0]
        reward = NEGATIVE_REWARD
        if arm_intuition == label:
            reward = POSITIVE_REWARD
        classifier_cum_reward += reward
        output_report['true_labels'].append(label)
        output_report['classifier_arms'].append(arm_intuition)
        output_report['classifier_reward'].append(classifier_cum_reward)

        exp_reward_list = causal_model.get_expected_reward(feat_vec, arm_intuition)
        selected_arm = causal_model.select_best_arm(exp_reward_list)
        reward = NEGATIVE_REWARD
        if selected_arm == label:
            reward = POSITIVE_REWARD
        causal_cum_reward += reward
        output_report['causal_arms'].append(selected_arm)
        output_report['causal_reward'].append(causal_cum_reward)
        causal_model.parameters_update(arm_intuition, selected_arm, feat_vec, reward)
        if verbose:
            print('The arm intuition by classifier is {}, the causal cmab selects arm {}, the actual label is {}.'
                .format(arm_intuition, selected_arm, label))

        exp_reward_list = cmab_model.get_expected_reward(feat_vec)
        selected_arm = cmab_model.select_best_arm(exp_reward_list)
        reward = NEGATIVE_REWARD
        if selected_arm == label:
            reward = POSITIVE_REWARD
        cmab_cum_reward += reward
        output_report['cmab_arms'].append(selected_arm)
        output_report['cmab_reward'].append(cmab_cum_reward)
        cmab_model.parameters_update(selected_arm, feat_vec, reward)
        if verbose:
            print('The cmab model selects arm {}, the actual label is {}.'
                  .format(selected_arm, label))

        iteration_count += 1


### RUN THE SCRIPT ###
output_report = run_bandit_simulation(X_train, y_train, verbose=False)
with open('./data/fashion/results/output_report.pickle', 'wb') as output_file:
    pickle.dump(output_report, output_file)


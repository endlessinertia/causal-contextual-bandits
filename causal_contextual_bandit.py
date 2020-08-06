import numpy as np

class CausalCMAB:

    '''
    ContextualBandit(d, context)

    Contextual bandit selection algorithm based on both Thompson Sampling and Upper Confidence Bound policies.

    Parameters
    --------
    d : int
        Length of the context, i.e. dimension of the context vector and mu parameter.
    context : 2D np.array
        Matrix of the contexts. Each row represent an arm to be selected based on its features.

    '''

    def __init__(self, d, context):
        self.mu = np.zeros(d)
        self.B = np.eye(d)
        self.f = np.zeros(d)
        self.context = context
        self.selected_arm = None

        # self.selected_arm_history = list()
        # self.reward_history = list()

    def TS_get_expected_reward(self):
        '''

        Compute the expected reward for each context vector following Thompson Sampling policy.
        Generate one sample mu_tilde from the Gaussian multivariate with parameters (self.mu, np.linalg.inv(self.B)).
        For each row (arm to be selected) compute the expected reward as a linear function of its context c_vec and mu_tilde

        Returns
        --------
        out : 1D np.array
            Array which contains the expected reward for each arm
        '''

        mu_tilde = np.random.multivariate_normal(self.mu, np.linalg.inv(self.B), size=1)[0]
        exp_reward_list = list()
        for c_vec in self.context:
            exp_reward = np.dot(mu_tilde[np.newaxis], c_vec[np.newaxis].T)
            exp_reward_list.extend(exp_reward[0])
        return np.array(exp_reward_list)

    def UCB_get_expected_reward(self, alpha=0.5):
        '''

        Compute the expected reward for each context vector following the Upper Confidence Bound policy.
        The expected reward is computed as the linear combination of the mu vector and the context vector c_vec plus
        the weighted standard deviation of mu distribution.

        Parameters
        --------
        alpha : float [0.0, 1.0] (default: 0.5)
            Trade-off parameter for exploration/exploitation.
            It gives the weighted importance of the variance of the model.

        Returns
        --------
        out : 1D np.array
            Array which contains the expected reward for each arm.
        '''

        exp_reward_list = list()
        for c_vec in self.context:
            ucb = np.sqrt(np.dot(np.dot(c_vec[np.newaxis], np.linalg.inv(self.B)), c_vec[np.newaxis].T))
            exp_reward = np.dot(self.mu, c_vec[np.newaxis].T) + alpha * ucb
            exp_reward_list.extend(exp_reward[0])
        return np.array(exp_reward_list)

    def select_best_arm(self, exp_reward_list):
        '''

        Select the argmax (best arm) among each computed expected reward.

        Parameters
        --------
        exp_reward_list : 1D np.array
            Array which contains the expected reward for each arm.

        Returns
        --------
        out : int
            Index of the max expected reward - i.e. the index of the selected arm.
        '''

        selected_arm = np.argmax(exp_reward_list, axis=0)
        self.selected_arm = selected_arm
        return selected_arm

    def parameters_update(self, reward):
        '''

        Incrementally updates the class parameters for the gaussian multivariate distribution
        of the contextual bandits (self.B, self.f and self.mu), given the last selected arm (self.selected_arm)
        and the actual reward received for that arm.

        Parameters
        --------
        reward : int or float
            The numeric reward received by selecting self.selected_arm. Usually is 0 or 1, but can be any float in R.

        '''
        temp_b = self.B
        temp_b += np.dot(self.context[self.selected_arm][np.newaxis].T, self.context[self.selected_arm][np.newaxis])
        self.B = temp_b

        temp_f = self.f
        temp_f += reward * self.context[self.selected_arm]
        self.f = temp_f
        new_mu = np.dot(np.linalg.inv(self.B), self.f.T)
        self.mu = new_mu

    # def parameters_update(self, reward):
    #     self.add_reward(reward)
    #     temp_sum_b = 0.0
    #     for arm in self.selected_arm_history:
    #         temp_sum_b += np.dot(self.context[arm][np.newaxis].T, self.context[arm][np.newaxis])
    #     B = np.eye(len(self.context[0])) + temp_sum_b
    #     self.B = B
    #
    #     temp_sum_mu = 0.0
    #     for arm, rew in zip(self.selected_arm_history, self.reward_history):
    #         temp_sum_mu += rew * self.context[arm]
    #     new_mu = np.dot(np.linalg.inv(B), temp_sum_mu.T)
    #     self.mu = new_mu



class CMAB_TS:

    '''
    CMAB_TS(n_arms, d)

    Contextual bandit for class prediction based on Thompson Sampling. Each arm is mapped to a class,
    dimension of the context d and number of arms n_arms could be different.

    Parameters
    --------
    n_arms : int
        Number of arms (classes) in the contextual bandit problem.
    d : int
        Length of the context, i.e. dimension of the context vector and mu parameter.

    '''

    def __init__(self, n_arms, d):
        self.arms_list = range(n_arms)
        self.mu_params_dict = dict()
        self.B_params_dict = dict()
        self.f_params_dict = dict()

        for arm in self.arm_list:
            self.mu_params_dict[arm] = np.zeros(d)
            self.B_params_dict[arm] = np.eye(d)
            self.f_params_dict[arm] = np.zeros(d)


    def get_expected_reward(self, context):
        '''

        Compute the expected reward for each arm following Thompson Sampling policy.
        Generate one sample mu_tilde for each arm from the Gaussian multivariate with parameters (mu, B^-1).
        Compute the expected reward as a linear function of the input context and mu_tilde.

        Parameters
        --------
        context : 1D np.array
            Array of the context (of dimension d) received as input for this round.

        Returns
        --------
        out : 1D np.array
            Array which contains the expected reward for each arm
        '''

        exp_reward_list = list()
        for arm in self.arms_list:
            mu_tilde = np.random.multivariate_normal(self.mu_params_dict[arm],
                                                     np.linalg.inv(self.B_params_dict[arm]),
                                                     size=1)[0]
            exp_reward = np.dot(mu_tilde, context)
            exp_reward_list.extend(exp_reward)
        return np.array(exp_reward_list)


    @staticmethod
    def select_best_arm(exp_reward_list):
        '''

        Select the argmax (best arm) among each computed expected reward.

        Parameters
        --------
        exp_reward_list : 1D np.array
            Array which contains the expected reward for each arm.

        Returns
        --------
        out : int
            Index of the max expected reward - i.e. the index of the selected arm.
        '''

        selected_arm = np.argmax(exp_reward_list, axis=0)
        return selected_arm


    def parameters_update(self, selected_arm, context, reward):
        '''

        Incrementally updates the class parameters for the gaussian multivariate distribution
        of the contextual bandits (self.B, self.f and self.mu), given the last selected arm (self.selected_arm)
        and the actual reward received for that arm.

        Parameters
        --------
        selected_arm: int
            The index of the arm selected for this round.
        context: 1D np.array
            Array of the context (of dimension d) received as input for this round.
        reward : int or float
            The numeric reward received by selecting selected_arm. Usually is 0 or 1, but can be any float in R.

        '''
        temp_b = self.B_params_dict[selected_arm]
        temp_b += np.dot(context.T, context)
        self.B_params_dict[selected_arm] = temp_b

        temp_f = self.f_params_dict[selected_arm]
        temp_f += reward * context
        self.f_params_dict[selected_arm] = temp_f
        new_mu = np.dot(np.linalg.inv(self.B_params_dict[selected_arm]), self.f_params_dict[selected_arm].T)
        self.mu_params_dict[selected_arm] = new_mu


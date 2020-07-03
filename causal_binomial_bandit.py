import numpy as np


class CausalTS:
    """
    CausalTS(n_arms)

    Causal (counterfactual) Thompson Sampling with n_arms arms based on Beta/Binomial distribution.

    Parameters
    --------
    n_arms : int
        Number of arms to be defined for the problem.

    """

    def __init__(self, n_arms):
        # 2D array (n_arms x n_arms) of ones for alpha params of the Beta distribution (arm_intuition x arm_counterfactual)
        self.alpha_params = np.ones((n_arms, n_arms))
        # 2D array (n_arms x n_arms) of ones for beta params of the Beta distribution (arm_intuition x arm_counterfactual)
        self.beta_params = np.ones((n_arms, n_arms))
        self.num_arms = n_arms


    @staticmethod
    def arm_selection(exp_reward_array):
        """
        Select the best arm (i.e. argmax) on a given array of expected rewards.
        :param exp_reward_array: the array of expected rewards for this round
        :return: the index of the max expected reward (i.e. the best arm)
        """
        #print('sampled exp reward = {}'.format(exp_reward_array))
        return np.argmax(exp_reward_array)


    def ts_rewards_sample(self, arm_intuition, bias_array):
        """
        Sample the reward of each arm from the current estimated Beta distribution using Thompson Sampling.
        The parameters of Beta is extracted from the row of arm_intuition and the sampled value is multiplied
        element-wise for the bias_array (of dimension num_arms)
        :param arm_intuition: the arm selected by the agent based on its intuition (before counterfactual analysis)
        :param bias_array: the array with the counterfactual bias for each arm under the arm_intuition condition
        :return: the array of estimated rewards for this round
        """
        #exp_rewards_list = [np.random.beta(a, b) for a, b in zip(np.diag(self.alpha_params), np.diag(self.beta_params))]
        exp_rewards_list = [np.random.beta(a, b) for a, b
                            in zip(self.alpha_params[arm_intuition], self.beta_params[arm_intuition])]
        return np.multiply(np.array(exp_rewards_list), bias_array)



    def expected_reward(self, f_arm, cf_arm):
        """
        The expected (mean) value for the Beta distribution with factual arm (intuition) f_arm and
        counterfactual arm cf_arm
        :param f_arm: the index of the factual arm
        :param cf_arm: the index of the counterfactual arm
        :return: the expected value of the Beta distribution
        """
        return float(self.alpha_params[f_arm, cf_arm]) / \
               float(self.alpha_params[f_arm, cf_arm] + self.beta_params[f_arm, cf_arm])


    def counterfactual_bias_array(self, arm_intuition):
        """
        Compute the bias array for all the candidate arms under the condition that the agent intended to
        select the arm_intuition
        :param arm_intuition: the index of the arm selected by the agent
        :return: the array of length num_arms with the bias for each candidate arms
        """
        bias_array = np.ones(self.num_arms)
        q2 = self.expected_reward(arm_intuition, arm_intuition)
        #print('\n Q2 = {}'.format(q2))

        for arm in range(0, self.num_arms):
            if arm == arm_intuition:
                pass
            else:
                q1 = self.expected_reward(arm_intuition, arm)
                #print('Q1 for arm {} = {}'.format(arm, q1))
                bias = 1.0 - abs(q1 - q2)
                if q1 > q2:
                    if bias < bias_array[arm_intuition]:
                        bias_array[arm_intuition] = bias
                else:
                    bias_array[arm] = bias

        return bias_array


    def update_arm_parameters(self, arm_intuition, arm_selection, success):
        """
        Update the parameters of the Beta distribution given the collected reward on arm_selection had
        the agent intended to select arm_intuition
        :param arm_intuition: the index of the arm the agent intended to pull
        :param arm_selection: the index of the arm selected to be pulled for this round
        :param success: the reward for the current round (0: failure, 1: success)
        """
        if success:
            self.alpha_params[arm_intuition, arm_selection] += 1
        else:
            self.beta_params[arm_intuition, arm_selection] += 1



class BinomialTS:
    """
    BinomialTS(n_arms)

    Standard implementation of Thompson Sampling with n_arms arms based on Beta/Binomial distribution.

    Parameters
    --------
    n_arms : int
        Number of arms to be defined for the problem.

    """

    def __init__(self, n_arms):
        self.alpha_params = np.ones(n_arms) #array of n_arms ones for alpha params of the Beta distribution
        self.beta_params = np.ones(n_arms) #array of n_arms ones for beta params of the Beta distribution
        self.num_arms = n_arms


    @staticmethod
    def arm_selection(exp_reward_array):
        """
        Select the best arm (i.e. argmax) on a given array of expected rewards.
        :param exp_reward_array: the array of expected rewards for this round
        :return: the index of the max expected reward (i.e. the best arm)
        """
        return np.argmax(exp_reward_array)


    def ts_rewards_sample(self):
        """
        Sample the reward of each arm from the current estimated Beta distribution using Thompson Sampling
        :return: the array of estimated rewards for this round
        """
        exp_rewards_list = [np.random.beta(a, b) for a, b in zip(self.alpha_params, self.beta_params)]
        return np.array(exp_rewards_list)


    def update_arm_parameters(self, arm, success):
        """
        Update the parameters of the Beta distribution given the collected reward on the selected arm
        :param arm: the index of the arm selected to be pulled for this round
        :param success: the reward for the current round (0: failure, 1: success)
        """
        if success:
            self.alpha_params[arm] += 1
        else:
            self.beta_params[arm] += 1

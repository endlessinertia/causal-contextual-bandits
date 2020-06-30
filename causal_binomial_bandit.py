import numpy as np


class CausalBinomialBandit:


    def __init__(self, n_arms):
        self.alpha_list = np.ones(n_arms)
        self.beta_list = np.ones(n_arms)
        self.num_arms = n_arms
        self.cum_regret = 0


    @staticmethod
    def arm_selection(exp_reward_array):
        return np.argmax(exp_reward_array)


    def expected_rewards_array(self, bias_list=None):
        exp_rewards_list = [np.random.beta(a, b) for a, b in zip(self.alpha_list, self.beta_list)]
        if bias_list is None:
            return np.array(exp_rewards_list)
        else:
            return np.multiply(np.array(exp_rewards_list), bias_list)


    def arm_expected_reward(self, arm):
        return np.random.beta(self.alpha_list[arm], self.beta_list[arm])

    def counterfactual_bias_array(self, selected_arm):
        bias_list = np.ones(self.num_arms)
        for arm in range(0, self.num_arms):
            if arm == selected_arm:
                pass
            else:
                q1 = self.arm_expected_reward(arm)
                q2 = self.arm_expected_reward(selected_arm)
                bias = 1.0 - abs(q1 - q2)
                if q1 > q2:
                    if bias < bias_list[selected_arm]:
                        bias_list[selected_arm] = bias
                else:
                    bias_list[arm] = bias
        return bias_list


    def update_arm_parameters(self, arm, success):
        if success:
            self.alpha_list[arm] += 1
        else:
            self.cum_regret += 1
            self.beta_list[arm] += 1





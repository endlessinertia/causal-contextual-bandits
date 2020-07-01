import numpy as np


class CausalTS:


    def __init__(self, n_arms):
        self.alpha_params = np.ones((n_arms, n_arms))
        self.beta_params = np.ones((n_arms, n_arms))
        self.num_arms = n_arms
        self.cum_regret = 0


    @staticmethod
    def arm_selection(exp_reward_array):
        return np.argmax(exp_reward_array)


    def expected_rewards_array(self, bias_list=None):
        exp_rewards_list = [np.random.beta(a, b) for a, b in zip(np.diag(self.alpha_params), np.diag(self.beta_params))]
        if bias_list is None:
            return np.array(exp_rewards_list)
        else:
            return np.multiply(np.array(exp_rewards_list), bias_list)


    def arm_expected_reward(self, f_arm, cf_arm):
        return np.random.beta(self.alpha_params[f_arm, cf_arm], self.beta_params[f_arm, cf_arm])


    def counterfactual_bias_array(self, arm_intuition):
        bias_list = np.ones(self.num_arms)
        q2 = self.arm_expected_reward(arm_intuition, arm_intuition)

        for arm in range(0, self.num_arms):
            if arm == arm_intuition:
                pass
            else:
                q1 = self.arm_expected_reward(arm_intuition, arm)
                bias = 1.0 - abs(q1 - q2)
                if q1 > q2:
                    if bias < bias_list[arm_intuition]:
                        bias_list[arm_intuition] = bias
                else:
                    bias_list[arm] = bias
        return bias_list


    def update_arm_parameters(self, arm_intuition, arm_selection, success):
        if success:
            self.alpha_params[arm_intuition, arm_selection] += 1
        else:
            self.cum_regret += 1
            self.beta_params[arm_intuition, arm_selection] += 1



class BinomialTS:

    def __init__(self, n_arms):
        self.alpha_params = np.ones(n_arms)
        self.beta_params = np.ones(n_arms)
        self.num_arms = n_arms
        self.cum_regret = 0


    @staticmethod
    def arm_selection(exp_reward_array):
        return np.argmax(exp_reward_array)


    def expected_rewards_array(self):
        exp_rewards_list = [np.random.beta(a, b) for a, b in zip(self.alpha_params, self.beta_params)]
        return np.array(exp_rewards_list)


    def arm_expected_reward(self, arm):
        return np.random.beta(self.alpha_params[arm], self.beta_params[arm])


    def update_arm_parameters(self, arm, success):
        if success:
            self.alpha_params[arm] += 1
        else:
            self.cum_regret += 1
            self.beta_params[arm] += 1

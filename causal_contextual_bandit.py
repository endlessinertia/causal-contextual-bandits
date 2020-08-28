import numpy as np
import itertools

class CausalCMAB:

    """
    CausalCMAB(n_arms, d)

    Causal contextual bandit for class prediction based on Thompson Sampling. Each arm is mapped to a class,
    dimension of the context d and number of arms n_arms could be different. The number of parameters for the Gaussian
    is (n_arms x n_arms), given that the model updates the parameters for each pair (arm_intuition, arm_selection).

    Parameters
    --------
    n_arms : int
        Number of arms (classes) in the contextual bandit problem.
    d : int
        Length of the context, i.e. dimension of the context vector and mu parameter.

    """

    def __init__(self, n_arms, d):

        self.arms_list = range(n_arms)

        self.mu_params_dict = dict()
        self.B_params_dict = dict()
        self.f_params_dict = dict()

        for arm_tuple in itertools.product(self.arms_list, self.arms_list):
            self.mu_params_dict[arm_tuple] = np.zeros(d)
            self.B_params_dict[arm_tuple] = np.eye(d)
            self.f_params_dict[arm_tuple] = np.zeros(d)


    def get_expected_reward(self, context, arm_intuition):
        """
        Compute the expected reward for each arm following Thompson Sampling policy.
        Generate one sample mu_tilde for each arm from the Gaussian multivariate with parameters (mu, B^-1).
        Compute the expected reward as a linear function of the input context and mu_tilde.

        Parameters
        --------
        context : 1D np.array
            Array of the context (of dimension d) received as input for this round.
        arm_intuition : int
            The index of the arm intuitively selected by the user (i.e. a more trivial decision maker) for this round.

        Returns
        --------
        out : 1D np.array
            Array which contains the expected reward for each arm, under the intuition of arm_intuition
        """

        exp_reward_list = list()
        for arm in self.arms_list:
            mu_tilde = np.random.multivariate_normal(self.mu_params_dict[(arm_intuition, arm)],
                                                     np.linalg.inv(self.B_params_dict[(arm_intuition, arm)]),
                                                     size=1)[0]
            exp_reward = np.dot(mu_tilde, context)
            exp_reward_list.append(exp_reward)
        return np.array(exp_reward_list)


    @staticmethod
    def select_best_arm(exp_reward_list):
        """
        Select the argmax (best arm) among each computed expected reward.

        Parameters
        --------
        exp_reward_list : 1D np.array
            Array which contains the expected reward for each arm.

        Returns
        --------
        out : int
            Index of the max expected reward - i.e. the index of the selected arm.
        """
        return np.argmax(exp_reward_list, axis=0)


    def parameters_update(self, arm_intuition, arm_selection, context, reward):
        """
        Incrementally updates the parameters for the gaussian multivariate distribution
        of the contextual bandits (self.B, self.f and self.mu), given the arm_intuition and arm_selection
        and the actual reward received for this arm pair.

        Parameters
        --------
        arm_intuition : int
            The index of the arm intuitively selected by the user (i.e. a more trivial decision maker) for this round.
        arm_selection : int
            The index of the arm finally selected by the system for this round.
        context: 1D np.array
            Array of the context (of dimension d) received as input for this round.
        reward : int or float
            The numeric reward received by selecting selected_arm under the intuition of selecting arm_intuition.
            Usually is 0 or 1, but can be any float in R.

        """
        temp_b = self.B_params_dict[(arm_intuition, arm_selection)]
        temp_b += np.dot(context.T, context)
        self.B_params_dict[(arm_intuition, arm_selection)] = temp_b

        temp_f = self.f_params_dict[(arm_intuition, arm_selection)]
        temp_f += reward * context
        self.f_params_dict[(arm_intuition, arm_selection)] = temp_f
        new_mu = np.dot(np.linalg.inv(self.B_params_dict[(arm_intuition, arm_selection)]), self.f_params_dict[(arm_intuition, arm_selection)].T)
        self.mu_params_dict[(arm_intuition, arm_selection)] = new_mu



class CMAB:

    """
    CMAB(n_arms, d)

    Contextual bandit for class prediction based on Thompson Sampling. Each arm is mapped to a class,
    dimension of the context d and number of arms n_arms could be different.

    Parameters
    --------
    n_arms : int
        Number of arms (classes) in the contextual bandit problem.
    d : int
        Length of the context, i.e. dimension of the context vector and mu parameter.

    """

    def __init__(self, n_arms, d):
        self.arms_list = range(n_arms)
        self.mu_params_dict = dict()
        self.B_params_dict = dict()
        self.f_params_dict = dict()

        for arm in self.arms_list:
            self.mu_params_dict[arm] = np.zeros(d)
            self.B_params_dict[arm] = np.eye(d)
            self.f_params_dict[arm] = np.zeros(d)


    def get_expected_reward(self, context):
        """
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
        """

        exp_reward_list = list()
        for arm in self.arms_list:
            mu_tilde = np.random.multivariate_normal(self.mu_params_dict[arm],
                                                     np.linalg.inv(self.B_params_dict[arm]),
                                                     size=1)[0]
            exp_reward = np.dot(mu_tilde, context)
            exp_reward_list.append(exp_reward)
        return np.array(exp_reward_list)


    @staticmethod
    def select_best_arm(exp_reward_list):
        """
        Select the argmax (best arm) among each computed expected reward.

        Parameters
        --------
        exp_reward_list : 1D np.array
            Array which contains the expected reward for each arm.

        Returns
        --------
        out : int
            Index of the max expected reward - i.e. the index of the selected arm.
        """

        return np.argmax(exp_reward_list, axis=0)


    def parameters_update(self, selected_arm, context, reward):
        """
        Incrementally updates the parameters for the gaussian multivariate distribution
        of the contextual bandits (self.B, self.f and self.mu), given the selected_arm
        and the actual reward received for that arm.

        Parameters
        --------
        selected_arm: int
            The index of the arm selected for this round.
        context: 1D np.array
            Array of the context (of dimension d) received as input for this round.
        reward : int or float
            The numeric reward received by selecting selected_arm. Usually is 0 or 1, but can be any float in R.
        """
        temp_b = self.B_params_dict[selected_arm]
        temp_b += np.dot(context.T, context)
        self.B_params_dict[selected_arm] = temp_b

        temp_f = self.f_params_dict[selected_arm]
        temp_f += reward * context
        self.f_params_dict[selected_arm] = temp_f
        new_mu = np.dot(np.linalg.inv(self.B_params_dict[selected_arm]), self.f_params_dict[selected_arm].T)
        self.mu_params_dict[selected_arm] = new_mu


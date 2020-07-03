from causal_binomial_bandit import *


def sample_context(p_d=0.5, p_b=0.5):
    """
    Sample a context for greedy casino - i.e. weather the player is drunk and weather the slot is blinking
    :param p_d: probability of the player being drunk
    :param p_b: probability of the slot to blink
    :return: the vector of the sampled context (drunk, blink)
    """
    return np.random.binomial(1, [p_d, p_b])


def payout_choice(d, b):
    """
    Generate the payout vector for the two arms based on the contextual conditions (drunk, blink)
    :param d: player is drunk (0: false, 1: true)
    :param b: slot is blinking (0: false, 1: true)
    :return: the payout vector of the two arms (indexed from 0)
    """
    if d == b:
        if d == 0:
            return [0.1, 0.5]
        else:
            return [0.2, 0.4]
    else:
        if d == 0:
            return [0.5, 0.1]
        else:
            return [0.4, 0.2]


def arm_intuition(d, b):
    """
    Player intuition on the arm to pull, based on XOR logical function
    :param d: player is drunk (0: false, 1: true)
    :param b: slot is blinking (0: false, 1: true)
    :return: the index of the arm the player wants to pull (starting from 0)
    """
    if d == b:
        return 0
    else:
        return 1


def sample_reward(payout_vector, arm):
    """
    Sample the reward of the chosen arm based on a Bernoulli with parameter selected
    from the sampled payout vector
    :param payout_vector: the array with the payout for each arm
    :param arm: the index of the chosen arm to be pulled
    :return: the sampled reward (1: win, 0: lose)
    """
    return np.random.binomial(1, payout_vector[arm])


def run_greedy_casino(n_trials = 1000, algo='obs', verbose=False):
    """
    Run a simulation for the greedy casino
    :param n_trials: number of trials - i.e. arm selections
    :param algo: algorithm used for the arm selection
    (obs: observational, ts: binomial Thompson sampling, c_ts: causal Thompson sampling)
    :param verbose: if True, report information of each trial
    """

    win_rate = 0.0
    n_arms = 2

    if algo == 'ts':
        BTS = BinomialTS(n_arms)
    elif algo == 'c_ts':
        CTS = CausalTS(n_arms)

    for i in range(n_trials):

        d, b = sample_context()
        player_choice = arm_intuition(d, b)
        payout_vector = payout_choice(d, b)

        if algo == 'obs':
            arm_choice = player_choice
            rew = sample_reward(payout_vector, arm_choice)
        if algo == 'ts':
            exp_reward_array = BTS.ts_rewards_sample()
            arm_choice = BTS.arm_selection(exp_reward_array)
            rew = sample_reward(payout_vector, arm_choice)
            BTS.update_arm_parameters(arm_choice, rew)
        if algo == 'c_ts':
            cf_bias_array = CTS.counterfactual_bias_array(player_choice)
            exp_reward_array = CTS.ts_rewards_sample(player_choice, cf_bias_array)
            arm_choice = CTS.arm_selection(exp_reward_array)
            rew = sample_reward(payout_vector, arm_choice)
            CTS.update_arm_parameters(player_choice, arm_choice, rew)

        win_rate += float(rew)

        if verbose:
            print('\n###### TRIAL {}/{} ######'.format(i + 1, n_trials))
            print('the player is drunk = {}, the slot is blinking = {}'.format(d, b))
            print('the player choose arm = {}'.format(player_choice))
            print('the system select arm = {}'.format(arm_choice))
            print('the sampled payout is {}'.format(payout_vector))
            print('the reward is = {}'.format(rew))
            if algo == 'ts':
                print('alpha parameters update = {}'.format(BTS.alpha_params))
                print('beta parameters update = {}'.format(BTS.beta_params))
            elif algo == 'c_ts':
                print('alpha parameters update =\n {}'.format(CTS.alpha_params))
                print('beta parameters update =\n {}'.format(CTS.beta_params))

    win_rate = win_rate / float(n_trials)
    print('\n\nEmpirical win rate for {} = {}'.format(algo, win_rate))



### RUN THE SCRIPT HERE ###
run_greedy_casino(n_trials=10000, algo='c_ts', verbose=True)



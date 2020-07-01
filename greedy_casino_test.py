from causal_binomial_bandit import *


def sample_context(p_d=0.5, p_b=0.5):
    return np.random.binomial(1, [p_d, p_b])


def payout_choice(d, b):
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
    if d == b:
        return 0
    else:
        return 1


def sample_reward(payout_vector, arm):
    return np.random.binomial(1, payout_vector[arm])


def run_greedy_casino(n_trials = 1000, algo='obs', verbose=False):

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
            exp_reward_array = BTS.expected_rewards_array()
            arm_choice = BTS.arm_selection(exp_reward_array)
            rew = sample_reward(payout_vector, arm_choice)
            BTS.update_arm_parameters(arm_choice, rew)
        if algo == 'c_ts':
            cf_bias_array = CTS.counterfactual_bias_array(player_choice)
            exp_reward_array = CTS.expected_rewards_array(cf_bias_array)
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
                print('cumulative regret = {}'.format(BTS.cum_regret))
                print('alpha parameters update = {}'.format(BTS.alpha_params))
                print('beta parameters update = {}'.format(BTS.beta_params))
            elif algo == 'c_ts':
                print('cumulative regret = {}'.format(CTS.cum_regret))
                print('alpha parameters update =\n {}'.format(CTS.alpha_params))
                print('beta parameters update =\n {}'.format(CTS.beta_params))

    win_rate = win_rate / float(n_trials)
    print('\n\nEmpirical win rate for {} = {}'.format(algo, win_rate))



### RUN THE SCRIPT HERE ###
run_greedy_casino(n_trials=10000, algo='c_ts', verbose=True)



import numpy as np
import gym
# import logz_pytorch as logz
# from logz import logz
import logz
import os
import time
import inspect

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

#============================================================================================#
# Utilities
#============================================================================================#


def pathlength(path):
    return len(path["reward"])


class MLP(nn.Module):

    # ========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # ========================================================================================#

    def __init__(self, input_size, output_size, n_layers=2, size=64, activation=F.tanh, output_activation=None):
        super(MLP, self).__init__()
        # YOUR_CODE_HERE
        self.act = activation
        self.act_out = output_activation
        self.layers = n_layers
        self.fc_first = nn.Linear(input_size, size)
        self.fc_hidden = nn.Linear(size, size)
        self.fc_last = nn.Linear(size, output_size)

    def forward(self, x):
        x = self.act(self.fc_first(x))
        for n in range(self.layers-1):
            x = self.act(self.fc_hidden(x))
        if self.act_out is None:
            x = self.fc_last(x)
        else:
            x = self.act_out(self.fc_last(x))

        return x.view(x.size(0), -1)




#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             # env_name='CartPole-v0',
             env_name='InvertedPendulum-v1',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=False,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             act_sigma=0.5
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getfullargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    #
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    #
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    #
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #TODO: REMOVE
    # #========================================================================================#
    # #                           ----------SECTION 4----------
    # # Networks
    # #
    # # Make symbolic operations for
    # #   1. Policy network outputs which describe the policy distribution.
    # #       a. For the discrete case, just logits for each action.
    # #
    # #       b. For the continuous case, the mean / log std of a Gaussian distribution over
    # #          actions.
    # #
    # #      Hint: use the 'build_mlp' function you defined in utilities.
    # #
    # #
    # #   2. Producing samples stochastically from the policy distribution.
    # #       a. For the discrete case, an op that takes in logits and produces actions.
    # #
    # #          Should have shape [None]
    # #
    # #       b. For the continuous case, use the reparameterization trick:
    # #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    # #
    # #               mu + sigma * z,         z ~ N(0, I)
    # #
    # #          This reduces the problem to just sampling z. (Hint: use random_normal!)
    # #
    # #          Should have shape [None, ac_dim]
    # #
    # #      Note: these ops should be functions of the policy network output ops.
    # #
    # #   3. Computing the log probability of a set of actions that were actually taken,
    # #      according to the policy.
    # #
    # #
    # #========================================================================================#
    #
    # if discrete:
    #     # YOUR_CODE_HERE
    #     sy_logits_na = TODO
    #     sy_sampled_ac = TODO # Hint: Use the multinomial op
    #     sy_logprob_n = TODO
    #
    # else:
    #     # YOUR_CODE_HERE
    #     sy_mean = TODO
    #     sy_logstd = TODO # logstd should just be a trainable variable, not a network output.
    #     sy_sampled_ac = TODO
    #     sy_logprob_n = TODO  # Hint: Use the log probability under a multivariate gaussian.

    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#

    mlp = MLP(ob_dim, ac_dim, n_layers=n_layers, size=size, activation=F.tanh, output_activation=None)
    if discrete:
        criterion = torch.nn.CrossEntropyLoss(reduce=False)
    else:
        criterion = torch.nn.MSELoss(reduce=False, size_average=False)
    update_op = optim.Adam(mlp.parameters(), lr=learning_rate)
    softmax=torch.nn.Softmax(1)
    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_mlp = MLP(ob_dim, 1, n_layers=n_layers, size=size)
        bs_opt = optim.Adam(baseline_mlp.parameters(), lr=learning_rate)
        bs_criterion = torch.nn.MSELoss()

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)

                raw_ac = mlp(Variable(torch.FloatTensor(ob[None])))
                if discrete:
                    ac = np.random.choice(np.arange(raw_ac.size(1)), 1, p=softmax(raw_ac).data[0].numpy())[0]
                else:
                    ac=[]
                    for n in np.arange(ac_dim):
                        # ac.append(np.random.normal(raw_ac.data[0][n], np.exp(raw_ac.data[0][n+1]), 1)[0])
                        # ac.append(np.random.normal(raw_ac.data[0][n], 0.5, 1)[0])
                        ac.append(raw_ac.data[0][n] + act_sigma*np.random.normal(0, 1, 1)[0])

                acs.append(ac)

                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs),
                    "reward" : np.array(rewards),
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        rewards = np.concatenate([path["reward"] for path in paths])

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages.
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t.
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
        #       entire trajectory (regardless of which time step the Q-value should be for).
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above.
        #
        # ==================================================================================== #

        # YOUR_CODE_HERE
        q_n = []
        if reward_to_go:
            for p in paths:
                # q_n_path = []
                for i in range(pathlength(p)):
                    r_ = p['reward'][i:]
                    g_ = [gamma ** k for k in range(len(r_))]
                    q_n.append(sum(g_*r_))

        else:
            for p in paths:
                r_ = p['reward']
                g_ = [gamma ** k for k in range(pathlength(p))]
                # q_n.append(np.array([sum(g_ * r_)]*pathlength(p)))
                q_n.extend(np.array([sum(g_ * r_)] * pathlength(p)))

        q_n = np.array(q_n)
        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            states_tensor = Variable(torch.FloatTensor(ob_no).view(-1, 1, ob_no.shape[1]))
            b_n = baseline_mlp(states_tensor).data.numpy()[:,0]
            b_n = (b_n - b_n.mean())/b_n.std()

            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # YOUR_CODE_HERE
            adv_n = (adv_n - adv_n.mean())/adv_n.std()
            pass


        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#

        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE

            bs_opt.zero_grad()
            states_tensor = Variable(torch.FloatTensor(ob_no).view(-1, 1, ob_no.shape[1]))
            q_values_predict_tensor = baseline_mlp(states_tensor)
            q_values = (q_n-q_n.mean())/q_n.std()
            bs_l = bs_criterion(q_values_predict_tensor, Variable(torch.from_numpy(q_values).float()))
            bs_l.backward()
            bs_opt.step()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        # YOUR_CODE_HERE

        # update_op.zero_grad()
        # adv_tensor = Variable(torch.FloatTensor(adv_n))
        # states_tensor = Variable(torch.FloatTensor(ob_no).view(-1, 1, ob_no.shape[1]))
        # actions_predict_tensor = mlp(states_tensor)
        # loss = criterion(actions_predict_tensor, Variable(torch.from_numpy(ac_na)))
        # loss = torch.sum(loss * adv_tensor)/len(paths)
        # loss.backward()
        # update_op.step()
        # print(loss.data[0])


        update_op.zero_grad()
        adv_tensor = Variable(torch.FloatTensor(adv_n)).view(-1,1)
        states_tensor = Variable(torch.FloatTensor(ob_no).view(-1, 1, ob_no.shape[1]))
        actions_predict_tensor = mlp(states_tensor)
        if discrete:
            loss = criterion(actions_predict_tensor, Variable(torch.from_numpy(ac_na)))
        else:
            loss = criterion(actions_predict_tensor, Variable(torch.FloatTensor(ac_na)))
        loss = torch.sum(loss * adv_tensor)/len(paths)
        loss.backward()
        update_op.step()
        print(loss.data[0])



        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--act_sigma', '-as', type=float, default=0.5)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                act_sigma=args.act_sigma
                )
        train_func()


if __name__ == "__main__":
    main()

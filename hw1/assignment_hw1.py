import numpy as np
from funcs import train_bc
from funcs import run_policy
from funcs import get_results_for_iterations
from get_expert_policy import get_expert_policy
from neural_nets import FFN
import matplotlib.pyplot as plt
from dagger import label_actions_by_expert_policy
from dagger import run_learned_policy
from dagger import aggregate


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sec2q1', action='store_true')
    parser.add_argument('--sec3q1', action='store_true')
    parser.add_argument('--sec3q2', action='store_true')
    parser.add_argument('--sec4q2', action='store_true')
    args = parser.parse_args()




    if args.sec2q1:
        expert_policy_hopper, expert_return = get_expert_policy('Hopper-v1', rollouts=10)
        ffn_hopper = FFN(expert_policy_hopper['states'][0].shape[0], 64, 128, 64,
                         expert_policy_hopper['actions'][0].shape[1])
        train_bc(expert_policy_hopper, ffn_hopper, n_iters=100, learning_rate=0.01, batch_size=100, is_plot=True)

    if args.sec3q1:
        print('-------')
        print('Ant Training: GOOD Behavioral Cloning')
        print('-------')
        env_name_ant = 'Ant-v1'
        expert_policy_ant, expert_return = get_expert_policy(env_name_ant, rollouts=100)
        ffn_ant = FFN(expert_policy_ant['states'][0].shape[0], 64, 128, 64,
                         expert_policy_ant['actions'][0].shape[1])
        train_bc(expert_policy_ant, ffn_ant, n_iters=500, learning_rate=0.01, batch_size=1000)

        print('-------')
        print('Walker2d Training: BAD Behavioral Cloning')
        print('-------')
        env_name_w = 'Walker2d-v1'
        expert_policy_w, expert_return = get_expert_policy(env_name_w, rollouts=100)
        ffn_w = FFN(expert_policy_w['states'][0].shape[0], 64, 128, 64,
                      expert_policy_w['actions'][0].shape[1])
        train_bc(expert_policy_w, ffn_w, n_iters=500, learning_rate=0.01, batch_size=1000)

        print('-------')
        print('RESULTS:')
        print('-------')
        print('Ant-v1')
        run_policy(ffn_ant, env_name_ant, rollouts=100, is_debug=True, is_render=False)
        print('Walker2d-v1')
        run_policy(ffn_w, env_name_w, rollouts=100, is_debug=True, is_render=False)


    if args.sec3q2:
        env_name_ant = 'Ant-v1'
        n_iters = 1500
        expert_policy_ant, expert_return = get_expert_policy(env_name_ant, rollouts=100)
        ffn_ant = FFN(expert_policy_ant['states'][0].shape[0], 64, 128, 64,
                         expert_policy_ant['actions'][0].shape[1])
        r_means = get_results_for_iterations(env_name_ant, expert_policy_ant, ffn_ant,
                                             n_iters=n_iters, test_every=50, learning_rate=0.01, batch_size=1000)

        plt.plot(np.arange(50,n_iters+1,50), r_means[:len(np.arange(50,n_iters+1,50))])
        plt.ylabel('Mean of Return')
        plt.xlabel('Number of Training (iterations)')
        plt.show()


    if args.sec4q2:
        env_name = 'Walker2d-v1'
        dataset, expert_return = get_expert_policy(env_name, rollouts=5)
        ffn = FFN(dataset['states'][0].shape[0], 64, 128, 64,
                  dataset['actions'][0].shape[1])

        means = []
        stds = []
        for i in range(20):
            print('\nStarted ',i+1,' iteration of DAgger')
            train_bc(dataset, ffn, n_iters=100, learning_rate=0.01, batch_size=1000)
            new_states = run_learned_policy(env_name, ffn, rollouts=1)
            new_actions = label_actions_by_expert_policy(new_states, env_name)
            dataset = aggregate(dataset, np.array(new_states), np.array(new_actions))
            print('\nRun Walker2d-v1 test')
            mean, std = run_policy(ffn, env_name, rollouts=30, is_debug=True, is_render=False)
            means.append(mean)
            stds.append(std)
            print(i+1, ' iteration finished\n')

        # train BC
        env_name_w = 'Walker2d-v1'
        expert_policy_w, expert_return = get_expert_policy(env_name_w, rollouts=100)
        ffn_w = FFN(expert_policy_w['states'][0].shape[0], 64, 128, 64,
                      expert_policy_w['actions'][0].shape[1])
        train_bc(expert_policy_w, ffn_w, n_iters=500, learning_rate=0.01, batch_size=1000)
        mean_bc, std_bc = run_policy(ffn_w, env_name_w, rollouts=100, is_debug=False, is_render=False)

        # plot results
        plt.errorbar(range(1,len(means)+1), means, yerr=stds, fmt='-o', label='DAgger')
        plt.axhline(y=mean_bc, xmin=0, xmax=1, hold=None, c='red', label='Behavioral Cloning')
        plt.axhline(y=expert_return, xmin=0, xmax=1, hold=None, c='green', label='Expert Policy')
        plt.ylabel('Mean of Return')
        plt.xlabel('Number of DAgger iterations')
        plt.legend(loc=4)
        plt.show()

if __name__ == '__main__':
    main()

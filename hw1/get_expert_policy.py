import load_policy
import tf_util
import tensorflow as tf
import numpy as np


def get_expert_policy(env_name, rollouts=30):
    print('------')
    print('Generate Expert Policy')
    print('------')
    policy = load_policy.load_policy('experts/'+env_name+'.pkl')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(env_name)
        max_steps = env.spec.timestep_limit

        returns = []
        states = []
        actions = []
        for i in range(rollouts):
            # print('iter', i)
            state = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy(state[None, :])
                states.append(state)
                actions.append(action)
                state, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                # if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print(rollouts, 'rollouts')
        print('Mean Return: ', np.mean(returns))
        print('Std of Return: ', np.std(returns))

        expert_state_actions = {'states': np.array(states),
                                'actions': np.array(actions)}

    return (expert_state_actions, np.mean(returns))

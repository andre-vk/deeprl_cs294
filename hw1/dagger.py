import torch
from torch.autograd import Variable
import load_policy
import tf_util
import tensorflow as tf
import numpy as np

def run_learned_policy(env_name, ffn, rollouts=10):
    import gym
    env = gym.make(env_name)
    max_steps = env.spec.timestep_limit

    states = []
    actions = []
    for i in range(rollouts):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            ffn_action_output = ffn( Variable(torch.from_numpy(state).view(1, -1).float()) )
            action = ffn_action_output.data.numpy()
            states.append(state)
            actions.append(action)
            state, r, done, _ = env.step(action)
            steps += 1
            if steps >= max_steps:
                break

    return (states)



def label_actions_by_expert_policy(new_states, env_name):

    policy = load_policy.load_policy('experts/'+env_name+'.pkl')

    with tf.Session():
        tf_util.initialize()
        actions = []
        for i in range(len(new_states)):
            state = new_states[i]
            action = policy(state[None, :])
            actions.append(action)

    return (actions)

def aggregate(dataset, states, actions):

    dataset['states'] = np.concatenate((dataset['states'], states), axis=0)
    dataset['actions'] = np.concatenate((dataset['actions'], actions), axis=0)

    return (dataset)



import time
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# get random sample
def training_batch(expert_policy, batch_size):
    random_idxs = np.arange(len(expert_policy['states']))
    random.shuffle(random_idxs)
    sampled_states = expert_policy['states'][random_idxs[:batch_size]]
    sampled_actions = expert_policy['actions'][random_idxs[:batch_size]]

    states_tensor = torch.from_numpy(sampled_states).view(-1, 1, sampled_states.shape[1]).float()
    actions_tensor = torch.from_numpy(sampled_actions)
    return Variable(states_tensor), Variable(actions_tensor)

# Training function
def train_ffn(states_tensor, actions_tensor, ffn, criterion, learning_rate=0.01):
    tmp_loss = 0

    for i in range(states_tensor.size()[0]):
        optimizer = optim.SGD(ffn.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        output = ffn(states_tensor[i])  # Predict action
        loss = criterion(output, actions_tensor[i])  # Calculate error
        tmp_loss += loss.data[0]
        loss.backward()  # Error backpropagation

        optimizer.step()

    return output, tmp_loss / states_tensor.size()[0]

def train_bc(expert_policy, ffn, learning_rate=0.01, n_iters=100, batch_size=100, is_plot=False):
    criterion = nn.MSELoss()
    # Train loop
    print_every = 20
    all_losses = []

    print ('------')
    print ('TRAIN')
    print ('------')
    start = time.time()
    for iter in range(1, n_iters + 1):
        output, loss = train_ffn(*training_batch(expert_policy, batch_size), ffn=ffn, criterion=criterion, learning_rate=learning_rate)
        all_losses.append(loss)

        if iter % print_every == 0:
            print('%s: (%d%%) MSE=%.4f' % (timeSince(start), iter / n_iters * 100, loss))

    if is_plot:
        plt.plot(all_losses)
        plt.ylabel('MSE Loss')
        plt.xlabel('Iteration')
        plt.show()

def train_without_batch(expert_policy, ffn, learning_rate=0.01, n_iters=1):
    criterion = nn.MSELoss()
    # Train loop
    print_every = 20
    all_losses = []

    print ('------')
    print ('TRAIN')
    print ('------')
    start = time.time()
    for iter in range(1, n_iters + 1):
        output, loss = train_ffn(*training_batch(expert_policy, expert_policy['states'].shape[0]), ffn=ffn, criterion=criterion, learning_rate=learning_rate)
        all_losses.append(loss)

        if iter % print_every == 0:
            print('%s: (%d%%) MSE=%.4f' % (timeSince(start), iter / n_iters * 100, loss))

    if is_plot:
        plt.plot(all_losses)
        plt.ylabel('MSE Loss')
        plt.xlabel('Iteration')
        plt.show()


def run_policy(ffn, env_name, rollouts=1, is_render=False, is_debug=True):
    import gym
    env = gym.make(env_name)
    max_steps = env.spec.timestep_limit

    returns = []
    states = []
    actions = []
    for i in range(rollouts):
        state = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            ffn_action_output = ffn( Variable(torch.from_numpy(state).view(1, -1).float()) )
            action = ffn_action_output.data.numpy()
            states.append(state)
            actions.append(action)
            state, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if is_render:
                env.render()
            # if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        # print('Total steps: '+str(steps))
        returns.append(totalr)

    # print('returns', returns)
    if is_debug:
        print(rollouts, 'rollouts')
        print('Mean Return: ', np.mean(returns))
        print('Std of Return: ', np.std(returns))

    return (np.mean(returns), np.std(returns))


def get_results_for_iterations(env_name, expert_policy, ffn, n_iters=1000, test_every=50, learning_rate=0.01, batch_size=100):
    r_means = []
    criterion = nn.MSELoss()
    print_every = 20
    start = time.time()
    for iter in range(1, n_iters + 1):
        output, loss = train_ffn(*training_batch(expert_policy, batch_size), ffn=ffn, criterion=criterion,
                                 learning_rate=learning_rate)

        if iter % print_every == 0:
            print('%s: (%d%%) MSE=%.4f' % (timeSince(start), iter / n_iters * 100, loss))

        if iter % test_every == 0:
            mean, std = run_policy(ffn, env_name, rollouts=30, is_render=False, is_debug=False)
            r_means.append(mean)
            print ('Tested ', iter, 'th iteration')

    return (r_means)
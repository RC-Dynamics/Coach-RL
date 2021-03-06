import argparse
import collections
import math
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym_coach_vss
import wandb

# Hyperparameters
actor_lr = 0.0005
critic_lr = 0.0003
gamma = 0.99
batch_size = 32
buffer_limit = 500000
soft_tau = 0.005  # for target network soft update
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float),\
            torch.tensor(a_lst, dtype=torch.float), \
            torch.tensor(r_lst), \
            torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs + 3, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Actor(nn.Module):
    def __init__(self, num_inputs):
        super(Actor, self).__init__()
        self.num_input = num_inputs
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 3)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        state = state.view(1, self.num_input)
        action = self.forward(state)
        return action.detach().cpu().numpy()


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15,
                 max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def train(critic, critic_target, actor, actor_target,
          critic_optim, actor_optim, memory):

    state_batch, action_batch,\
        reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

    n_inputs = state_batch.size()[1]*state_batch.size()[2]
    state_batch = state_batch.view(batch_size, n_inputs)
    next_state_batch = next_state_batch.view(batch_size, n_inputs)

    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device).squeeze()
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    actor_loss = critic(state_batch, actor(state_batch))
    actor_loss = -actor_loss.mean()

    next_actions_target = actor_target(next_state_batch)
    q_targets = critic_target(next_state_batch, next_actions_target)
    targets = reward_batch + (1.0 - done_batch)*gamma*q_targets

    q_values = critic(state_batch, action_batch)
    critic_loss = F.smooth_l1_loss(q_values, targets.detach())
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    for target_param, param in zip(critic_target.parameters(),
                                   critic.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    for target_param, param in zip(actor_target.parameters(),
                                   actor.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    return actor_loss.item(), critic_loss.item()


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    res = math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])
    if idx > 0 and (idx == len(array) or res):
        return idx-1
    else:
        return idx


def mapped_action(action):
    action_list = ["000", "001", "002", "010", "011", "012", "020", "021",
                   "022", "100", "101", "102", "110", "111", "112", "120",
                   "121", "122", "200", "201", "202", "210", "211", "212",
                   "220", "221", "222"]
    act = ''
    for x in action:
        if x < -0.34:
            act += '0'
        elif x < 0.34:
            act += '1'
        else:
            act += '2'
    return action_list.index(act)


def main(load_model=False, test=False):
    try:
        if not test:
            wandb.init(name="CoachRL-DDPG", project="CoachRL")
        env = gym.make('CoachVssContinuous-v0')
        ou_noise = OUNoise(env.action_space)

        n_inputs = env.observation_space.shape[0] * \
            env.observation_space.shape[1]

        actor = Actor(n_inputs).to(device)
        actor_target = Actor(n_inputs).to(device)
        critic = Critic(n_inputs).to(device)
        critic_target = Critic(n_inputs).to(device)
        critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
        actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)

        if load_model or test:
            actor_dict = torch.load('models/DDPG_ACTOR.model')
            critic_dict = torch.load('models/DDPG_CRITIC.model')
            actor.load_state_dict(actor_dict)
            critic.load_state_dict(critic_dict)
            if not test:
                actor_optim_dict = torch.load(f'models/DDPG_ACTOR.optim')
                actor_optim.load_state_dict(actor_optim_dict)
                critic_optim_dict = torch.load(f'models/DDPG_CRITIC.optim')
                critic_optim.load_state_dict(critic_optim_dict)

        actor_target.load_state_dict(actor.state_dict())
        critic_target.load_state_dict(critic.state_dict())

        memory = ReplayBuffer()
        update_interval = 10
        total_steps = 0
        for n_epi in range(30):
            s = env.reset()
            done = False
            score = 0.0
            epi_step = 0
            episode = list()
            while not done:  # maximum length of episode is 200 for Pendulum-v0
                a = actor.get_action(s)
                if not test:
                    a = ou_noise.get_action(a, epi_step)[0]
                else:
                    a = a[0]
                action = mapped_action(a)
                s_prime, r, done, _ = env.step(action)
                done_mask = 0.0 if done else 1.0
                # episode.append((s, a, r, s_prime, done_mask))
                memory.put((s, a, r, s_prime, done_mask))
                score += r
                s = s_prime
                total_steps += 1
                epi_step += 1
                if done:
                    print('Reset')
                if memory.size() > batch_size and not test:
                    act_loss, critic_loss = train(critic, critic_target, actor,
                                                  actor_target, critic_optim,
                                                  actor_optim, memory)
                    wandb.log({'Loss/DDPG/Actor': act_loss,
                               'Loss/DDPG/Critic': critic_loss},
                              step=total_steps)
                    if total_steps % 1000 == 0:
                        torch.save(critic.state_dict(),
                                   f'models/DDPG_CRITIC_{n_epi:06d}.model')
                        torch.save(actor.state_dict(),
                                   f'models/DDPG_ACTOR_{n_epi:06d}.model')
                        torch.save(actor_optim.state_dict(),
                                   f'models/DDPG_ACTOR_{n_epi:06d}.optim')
                        torch.save(critic_optim.state_dict(),
                                   f'models/DDPG_CRITIC_{n_epi:06d}.optim')

            # if not env.broken:
            #     for exp in episode:
            #         memory.put(exp)

            if not test and not env.broken:
                goal_diff = env.goal_prev_yellow - env.goal_prev_blue
                wandb.log({'Rewards/total': score,
                           'Rewards/goal_diff': goal_diff,
                           'Rewards/num_penalties': env.num_penalties,
                           }, step=total_steps)

        env.close()
    except Exception as e:
        env.close()
        raise e


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Predicts your time series')
    PARSER.add_argument('--test', default=False,
                        action='store_true', help="Test mode")
    PARSER.add_argument('--load', default=False,
                        action='store_true',
                        help="Load models from examples/models/")
    ARGS = PARSER.parse_args()
    if not os.path.exists('./models'):
        os.makedirs('models')

    main(load_model=ARGS.load, test=ARGS.test)

import argparse
import collections
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
from gym_coach_vss import CoachEnv


random.seed(42)
# Hyperparameters
learning_rate = 0.0005
gamma = 0.94  # 0.9
buffer_limit = 500000
batch_size = 32
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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, num_input, actions):
        super(Qnet, self).__init__()
        self.actions = actions
        self.num_input = num_input
        self.fc1 = nn.Linear(num_input, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float().to(device)
        obs = obs.view(1, self.num_input)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.actions-1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    losses = list()
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_prime = s_prime.to(device)
        done_mask = done_mask.to(device)

        n_inputs = s.size()[1]*s.size()[2]

        s = s.view(batch_size, n_inputs)
        s_prime = s_prime.view(batch_size, n_inputs)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses


def main(load_model=False, test=False, use_render=False):
    try:
        if not test:
            wandb.init(name="CoachRL-DQN", project="CoachRL")
        env = gym.make('CoachVss-v0', render=use_render)
        n_inputs = env.observation_space.shape[0] * \
            env.observation_space.shape[1]
        q = Qnet(n_inputs, env.action_space.n).to(device)
        q_target = Qnet(n_inputs, env.action_space.n).to(device)
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer()
        optimizer = optim.Adam(q.parameters(), lr=learning_rate)

        if load_model or test:
            q_dict = torch.load('models/DQN_best.model',
                                map_location=lambda storage, loc: storage)
            q.load_state_dict(q_dict)
            q_target.load_state_dict(q_dict)
            if not test:
                optim_dict = torch.load(f'models/DQN.optim')
                optimizer.load_state_dict(optim_dict)

        update_interval = 10
        total_steps = 0
        for n_epi in range(2000):
            s = env.reset()
            done = False
            epi_steps = 0
            score = 0.0
            episode = list()
            while not done:
                epsilon = 0.01 + (0.99 - 0.01) * \
                    np.exp(-1. * total_steps / 30000)
                epsilon = epsilon if not test else 0.01
                a = q.sample_action(s, epsilon)
                s_prime, r, done, info = env.step(a)
                done_mask = 0.0 if done else 1.0
                episode.append((s, a, r, s_prime, done_mask))
                s = s_prime
                score += r
                total_steps += 1
                epi_steps += 1
                if done:
                    print('Reset')

            if not env.broken:
                for exp in episode:
                    memory.put(exp)

            if memory.size() > batch_size and not test:
                losses = train(q, q_target, memory, optimizer)
                wandb.log({'Loss/DQN': np.mean(losses)},
                          step=total_steps, commit=False)
                torch.save(q.state_dict(), 'models/DQN_best.model')

            if n_epi % 100 == 0:
                torch.save(q.state_dict(), f'models/DQN_{n_epi:06d}.model')
                torch.save(optimizer.state_dict(),
                           f'models/DQN_{n_epi:06d}.optim')

            if n_epi % update_interval == 0 and n_epi > 0 and not test:
                q_target.load_state_dict(q.state_dict())

            if not test and not env.broken:
                goal_diff = env.goal_prev_yellow - env.goal_prev_blue
                wandb.log({'Rewards/total': score,
                           'Loss/epsilon': epsilon,
                           'Rewards/goal_diff': goal_diff,
                           'Rewards/num_penalties': env.num_penalties,
                           'Rewards/num_atk_faults': env.num_atk_faults
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
    PARSER.add_argument('-r','--render', default=False,
                        action='store_true',
                        help="Use render")

    ARGS = PARSER.parse_args()
    if not os.path.exists('./models'):
        os.makedirs('models')

    main(load_model=ARGS.load, test=ARGS.test, use_render=ARGS.render)

import collections
import math
import random
import re

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
from gym_coach_vss import CoachEnv

wandb.init(name="CoachRL-TriDQN", project="CoachRL")

random.seed(42)
# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    def __init__(self, num_imput, actions):
        super(Qnet, self).__init__()
        self.actions = actions
        self.num_imput = num_imput
        self.fc1 = nn.Linear(num_imput, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float().to(device)
        obs = obs.view(1, self.num_imput)
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


def concat(x, y, qtde_steps=2):
    in1 = torch.Tensor(x)
    in2 = torch.Tensor([y for _ in range(qtde_steps)]).unsqueeze(1)
    dog = torch.cat([in1, in2], dim=1)
    return dog.numpy()


def main():
    action_list = ["000", "001", "002", "010", "011", "012", "020", "021",
                   "022", "100", "101", "102", "110", "111", "112", "120",
                   "121", "122", "200", "201", "202", "210", "211", "212",
                   "220", "221", "222"]
    try:
        env = CoachEnv()
        n_inputs = env.observation_space.shape[0] * \
            env.observation_space.shape[1]
        q_head = Qnet(n_inputs, 3).to(device)
        q_head_target = Qnet(n_inputs, 3).to(device)
        q_head_target.load_state_dict(q_head.state_dict())
        memory_head = ReplayBuffer()

        q_daughter = Qnet(n_inputs+env.window_size, 3).to(device)
        q_daughter_target = Qnet(n_inputs+env.window_size, 3).to(device)
        q_daughter_target.load_state_dict(q_daughter.state_dict())
        memory_daughter = ReplayBuffer()

        q_son = Qnet(n_inputs+2*env.window_size, 3).to(device)
        q_son_target = Qnet(n_inputs+2*env.window_size, 3).to(device)
        q_son_target.load_state_dict(q_son.state_dict())
        memory_son = ReplayBuffer()

        update_interval = 10
        score = 0.0
        optimizer_head = optim.Adam(q_head.parameters(), lr=learning_rate)
        optimizer_daughter = optim.Adam(
            q_daughter.parameters(), lr=learning_rate)
        optimizer_son = optim.Adam(q_son.parameters(), lr=learning_rate)

        total_steps = 0
        for n_epi in range(1000):
            s = env.reset()
            done = False
            epi_steps = 0
            score = 0.0
            a_head = -1
            a_daughter = -1
            while not done:
                epsilon = 0.01 + (0.99 - 0.01) * \
                    math.exp(-1. * total_steps / 30000)
                state_daughter = concat(s, a_head, env.window_size)
                state_son = concat(state_daughter,
                                   a_daughter, env.window_size)

                a_head = q_head.sample_action(s, epsilon)
                a_daughter = q_daughter.sample_action(state_daughter, epsilon)
                a_son = q_son.sample_action(state_son, epsilon)

                a_str = str(a_head) + str(a_daughter) + str(a_son)
                a = action_list.index(a_str)
                s_prime, r, done, info = env.step(a)

                more_than_one_goalie = [m.start(0)
                                        for m in re.finditer('2', a_str)]
                rews = [0, 0]
                if len(more_than_one_goalie) > 1:
                    iters = 1
                    for i, rbt_i in enumerate(more_than_one_goalie):
                        if rbt_i == 0:
                            continue
                        rews[i-1] = -5*rbt_i

                done_mask = 0.0 if done else 1.0
                memory_head.put((s, a_head, r, s_prime, done_mask))

                s_prime_daughter = concat(s_prime, a_head, env.window_size)
                memory_daughter.put((state_daughter, a_daughter,
                                     r+rews[0], s_prime_daughter, done_mask))

                s_prime_son = concat(
                    s_prime_daughter, a_daughter, env.window_size)
                memory_son.put((state_son, a_son, r+rews[1],
                                s_prime_son, done_mask))
                s = s_prime
                score += r
                total_steps += 1
                epi_steps += 1
                if done:
                    print('Reset')

            if memory_head.size() > batch_size:
                losses = train(q_head, q_head_target,
                               memory_head, optimizer_head)
                losses_daughter = train(q_daughter, q_daughter_target,
                                        memory_daughter, optimizer_daughter)
                losses_son = train(q_son, q_son_target,
                                   memory_son, optimizer_son)
                wandb.log({'Loss/Head': np.mean(losses)})
                wandb.log({'Loss/Daughter': np.mean(losses_daughter)})
                wandb.log({'Loss/Son': np.mean(losses_son)})

            if n_epi % update_interval == 0 and n_epi > 0:
                q_head_target.load_state_dict(q_head.state_dict())
                q_daughter_target.load_state_dict(q_daughter.state_dict())
                q_son_target.load_state_dict(q_son.state_dict())
            wandb.log({'rewards/total': score,
                       'Loss/epsilon': epsilon})
        env.close()
    except Exception as e:
        env.close()
        raise e


if __name__ == '__main__':
    main()

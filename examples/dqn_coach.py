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
from collections import defaultdict

random.seed(42)
# Hyperparameters
learning_rate = 0.0005
gamma = 0.94  # 0.9
buffer_limit = 500000
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vss_to_agressive = {0: 26, 1: 25, 2: 22, 3: 24,
                    4: 19, 5: 16, 6: 21, 7: 15,
                    8: 10, 9: 23, 10: 18, 11: 14,
                    12: 17, 13: 7, 14: 6, 15: 13,
                    16: 5, 17: 3, 18: 20, 19: 12,
                    20: 9, 21: 11, 22: 4, 23: 2,
                    24: 8, 25: 1, 26: 0}

num_2_role = {0: 'G', 1: 'Z', 2: 'A'}
num_2_role = ['G', 'Z', 'A']

vss_order = {0:  'AAA', 1:  'AAZ', 2:  'AAG', 3:  'AZA',
             4:  'AZZ', 5:  'AZG', 6:  'AGA', 7:  'AGZ',
             8:  'AGG', 9:  'ZAA', 10: 'ZAZ', 11: 'ZAG',
             12: 'ZZA', 13: 'ZZZ', 14: 'ZZG', 15: 'ZGA',
             16: 'ZGZ', 17: 'ZGG', 18: 'GAA', 19: 'GAZ',
             20: 'GAG', 21: 'GZA', 22: 'GZZ', 23: 'GZG',
             24: 'GGA', 25: 'GGZ', 26: 'GGG'}
formation_2_idx = {key: value for (value, key) in vss_order.items()}

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
        out = self.forward(obs) #[1, 27]
        coin = random.random()

        if coin < epsilon:
            #return random.randint(0, 26)
            play0 = num_2_role[random.randint(0, 2)]
            play1 = num_2_role[random.randint(0, 2)]
            play2 = num_2_role[random.randint(0, 2)]
            idx   = formation_2_idx[play0+play1+play2]
            
            return idx

        else:
            #return out.argmax().item()
            play0 = num_2_role[out[0, :3].argmax().item()]
            play1 = num_2_role[out[0, 3:6].argmax().item()]
            play2 = num_2_role[out[0, 6:9].argmax().item()]
            idx   = formation_2_idx[play0+play1+play2]
            
            return idx

def out_to_action(out, num_roles=3):
    # pred = [batch_size, n_actions]
    out0 = out[:,  :num_roles].argmax(1)
    out1 = out[:, num_roles:num_roles*2].argmax(1)
    out2 = out[:, num_roles*2:num_roles*3].argmax(1)

    idx   = [formation_2_idx[num_2_role[o0]+num_2_role[o1]+num_2_role[o2]] for o0, 
            o1, o2 in zip(out0, out1, out2)]
    code = 26 - (out2 + (out1 * num_roles) + (out0 * num_roles * num_roles))
    # Magic Number that works
    #if (torch.Tensor(idx).cuda() != code).sum():
    #    breakpoint()

    return code.unsqueeze(1)    

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
        q_a = out_to_action(q_out)
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
        env = gym.make('CoachVss-v0', render=use_render, fast_mode=False)
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
            actions_per_epi = []
            done = False
            epi_steps = 0
            score = 0.0
            episode = list()
            count_form = defaultdict(lambda:0)

            while not done:
                epsilon = 0.01 + (0.99 - 0.01) * \
                    np.exp(-1. * total_steps / 30000)
                epsilon = epsilon if not test else 0.01
                a = q.sample_action(s, epsilon)
                s_prime, r, done, info = env.step(a)
                actions_per_epi.append(vss_to_agressive[a])
                count_form[vss_to_agressive[a]] += 1
                
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
                torch.save(optimizer.state_dict(),
                           f'models/DQN_best.optim')

            if n_epi % 100 == 0:
                torch.save(q.state_dict(), f'models/DQN_{n_epi:06d}.model')
                torch.save(optimizer.state_dict(),
                           f'models/DQN_{n_epi:06d}.optim')

            if n_epi % update_interval == 0 and n_epi > 0 and not test:
                q_target.load_state_dict(q.state_dict())

            if not test and not env.broken:
                print(count_form.items())
                goal_diff = env.goal_prev_yellow - env.goal_prev_blue
                wandb.log({'Rewards/total': score,
                           'Loss/epsilon': epsilon,
                           'Rewards/goal_diff': goal_diff,
                           'Rewards/num_penalties': env.num_penalties,
                           'Rewards/num_atk_faults': env.num_atk_faults,
                           'Actions/hist': wandb.Histogram(actions_per_epi, 
                                            num_bins=27)
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

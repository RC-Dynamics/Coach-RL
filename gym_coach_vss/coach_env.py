import os
import socket
import struct
import subprocess
import time

import gym
import numpy as np
from gym.spaces import Box, Discrete

from gym_coach_vss.fira_parser import FiraParser
from gym_coach_vss.Game import History, Stats

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'


class CoachEnv(gym.Env):

    def __init__(self, addr='224.5.23.2', fira_port=10020,
                 sw_port=8084, qtde_steps=100, fast_mode=True,
                 render=False, sim_path=None, is_discrete=False,
                 versus='determistic'):

        super(CoachEnv, self).__init__()
        self.addr = addr
        self.sw_port = sw_port
        self.fira_port = fira_port
        self.fira = None
        self.sw_conn = None
        self.fast_mode = fast_mode
        self.do_render = render
        self.sim_path = sim_path
        self.history = History(qtde_steps)
        self.qtde_steps = qtde_steps
        self.agent_blue_process = None
        self.agent_yellow_process = None
        self.versus = versus
        self.sim_time = None
        self.time_limit = (5 * 60 * 1000)
        self.goal_prev_yellow = 0
        self.goal_prev_blue = 0
        self.is_discrete = is_discrete
        if not self.is_discrete:
            self.observation_space = Box(
                low=-1.0, high=1.0, shape=(qtde_steps, 29), dtype=np.float32)
        self.action_space = Discrete(18)

    def start_agents(self):
        if self.versus == 'determistic':
            command_blue = [BIN_PATH + 'VSSL_blue']
            command_blue.append('-H')
        elif self.versus == 'deep':
            command_blue = [BIN_PATH + 'VSSL_blue']
            command_blue.append('-H')
        else:
            raise ValueError(f'No team with {self.versus} type')
        command_yellow = [BIN_PATH + 'VSSL_yellow']
        command_yellow.append('-H')
        self.agent_blue_process = subprocess.Popen(command_blue)
        self.agent_yellow_process = subprocess.Popen(command_yellow)
        time.sleep(5)
        self.sw_conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sw_conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sw_conn.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.sw_conn.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sw_conn.bind(('0.0.0.0', 8084))

    def stop_agents(self):
        self.agent_blue_process.terminate()
        self.agent_blue_process.wait()
        self.agent_yellow_process.terminate()
        self.agent_yellow_process.wait()
        self.sw_conn.close()
        self.sw_conn = None

    def start(self):
        if self.fira is None:
            self.fira = FiraParser('224.5.23.2', port=self.fira_port,
                                   fast_mode=self.fast_mode,
                                   render=self.do_render,
                                   sim_path=self.sim_path)
        self.fira.start()
        if self.agent_blue_process is None:
            self.start_agents()

    def stop(self):
        if self.fira:
            self.stop_agents()
            self.fira.stop()

    def _receive_state(self):
        data = self.fira.receive()
        self.history.update(data)
        if self.is_discrete:
            state = self.history.disc_states
        else:
            state = self.history.cont_states
        return np.array(state)

    def reset(self):
        if self.fira:
            self.fira.stop()
        self.start()
        self.goal_prev_blue = 0
        self.goal_prev_yellow = 0
        state = self._receive_state()
        return np.array(state)

    def compute_rewards(self):
        diff_goal_blue = self.goal_prev_blue - self.history.data.goals_blue
        diff_goal_yellow = self.history.data.goals_yellow - self.goal_prev_yellow

        reward = 0
        if diff_goal_blue < 0:
            print('********************GOAL BLUE*********************')
            self.goal_prev_blue = self.history.data.goals_blue
            print(
                f'Blue {self.goal_prev_blue} vs {self.goal_prev_yellow} Yellow'
            )
            reward += diff_goal_blue*1000

        if diff_goal_yellow > 0:
            print('********************GOAL YELLOW*******************')
            self.goal_prev_yellow = self.history.data.goals_yellow
            print(
                f'Blue {self.goal_prev_blue} vs {self.goal_prev_yellow} Yellow'
            )
            reward += diff_goal_yellow*1000

        return reward

    def step(self, action):
        reward = 0
        for _ in range(self.qtde_steps):
            out_str = struct.pack('i', int(action))
            self.sw_conn.sendto(out_str, ('0.0.0.0', 4098))
            state = self._receive_state()
            done = True if self.history.time > self.time_limit else False
            reward += self.compute_rewards()
            if done:
                break
        return state, reward, done, self.history

    def render(self, mode='human'):
        pass

    def close(self):
        self.stop()

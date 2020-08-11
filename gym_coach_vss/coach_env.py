import os
import socket
import struct
import subprocess

import gym
import numpy as np
from gym.spaces import Box

from gym_coach_vss.fira_parser import FiraParser
from gym_coach_vss.Game import History, Stats

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'


class CoachEnv(gym.Env):

    def __init__(self, addr='224.5.23.2', fira_port=10020,
                 sw_port=8084, qtde_steps=10, fast_mode=True,
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
        # self.observation_space = Box(low=-1.0, high=1.0, shape=())

    def start_agents(self):
        command_blue = BIN_PATH + 'VSSL_blue'
        command_yellow = BIN_PATH + 'VSSL_yellow'
        self.agent_blue_process = subprocess.Popen(command_blue)
        self.agent_yellow_process = subprocess.Popen(command_yellow)
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
                                   render=self.render,
                                   sim_path=self.sim_path)
        self.fira.start()
        self.start_agents()

    def stop(self):
        self.stop_agents()
        self.fira.stop()

    def _receive_state(self):
        data = self.fira.receive()
        self.history.update(data)
        if self.is_discrete:
            state = np.array(self.history.disc_states)
        else:
            state = np.array(self.history.cont_states)
        return state

    def reset(self):
        self.stop()
        self.start()
        state = self._receive_state()
        return state

    def compute_rewards(self):
        diff_goal_blue = self.goal_prev_blue - self.history.data.goals_blue
        diff_goal_yellow = self.history.data.goals_yellow -\
            self.goal_prev_yellow

        reward = 0
        if diff_goal_blue < 0:
            self.goal_prev_blue = self.history.data.goals_blue
            reward += diff_goal_blue*1000

        if diff_goal_yellow < 0:
            self.goal_prev_yellow = self.history.data.goals_yellow
            reward += diff_goal_yellow*1000

        return reward

    def step(self, action):
        for _ in range(self.qtde_steps):
            out_str = struct.pack('i', int(action))
            self.sw_conn.sendto(out_str, ('0.0.0.0', 4098))
            state = self._receive_state()
        reward = self.compute_rewards()
        done = True if self.history.time > self.time_limit else False
        return state, reward, done, self.history

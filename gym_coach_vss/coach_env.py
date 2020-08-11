import os
import socket
import subprocess

import gym
from gym.spaces import Box

from gym_coach_vss.fira_parser import FiraParser
from gym_coach_vss.Game import History, Stats

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'


class CoachEnv(gym.Env):

    def __init__(self, addr='224.5.23.2', fira_port=10020,
                 sw_port=8084, max_history=10, fast_mode=True,
                 render=False, sim_path=None, versus='determistic'):

        super(CoachEnv, self).__init__()
        self.addr = addr
        self.sw_port = sw_port
        self.fira_port = fira_port
        self.fira = None
        self.sw_conn = None
        self.fast_mode = fast_mode
        self.do_render = render
        self.sim_path = sim_path
        self.history = History(max_history)
        self.agent_blue_process = None
        self.agent_yellow_process = None
        self.versus = versus
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
        return 1

    def reset(self):
        self.stop()
        self.start()
        state = self._receive_state()
        return state

    def step(self, action):
        pass

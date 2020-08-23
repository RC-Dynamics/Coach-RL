import os
import socket
import struct
import subprocess
import time
import math

import gym
import numpy as np
from gym.spaces import Box, Discrete

from gym_coach_vss.fira_parser import FiraParser
from gym_coach_vss.Game import History, Stats
import random

BIN_PATH = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/bin/'


w_grad_ball_potential = (0.08, 1)


class CoachEnv(gym.Env):

    def __init__(self, addr='224.5.23.2', fira_port=10020,
                 sw_port=8084, qtde_steps=60,
                 update_interval=15, fast_mode=True,
                 render=False, sim_path=None, is_discrete=True,
                 versus='determistic', logger_path='log.txt', yellow_name='yellow'):

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
        self.prev_time = 0.0
        self.num_penalties = 0
        self.num_atk_faults = 0
        self.full_atk_time = 0
        self.done = False
        self.logger_path = logger_path
        self.yellow_name = yellow_name
        self.update_interval = update_interval
        self.window_size = (qtde_steps//update_interval)
        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=(self.window_size, 30),
                                     dtype=np.float32)
        if self.is_discrete:
            self.action_space = Discrete(27)
        else:
            self.action_space = Box(low=-1.0, high=1.0,
                                    shape=(1,),
                                    dtype=np.float32)

    def start_agents(self):
        if self.versus == 'determistic':
            command_blue = [BIN_PATH + 'VSSL_blue']
            command_blue.append('-H')
        elif self.versus == 'deep':
            command_blue = [BIN_PATH + 'VSSL_blue']
            command_blue.append('-H')
        elif self.versus == 'goalie':
            command_blue = [BIN_PATH + 'VSSL_blue_goalie']
            command_blue.append('-H')
        elif self.versus == 'threezones':
            command_blue = [BIN_PATH + 'VSSL_blue_threezones']
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

    def check_agents(self):
        r_blue = self.agent_blue_process.poll() == None
        r_yellow = self.agent_yellow_process.poll() == None

        if not r_blue:
            print("Warring: Blue team Down", self.versus)
        if not r_yellow:
            print("Warring: Yellow Team Down")

        return r_blue and r_yellow


    def stop_agents(self):
        self.agent_blue_process.terminate()
        self.agent_blue_process.wait()
        self.agent_yellow_process.terminate()
        self.agent_yellow_process.wait()
        self.sw_conn.close()
        self.sw_conn = None
        self.agent_yellow_process = None
        self.agent_blue_process = None

    def start(self):
        if self.agent_yellow_process is None:
            self.start_agents()
            time.sleep(2)
        if self.fira is None:
            self.fira = FiraParser('224.5.23.2', port=self.fira_port,
                                   fast_mode=self.fast_mode,
                                   render=self.do_render,
                                   sim_path=self.sim_path)
        self.fira.start()

    def stop(self):
        if self.fira:
            self.stop_agents()
            self.fira.stop()

    def _receive_state(self, reset=False):
        data = self.fira.receive()
        self.history.update(data, reset=reset)
        state = self.history.cont_states
        state = np.array(state)
        state = state[self.update_interval-1::self.update_interval]
        return state

    def write_log(self, is_first=False):                
        with open(self.logger_path, 'a') as log:
            if is_first:
                log.write(f"{self.yellow_name}, {self.versus}\n")
            else:
                log.write(f"{self.goal_prev_yellow}, {self.goal_prev_blue}\n")
                

    def reset(self):
        is_first = not (self.fira and self.agent_yellow_process)
        self.write_log(is_first)

        if not is_first:
            print( f"Coach {self.goal_prev_yellow} x {self.goal_prev_blue} {self.versus}")
            self.stop()

        self.start()
        self.goal_prev_blue = 0
        self.goal_prev_yellow = 0
        self.num_atk_faults = 0
        self.num_penalties = 0
        self.history = History(self.qtde_steps)
        state = self._receive_state(reset=True)
        
        if self.versus == 'determistic':
            options = [18, 21]
            if self.full_atk_time % 10 == 0:
                option = 0
            else:
                option = int(random.choice(options))
            self.full_atk_time += 1
            out_str = struct.pack('i', option)
            self.sw_conn.sendto(out_str, ('0.0.0.0', 4097))
            if option == 0:
                option = 'AAA'
            elif option == 18:
                option = 'GAA'
            else:
                option = 'GZA'
                
        else:
            option = self.versus
        print(f'************* Against {option} *************')

        return np.array(state)

    def ball_potential(self, step=-1):
        dx_d = 0 - self.history.balls[step].x  # distance to defence
        dx_a = 170.0 - self.history.balls[step].x  # distance to attack
        dy = 65.0 - self.history.balls[step].y
        potential = ((-math.sqrt(dx_a ** 2 + 2 * dy ** 2)
                      + math.sqrt(dx_d ** 2 + 2 * dy ** 2)) / 170 - 1) / 2

        return potential

    def bound(self, value, floor, ceil):
        if value < floor:
            return floor
        elif value > ceil:
            return ceil
        else:
            return value

    def is_atk_fault(self):
        atk_fault = False
        bx, by = self.history.data.frame.ball.x, self.history.data.frame.ball.y
        if bx < -0.6 and abs(by) < 0.35:
            one_in_pen_area = False
            for robot in self.history.data.frame.robots_yellow:
                rx, ry = robot.x, robot.y
                if rx < -0.6 and abs(ry) < 0.35:
                    if (one_in_pen_area):
                        atk_fault = True
                    else:
                        one_in_pen_area = True
        return atk_fault

    def is_penalty(self):
        penalty = False
        bx, by = self.history.data.frame.ball.x, self.history.data.frame.ball.y
        if bx > 0.6 and abs(by) < 0.35:
            one_in_pen_area = False
            for robot in self.history.data.frame.robots_yellow:
                rx, ry = robot.x, robot.y
                if rx > 0.6 and abs(ry) < 0.35:
                    if (one_in_pen_area):
                        penalty = True
                    else:
                        one_in_pen_area = True
        return penalty

    def compute_rewards(self):
        diff_goal_blue = self.goal_prev_blue - self.history.data.goals_blue
        diff_goal_yellow = self.history.data.goals_yellow -\
            self.goal_prev_yellow

        ball_potential = self.ball_potential()
        prev_ball_potential = self.ball_potential((-2))

        if self.prev_time < self.history.time:
            dt = self.history.time - self.prev_time
            self.prev_time = self.history.time
        else:
            self.history.time = self.prev_time = 0.0
            dt = 0
            self.done = True

        if self.history.balls and dt > 0:
            grad_ball_potential = self.bound(
                (ball_potential - prev_ball_potential) * 4000 / dt,
                -1.0, 1.0)  # (-1,1)
        else:
            grad_ball_potential = 0.0

        reward = 0.0

        if self.is_penalty():
            self.num_penalties += 1
            reward -= 35
        if self.is_atk_fault():
            self.num_atk_faults += 1
            reward -= 10

        if diff_goal_blue < 0.0:
            print('********************GOAL BLUE*********************')
            self.goal_prev_blue = self.history.data.goals_blue
            print(
                f'Blue {self.goal_prev_blue} vs {self.goal_prev_yellow} Yellow'
            )
            reward += diff_goal_blue*100.0

        if diff_goal_yellow > 0.0:
            print('********************GOAL YELLOW*******************')
            self.goal_prev_yellow = self.history.data.goals_yellow
            print(
                f'Blue {self.goal_prev_blue} vs {self.goal_prev_yellow} Yellow'
            )
            reward += diff_goal_yellow*100.0

        reward += grad_ball_potential * w_grad_ball_potential[0]

        return reward

    def step(self, action):
        self.done = False
        reward = 0
        out_str = struct.pack('i', int(action))
        self.sw_conn.sendto(out_str, ('0.0.0.0', 4098))
        for _ in range(self.qtde_steps):
            state = self._receive_state()
            reward += self.compute_rewards()
            if not self.check_agents():
                self.done = True

            if self.done:
                break
        return state, reward, self.done, self.history

    def render(self, mode='human'):
        pass

    def close(self):
        self.stop()


class CoachEnvContinuous(CoachEnv):

    def __init__(self):
        super(CoachEnvContinuous, self).__init__(is_discrete=False)

import math
import os
import socket
import subprocess

from gym_coach_vss.fira_client import FiraClient


class FiraParser(object):

    def __init__(self, ip='224.5.23.2', port=10020,
                 fast_mode=True, render=False, sim_path=None):
        # -- Connection
        self.ip = ip
        self.port = port
        self.conn = FiraClient(ip=ip, port=self.port)
        self.com_socket = None
        self.address = None
        self.is_running = False
        self.process = None
        self.goals_left = 0
        self.goals_right = 0
        self.fast_mode = fast_mode
        self.render = render
        if sim_path is not None:
            self.simulator_path = sim_path
        else:
            home = '/home/' + os.getenv('user')
            self.simulator_path = home + 'FIRASim/bin/FIRASim'

    # Simulation methods
    # ----------------------------

    def start(self):
        self._start_simulation()
        self._connect()
        self.is_running = True
        self.goals_left = 0
        self.goals_right = 0

    def stop(self):
        self.is_running = False
        self._disconnect()
        self._stop_simulation()

    def _start_simulation(self):
        command = [self.simulator_path]
        if not self.render:
            command.append('-H')
        if self.fast_mode:
            command.append('--xlr8')
        self.process = subprocess.Popen(command)

    def _stop_simulation(self):
        self.process.terminate()
        self.process.wait()

    def reset(self):
        self.stop()
        self.start()

    def receive(self):
        data = self.conn.receive()
        return data

    def _disconnect(self):
        pass

    def _connect(self):
        self.com_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = (self.ip, self.port+1)
        self.conn.connect()

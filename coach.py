import argparse
import random
import socket
import struct
import time

import numpy as np

from dqn import DQN
from fira_parser import FiraParser


class Coach:
    # number of valid options
    num_formacoes = 3
    # seconds until next change
    end_game = 5*60

    def __init__(self):
        START = (self.end_game + 1)*(-1) + time.time()
        self.fira = FiraParser()
        self.fira.start()

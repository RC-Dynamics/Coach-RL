from gym.envs.registration import register

from .coach_env import *  # noqa
from .fira_client import *  # noqa
from .fira_parser import *  # noqa
from .fira_rec import *  # noqa
from .Game import *  # noqa
from .pb_fira import *

register(
    id='coach_vss-v0',
    entry_point='gym_coach_vss.coach_env:CoachEnv'
)

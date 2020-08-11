import gym
from gym_coach_vss import CoachEnv

env = CoachEnv()

try:
    for _ in range(10):
        state = env.reset()
        print(state.shape)
        done = False
        while not done:
            action = env.action_space.sample()
            print(action)
            next_state, reward, done, info = env.step(action)
except KeyboardInterrupt:
    env.close()

from gym_coach_vss import CoachEnv

env = CoachEnv(render=True)

try:
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
except KeyboardInterrupt:
    env.close()

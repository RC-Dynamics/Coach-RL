import collections

import numpy as np


class Stats:

    def __init__(self, data, n_player_per_team=3, control_radius=0.08):
        # width: 1.3, length: 1.5, goal_width: 0.4, goal_depth: 0.1
        field_width = data.field.width
        field_length = data.field.length
        self.goal = np.array([field_length/2, field_width/2])
        self.control_radius = control_radius

        self.court_offset = 0.2  # lembrar de conferir essa constante
        self.court_right_x = field_length/2 - self.court_offset
        self.court_left_x = -1 * self.court_right_x
        self.n_player_per_team = n_player_per_team

        self.eps = 1e-10

    def get_DDB(self, data):
        sbd = []

        ball = data.frame.ball

        for robot in data.frame.robots_blue:
            sbd.append(np.sqrt((robot.x - ball.x) **
                               2 + (robot.y - ball.y)**2))  # y
        for robot in data.frame.robots_yellow:
            sbd.append(np.sqrt((robot.x - ball.x) **
                               2 + (robot.y - ball.y)**2))  # y

        ddb_score = np.sum(sbd[:self.n_player_per_team]) - \
            np.sum(sbd[self.n_player_per_team:])

        return ddb_score

    def get_ball_dist_to_goal(self, data):
        ball = data.frame.ball
        ball = np.array([ball.x, ball.y])
        return np.linalg.norm(ball - self.goal)

    def get_param(self, history):
        phi_t = self.get_phi(history)
        c_t = self.get_control(history)
        return phi_t, c_t

    def get_control(self, history):
        cm, co = 0, 0

        ball = history.getBalls()
        blues = history.getBlueRobots()
        yellows = history.getYellowRobots()

        for i in range(len(ball)):
            cm += sum([self.is_controling(blue[i], ball[i]) for blue in blues])
            co += sum([self.is_controling(yellow[i], ball[i])
                       for yellow in yellows])
        c_t = cm / (cm + co + self.eps)

        return c_t

    def get_phi(self, history):
        bm, bo = 0, 0

        ball = history.getBalls()

        for ball_state in ball:
            bm += self.ball_in_area(ball=ball_state, use_right_goal=True)
            bo += self.ball_in_area(ball=ball_state, use_right_goal=False)

        phi_t = bm / (bm + bo + self.eps)

        return phi_t

    def is_controling(self, robot_pos, ball_pos):
        dist = np.sqrt((robot_pos.x-ball_pos.x)**2 +
                       (robot_pos.y-ball_pos.y)**2)
        return dist <= self.control_radius

    def ball_in_area(self, ball, use_right_goal=True):

        if use_right_goal:
            return int(ball.x >= self.court_right_x)
        return int(ball.x <= self.court_left_x)


class History:
    def __init__(self, MAX, num_robotsBlue=3, num_robotsYellow=3):
        self.MAX = MAX
        self.balls = collections.deque(maxlen=MAX)
        self.listOfBlueRobots = [
            collections.deque(maxlen=MAX)] * num_robotsBlue
        self.listOfYellowRobots = [
            collections.deque(maxlen=MAX)] * num_robotsYellow
        self.cont_states = collections.deque(maxlen=MAX)
        self.disc_states = collections.deque(maxlen=MAX)
        self.num_robotsBlue = num_robotsBlue
        self.num_robotsYellow = num_robotsYellow
        self.num_insertions = 0
        self.time = 0
        self.data = None
        self.stats = None

    def start_lists(self, data):
        for _ in range(self.MAX):
            self.balls.append(data.frame.ball)
            cont_state = []
            cont_state += [self.data.frame.ball.x, self.data.frame.ball.y]
            for robot in self.data.frame.robots_blue:
                cont_state += [robot.x, robot.y, robot.vx, robot.vy]
                self.listOfBlueRobots[robot.robot_id].append(robot)
            for robot in self.data.frame.robots_yellow:
                cont_state += [robot.x, robot.y, robot.vx,
                               robot.vy, robot.orientation]
                self.listOfYellowRobots[robot.robot_id].append(robot)
            cont_state += [0]
            self.cont_states.append(cont_state)

    def update(self, data, reset):
        self.data = data
        if reset:
            self.start_lists(self.data)
        self.stats = Stats(self.data)
        cont_state = []
        cont_state += [self.data.frame.ball.x, self.data.frame.ball.y]
        self.balls.append(self.data.frame.ball)
        for robot in self.data.frame.robots_blue:
            cont_state += [robot.x, robot.y, robot.vx, robot.vy]
            self.listOfBlueRobots[robot.robot_id].append(robot)
        for robot in self.data.frame.robots_yellow:
            cont_state += [robot.x, robot.y, robot.vx,
                           robot.vy, robot.orientation]
            self.listOfYellowRobots[robot.robot_id].append(robot)
        cont_state += [(data.goals_yellow - data.goals_blue) / 10]

        self.cont_states.append(cont_state)
        self.time = self.data.step

        self.num_insertions += 1

    def getBalls(self):
        return self.balls

    def getBlueRobots(self):
        return self.listOfBlueRobots

    def getYellowRobots(self):
        return self.listOfYellowRobots

    def getNumBlues(self):
        return self.num_robotsBlue

    def getNumYellows(self):
        return self.num_robotsYellow

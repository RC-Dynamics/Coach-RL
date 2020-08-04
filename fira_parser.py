import subprocess
from firaclient import *
import math


class FiraParser(object):

    def __init__(self, ip=None, port=10020):
        # -- Connection
        self.ip = ip
        self.port = port
        self.conn = FiraClient(port=self.port)




    # Simulation methods
    # ----------------------------
    def start(self):
        self._connect()
        self.is_running = True
        self.goals_left = 0
        self.goals_right = 0

    def stop(self):
        self.is_running = False
        self._disconnect()
        

    def reset(self):
        self.stop()
        self.start()

    # Network methods
    # ----------------------------
    def convert_ssl_to_sim_coord(self, pose, v_pose):
        width = 1.3/2.0
        lenght = (1.5/2.0) + 0.1
        '''if(self.is_team_yellow):
            pose.x = (lenght - pose.x)*100
            pose.y = (width + pose.y)*100
            pose.yaw = math.pi - pose.yaw
            v_pose.x *= -100
            v_pose.y *= 100
            v_pose.yaw = v_pose.yaw
        else:
            pose.x = (lenght+pose.x)*100
            pose.y = (width - pose.y)*100
            pose.yaw = -pose.yaw
            v_pose.x *= 100
            v_pose.y *= -100
            v_pose.yaw = v_pose.yaw'''

    def receive(self):
        data = self.conn.receive()
        '''state = Global_State()
        state.time = data.step
        state.goals_yellow = data.goals_yellow
        state.goals_blue = data.goals_blue

        # Ball
        ball = data.frame.ball
        ball_state = state.balls.add()
        ball_state.pose.x = ball.x
        ball_state.pose.y = ball.y
        ball_state.v_pose.x = ball.vx
        ball_state.v_pose.y = ball.vy
        self.convert_ssl_to_sim_coord(ball_state.pose, ball_state.v_pose)

        # Robots Yellow
        for robot in data.frame.robots_yellow:
            robot_state = state.robots_yellow.add()
            robot_state.pose.x = robot.x
            robot_state.pose.y = robot.y
            robot_state.pose.yaw = robot.orientation
            robot_state.v_pose.x = robot.vx
            robot_state.v_pose.y = robot.vy
            robot_state.v_pose.yaw = robot.vorientation
            self.convert_ssl_to_sim_coord(robot_state.pose,
                                          robot_state.v_pose)

        # Robots Blue
        for robot in data.frame.robots_blue:
            robot_state = state.robots_blue.add()
            robot_state.pose.x = robot.x
            robot_state.pose.y = robot.y
            robot_state.pose.yaw = robot.orientation
            robot_state.v_pose.x = robot.vx
            robot_state.v_pose.y = robot.vy
            robot_state.v_pose.yaw = robot.vorientation
            self.convert_ssl_to_sim_coord(robot_state.pose,
                                          robot_state.v_pose)
        # print(state.robots_blue[0].v_pose)'''
        #print(data.frame.ball.x)
        #print(data.frame.ball.y)
        return data

    def _disconnect(self):
        pass

    def _connect(self):
        self.com_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = ("127.0.0.1", self.port+1)
        self.conn.connect()



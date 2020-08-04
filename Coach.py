import socket
import time
import struct
import random
import argparse

from fira_parser import *

fira = FiraParser()
fira.start()

serv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serv.bind(('0.0.0.0', 8084))


NUM_FORMACOES = 3 #number of valid options for (formations/players/behaviors/We need a better name for this....)
END_GAME = 5*60 #seconds until next change
START = (END_GAME + 1)*(-1) + time.time()


parser = argparse.ArgumentParser()
parser.add_argument("--default", help="use --default if you want to send a default value, instead of a random one")
args = parser.parse_args()



try:
    while True:
        if ((time.time() - START) > END_GAME):
            if args.default:
                print("default value turned on:")
                print(args.default)
                aux = args.default
                out_str = struct.pack('i',int(aux))
                serv.sendto(out_str, ('0.0.0.0', 4098))
                break
            else:
                print("mudando a formacao para")
                aux = (random.randint(0,NUM_FORMACOES))
                print(aux)
            out_str = struct.pack('i',int(aux))
            serv.sendto(out_str, ('0.0.0.0', 4098))
            START = time.time()

        #Recebendo dados do simulador...
        data = fira.receive()
        data.step #time
        data.goals_yellow #gols yellow
        data.goals_blue #gols blue

        # Ball
        ball = data.frame.ball
        ball.x 
        ball.y
        ball.vx
        ball.vy
        

        # Robots Yellow
        for robot in data.frame.robots_yellow:

            robot.x
            robot.y
            robot.orientation
            robot.vx
            robot.vy
            robot.vorientation
            

        # Robots Blue
        for robot in data.frame.robots_blue:

            robot.x
            robot.y
            robot.orientation
            robot.vx
            robot.vy
            robot.vorientation
            

        print(data.frame.ball.x)
        print(data.frame.ball.y)
			


except Exception as e: 
    print(e)
    serv.close()
    fira.stop()

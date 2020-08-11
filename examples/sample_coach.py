import socket
import struct
import time

from gym_coach_vss.fira_parser import FiraParser
from gym_coach_vss.Game import History, Stats

fira = FiraParser('224.5.23.2')
fira.start()

serv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serv.bind(('0.0.0.0', 8084))  # 8084 Yellow team; 4097 Blue Team

# number of valid options for (formations/players/behaviors/We need a better name for this....)
NUM_FORMACOES = 3
END_GAME = 20  # seconds until next change
START = (END_GAME + 1)*(-1) + time.time()

data = fira.receive()

default = False
stats = Stats(data)
history = History(10)
idx = 0
aux = 0
try:
    while True:

        time.sleep(0.003)

        # Recebendo dados do simulador...
        data = fira.receive()
        history.update(data)

        if idx % 100 == 0:
            aux = data.frame.ball.x > 0  # (aux + 1)%2

            out_str = struct.pack('i', int(aux))
            serv.sendto(out_str, ('0.0.0.0', 4098))
            print("\r Change Formation to:", aux, end='', flush=True)

        idx = idx+1

        #print(f'\r {data.frame.ball.x:.03f} {data.frame.ball.y:.03f}, {stats.get_param(history)}', end='', flush=True)

except Exception as e:
    print(e)
    serv.close()
    fira.stop()

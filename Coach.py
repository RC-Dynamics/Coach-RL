import socket
import time
import struct
import random

serv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serv.bind(('0.0.0.0', 8084))


NUM_FORMACOES = 3 #number of valid options for (formations/players/behaviors/We need a better name for this....)
END_GAME = 5*60 #seconds until next change
START = (END_GAME + 1)*(-1) + time.time()



try:
    while True:
        if ((time.time() - START) > END_GAME):
            print("mudando a formacao para")
            aux = (random.randint(0,NUM_FORMACOES))
            print(aux)
            out_str = struct.pack('i',int(aux))
            serv.sendto(out_str, ('0.0.0.0', 4097))
            START = time.time()
			


except Exception as e: 
    print(e)
    serv.close()

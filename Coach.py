import socket
import time
import struct
import random
import argparse

from fira_parser import *
from Game import Stats, History


def main():
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
        history = History(100,3,3)
        
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
            history.update(data)
            stats = Stats(data)
            bp = stats.get_bp(data)
            ctrl = stats.get_controll(history)
            DDB = stats.get_DDB(data)
            phi, c = stats.get_param(history)

    except Exception as e: 
        print(e)
        serv.close()
        fira.stop()

if __name__ == "__main__":
    main()
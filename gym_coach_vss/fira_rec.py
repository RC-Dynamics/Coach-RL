from fira_parser import *


fira = FiraParser()
fira.start()


while(1):
    print('rodando')
    data = fira.receive()

fira.stop()
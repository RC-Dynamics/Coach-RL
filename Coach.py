import socket

import struct
from torch import nn

serv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serv.bind(('0.0.0.0', 8084))


#serv.listen(5)



try:

    while True:
        #d, client = serv.recvfrom(4096)
        #if not d: break
        aux = input("escreve um inteiro para a zona 0, 1 ou 2")
        out_str = struct.pack('i',int(aux))
        #for i in range(0,2):
        #    print(client, struct.unpack_from('f',out_str,4*i)[0])
        serv.sendto(out_str, ('0.0.0.0', 4097))
    #conn.close()
    print ('client disconnected')

except Exception as e: 
    print(e)
    print("Close Server!")
    #conn.close()
    serv.close()

        

import binascii
import socket

from gym_coach_vss.pb_fira.packet_pb2 import Environment


class FiraClient:

    def __init__(self, ip='224.5.23.2', port=10020):
        """
        Init SSLClient object.
        Extended description of function.
        Parameters
        ----------
        ip : str
            Multicast IP in format '255.255.255.255'. 
        port : int
            Port up to 1024. 
        """

        self.ip = ip
        self.port = port
        self.sock = None

    def connect(self):
        """Binds the client with ip and port and configure to UDP multicast."""

        if not isinstance(self.ip, str):
            raise ValueError('IP type should be string type')
        if not isinstance(self.port, int):
            raise ValueError('Port type should be int type')

        self.sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 128)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.bind((self.ip, self.port))

    def receive(self):
        """Receive package and decode."""

        data, _ = self.sock.recvfrom(1024)
        decoded_data = Environment().FromString(data)
        return decoded_data

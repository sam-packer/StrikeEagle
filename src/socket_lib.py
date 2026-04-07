import socket
import struct
import sys
import threading

hostname = socket.gethostname()
HOST = socket.gethostbyname(hostname)

message_buffer = {}
send_message_thread = None
condition = threading.Condition()
logger = ""
sock = 0


def connect_socket(c_host, port):
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Disable Nagle's algorithm — send every message immediately instead of
    # buffering small writes. Critical for low-latency request/response patterns.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        try:
            return sock.connect((c_host, port)) == 0
        except Exception:
            pass


def close_socket():
    sock.close()


def send_message(message):
    size = len(message)
    sizeb = size.to_bytes(4, byteorder="big")
    sock.sendall(sizeb + message)


def send_messages_batch(messages):
    """Send multiple messages in a single sendall call.

    Each message is a bytes object. They are framed with 4-byte big-endian
    length headers and concatenated into one TCP write, reducing syscall
    overhead for fire-and-forget commands.
    """
    buf = bytearray()
    for msg in messages:
        size = len(msg)
        buf.extend(size.to_bytes(4, byteorder="big"))
        buf.extend(msg)
    sock.sendall(buf)


def get_answer_header():
    global logger
    try:
        received = sock.recv(4)
        while len(received) > 0 and len(received) < 4:
            received += sock.recv(4 - len(received))
        if len(received) <= 0:
            return None
        size = int.from_bytes(received, "big")
        return size
    except Exception:
        logger = "Error: Crash socket get_answer_header\n {0}".format(
            sys.exc_info()[0]
        )
        return None


def get_answer(with_id=False, max_size_before_flush=-1):
    global logger
    try:
        size = get_answer_header()
        if size is None or (max_size_before_flush != -1 and size > max_size_before_flush):
            return None
        received = sock.recv(size)
        while len(received) < size:
            received += sock.recv(size - len(received))
        return received
    except Exception:
        logger = "Error: Crash socket get answer\n {0}".format(sys.exc_info()[0])
        return None

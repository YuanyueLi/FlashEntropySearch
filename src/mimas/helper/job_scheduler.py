import numpy as np
import socket
import struct
import threading
from threading import Thread
from queue import Queue
import time

import cloudpickle

_PACKET_SIZE = 8*1024


def pack_int(value):
    return struct.pack('q', value)


def unpack_int(value):
    return struct.unpack('q', value)[0]


def send_data(conn, data):
    try:
        data = cloudpickle.dumps(data)
        conn.sendall(pack_int(len(data)))
        response = conn.recv(1024)
        if not response:
            return False
        data_list = [data[i:i+_PACKET_SIZE] for i in range(0, len(data), _PACKET_SIZE)]
        for d in data_list:
            conn.sendall(d)
        response = conn.recv(1024)
        if not response:
            return False
        return True
    except:
        return False


def receive_data(conn):
    try:
        data_length = unpack_int(conn.recv(1024))
        conn.sendall(b'\1')
        data_list = []
        while data_length > 0:
            data = conn.recv(_PACKET_SIZE)
            data_list.append(data)
            data_length -= len(data)
        data = b''.join(data_list)
        conn.sendall(b'\1')
        data = cloudpickle.loads(data)
        return data
    except:
        return None


class JobScheduler:
    def __init__(self, func_run, func_merge=None,
                 para_share=None, output_result_in_order=True,
                 host='localhost', port=5678):
        if func_merge is None:
            def func_merge(f_final_result, f_cur_result):
                if f_final_result is None:
                    f_final_result = [f_cur_result]
                else:
                    f_final_result.append(f_cur_result)
                return f_final_result

        self._func_merge = func_merge
        self._func_run = func_run

        self._para_share = para_share

        self._output_result_in_order = output_result_in_order
        self._final_result = None

        self._total_output_num = 0
        self._total_input_num = 0

        self._q_input = Queue()
        self._q_output = Queue()

        self.job_host = JobHost(self._q_input, self._q_output, host=host, port=port)

    def add_job(self, para):
        self._q_input.put((self._total_input_num, self._func_run, para))
        self._total_input_num += 1

    def get_result(self):
        while self._total_input_num > self._total_output_num:
            result = self._q_output.get(block=True)
            self._final_result = self._func_merge(self._final_result, result)
            self._total_output_num += 1

        # The last one is for the connection which triggers the job_host to stop
        for _ in range(self.job_host.get_worker_number()):
            self._q_input.put(None)

        self.job_host.stop()
        return self._final_result


class JobHost:
    def __init__(self, q_input, q_output, host, port):
        self.host = host
        self.port = port

        self._threads = []
        self._worker_sockets = {}
        self._next_client_id = 0

        self.q_input = q_input
        self.q_output = q_output

        self._deamon_thread = Thread(target=self._deamon_thread_func)
        self._deamon_thread.start()

        self._stop = 0
        self._socket = None

    def _deamon_thread_func(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            self._socket = s
            s.bind((self.host, self.port))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 3)
            s.listen()
            while True:
                conn, addr = s.accept()
                if self._stop:
                    break
                thread = Thread(target=self._accept_new_worker, args=(conn, self._next_client_id))
                thread.start()
                self._threads.append(thread)
                self._next_client_id += 1
            print('Server stopped')
        print("Deamon thread finished")

    def _accept_new_worker(self, conn, client_id):
        with conn:
            lock = threading.Lock()
            lock.acquire()
            self._worker_sockets[client_id] = conn
            lock.release()
            print(f"Connected by client_id: {client_id}, total clients: {len(self._worker_sockets)}")

            while True:
                q_item = self.q_input.get(block=True, timeout=1)
                if q_item is None or self._stop:
                    print(f"Client_id: {client_id} finished, total clients: {len(self._worker_sockets)}")
                    break
                else:
                    print(f"Get job from input queue, total jobs: {self.q_input.qsize()}")
                    i, func, para = q_item
                    send_data(conn, (i, func, para))
                    result = receive_data(conn)
                    if result is None:
                        # The worker didn't return any result, so we assume it is dead, re-calulate the result
                        print(f"Error: Client_id: {client_id} is dead, re-calculating")
                        self.q_input.put(q_item)
                        break
                    assert result[0] == i
                    self.q_output.put((i, result))

            lock.acquire()
            self._worker_sockets.pop(client_id)
            lock.release()
        print(f"Disconnected by client_id: {client_id}, total clients: {len(self._worker_sockets)}")

    def stop(self):
        self._stop = 1
        # Send a new connection to the server to stop it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
        self._socket.close()
        for thread in self._threads:
            thread.join()
        self._deamon_thread.join()

    def get_worker_number(self):
        return len(self._worker_sockets)


def run_as_guest(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        conn.connect((host, port))
        while True:
            job = receive_data(conn)
            if not job:
                break

            job_id, func, parameters = job
            result = func(*parameters)
            host_response = send_data(conn, (job_id, result))
            if not host_response:
                break


if __name__ == '__main__':
    run_as_guest(host='localhost', port=5678)

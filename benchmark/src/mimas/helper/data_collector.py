import struct
import pickle


class DataCollector:
    def __init__(self, filename, mode):
        if mode == "read":
            self.file = open(filename, 'rb')
            information_data_location = self.read_int()
            information_data_length = self.read_int()
            self.file.seek(information_data_location)
            self.information_data = pickle.loads(self.file.read(information_data_length))

        elif mode == "write":
            self.file = open(filename, 'wb')
            self.write_int(0)
            self.write_int(0)
            self.information_data = []
        else:
            raise ValueError("Unknown mode: {}".format(mode))

    def add_data(self, data):
        binary_data = pickle.dumps(data)
        self.file.seek(0, 2)
        self.information_data.append((self.file.tell(), len(binary_data)))
        self.file.write(binary_data)
        return len(self.information_data) - 1

    def close(self):
        binary_data = pickle.dumps(self.information_data)
        self.file.seek(0, 2)
        information_data_location = self.file.tell()
        self.file.write(binary_data)
        self.file.seek(0)
        self.write_int(information_data_location)
        self.write_int(len(binary_data))

    def get_size(self):
        return len(self.information_data)

    def get_data(self, index):
        self.file.seek(self.information_data[index][0])
        return pickle.loads(self.file.read(self.information_data[index][1]))

    def read_int(self):
        return struct.unpack("q", self.file.read(8))[0]

    def write_int(self, value):
        self.file.write(struct.pack("q", value))

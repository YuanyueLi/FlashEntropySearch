import argparse
import datetime
import json
import os
import pprint
import sys
import logging
import types
from pathlib import Path
from argparse import Namespace as NamespaceOld

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


class Namespace(NamespaceOld):
    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return default

    def update(self, other):
        for k, v in other.items():
            setattr(self, k, v)
    
    def dict(self):
        return self.__dict__


class Arguments(argparse.ArgumentParser):
    """
    Compare to standard ArgumentParser, this class has following features:
    1. Can read json file as parameter.
    2. Automatically create output path if not existed.
    3. Save parameter to log file.
    """

    def __init__(self, *args, **kwargs):
        super(Arguments, self).__init__(add_help=True, *args, **kwargs)
        self.formatter_class = lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=100, width=200)
        self.add_argument('-parameter_file', type=argparse.FileType("r"), default=None,
                          help="Read parameter from a file in json format.")
        self.add_argument('-output_parameter', type=int, default=0)
        self.add_argument('-threads', type=int, default=1)
        self.add_argument('-debug', type=int, default=0)
        # self.add_argument('-path_output', type=str)

    def add_argument_from_dictionary(self, arguments: dict) -> None:
        """"
        Over current parameter
        """

        for item in arguments:
            if "-" + item not in self._option_string_actions:
                if isinstance(arguments[item], list):
                    self.add_argument("-" + item, nargs='*', default=arguments[item],
                                      help=f"Default value: {arguments[item]}")
                else:
                    self.add_argument(
                        "-" + item, type=type(arguments[item]),
                        default=arguments[item],
                        help=f"Default value: {arguments[item]}")
            else:
                action = self._option_string_actions["-" + item]
                action.default = arguments[item]

    def parse_args(self, print_parameter=True, auto_create_path=True, args=None, output_to_log=False):
        parsed_args = super(Arguments, self).parse_args(args=args, namespace=Namespace())

        # Deal with the json file
        if parsed_args.parameter_file:
            parameter_file = json.load(parsed_args.parameter_file)
            for item in parameter_file:
                setattr(parsed_args, item, parameter_file[item])
            raise NotImplementedError("Check the code before use this function")

        # Smart fill the path_output
        if "path_output" not in parsed_args.__dict__:
            setattr(parsed_args, "path_output", None)
        if parsed_args.path_output is None:
            if "file_output" in parsed_args.__dict__:
                parsed_args.path_output = Path(parsed_args.file_output).parent
            else:
                parsed_args.path_output = Path(sys.argv[0]).parent
            parsed_args.path_output.mkdir(parents=True, exist_ok=True)

        # Convert all pathname to absolute path, and create the directory if not existed
        for para_name in parsed_args.__dict__:
            if para_name.startswith("path_") or para_name.startswith("file_"):
                value = getattr(parsed_args, para_name)
                if isinstance(value, str):
                    value = Path(value)
                elif isinstance(value, list):
                    value = [Path(item) for item in value]
                setattr(parsed_args, para_name, value)

                # Create path/file if not existed.
                if auto_create_path:
                    if isinstance(value, Path):
                        if para_name.startswith("path_"):
                            value.mkdir(parents=True, exist_ok=True)
                        else:
                            value.parent.mkdir(parents=True, exist_ok=True)
                    elif isinstance(value, list):
                        for item in value:
                            if para_name.startswith("path_"):
                                item.mkdir(parents=True, exist_ok=True)
                            else:
                                item.parent.mkdir(parents=True, exist_ok=True)

        if print_parameter:
            pprint.pprint(vars(parsed_args))

        # Save parameter to log file
        log_output = Path(parsed_args.path_output) / \
            ("parameter-" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.log')
        if output_to_log:
            stdout_logger = logging.getLogger('STDOUT')
            sl = StreamToLogger(stdout_logger, logging.INFO)
            sys.stdout = sl
            setattr(parsed_args, "output_parameter", True)

        if parsed_args.output_parameter:
            logging.basicConfig(filename=log_output, level=logging.DEBUG,
                                format='%(asctime)s %(message)s')
            logging.debug(pprint.pformat(vars(parsed_args)))
        return parsed_args


if __name__ == "__main__":
    args = Arguments()
    args.add_argument("-test", type=str)

    args.add_argument_from_dictionary({
        "test3": "def"
    })
    para = args.parse_args()
    print(para)

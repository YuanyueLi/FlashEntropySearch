import multiprocessing as mp
import numpy as np
from functools import reduce
import copy
import pickle
import tempfile
import os


def _func_worker_no_auto_stop(func_run, func_para_share, q_input, q_output, **kwargs):
    if kwargs.get("copy_shared_para", False):
        func_para_share = copy.deepcopy(func_para_share)
    if kwargs.get("pass_value_via_disk", False):
        pass_value_via_disk = True
    else:
        pass_value_via_disk = False

    while 1:
        try:
            q_item = q_input.get(block=True, timeout=1)
        except:
            continue

        if q_item is None:
            break

        i, para = q_item
        if func_para_share is not None:
            para += func_para_share

        try:
            result = func_run(*para)
        except Exception as e:
            result = None
            print(f"Error found when processing {para}: {e}")

        if pass_value_via_disk:
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
                pickle.dump(result, f)
                result = f.name

        q_output.put((i, result))
    return


class MPRunner:
    def __init__(self, func_run, func_merge=None, func_worker=None,
                 para_share=None, para_merge=(), copy_shared_para=False,
                 threads=1, max_job_in_queue=None, output_result_in_order=True,
                 pass_value_via_disk=False):
        if threads is None:
            threads = 1
        if func_merge is None:
            def func_merge(f_final_result, f_cur_result):
                if f_final_result is None:
                    f_final_result = [f_cur_result]
                else:
                    f_final_result.append(f_cur_result)
                return f_final_result
        if func_worker is None:
            func_worker = _func_worker_no_auto_stop

        self._func_merge = func_merge
        self._func_run = func_run

        self._para_merge = para_merge
        self._para_share = para_share

        self._q_input = mp.Queue()
        self._q_output = mp.Queue()
        if max_job_in_queue is None:
            self._q_max_item_in_queue = None
        else:
            self._q_max_item_in_queue = threads * max_job_in_queue

        self._temp_result = {}
        self._final_result = None
        self._output_result_in_order = output_result_in_order
        self._pass_value_via_disk = pass_value_via_disk

        self._total_output_num = 0
        self._total_input_num = 0

        self._workers = [mp.Process(target=func_worker,
                                    args=(func_run, para_share, self._q_input, self._q_output),
                                    kwargs={"copy_shared_para": copy_shared_para,
                                            "pass_value_via_disk": pass_value_via_disk})
                         for _ in range(threads)]

        for w in self._workers:
            w.start()

    def add_job(self, *args):
        self.add_parameter_for_job(args)

    def get_result(self):
        return self.wait_for_result()

    def add_parameter_for_job(self, cur_para, debug=0):
        if debug == 1:
            if self._para_share is None:
                return self._func_run(*cur_para)
            else:
                return self._func_run(*cur_para, *self._para_share)

        if self._q_max_item_in_queue is not None:
            if (self._q_output.qsize() > self._q_max_item_in_queue) or \
                    (self._q_input.qsize() > self._q_max_item_in_queue):
                cur_result_num, cur_result = self._q_output.get()
                if self._pass_value_via_disk:
                    filename = cur_result
                    cur_result = pickle.load(open(filename, "rb"))
                    os.remove(filename)

                if self._output_result_in_order:
                    self._temp_result[cur_result_num] = cur_result
                    while self._total_output_num in self._temp_result:
                        self._final_result = self._func_merge(
                            self._final_result, self._temp_result.pop(self._total_output_num), *self._para_merge)
                        self._total_output_num += 1
                else:
                    self._final_result = self._func_merge(
                        self._final_result, cur_result, *self._para_merge)
                    self._total_output_num += 1

        self._q_input.put((self._total_input_num, cur_para,))
        self._total_input_num += 1

    def wait_for_result(self):
        for _ in self._workers:
            self._q_input.put(None)

        while self._total_output_num < self._total_input_num:
            cur_result_num, cur_result = self._q_output.get()
            if self._pass_value_via_disk:
                filename = cur_result
                cur_result = pickle.load(open(filename, "rb"))
                os.remove(filename)

            self._temp_result[cur_result_num] = cur_result

            while self._total_output_num in self._temp_result:
                self._final_result = self._func_merge(
                    self._final_result, self._temp_result.pop(self._total_output_num), *self._para_merge)
                self._total_output_num += 1

        for w in self._workers:
            w.join()
            w.close()
        return self._final_result


def run_multiple_process(func_run, func_merge=None, func_worker=None,
                         all_para_individual=(), para_share=None, threads=1):
    if func_merge is None:
        def func_merge(f_final_result, f_cur_result):
            if f_final_result is None:
                f_final_result = [f_cur_result]
            else:
                f_final_result.append(f_cur_result)
            return f_final_result

    if func_worker is None:
        func_worker = _func_worker_no_auto_stop

    q_input = mp.Queue()
    for cur_result_num, para in enumerate(all_para_individual):
        q_input.put((cur_result_num, para,))

    q_output = mp.Queue()

    workers = [mp.Process(target=func_worker, args=(func_run, para_share, q_input, q_output))
               for _ in range(threads)]

    for w in workers:
        w.start()

    temp_result = {}
    final_result = None
    cur_processing_num = 0
    total_item_num = len(all_para_individual)
    while cur_processing_num < total_item_num:
        cur_result_num, cur_result = q_output.get()
        temp_result[cur_result_num] = cur_result

        while cur_processing_num in temp_result:
            final_result = func_merge(final_result, temp_result.pop(cur_processing_num))
            cur_processing_num += 1

    for w in workers:
        q_input.put(None)

    for w in workers:
        w.join()
    return final_result


def convert_numpy_array_to_shared_memory(np_array, array_c_type=None):
    """
    The char table of shared memory can be find at:
    https://docs.python.org/3/library/struct.html#format-characters
    https://docs.python.org/3/library/array.html#module-array (This one is wrong!)
    The documentation of numpy.frombuffer can be find at:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.frombuffer.html
    Note: the char table is different from the char table in numpy
    """
    dim = np_array.shape
    num = reduce(lambda x, y: x * y, dim)
    if array_c_type is None:
        array_c_type = np_array.dtype.char
    base = mp.Array(array_c_type, num, lock=False)
    np_array_new = np.frombuffer(base, dtype=np_array.dtype).reshape(dim)
    np_array_new[:] = np_array
    return np_array_new

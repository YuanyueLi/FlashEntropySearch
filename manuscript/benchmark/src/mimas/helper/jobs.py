import hashlib
import os
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import cloudpickle
import sys
import re
import subprocess
import shutil


def _func_merge_result(f_final_result, f_cur_result):
    if f_final_result is None:
        f_final_result = [f_cur_result]
    else:
        f_final_result.append(f_cur_result)
    return f_final_result


class Jobs:
    def __init__(self, path_for_job_storage, func_run, func_merge=None,
                 output_result_in_order=True, job_batch_size=1, module_to_load=None):
        self._path_for_job_storage = Path(path_for_job_storage)
        self._path_for_job_storage.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(Path(__file__).resolve()), self._path_for_job_storage)
        if module_to_load is not None:
            for m in module_to_load:
                cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES.add(m)
        while True:
            cloudpickle.dump(func_run, open(self._path_for_job_storage/'function.pkl', 'wb'))
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(self._path_for_job_storage)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if len(stderr) > 0:
                print(stderr.decode())
            if len(stdout) > 0:
                stdout = stdout.decode()
                cloudpickle.cloudpickle._PICKLE_BY_VALUE_MODULES.add(stdout.rstrip())
            else:
                break
        # self._func_run = func_run
        if func_merge is None:
            self._func_merge = _func_merge_result
        else:
            self._func_merge = func_merge

        self._output_result_in_order = output_result_in_order
        self._job_batch_size = job_batch_size
        self._job_cache = []
        self._final_result = None

        self._input_para_list = []

    def add_job(self, *args, **kwargs):
        para = (args, kwargs)
        self._job_cache.append(para)
        if len(self._job_cache) >= self._job_batch_size:
            self._write_jobs_to_disk()

    def _write_jobs_to_disk(self):
        job = self._job_cache
        job_bytes = cloudpickle.dumps(job)
        job_md5 = hashlib.md5(job_bytes).hexdigest()
        # Write job to file
        file_output = self._path_for_job_storage / job_md5[:2] / (job_md5 + '.job')
        file_output.parent.mkdir(parents=True, exist_ok=True)
        with file_output.open('wb') as f:
            f.write(job_bytes)
        # Add job to list
        self._input_para_list.append(job_md5)
        # Clear job cache
        self._job_cache = []

    def get_result(self, check_interval_in_second=10, remove_result_after_read=False):
        if self._job_cache:
            self._write_jobs_to_disk()
        # Collect all result
        for job_md5 in self._input_para_list:
            file_job = self._path_for_job_storage / job_md5[:2] / (job_md5 + '.job')
            file_job_result = self._path_for_job_storage / job_md5[:2] / (job_md5 + '.result')
            file_job_lock = self._path_for_job_storage / job_md5[:2] / (job_md5 + '.lock')
            # print(f"Get result for {job_md5}")
            while file_job_lock.is_file() or (not file_job_result.is_file()):
                print("Waiting for job {} to finish.".format(job_md5))
                time.sleep(check_interval_in_second)

            with file_job_result.open('rb') as f:
                all_results = cloudpickle.loads(f.read())
            for result in all_results:
                self._final_result = self._func_merge(self._final_result, result)
            if remove_result_after_read:
                file_job.unlink()
                file_job_result.unlink()

        return self._final_result


class JobWorker:
    def __init__(self, path_for_job_storage) -> None:
        self._path_for_job_storage = Path(path_for_job_storage)
        self._path_for_job_storage.mkdir(parents=True, exist_ok=True)
        self._thread = None
        self._queue_single = Queue()
        self._job_func = self.load_function()

    def work(self, lock_file_expire_time_in_second=30):
        # First select a job withouth lock
        for file_job in self._path_for_job_storage.glob('**/*.job'):
            # If the result file exists, skip this job
            file_result = file_job.parent / (file_job.stem + '.result')
            if file_result.is_file():
                continue

            # If the lock file exists, skip this job
            file_lock = file_job.parent / (file_job.stem + '.lock')
            if file_lock.is_file():
                continue
            try:
                fd = os.open(file_lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
            except:
                continue
            self._run_a_job(file_job, file_result, file_lock)
        return 0

        # Then select a job with expired lock
        for file_job in self._path_for_job_storage.glob('**/*.job'):
            file_lock = file_job.parent / (file_job.stem + '.lock')
            file_result = file_job.parent / (file_job.stem + '.result')
            if not file_lock.is_file():
                continue
            try:
                last_modify_time = file_lock.stat().st_mtime
                if time.time() - last_modify_time > lock_file_expire_time_in_second:
                    file_lock.unlink()
                fd = os.open(file_lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
            except:
                continue

            self._run_a_job(file_job, file_result, file_lock)
        return 0

    def _run_a_job(self, file_job, file_result, file_lock):
        # Create a thread to keep the lock alive
        self._thread = Thread(target=self.keep_lock_alive, args=(file_lock,))
        self._thread.start()
        try:
            # Run the job
            with file_job.open('rb') as f:
                jobs_list = cloudpickle.loads(f.read())
                print(f'Run job {file_job.name}', self._job_func.__name__, str(jobs_list[0][0])[:30], str(jobs_list[0][1])[:30])
                result_list = []
                for args, kwargs in jobs_list:
                    result = self._job_func(*args, **kwargs)
                    result_list.append(result)
                # Write result to file
                with file_result.open('wb') as f:
                    f.write(cloudpickle.dumps(result_list))
            # Remove the lock file
            file_lock.unlink()
        except Exception as e:
            # Write error to file
            file_error = file_job.parent / (file_job.stem + '.error')
            with file_error.open('wt') as f:
                f.write(str(e))

        # Release the lock file
        self._queue_single.put(1)
        self._thread.join()

    def keep_lock_alive(self, filename, check_interval_in_second=10):
        while True:
            try:
                data = self._queue_single.get(timeout=check_interval_in_second)
                if data == 1:
                    break
            except:
                pass
            with filename.open('wt') as f:
                f.write(str(time.time()))

    def load_function(self):
        try:
            return cloudpickle.load(open(self._path_for_job_storage/"function.pkl", 'rb'))
        except AttributeError as e:
            module_name = re.findall(r"module\ '(.+)'\ from", str(e))[0]
            # print(f"Error in loading module {module_name}, please add the '{module_name}' module when initializing the JobWorker.")
            print(module_name)


if __name__ == '__main__':
    # worker = JobWorker("/share/fiehnlab/users/yli/jobs")
    # worker.work()
    if len(sys.argv) > 1:
        work_path = sys.argv[1]
        worker = JobWorker(work_path)
        worker.work()
    else:
        print(f"Usage: {sys.argv[0]} <path_to_job_storage>")

from job_scheduler import JobScheduler
import numpy as np


def test_host():
    test_host = JobScheduler(func_run=lambda a, b: np.add(a, b))
    test_host.add_job((1, 2))
    test_host.add_job((3, 4))
    test_host.add_job((5, 6))
    test_host.add_job((7, 8))
    test_host.add_job((9, 10))
    all_result = test_host.get_result()
    print(all_result)
    return 1


if __name__ == '__main__':
    test_host()

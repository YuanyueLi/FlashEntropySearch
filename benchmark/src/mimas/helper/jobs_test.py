from jobs import Jobs
import numpy as np


def test_host():
    test_host = Jobs(path_for_job_storage="/tmp/tmp_job", func_run=lambda a, b: np.add(a, b))
    test_host.add_job(1, b=2)
    test_host.add_job(1, 2)
    test_host.add_job(1, 2)
    test_host.add_job(1, 2)
    test_host.add_job(1, 2)
    all_result = test_host.get_result()
    print(all_result)
    return 1


if __name__ == '__main__':
    test_host()

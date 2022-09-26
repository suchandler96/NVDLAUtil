import os
import sys
from multiprocessing import Pool
import random
import time


def run_test(i, tests):
    print("running test %d of %d" % (i + 1, len(tests)))
    os.system(tests[i])
    print("finish test %d of %d" % (i + 1, len(tests)))


if __name__ == '__main__':
    with open(sys.argv[1]) as fp:
        lines = fp.readlines()

    tests = []
    for line in lines:
        if "gem5.opt" in line:
            tests.append(line.strip())

    os.chdir("/home/lactose/gem5-rtl/")

    for i in range(len(tests)):
        words = tests[i].split()
        idx_of_d = words.index("-d")
        if not os.path.exists(words[idx_of_d + 1]):
            os.mkdir(words[idx_of_d + 1])
        if "tee" not in tests[i]:
            tests[i] += " | tee " + words[idx_of_d + 1] + "/system.log"

    # print(tests)
    pool = Pool(12)
    print(len(tests))
    for i in range(len(tests)):
        pool.apply_async(run_test, args=(i, tests))

    print("waiting...")

    pool.close()
    pool.join()

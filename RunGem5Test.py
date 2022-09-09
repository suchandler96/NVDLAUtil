import os
import sys
from multiprocessing import Pool


def run_test(i, tests):
    print("running test %d of %d" % i + 1, len(tests))
    os.system(tests[i])
    print("finish test %d of %d" % i + 1, len(tests))


if __name__ == '__main__':
    with open(sys.argv[1]) as fp:
        lines = fp.readlines()

    tests = []
    for line in lines:
        if "gem5.opt" in line:
            tests.append(line)

    os.chdir("/home/lactose/gem5-rtl/")

    for test in tests:
        words = test.split()
        idx_of_d = words.index("-d")
        os.mkdir(words[idx_of_d + 1])

        if "tee" not in test:
            test += " | tee " + words[idx_of_d + 1] + "/system.log"

    pool = Pool(processes=12)
    for i in range(len(tests)):
        pool.apply_async(run_test, (i, tests))

    pool.close()
    pool.join()

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    with open(sys.argv[1]) as fp:
        lines = fp.readlines()

    addr2time_mapping = dict()

    return_time_counter = Counter()

    for line in lines:
        if line[0] != '(':
            continue

        if "DBB: read request from dla" in line:
            pos_of_right_bracket = line.find(')')
            pos_of_addr = line.find("addr ") + 5
            addr = int(line[pos_of_addr:pos_of_addr + 8], 16)

            cycle = int(line[1:pos_of_right_bracket])
            print(line)
            assert addr not in addr2time_mapping.keys()
            addr2time_mapping[addr] = cycle
            continue

        if "returned by gem5" in line:
            pos_of_right_bracket = line.find(')')
            cycle = int(line[1:pos_of_right_bracket])

            if "0x" in line:
                pos_of_addr = line.find("0x") + 2
            else:
                pos_of_addr = line.find("addr ") + 5

            addr = int(line[pos_of_addr:pos_of_addr + 8], 16)

            if addr not in addr2time_mapping.keys():
                continue
                # this happens sometimes in spm
            issue_cycle = addr2time_mapping[addr]
            addr2time_mapping.pop(addr)
            assert(issue_cycle != 0)

            real_cycles = (cycle - issue_cycle) // 2

            return_time_counter[real_cycles] += 1

    x_coords = []
    y_coords = []

    sorted_items = sorted(return_time_counter.items(), key=lambda x: x[0])

    for key, value in sorted_items:
        x_coords.append(key)
        y_coords.append(value)

    total_y = sum(y_coords)
    y_coords = [y / total_y for y in y_coords]

    plt.bar(x_coords, y_coords)

    plt.title("Read Request Response Time Distribution of " + sys.argv[1].split('/')[-2], fontdict={'fontsize': 16})
    plt.xlabel("Cycles", fontdict={'fontsize': 14})
    plt.ylabel("Frequency", fontdict={'fontsize': 14})
    # plt.rcParams['figure.figsize'] = (6.0, 3.0)

    # plt.savefig(sys.argv[1].replace("system.log", "read_request_response_time_distribution.png"), dpi=300)
    plt.show()

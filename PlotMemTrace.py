import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        lines = f.readlines()

    addresses = [int(line, 16) for line in lines]

    x_coords = np.arange(0, len(addresses))

    fig = plt.figure()
    ax = fig.gca()
    plt.scatter(x_coords, addresses, s=1)

    ylabels = map(lambda t: '0x%08x' % int(t), ax.get_yticks())
    ax.set_yticklabels(ylabels)

    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import helper as hp
import os

if __name__ == '__main__':

    fname = 'log.txt'
    with open(fname) as r:
        ldr = []
        ldf = []
        lds = []
        g = []
        title = r.readline()
        while title and title!='\n':
            num = float((r.readline().split('[')[1].split()[0]))
            ldr.append(num)
            num = float((r.readline().split('[')[1].split()[0]))
            ldf.append(num)
            ds = float(r.readline())
            num = float((r.readline().split('[')[1].split()[0]))
            g.append(num)
            title = r.readline()

        plt.plot(ldr, color='r')
        plt.plot(ldf, color='g')
        plt.plot(g, color='b')
        plt.show()
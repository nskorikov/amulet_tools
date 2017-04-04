from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
import re


def main():

    v, e = read_data('energy.dat')
    vols = np.array(v)
    energies = np.array(e)

    eos = birch_murnaghan
    x0 = np.array([energies.mean(), 0.1, 0.1, vols.mean()])  # initial guess of parameters
    plsq = leastsq(cost, x0, args=(energies, vols, eos))
    print('Fitted parameters = {0}'.format(plsq[0]))

    eos = murnaghan
    x0 = np.array([energies.mean(), 0.1, 0.1, vols.mean()])  # initial guess of parameters
    plsq = leastsq(cost, x0, args=(energies, vols, eos))
    print('Fitted parameters = {0}'.format(plsq[0]))

    plt.plot(vols, energies, 'ro')

    # plot the fitted curve on top
    x = np.linspace(min(vols), max(vols), 50)
    y = eos(plsq[0], x)
    plt.plot(x, y, 'k-')
    plt.xlabel('Volume')
    plt.ylabel('Energy')
    plt.savefig('nonlinear-curve-fitting1.png')


def murnaghan(parameters, vol):
    """From Phys. Rev. B 28, 5480 (1983)"""
    e0, b0, b_p, v0 = parameters
    e = e0 + b0 * vol / b_p * (((v0 / vol)**b_p) / (b_p - 1) + 1) - v0 * b0 / (b_p - 1.0)
    return e


def birch_murnaghan(parameters, vol):
    """Birch-Murnaghan EOS"""
    e0, b0, b_p, v0 = parameters
    v0v = v0/vol
    e = e0 + 9*b0*v0*((v0v**(2/3)-1)**3*b_p + (v0v**(2/3)-1)**2 * (6-4*v0v**(2/3)))/16
    return e


def cost(param, y, x, eos):
    """Cost function"""
    err = y - eos(param, x)
    return err


def read_data(fname):
    e = []
    v = []
    with open(fname, 'r') as f:
        for line in f:
            if not re.match(r'^#', line):
                tmp = line.split()
                if len(tmp) > 1:
                    v.append(float(tmp[0].strip()))
                    e.append(float(tmp[1].strip()))
    # for vv, ee in zip(V,E):
    #     print(vv,ee)
    return v, e

if __name__ == "__main__":
    main()

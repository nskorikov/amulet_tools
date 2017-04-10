from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
import re

# Bohr in SI units
bohr_si = 0.52917720859e-10
# electron mass in SI units
emass_si = 9.10938215e-31
# atomic unit of time in SI units
autime_si = 2.418884326505e-17
# atomic pressure unit in GPa
aupress_gpa = 1.e-9 * emass_si / (bohr_si * autime_si**2)


def main():
    ha2ev = 27.21138505
    ae2aa = 0.52917721092
    ae2aa3 = 0.148184711

    v, e = read_data('energy.dat')
    vols = np.array(v)
    energies = np.array(e)

    eos = birch_murnaghan
    x0 = np.array([energies.mean(), 0.1, 0.1, vols.mean()])  # initial guess of parameters
    plsq = leastsq(cost, x0, args=(energies, vols, eos))
    # print('Fitted parameters = {0}'.format(plsq[0]))
    V0 = plsq[0][3]
    B0 = plsq[0][1]
    a0 = V0**(1/3) * ae2aa
    print('V0       =  {:10.7f}'.format(V0))
    print('a0       =  {:10.7f}'.format(a0))
    print('E0       =  {:10.7f}'.format(plsq[0][0]))
    print('B0       =  ', B0)
    print('B0(GPa)  =  ', B0 * aupress_gpa)
    print('B1       =  {:10.7f}'.format(plsq[0][2]))

    # eos = murnaghan
    # x0 = np.array([energies.mean(), 0.1, 0.1, vols.mean()])  # initial guess of parameters
    # plsq = leastsq(cost, x0, args=(energies, vols, eos))
    # print('Fitted parameters = {0}'.format(plsq[0]))

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
    E = e0 + b0 * vol / b_p * (((v0 / vol)**b_p) / (b_p - 1) + 1) - v0 * b0 / (b_p - 1.0)
    return E


def birch_murnaghan(parameters, vol):
    """Birch-Murnaghan EOS"""
    e0, b0, b_p, v0 = parameters
    v0v = v0/vol
    e = e0 + 9/16*b0*v0*((v0v**(2/3)-1)**3*b_p + (v0v**(2/3)-1)**2 * (6-4*v0v**(2/3)))
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

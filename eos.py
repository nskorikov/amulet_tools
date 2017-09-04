from scipy.optimize import curve_fit
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
# hartree in eV units
ha2ev = 27.21138505
# Bohr in \AA units
ae2aa = 0.52917721092
# Bohr^3 in \AA^3 units
ae2aa3 = 0.148184711


def main():

    v, e, de = read_data('energy1.dat')
    vols = np.array(v)
    energies = np.array(e)

    # eos = murnaghan
    eos = birch_murnaghan
    x0 = np.array([energies.mean(), 1.1, 1.1, vols.mean()])  # initial parameters
    popt, pcov = curve_fit(func, xdata=vols, ydata=energies, sigma=de)
    err = np.sqrt(np.diag(pcov))
    # popt = leastsq(cost, x0, args=(energies, vols, eos))
    # if popt[1]:
    #     write_data(eos, popt, energies, vols, de)
    # else:
    #     print('Solution was not found')
    write_data(eos, popt, err, energies, vols, de)

def murnaghan(parameters, vol):
    """
    EOS from Phys. Rev. B 28, 5480 (1983)
    """
    e0, b0, b_p, v0 = parameters
    e = e0 + b0 * vol / b_p * (((v0 / vol)**b_p) / (b_p - 1) + 1) - v0 * b0 / (b_p - 1.0)
    return e


def birch_murnaghan(parameters, vol):
    """
    Birch-Murnaghan EOS
    """
    e0, b0, b_p, v0 = parameters
    v0v = v0/vol
    e = e0 + 9/16*b0*v0*((v0v**(2/3)-1)**3*b_p + (v0v**(2/3)-1)**2 * (6-4*v0v**(2/3)))
    return e


def func(vol, e0, b0, b_p, v0):
    v0v = v0/vol
    e = e0 + 9/16*b0*v0*((v0v**(2/3)-1)**3*b_p + (v0v**(2/3)-1)**2 * (6-4*v0v**(2/3)))
    return e


def cost(param, y, x, eos):
    """
    Cost function
    """
    err = y - eos(param, x)
    return err


def deviation(eos, param, y, x):
    """
    :param eos:  function for calculation
    :param param:  array of functions parameters
    :param x, y:  arrays of data points
    :return:  value of standard deviation
    """
    return np.sum(np.square(cost(param, y, x, eos)))/len(x)


def read_data(fname):
    a = []
    e = []
    edft = []
    edmft = []
    de = []
    v = []
    with open(fname, 'r') as f:
        for line in f:
            if not re.match(r'^#', line):
                tmp = line.split()
                if len(tmp) > 1:
                    a.append(float(tmp[0].strip()))
                    v.append(float(tmp[1].strip()) * ae2aa3)
                    edft.append(float(tmp[2].strip()) * ha2ev)
                    edmft.append(float(tmp[3].strip()))
                    de.append(float(tmp[4].strip()))
    # TODO: add some automatic for this shift. But the shift should be
    # conformed between different phases of one system.
    # emin = np.min(edft)    # this break consistence between phases
    # edft -= emin
    for i in range(len(edft)):
        edft[i] -= -138538.444909219
    print('Zero of energy is shifted to -138538.444909219 eV')

    for e1, e2 in zip(edft, edmft):
        e.append(e1+e2)

    # for vv, ee in zip(V,E):
    #     print(vv,ee)
    return v, e, de


def write_data(eos, results, err, e, v, de):

    v0 = results[3]
    dv0 = err[3]
    b0 = results[1]
    db0 = err[1]
    a0 = v0**(1/3)

    x = np.linspace(min(36.0, min(v)), max(max(v), 50.0), 50)
    y = eos(results, x)

    print('V0(AA^3) =  {:10.7f}+/-{:10.7f}'.format(v0, dv0))
    print('a0       =  {:10.7f}'.format(a0))
    print('E0(eV)   =  {:10.7f}+/-{:10.7f}'.format(results[0], err[0]))
    print('B0       =  {:10.7f}+/-{:10.7f}'.format(b0, db0))
    print('B0(Mbar) =  {:10.7f}'.format(b0 * 1.6021765))
    print('B1       =  {:10.7f}+/-{:10.7f}'.format(results[2], err[2]))

    with open('fit.dat', 'w') as f:
        f.write('# STD: {:10.7f}\n'.format(deviation(eos, results, e, v)))
        f.write('#V0(AA^3) =  {:10.7f}+/-{:10.7f}\n'.format(v0, dv0))
        f.write('# a0       =  {:10.7f}\n'.format(a0))
        f.write('# E0(eV)   =  {:10.7f}+/-{:10.7f}\n'.format(results[0], err[0]))
        f.write('# B0       =  {:10.7f}+/-{:10.7f}\n'.format(b0, db0))
        f.write('# B0(Mbar) =  {:10.7f}\n'.format(b0 * 1.6021765))
        f.write('# B1       =  {:10.7f}+/-{:10.7f}\n'.format(results[2], err[2]))
        f.write('#\n')
        f.write('# Initial points:\n')
        for vi, ei, dei in zip(v, e, de):
            f.write('{0:8.4f}{1:8.4f}{2:8.4f}\n'.format(vi, ei, dei))
        f.write('\n\n')
        f.write('# Fitted data:\n')
        for xi, yi in zip(x, y):
            f.write('{0:8.4f}{1:8.4f}\n'.format(xi, yi))

    # plot_picture(x, y, v, e, de)
    plt.errorbar(v, e, yerr=de, fmt='.')
    plt.text(40, -3.5, 'a0       =  {:10.7f}'.format(a0))
    plt.text(40, -3.7, 'E0(eV)   =  {:10.7f}+/-{:10.7f}'.format(results[0],
             deviation(eos, results, e, v)))
    plt.text(40, -3.9, 'B0(Mbar) =  {:10.7f}'.format(b0 * 1.6021765))
    plt.text(40, -4.1, 'B1       =  {:10.7f}'.format(results[2]))
    plt.plot(x, y, 'k-')
    plt.xlabel('Volume, AA^3')
    plt.ylabel('Energy, eV')
    plt.savefig('nonlinear-curve-fitting1.eps')


if __name__ == "__main__":
    main()

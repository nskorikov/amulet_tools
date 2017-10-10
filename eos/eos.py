from scipy.optimize import curve_fit, OptimizeWarning
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

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

    data_file, eos, zero_e = handle_commandline()

    fit1 = EosFit(data_file, eos, zero_e)
    fit1.fit()
    fit1.write_data()


def handle_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='energy1.dat',
                        help='Name of file with input data\n'
                        '[default: %(default)s]')
    parser.add_argument('-eos', default='birch_murnaghan',
                        help='Kind of EOS for fit, for now available Murnaghan '
                             '(Phys. Rev. B 28, 5480 (1983) and Birch_Murnaghan equations\n'
                             '[default: %(default)s]')
    parser.add_argument('-zero_e', default=-138538.444909219,
                        help='Shift initial energies closer to zero\n'
                             '[default: %(default)s]. Does not ask why so.')
    args = parser.parse_args()
    return args.f, args.eos.lower(), args.zero_e


class EosFit:

    def __init__(self, infile, eos, zeroenergy=0.0):
        if eos == 'murnaghan':
            self.eos = self.murnaghan
        elif eos == 'birch_murnaghan':
            self.eos = self.birch_murnaghan
        else:
            raise NameError('Undefined EOS equation')

        self.eos_parameters = np.zeros(4, np.float)
        self.zeroenergy = zeroenergy
        v, e, de = self.read_data(infile)
        self.vols = np.array(v)
        self.energies = np.array(e)
        self.sigma = np.array(de)
        self.errors = np.zeros(4, np.float)

    def fit(self):
        p0 = np.array([self.energies.mean(), 1.1, 1.1, self.vols.mean()])
        try:
            self.eos_parameters, pcov = curve_fit(self.eos, method='lm', p0=p0, xdata=self.vols,
                                                  ydata=self.energies, sigma=self.sigma)
        except RuntimeError:
            print('The least-squares minimization fails')
        except OptimizeWarning:
            print('Covariance of the parameters can not be estimated')
        except ValueError:
            print('Something wrong with input points')
        else:
            self.errors = np.sqrt(np.diag(pcov))

    @staticmethod
    def murnaghan(vol, e0, b0, b1, v0):
        """
        EOS from Phys. Rev. B 28, 5480 (1983)
        """
        return e0 + b0 * vol / b1 * (((v0 / vol) ** b1) / (b1 - 1) + 1) - v0 * b0 / (b1 - 1.0)

    @staticmethod
    def birch_murnaghan(vol, e0, b0, b1, v0):
        """
        Birch-Murnaghan EOS
        """
        v0v = v0/vol
        return e0 + 9/16*b0*v0*((v0v**(2/3)-1) ** 3 * b1 + (v0v ** (2 / 3) - 1) ** 2 * (6 - 4 * v0v ** (2 / 3)))

    def wrap_eos(self, x):
        e0 = self.eos_parameters[0]
        b0 = self.eos_parameters[1]
        b1 = self.eos_parameters[2]
        v0 = self.eos_parameters[3]
        return self.eos(x, e0, b0, b1, v0)

    def read_data(self, fname):
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
                    if len(tmp) > 5:
                        a.append(float(tmp[0].strip()))
                        v.append(float(tmp[1].strip()) * ae2aa3)
                        edft.append(float(tmp[2].strip()) * ha2ev)
                        edmft.append(float(tmp[3].strip()))
                        de.append(float(tmp[4].strip()))
        if len(a) < 4:
            print('\n Warning! There are less than 4 points for fitting of 4 parameters!')
            print('Most likely the results of the fit will be bad.')
        elif len(a) < 6:
            print('\n Warning! There are less than 6 points for fitting of 4 parameters!\n')
        else:
            print('There are {:3d} input points for fit'.format(len(a)))

        # TODO: add some automatic for this shift. But the shift should be
        # conformed between different phases of one system.
        # for i in range(len(edft)):
        #     edft[i] -= self.zeroenergy

        print('Zero of energy is shifted to {:17.9f} eV\n'.format(self.zeroenergy))

        for e1, e2 in zip(edft, edmft):
            e.append(e1+e2-self.zeroenergy)

        # for vv, ee in zip(V,E):
        #     print(vv,ee)
        return v, e, de

    def write_data(self):
        v0 = self.eos_parameters[3]
        dv0 = self.errors[3]
        b0 = self.eos_parameters[1]
        db0 = self.errors[1]
        a0 = v0**(1.0/3.0)
        # \delta a=da/dv * \delta V
        da0 = v0**(-2/3)*dv0/3

        # TODO: fix plot interval
        x = np.linspace(min(36.0, min(self.vols)), max(max(self.vols), 50.0), 50)
        y = self.wrap_eos(x)

        print('V0(AA^3) =  {:10.7f} +/-{:10.7f}'.format(v0, dv0))
        print('a0       =  {:10.7f} +/-{:10.7f}'.format(a0, da0))
        print('E0(eV)   =  {:10.7f} +/-{:10.7f}'.format(self.eos_parameters[0], self.errors[0]))
        print('B0       =  {:10.7f} +/-{:10.7f}'.format(b0, db0))
        print('B0(Mbar) =  {:10.7f}'.format(b0 * 1.6021765))
        print('B1       =  {:10.7f} +/-{:10.7f}'.format(self.eos_parameters[2], self.errors[2]))

        with open('fit.dat', 'w') as f:
            f.write('# STD: {:10.7f}\n'.format(self.deviation()))
            f.write('#V0(AA^3) =  {:10.7f} +/-{:10.7f}\n'.format(v0, dv0))
            f.write('# a0       =  {:10.7f} +/-{:10.7f}\n'.format(a0, da0))
            f.write('# E0(eV)   =  {:10.7f} +/-{:10.7f}\n'.format(self.eos_parameters[0],
                                                                  self.errors[0]))
            f.write('# B0       =  {:10.7f} +/-{:10.7f}\n'.format(b0, db0))
            f.write('# B0(Mbar) =  {:10.7f}\n'.format(b0 * 1.6021765))
            f.write('# B1       =  {:10.7f} +/-{:10.7f}\n'.format(self.eos_parameters[2],
                                                                  self.errors[2]))
            f.write('#\n')
            f.write('# Initial points:\n')
            for vi, ei, dei in zip(self.vols, self.energies, self.sigma):
                f.write('{0:8.4f}{1:8.4f}{2:8.4f}\n'.format(vi, ei, dei))
            f.write('\n\n')
            f.write('# Fitted data:\n')
            for xi, yi in zip(x, y):
                f.write('{0:8.4f}{1:8.4f}\n'.format(xi, yi))

        plt.errorbar(self.vols, self.energies, yerr=self.sigma, fmt='.')
        plt.plot(x, y, 'k-')

        axis = plt.axis()

        x_txt = axis[0]+5

        dy = (axis[3]-axis[2])/15
        y_txt = axis[3]-dy

        plt.text(x_txt, y_txt, 'a0      =  {:10.7f}+/-{:10.7f}'.format(a0, da0))
        y_txt -= dy

        plt.text(x_txt, y_txt, 'E0(eV)  =  {:10.7f}+/-{:10.7f}'.format(self.eos_parameters[0],
                                                                       self.deviation()))
        y_txt -= dy

        plt.text(x_txt, y_txt, 'B0(Mbar)=  {:10.7f}+/-{:10.7f}'.format(b0 * 1.6021765,
                                                                       db0 * 1.6021765))
        y_txt -= dy

        plt.text(x_txt, y_txt, 'B1      =  {:10.7f}+/-{:10.7f}'.format(self.eos_parameters[2],
                                                                       self.errors[2]))
        plt.xlabel('Volume, AA^3')
        plt.ylabel('Energy, eV')
        plt.savefig('nonlinear-curve-fitting1.eps')

    def cost(self):
        """
        Difference between initial points and points of the fitted curve
        """
        err = self.energies-self.wrap_eos(self.vols)
        return err

    def deviation(self):
        """
        Standard deviation
        """
        return np.sum(np.square(self.cost()))/(len(self.vols)-1)


if __name__ == "__main__":
    main()

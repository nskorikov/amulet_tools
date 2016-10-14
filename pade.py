#!/usr/bin/env python
# coding=utf-8
# Program make analitical continuation of complex function defined on Matsubara
# frequency to real energy using Pade approximant. Universal version.
import time
import os.path
import argparse
import sys
import random
from math import ceil

debug = True


def main():
    print('Start at %s ' % time.ctime())
    start_time = time.time()
    inp = pade_input()          # parsing of command line
    p = pade_stuff(inp)         # reading of sigma, preparation of sets,
                                # selection of version keep it all in p
    p.do_pade()                 # calc huge array of pade coefficient for sets
                                # all non-physical solution are dropped

    # results, deltas, solutions = do_pade(p)


def analitical(f):
    analitical = True
    for ff in f:
        if ff.imag > 0.0:
            analitical = False
            break
    return analitical


def residual(f1, f2):
    """
    Return normed difference between two complex functions of equal length
    \Delta=\sqrt( \Sum( (f1_i-f2_i)^2 ) )/N, where N -- length of functions.
    """
    l1 = len(f1)
    if l1 != len(f2):
        print('WARNING: calc_residual')
        print('Lengths of f1 and f2 are different!\n')
    d = sum([abs(f1[i] - f2[i]) for i in range(l1)])
    d /= l1
    return d


class pade_stuff():
    """
    Class sets links to used subroutins in dependance of input parameters
    (random, use_moments, mlib etc.). Additionaly it reads input function
    from disk and sets type of arrays in acordance with mlib.
    """

    def __init__(self, inp):
        tmp = self.prepare_pade(inp)
        self.sets = tmp[0]
        self.p_c = tmp[1]
        self.pade = tmp[2]
        self.points = tmp[3]
        self.iw = tmp[4]
        self.f = tmp[5]
        self.e = tmp[6]
        self.sigre = []
        self.sigim = []
        self.delta = []

    def prepare_pade(self, ii):
        sets = []
        for ipo in range(ii.npo[0], ii.npo[1], 1):
            for ine in range(ii.ne[0], ii.ne[1]):
                if (ine + ipo) % 2 != 0:
                    continue
                if ii.ls:
                    nls = ipo // 4
                else:
                    nls = 1
                for ils in range(0, nls, 2):
                    for irand in range(0, ii.nrandomcycle):
                        set = [ipo, ine, ils, irand]
                        sets.append(set)

        iw, sigma = self.readsigma(ii.infile)
        emesh = self.make_e_mesh(ii.emin, ii.de, ii.npts)
        pade, pade_coef, points= self.choose_version(ii)
        return sets, pade_coef, pade, points, iw, sigma, emesh

    def do_pade(self):
        for s in self.sets:
            iw1, f1 = self.points(s[0], s[1])
            pq, success, solver = self.make_coef(s, iw1, f1)
            gr = self.pade(pq, self.e)
            if not analitical(gr):
                print(len(self.sets))
                self.sets.remove(s)
                print(len(self.sets))
                print('Set', s,'gives not analitical solution\n')
#                 print(self.sets, '\n')
                continue
            gi = self.pade(pq, self.iw)
            if not analitical(gi):
                print('Set', s,'gives not analitical solution\n')
                print(len(self.sets))
                self.sets.remove(s)
                print(len(self.sets))
                continue
            self.sigre.append(gr)
            self.sigim.append(gi)
            self.delta.append(residual(gi, self.f))
        print(len(self.sets), len(self.sigre), len(self.sigim))
        for d in self.delta:
            print(d)

    def make_coef(self, s, iw1, f1):
        m = len(iw1)
        r = (s[0] + s[1] - s[2]) // 2
        a = []
        b = []
        for i in range(0, m):
            b.append(f1[i] * iw1[i] ** r)
            aa = []
            for j in range(0, r):
                aa.append(iw1[i] ** j)
#             a.append(aa)
#             aa = []
            for j in range(r, 2 * r):
                aa.append(-f1[i] * iw1[i] ** (j - r))
            a.append(aa)
                
        return self.p_c(a, b)
#         pq, success, used_solver = self.p_c(a, b)

    def choose_version(self, ii):
        if ii.mlib == 'numpy':
            if ii.use_moments:
                pade = self.pade_n_m
            else:
                pade = self.pade_n
            if ii.ls:
                pade_coef = self.pade_coef_n_ls
            else:
                pade_coef = self.pade_coef_n
        elif ii.mlib == 'mpmath':
            if ii.use_moments:
                pade = self.pade_m_m
            else:
                pade = self.pade_m
            if ii.ls:
                pade_coef = self.pade_coef_m_ls
            else:
                pade_coef = self.pade_coef_m
        else:
            raise('mlib != numpy and mlib != numpy')
        if ii.random:
            points = self.choose_prandom_points
        else:
            points = self.choose_seq_points
        return pade, pade_coef, points

    def readsigma(self, filename):
        """
        Read sigma in format of AMULET.
        """
        print('Input file contains:')
        with open(filename, 'r') as f:
            data = f.readlines()

        nlines = len(data)

        # Count pairs of blank lines to determine number of datasets.
        # Each dataset should be ended by 2 blank lines.
        ndatasets = 0
        for i in range(0, nlines - 1):
            if not data[i].split() and not data[i + 1].split():
                ndatasets += 1
                blank_lines_ending = 2
        # If there are no blank lines at end of file
        if ndatasets == 0 and not data[-1].split():
            ndatasets += 1
            blank_lines_ending = 1
            nlinesperblock = (nlines + 1) // ndatasets
        elif ndatasets == 0 and data[-1].split():
            ndatasets += 1
            nlinesperblock = (nlines + 2) // ndatasets
        else:
            nlinesperblock = nlines // ndatasets
        print(" Number of datasets: %i " % ndatasets)

        # nlinesperblock = nlines / ndatasets
        print(" Number of lines per block: %i " % nlinesperblock)

        # Take the last dataset from data file
        data = data[nlinesperblock * (ndatasets - 1): 
                    (nlinesperblock * ndatasets) - blank_lines_ending]
        nlines = len(data)
        s = data[0].split()

        # Structure of Sigma file:
        # first column -- energy, the next two column are Re and Im parts of
        # sigma majority last two column are Re and Im parts of sigma minority
        # (if calc is spin-polarized)
        l = len(s)
        if l == 3:
            e = []
            z = []
            for i in range(0, nlines):
                s = data[i].split()
                e.append(1j * float(s[0]))
                z.append(float(s[1]) + 1j * float(s[2]))
        elif l == 5:
            e = []
            z = []
            for i in range(0, nlines):
                s = data[i].split()
                e.append(1j * float(s[0]))
                z.append([float(s[1]) + 1j * float(s[2]),
                          float(s[3]) + 1j * float(s[4])])
        else:
            print("unknown data format")

        return e, z

    def make_e_mesh(self, t, d, n):
        return [t + i * d + 1j * 0.01 for i in range(n)]

    def pade_n(self, coef, e):
        """
        Calculation of analitical function on a arbitrary mesh for a given
         Pade coefficient.
        """
        if debug: 
            print('pade_n')
        nlines = len(e)
        r = len(coef) // 2
        f = np.zeros(nlines, dtype=np.complex128)
        pq = np.ones(r * 2 + 1, dtype=np.complex128)
        for i in range(0, r):
            pq[i] = coef[i]
            pq[i + r] = coef[i + r]
        for iw in range(0, nlines):
            p = np.complex128(0.0)
            q = np.complex128(0.0)
            for i in range(0, r):
                p += pq[i] * e[iw] ** i
            for i in range(0, r + 1):
                q += pq[i + r] * e[iw] ** i
            f[iw] = np.divide(p, q)
        return f.tolist()

    def pade_n_m(self):
        if debug: 
            print('pade_n_m')
        pass

    def pade_m(self):
        if debug:
            print('pade_m')
        pass

    def pade_m_m(self):
        if debug:
            print('pade_m_m')
        pass

    def pade_coef_m(self, a, b):
        """
        Subroutine pade_ls_coeficients() finds coefficient of Pade approximant
         solving equation aX=b. The general mpmath.inverse() routine used
         for inversion of matrix 'a'. In this version, number of 
         coefficients is equal to number of complex points where the function
         is defined 
        """
        if debug:
            print('pade_coef_m')
        solver = 'Mpmath inverse'
        try:
            a = mp.inverse(a)
        except ZeroDivisionError as err:
            if 'matrix is numerically singular' in err.message:
                pq = 123456.7
                success = False
            else:
                raise
        else:
            x = a * b
            success = True
        return x, success, solver

    def pade_coef_m_ls(self, a, b):
        """
        Subroutine pade_ls_coeficients() finds coefficient of Pade approximant
         solving equation aX=b. The Least Squares method using 
         mpmath.lu_solve() or mpmath.qr_solve() is utilized. In this version,
         number of coefficients is less then number of complex points where
         the function is defined 
        """
        if debug:
            print('pade_coef_m_ls')
        success = True
        solver = 'Mpmath LU solver'
        try:
            x = mp.lu_solve(a, b)
        except (ZeroDivisionError, ValueError):
            # if 'matrix is numerically singular' in err.message:
            try:
                x = mp.qr_solve(a, b)
            # success = True
            except ValueError:
                # if 'matrix is numerically singular' in err.message:
                success = False
                x = 123456.7
                # else:
                #     raise
            else:
                x = x[0]
                solver = 'Mpmath QR solver'
        if success is True:
            x.rows += 1
            x[n, 0] = mp.mpc(1, 0)
        return x, success, solver

    def pade_coef_n(self, a, b):
        """
        Subroutine pade_ls_coeficients() finds coefficient of Pade approximant
         solving equation aX=b. The general numpy.linalg.inv() routine used
         for inversion of matrix 'a'. In this version, number of 
         coefficients is equal to number of complex points where the function
         is defined 
        """
        if debug:
            print('pade_coef_n')
        # Solving the equation: aX=b, where
        #                       |p|
        #                   X = | |
        #                       |q|
        solver = 'Numpy linalg.inv'
        try:
            a = np.linalg.inv(a)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                x = 123456.7
                success = False
            else:
                raise
        else:
            x = np.dot(a, b)
            success = True
        return x, success, solver

    def pade_coef_n_ls(self, a, b):
        """
        Subroutine pade_ls_coeficients() finds coefficient of Pade approximant
         solving equation aX=b. The Least Squares method using 
         numpy.linalg.lstsq is utilized. In this version, number of 
         coefficients is less then number of complex points where the function
         is defined 
        """
        if debug:
            print('pade_coef_n_ls')
        # Solving the equation: aX=b, where
        #                       |p|
        #                   X = | |
        #                       |q|
        try:
            x = np.linalg.lstsq(a, b)[0]
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                x = 123456.7
                success = False
            else:
                raise
        else:
            success = True
        return x, success

    def choose_prandom_points(self, npos, nneg):
        """
        Subroutine selects from input nneg+npos points: first nneg+npos-nrnd
        points are selected sequently, then nrnd points are picked randomly.
        Number of randomly selected points nrnd is determined randomly in
        interval from 1/16 to 1/3 of total number of points.
        e -- input complex array with energy points
        f -- input complex array with values of function in points e[i]
        """
        e = self.iw
        f = self.f 
        if (nneg + npos) % 2 != 0:
            print('Number of chosen points should be even!',
                  nneg, npos, nneg + npos)
            npos += 1
        q = nneg + npos
        nrnd = random.randrange(q // 16, q // 3, 2)
        ee = []
        ff = []
        for i in range(0, nneg):
            ee.append(mp.conj(e[nneg - 1 - i]))
            ff.append(mp.conj(f[nneg - 1 - i]))
        for i in range(nneg, q - nrnd):
            ee.append(e[i - nneg])
            ff.append(f[i - nneg])
        # Make list of random points
        pp = random.sample(range(q - nrnd, len(e) - 1), nrnd)
        # Sort them
        pp.sort()
        # Fix repeated points (if there are present)
        for i in range(0, nrnd - 1):
            if pp[i] == pp[i + 1]:
                pp[i + 1] += 1
        # The last two points should be sequential to fix tail of F(z).
        if nrnd != 0:
            pp[nrnd - 1] = pp[nrnd - 2] + 1
        # Append selected points
        for i in range(0, nrnd):
            ee.append(e[pp[i]])
            ff.append(f[pp[i]])
        return ee, ff

    def choose_seq_points(self, npos, nneg):
        # Pick first nneg+npos points
        e = self.iw
        f = self.f 
        if (nneg + npos) % 2 != 0:
            print('Number of chosen points should be even!', nneg, npos, nneg + npos)
            npos += 1
        ee = []
        ff = []
        for i in range(0, nneg):
            ee.append(e.conj()[nneg - 1 - i])
            ff.append(f.conj()[nneg - 1 - i])
        for i in range(nneg, nneg + npos):
            ee.append(e[i - nneg])
            ff.append(f[i - nneg])
        return ee, ff


class pade_input():

    def __init__(self):
        commandline = self.handle_commandline()
        self.emin = commandline['emin']
        self.emax = commandline['emax']
        self.de = commandline['de']
        self.npts = commandline['npts']
        self.use_moments = commandline['use_moments']
        self.random = commandline['random']
        self.nrandomcycle = commandline['nrandomcycle']
        self.ls = commandline['ls']
        self.npo = commandline['npo']
        self.use_ne = commandline['use_ne']
        self.ne = commandline['ne']
        self.infile = os.path.abspath(commandline['f'])
        self.logfile = commandline['logfile']
        self.m = commandline['m']
        self.mlib = commandline['mlib']
        self.prec = commandline['prec']
        self.validate_input()
        self.select_mlib()
        self.print_input()

    def select_mlib(self):
        """
        Load appropriate mathematical module for matrix inversion and
        for another mathematical things.
        """
        if self.mlib == 'auto':
            try:
                from mpmath import mp, im, re, fdiv, mpc, fp, workdps
            except ImportError:
                print('There are no mpmath module')
                try:
                    import numpy as np
                except ImportError:
                    raise('The are no numpy too, please install something')
                else:
                    global np
            else:
                global mp, im, re, fdiv, mpc, fp, workdps
                fp.dps = 12
                mp.prec = self.prec
        if self.mlib == 'mpmath':
            global mp, im, re, fdiv, mpc, fp, workdps
            try:
                from mpmath import mp, im, re, fdiv, mpc, fp, workdps
            except ImportError:
                raise('Please install mpmath Python module or use '
                      'numpy as mlib')
            else:
                fp.dps = 12
                mp.prec = self.prec
        if self.mlib == 'numpy':
            global np
            try:
                import numpy as np
            except ImportError:
                raise('Please install numpy Python module or use '
                      'mpmath as mlib')

    def validate_input(self):
        if not self.random:
            self.nrandomcycle = 1
        if self.random and self.nrandomcycle == 1:
            print('You set switch "-random"')
            raise ValueError('In such case you should set "-nrandomcycle">1')
        if not self.use_ne:
            self.ne = (0, 1)
        if not os.path.exists(self.infile):
            raise TypeError('File %s does not exist' % self.infile)
        if self.emax < self.emin:
            raise ValueError('Incorrect input: emax is less then emin')
        if (self.emax - self.emin) / self.npts != self.de:
            print('Check parameters of real energy')
            print('Continuation will be performed to interval'
                  '[%5.2f,%5.2f] with step %4.3f' % (emin, emax, de))
            self.npts = (self.emax - self.emin) // self.de
        if self.use_moments and self.m == (0.0, 0.0, 0.0):
            print('You set switch "-use_moments"')
            raise ValueError(
                'Values of moments should be defined in commandline')
        if self.mlib not in ('mpmath', 'numpy', 'auto'):
            raise ValueError(
                'Set "-mlib" parameter to one of: mpmath, numpy, auto:'
                'or ommit it to use default')

    def print_input(self, direction='sys.stdout'):
        old = sys.stdout
        if direction != 'sys.stdout':
            print('direction=%s' % direction)
            sys.stdout = open(direction, 'a')
        if self.use_ne:
            print('The symmetry of Green function G(z*)=-G*(z) will be'
                  ' accounted')
            print('The number of negative points will be varied in interval:'
                  ' [%4i, %4i] ' % (self.ne[0], self.ne[1]))
        else:
            print('The symmetry of Green function will not be accounted')
        print('The number of positive points will be varied in interval:'
              '[%4i, %4i]' % (self.npo[0], self.npo[1]))
        if self.random:
            print('Some random points will be added to sequential set of'
                  ' points')
        if self.use_moments:
            print('Momenta of function will be accounted in continuation: '
                  '%5.2f %5.2f %5.2f' % self.m)
        if self.ls:
            print('Coefficients of Pade polinom will be finded by Least'
                  ' Squares method')
        print('Continuation will be performed to interval [%5.2f,%5.2f]'
              ' with step %4.3f' % (self.emin, self.emax, self.de))
        print('Function from %s will be continued to real axis' % self.infile)
        print('Log of execution will be duplicated to %s' % self.logfile)
        print('For mathematical tasks we will use "%s"' % self.mlib)
        if debug:
            ss = sys.modules
            if 'numpy' in ss:
                print('Numpy loaded')
            if 'mpmath' in ss:
                print('Mpmath loaded')
        sys.stdout.flush()
        sys.stdout = old

    def handle_commandline(self):
        """
        Method defines possible commandline keys and them default values,
        parse commandline and return dictionary 'inputdata'
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', default='g.dat',
                            help='file with input function '
                            '[default: %(default)s]')
        parser.add_argument('-logfile', default='sets.dat',
                            help='file for log, new data will be appended to'
                            ' existent [default: %(default)s]')
        parser.add_argument('-emin', default=-10.0, type=float,
                            help='minimum energy on real axis '
                            '[default: %(default)d]')
        parser.add_argument('-emax', default=10.0, type=float,
                            help='maximum energy on real axis ')
        parser.add_argument('-de', default=0.01, type=float,
                            help='energy step on real axis '
                            '[default: %(default)f]')
        parser.add_argument('-npts', type=int, default=2000,
                            help='number of points on real energy axis'
                            '[default: %(default)i]')
        parser.add_argument('-use_moments', action='store_true',
                            help='Use or not external information about '
                            'momenta [default: %(default)s]')
        parser.add_argument('-m', nargs=3, default=(0.0, 0.0, 0.0), type=float,
                            help='first momenta of function: m0, m1, m2')
        parser.add_argument('-pm', '--print_moments', action='store_true',
                            help='Print or not estimated values of momenta'
                            '[default: %(default)s]')
        parser.add_argument('-random', action='store_true',
                            help='Use or not randomly picked points in input '
                            'set [default: %(default)s]')
        parser.add_argument('-nrandomcycle', type=int, default=200,
                            help='number cycles with random points'
                            '[default: %(default)i]')
        parser.add_argument('-ls', action='store_true',
                            help='Use non-diagonal form of Pade coefficients '
                            'matrix [default: %(default)s]')
        parser.add_argument('-npo', nargs=2, default=(10, 90), type=int,
                            help='number of input iw points'
                            '[default: %(default)s]')
        parser.add_argument('-use_ne', action='store_true',
                            help='use symmetry of input function: '
                            'G(z^*)=G^*(z) [default: %(default)s]')
        parser.add_argument('-ne', nargs=2, default=(0, 5), type=int,
                            help='number of negative iw points'
                            '[default: %(default)s]')
        parser.add_argument('-mlib', default='numpy',
                            help='Choose mathematical library: numpy or mpmath'
                            ' or auto [default: %(default)s]')
        parser.add_argument('-precision', type=int, default=256,
                            help='precision of floating point numbers (in '
                            'bits) for use in mpmath [default: %(default)i]')
        parser.add_argument('-debug', default='False',
                            help='debug mode [default: %(default)s]')

        args = parser.parse_args()

        inputdata = {'emin': args.emin,
                     'emax': args.emax,
                     'de': args.de,
                     'npts': args.npts,
                     'f': args.f,
                     'use_moments': args.use_moments,
                     'm': args.m,
                     'pm': args.print_moments,
                     'random': args.random,
                     'ls': args.ls,
                     'npo': args.npo,
                     'use_ne': args.use_ne,
                     'ne': args.ne,
                     'nrandomcycle': args.nrandomcycle,
                     'logfile': args.logfile,
                     'mlib': args.mlib,
                     'prec': args.precision
                     }

        return inputdata


def calc_residual(f1, f2):
    l1 = len(f1)
    if l1 != len(f2):
        raise('WARNING: calc_residual: Lengths of f1 and f2 are different!\n')
    d = pow(sum([pow(f1[i] - f2[i], 2) for i in range(l1)]), 0.5)
    d /= l1
    return float(abs(d))

if __name__ == "__main__":
    main()

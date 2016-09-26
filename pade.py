#!/usr/bin/env python
# coding=utf-8
# Program make analitical continuation of complex function defined on Matsubara
# frequency to real energy using Pade approximant. Universal version.
import time
import os.path
import argparse
import sys

debug = True

def main():
    print('Start at %s ' % time.ctime())
    start_time = time.time()
    inp = pade_input()
    p = pade_stuff(inp)
    p.pade()
    p.p_c()
    p.make_coef()
    
    # results, deltas, solutions = do_pade(p)
    
    
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
        self.w = tmp[3]
        self.f = tmp[4]
        self.e = tmp[5]

    def make_coef(self):
        for s in self.sets:
            print(len(s), s)


    def prepare_pade(self, ii):
        sets = []
        for ipo in range(ii.npo[0], ii.npo[1], 1):
            for ine in range(ii.ne[0], ii.ne[1]):
                if ii.ls:
                    qq2 = ipo//4
                else:
                    qq2 = 1
                for q1 in range(0, qq2, 2):
                    for qq in range(0, ii.nrandomcycle):
                        if (ine + ipo) % 2 != 0:
                            continue
                        set = [ipo, ine, q1, qq]
                        sets.append(set)
    
        w, f = self.readsigma(ii.infile)
        e = self.make_e_mesh(ii.emin, ii.de, ii.npts)
        suffix_pade = ''
        suffix_pade_coef = ''
        if ii.ls:
            suffix_pade_coef += '_ls'
        if ii.mlib == 'numpy':
            w = np.array(w, dtype = np.complex128)
            f = np.array(f, dtype = np.complex128)
            if ii.use_moments:
                pade = self.pade_n_m
            else:
                pade = self.pade_n
            if ii.ls:
                pade_coef = self.pade_coef_n_ls
            else:
                pade_coef = self.pade_coef_n
        elif ii.mlib == 'mpmath':
            w = fp.matrix(w) 
            f = fp.matrix(f)
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
        return sets, pade_coef, pade, w, f, e
    
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
        data = data[nlinesperblock * (ndatasets - 1): (nlinesperblock * ndatasets) -
                                                      blank_lines_ending]
        nlines = len(data)
        s = data[0].split()
    
        # Structure of Sigma file:
        # first column -- energy, the next two column are Re and Im parts of sigma majority
        # last two column are Re and Im parts of sigma minority (if calc is spin-polarized)
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
                z.append([float(s[1]) + 1j * float(s[2]), float(s[3]) + 1j * float(s[4])])
        else:
            print("unknown data format")
    
        return e, z
    
    def make_e_mesh(self, t, d, n):
        return [t + i * d + 1j * 0.01 for i in range(n)]

    def pade_n(self):
        print('pade_n')
        pass
    
    def pade_n_m(self):
        print('pade_n_m')
        pass
    
    def pade_m(self):
        print('pade_m')
        pass
    
    def pade_m_m(self):
        print('pade_m_m')
        pass
    
    def pade_coef_m(self):
        print('pade_coef_m')
        pass
    
    def pade_coef_m_ls(self):
        print('pade_coef_m_ls')
        pass
    
    def pade_coef_n(self):
        """
        Subroutine pade_coeficients() finds coefficient of Pade approximant
        f - values of complex function for approximation
        e - complex points in which function f is determined
        """
        if debug:
            print('pade_coef_n')
        r = len(self.e) // 2
        s = np.zeros(2 * r, dtype=np.complex128)
        x = np.zeros((2 * r, 2 * r), dtype=np.complex128)
        for i in range(0, 2 * r):
            s[i] = f[i] * e[i] ** r
            for j in range(0, r):
                x[i, j] = e[i] ** j
            for j in range(r, 2 * r):
                x[i, j] = -f[i] * e[i] ** (j - r)
        # Solving the equation: |p|
        #                       | |=X^{-1}*s
        #                       |q|
        try:
            x = np.linalg.inv(x)
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                pq = 123456.7
                success = False
            else:
                raise
        else:
            pq = np.dot(x, s)
            success = True
        return pq, success

    def pade_coef_n_ls(self):
        if debug:
            print('pade_coef_n_ls')
        """
        Subroutine pade_ls_coeficients() finds coefficient of Pade approximant by Least Squares method
        f - values of complex function for approximation
        e - complex points in which function z is determined
        n - number of coefficients, should be less than number of points in e (n<m)
        """
        m = len(self.e)
        r = n // 2
        s = np.zeros(m, dtype=np.complex128)
        x = np.zeros((m, n), dtype=np.complex128)
        for i in range(0, m):
            s[i] = f[i] * e[i] ** r
            for j in range(0, r):
                x[i, j] = e[i] ** j
            for j in range(r, 2 * r):
                x[i, j] = -f[i] * e[i] ** (j - r)
        # Solving the equation: aX=b, where
        # a=x, b=s,
        #                       |p|
        #                   X = | |
        #                       |q|
        try:
            self.pq = np.linalg.lstsq(x, s)[0]
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                pq = 123456.7
                success = False
            else:
                raise
        else:
            success = True
        return pq, success

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
            try:
                from mpmath import mp, im, re, fdiv, mpc, fp, workdps
            except ImportError:
                raise('Please install mpmath Python module or use numpy as mlib')
            else:
                global mp, im, re, fdiv, mpc, fp, workdps
                fp.dps = 12
                mp.prec = self.prec
        if self.mlib == 'numpy':
            try:
                import numpy as np
            except ImportError:
                raise('Please install numpy Python module or use mpmath as mlib')
            else:
                global np
            
            
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
                            help='Use or not external information about momenta'
                            '[default: %(default)s]')
        parser.add_argument('-m', nargs=3, default=(0.0, 0.0, 0.0), type=float,
                            help='first momenta of function: m0, m1, m2')
        parser.add_argument('-pm', '--print_moments', action='store_true',
                            help='Print or not estimated values of momenta'
                            '[default: %(default)s]')
        parser.add_argument('-random', action='store_true',
                            help='Use or not randomly picked points in input set'
                            '[default: %(default)s]')
        parser.add_argument('-nrandomcycle', type=int, default=200,
                            help='number cycles with random points'
                            '[default: %(default)i]')
        parser.add_argument('-ls', action='store_true',
                            help='Use non-diagonal form of Pade coefficients matrix'
                            '[default: %(default)s]')
        parser.add_argument('-npo', nargs=2, default=(10, 90), type=int,
                            help='number of input iw points'
                            '[default: %(default)s]')
        parser.add_argument('-use_ne', action='store_true',
                            help='use symmetry of input function: G(z^*)=G^*(z)'
                            '[default: %(default)s]')
        parser.add_argument('-ne', nargs=2, default=(0, 5), type=int,
                            help='number of negative iw points'
                            '[default: %(default)s]')
        parser.add_argument('-mlib', default='numpy',
                            help='Choose mathematical library: numpy or mpmath'
                            ' or auto [default: %(default)s]')
        parser.add_argument('-precision', type=int, default=256,
                        help='precision of floating point numbers (in bits)'
                             ' for use in mpmath [default: %(default)i]')

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


if __name__ == "__main__":
    main()

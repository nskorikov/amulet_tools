#!/usr/bin/python
# coding=utf-8
# Program make analitical continuation of complex function defined on Matsubara frequency
# to real energy using Pade approximant. This version is mpmath based. 

import sys
import argparse
import os.path

try:
    from mpmath import mp, im, re, fdiv, mpc, fp, workdps
except ImportError:
    print('Please install mpmath Python module or use numpy based version of pade.py')
    raise
import math
import time
import random
import resource

fp.dps = 12

def main():
    print('Start at %s ' % time.ctime())
    start_time = time.time()

    handle_input()
    print_params(logfile)
    print_params()
    fp.dps = 12
    w, f = readsigma(infile)
    e = make_e_mesh(emin, de, npts)
    w1 = []
    w2 = []
    w3 = []
    solutions = []
    sets = []
    mmts = []

    if not use_moments:
        print('%4s %4s %4s %4s %20s %11s %12s %12s\b' % ('npo', 'nne', 'try', 'ils',
                                                         'delta', 'm0', 'm1', 'm2'))
    else:
        print('%4s %4s %4s %4s %10s' % ('npo', 'nne', 'try', 'ils', 'delta'))

    for ipo in range(npo[0], npo[1], 1):
        for ine in range(ne[0], ne[1]):
            for q1 in range(0, ipo//4, 2):
#             for q1 in range(0, qq2, 2):
                sys.stdout.flush()
                for qq in range(0, nrandomcycle):
                    if (ine + ipo) % 2 != 0:
                        continue

                    sys.stdout.write('%4i %4i %4i %4i %s\r' % (ipo, ine, qq, q1, " " * 16))

                    if randomp:
                        # tmp = choose_prandom_points(w, f, ine, ipo)      # the best shape, the worst moments
                        e1, f1 = choose_prandom_points(w, f, ine, ipo)
                    else:
                        # tmp = choose_seq_points_plus(w, f, ine, ipo)     # work only with pade_ls_coeff
                        # tmp = choose_seq_points(w, f, ine, ipo)          # work only with pade_ls_coeff
                        e1, f1 = choose_seq_points(w, f, ine, ipo)
                    # tmp = choose_points(w, f, nne, npo)                  # manual selection
                    # tmp = choose_random_points(w, f, nne, npo)           # bad idea

                    if use_moments:
                        f1[:, 0] = make_f_prime(f1[:, 0], e1, m)

                    pq, success, used_solver = pade_ls_coefficients(f1[:, 0], e1, ipo + ine - q1)

                    if not success:
                        continue

                    # use external information about momenta

                    if use_moments:
                        sigim = pade_m(pq, w, m)
                    else:
                        sigim = pade(pq, w)

                    if not is_imag_negative(w, sigim):
                        continue
                    delta = calc_residual(f, sigim)
                    # if delta > 1.00001:
                    #     continue
                    if use_moments:
                        sigre = pade_m(pq, e, m)
                    else:
                        sigre = pade(pq, e)

                    if not is_imag_negative(e, sigre):
                        continue

                    memusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if not use_moments:
                        moments = get_moments(pq)
                        mmts.append(moments)
                        # mmts = np.vstack([mmts, moments])
                        s = ''
                        for mm in moments:
                            s += '{:12.4f}'.format(mm)
                            # s += '{:18s}'.format(mp.nstr(mm, n=4))
                        s = '{:4d} {:4d} {:4d} {:4d}     {:18.12f} {:s}  {:24s}'. \
                            format(ipo, ine, qq, q1, float(delta), s, used_solver)
                        print('%s' % s)
                        s += '\n'
                    else:
                        s = '{:4d} {:4d} {:4d} {:4d}     {:18.12f}    {:24s} {:d}'. \
                            format(ipo, ine, qq, q1, float(delta), used_solver, memusage)
                        print('%s' % s)
                        s += '\n'

                    # Save "good" results for next consideration
                    # with workdps(12):
                    sets.append(s)
                    w1.append(delta)

                    tmp = []
                    for tt in sigre:
                        tmp.append(complex(tt))
                    solutions.append(tmp)

                    # Write continued function to the disk
                    if delta < 0.100000005:
                        outfile = 'imG_' + str(ipo) + '_' + str(ine) + '_' + \
                                  str(qq) + '_' + str(q1) + '.dat'
                        write_g_im(outfile, w, sigim)

                        outfile = 'reG_' + str(ipo) + '_' + str(ine) + '_' + \
                                  str(qq) + '_' + str(q1) + '.dat'
                        write_g_re(outfile, e, sigre)

    # mp.dps = 12
    if not use_moments:
        mmean = calc_mean(mmts)
        print('Mean moments: %12.3f %12.3f %12.3f' % (float(mmean[0]), float(mmean[1]), float(mmean[2])))

    qq = len(w1)
    with open(logfile, 'a') as f:
        if not use_moments:
            s = '{:4s} {:4s} {:4s} {:4s} {:9s} {:15s} {:16s} {:16s}\n'. \
                format('npo', 'nne', 'try', 'ils', 'delta', 'm0', 'm1', 'm2')
            f.write(s)
        else:
            s = '{:4s} {:4s} {:4s} {:4s} {:9s}\n'.format('npo', 'nne', 'try', 'ils', 'delta')
            f.write(s)
        for i in range(0, qq):
            f.write(sets[i])
        f.write('\n')
        if not use_moments:
            s = '{0:38s}{1:12.3f}{2:12.3f}{3:12.3f}\n'. \
                format('Mean moments', float(mmean[0]), float(mmean[1]), float(mmean[2]))
            f.write(s)
        f.write('\n\n\n')

    if qq == 0:
        print('There are no solutions')
        print("Stop at %s" % time.ctime())
        end_time = time.time()
        run_time = end_time - start_time
        print(run_time)
        hour = int((run_time / 60) / 60)
        minute = int((run_time / 60) - (hour * 60))
        sec = run_time % 60
        print('Program runtime = %2i:%2i:%2i' % (hour, minute, sec))
        sys.exit()

    w = 0.0
    w2 = mp.zeros(len(w1), 1)
    w3 = mp.zeros(len(w1), 1)
    for i in range(0, qq):
        # w1[i] = 1 / w1[i]
        w1[i] = float(w1[i])
        w3[i] = 200000 * (min(w1) - w1[i])
        w2[i] = math.exp(w3[i])
    for i in range(0, qq):
        w3[i] = w1[i]

    for i in range(0, qq):
        w1[i] = 1 / w1[i]

    w1sum = sum(w1)
    w2sum = sum(w2)
    for i in range(0, qq):
        w1[i] /= w1sum
        w2[i] /= w2sum

    with open('weights.dat', 'w') as f:
        for i in range(0, qq):
            s = '{:16.12f} {:16.12f} {:16.12f}\n'.format(float(w3[i]), float(w1[i]), float(w2[i]))
            # s = '{:d} {:16.12f} {:16.12f}\n'.format(i, float(w1[i]), float(w2[i]))
            f.write(s)

    sigre = mp.zeros(len(e), 1)
    for i in range(0, qq):
        for j in range(0, len(e)):
            sigre[j] += solutions[i][j] * w1[i]
    outfile = 'solution_w1.dat'
    write_g_re(outfile, e, sigre)
    sigre[:] = 0.0
    for i in range(0, qq):
        for j in range(0, len(e)):
            sigre[j] += solutions[i][j] * w2[i]
    outfile = 'solution_w2.dat'
    write_g_re(outfile, e, sigre)

    print("Stop at %s" % time.ctime())
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
    hour = int((run_time / 60) / 60)
    minute = int((run_time / 60) - (hour * 60))
    sec = run_time % 60

    print('Program runtime = %i:%i:%i' % (hour, minute, sec))
    with open(logfile, 'a') as f:

        f.write('       %f       ' % run_time)
        f.write('Program runtime = %i:%i:%i' % (hour, minute, sec))
    

# class InData:
#
#     def makeinput(self):
#         infile, emin, de, npts = handle_commandline()

def handle_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="g.dat",
                        help="file with input function "
                             "[default: %(default)s]")
    parser.add_argument("-logfile", default="sets.dat",
                        help="file for log, new data will be appended to existent "
                             "[default: %(default)s]")
    parser.add_argument("-emin", default=-5.0, type=float,
                        help="minimum energy on real axis "
                             "[default: %(default)d]")
    parser.add_argument("-emax", default=-1000.0, type=float,
                        help="maximum energy on real axis ")
    parser.add_argument("-de", default=0.01, type=float,
                        help="energy step on real axis "
                             "[default: %(default)f]")
    parser.add_argument("-npts", type=int, default=1000,
                        help="number of points on real energy axis"
                             "[default: %(default)i]")
    parser.add_argument("-use_moments", action='store_true',
                        help="Use or not external information about momenta "
                             "[default: %(default)s]")
    parser.add_argument("-m", nargs=3, type=float,
                        help="first momenta of function: m0, m1, m2 ")
    parser.add_argument("-pm", "--print_moments", action='store_true',
                        help="Print or not estimated values of momenta"
                             "[default: %(default)s]")
    parser.add_argument("-random", action='store_true',
                        help="Use or not randomly picked points in input set "
                             "[default: %(default)s]")
    parser.add_argument("-nrandomcycle", type=int, default=100,
                        help="number cycles with random points"
                             "[default: %(default)i]")
    parser.add_argument("-ls", action='store_true',
                        help="Use non-diagonal form of Pade coefficients matrix "
                             "[default: %(default)s]")
    parser.add_argument("-npo", nargs=2, default=(30, 90), type=int,
                        help="number of input iw points "
                             "[default: %(default)s]")
    parser.add_argument("-use_ne", action='store_true',
                        help="use symmetry of input function: G(z^*)=G^*(z) "
                             "[default: %(default)s]")
    parser.add_argument("-ne", nargs=2, default=(0, 5), type=int,
                        help="number of negative iw points "
                             "[default: %(default)s]")
    parser.add_argument("-precision", type=int, default=256,
                        help="precision of floating point numbers (in bits)"
                             "[default: %(default)i]")

    args = parser.parse_args()

    if args.use_moments and not args.m:
        raise TypeError("moments values are not given")

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
                 'prec': args.precision
                 }

    return inputdata


def handle_input():
    global emin, de, npts, use_moments, randomp, ls, npo, use_ne, ne, infile
    global infile, logfile, m, nrandomcycle

    inputdata = handle_commandline()
    mp.prec = inputdata['prec']
    emin = inputdata['emin']
    de = inputdata['de']
    npts = inputdata['npts']
    use_moments = inputdata['use_moments']
    randomp = inputdata['random']
    if randomp:
        nrandomcycle = inputdata['nrandomcycle']
    else:
        nrandomcycle = 1
    ls = inputdata['ls']
    npo = inputdata['npo']
    use_ne = inputdata['use_ne']
    if use_ne:
        ne = inputdata['ne']
    else:
        ne = (0,1)
    # ne = inputdata['ne']
    # infile = inputdata['f']
    infile = os.path.abspath(inputdata['f'])
    if not os.path.exists(infile):
        raise TypeError('File %s not exist' % infile)
    logfile = inputdata['logfile']
    if 'emax' in inputdata:
        if inputdata['emax'] != -1000:
            emax = inputdata['emax']
            if (emax - emin) / npts != de:
                print('Parameters of real energy are not selfconsistent!!! ')
                print('Default values will be used!!! ')
                de = (emax - emin) / (npts + 1)

    if 'm' in inputdata:
        m = inputdata['m']
    else:
        m = (1, 2, 3)


def print_params(direction='sys.stdout'):
    old = sys.stdout
    if direction != 'sys.stdout':
        print('direction=%s' % direction)
        sys.stdout = open(direction, 'a')
    if use_ne:
        print("The symmetry of Green function will be accounted")
        print("The number of negative points will be varied in interval: [%4i, %4i] " % (ne[0], ne[1]))
    else:
        print("The symmetry of Green function will not be accounted")
    print("The number of positive points will be varied in interval: [%4i, %4i] " % (npo[0], npo[1]))
    if randomp:
        print("Some random points will be added to sequential set of points")
    if use_moments:
        print("Momenta of function will be accounted in continuation: "
              "%5.2f %5.2f %5.2f" % (m[0], m[1], m[2]))
    if ls:
        print("Coefficients of Pade polinom will be finded by Least Squares method")
    emax = emin + de * npts
    print("Continuation will be performed to interval "
          "[%5.2f,%5.2f] with step %4.3f" % (emin, emax, de))
    print("Function from %s will be continued to real axis" % infile)
    print("Log of execution will be duplicated to %s" % logfile)
    print(mp)
    sys.stdout.flush()
    sys.stdout = old


def make_e_mesh(t, d, n):
    return [t + i * d + 1j * 0.01 for i in range(n)]
    # e = fp.matrix(n, 1)
    # for i in range(0, n):
    #     e[i] = t + 1j * 0.01
    #     t += d
    # return e


def choose_random_points(e, f, nneg, npos):
    # Subroutine selects nneg+npos points randomly
    # temporary it does not select negative points
    # It is a very bad idea to pick all points randomly
    # Use for testing purpose

    if (nneg + npos) % 2 != 0:
        print('Number of chosen points should be even!', nneg, npos, nneg + npos)
        npos += 1
    q = nneg + npos
    r = len(e)

    points = random.sample(range(0, r - 1), q - 1)

    # points = np.random.randint(low=0, high=r - 1, size=q - 1)
    points.sort()
    for i in range(0, q - 2):
        if points[i] == points[i + 1]:
            points[i + 1] += 1

    ee = mp.zeros(q, 1)
    ff = mp.zeros(q, f.cols)
    for i in range(0, q - 1):
        ee[i] = e[points[i]]
        ff[i] = f[points[i]]
    return ee, ff


def choose_prandom_points(e, f, nneg, npos):
    """
    Subroutine selects from input nneg+npos points: first nneg+npos-nrnd points are selected 
    sequently, then nrnd points are picked randomly. Number of randomly selected points nrnd 
    is determined randomly in interval from 1/16 to 1/3 of total number of points.
    e -- input complex array with energy points
    f -- input complex array with values of function in points e[i]
    """
    if (nneg + npos) % 2 != 0:
        print('Number of chosen points should be even!', nneg, npos, nneg + npos)
        npos += 1
    q = nneg + npos
    nrnd = random.randrange(q//16, q//3, 2)
    r = len(e)
    ee = fp.zeros(q, 1)
    ff = fp.zeros(q, f.cols)
    for i in range(0, nneg):
        ee[i] = mp.conj(e[nneg - 1 - i])
        ff[i] = mp.conj(f[nneg - 1 - i])
    for i in range(nneg, q - nrnd):
        ee[i] = e[i - nneg]
        ff[i] = f[i - nneg]
    # Make list of random points
    pp = random.sample(range(q - nrnd, r - 1), nrnd)
    # Sort them
    pp.sort()
    # Fix repeated points (if there are)
    for i in range(0, nrnd - 1):
        if pp[i] == pp[i + 1]:
            pp[i + 1] += 1
    # The last two points should be sequential to fix diff at the tail of F(z).
    pp[nrnd - 1] = pp[nrnd - 2] + 1

    # Construct
    for i in range(0, nrnd):
        ee[q - nrnd + i] = e[pp[i]]
        ff[q - nrnd + i] = f[pp[i]]

    return ee, ff


def choose_points(e, f, nneg, npos):
    # Choose first npo+nneg points,
    # than pick some points by hands
    # For testing purposes

    if (nneg + npos) % 2 != 0:
        print('Number of chosen points should be even!', nneg, npos, nneg + npos)
        npos += 1
    q = nneg + npos
    # ee = np.zeros(q, dtype=mp.mpc)
    # ff = np.zeros((q, f.shape[1]), dtype=mp.mpc)
    ee = fp.zeros(q, 1)
    ff = fp.zeros(q, f.cols)
    for i in range(0, nneg):
        ee[i] = e.conj()[nneg - 1 - i]
        ff[i] = f.conj()[nneg - 1 - i]
    for i in range(nneg, nneg + npos - 6):
        ee[i] = e[i - nneg]
        ff[i] = f[i - nneg]

    ee[q - 6] = e[750]
    ff[q - 6] = f[750]
    ee[q - 5] = e[900]
    ff[q - 5] = f[900]
    ee[q - 4] = e[937]
    ff[q - 4] = f[937]
    ee[q - 3] = e[960]
    ff[q - 3] = f[960]
    ee[q - 2] = e[999]
    ff[q - 2] = f[999]
    ee[q - 1] = e[1000]
    ff[q - 1] = f[1000]

    return ee, ff


def choose_seq_points(e, f, nneg, npos):
    # Pick first nneg+npos points

    if (nneg + npos) % 2 != 0:
        print('Number of chosen points should be even!', nneg, npos, nneg + npos)
        npos += 1
    q = nneg + npos
    # ee = np.zeros(q, dtype=mp.mpc)
    # ff = np.zeros((q, f.shape[1]), dtype=mp.mpc)
    ee = fp.zeros(q, 1)
    ff = fp.zeros(q, f.cols)
    for i in range(0, nneg):
        ee[i] = mp.conj(e[nneg - 1 - i])
        ff[i] = mp.conj(f[nneg - 1 - i])
    for i in range(nneg, nneg + npos):
        ee[i] = e[i - nneg]
        ff[i] = f[i - nneg]
    return ee, ff


def choose_seq_points_plus(e, f, nneg, npos):
    # Choose first nneg+npos-2 and last two points

    if (nneg + npos) % 2 != 0:
        print('Number of chosen points should be even!', nneg, npos, nneg + npos)
        npos += 1
    q = nneg + npos
    # ee = np.zeros(q, dtype=mp.mpc)
    # ff = np.zeros((q, f.shape[1]), dtype=mp.mpc)
    ee = fp.zeros(q, 1)
    ff = fp.zeros(q, f.cols)
    for i in range(0, nneg):
        ee[i] = e.conj()[nneg - 1 - i]
        ff[i] = f.conj()[nneg - 1 - i]
    for i in range(nneg, nneg + npos - 2):
        ee[i] = e[i - nneg]
        ff[i] = f[i - nneg]
    for i in range(-2, 0):
        ee[i] = e[i]
        ff[i] = f[i]
    return ee, ff


def pade_coeficients(f, e):
    # Subroutine pade_coeficients() finds coefficient of Pade approximant
    # f - values of complex function for approximation
    # e - complex points in which function f is determined

    r = len(e) / 2
    # Preparation of arrays to calc Pade coefficiens
    s = mp.zeros(2 * r, 1)
    x = mp.zeros(2 * r)
    for i in range(0, 2 * r):
        s[i] = f[i] * e[i] ** r
        for j in range(0, r):
            x[i, j] = e[i] ** j
        for j in range(r, 2 * r):
            x[i, j] = -f[i] * e[i] ** (j - r)

    # Solving the equation: |p|
    #                       | |=X^{-1}*s
    #                       |q|
    # Here we should catch exception in linalg!!
    try:
        x = mp.inverse(x)
        pq = x * s
        # x **= -1
        # pq = np.dot(x, s)
        success = True
    except ZeroDivisionError as err:
        if 'matrix is numerically singular' in err.message:
            pq = 123456.7
            success = False
        else:
            raise
    return pq, success
    # return pq


def pade_ls_coefficients(f, e, n):
    # Subroutine pade_ls_coeficients() finds coefficient of Pade approximant by Least Squares method
    # f - values of complex function for approximation
    # e - complex points in which function z is determined
    # n - number of coefficients, should be less than number of points in e (n<m)

    m = len(e)
    r = n / 2
    # Preparation of arrays to calc Pade coefficiens
    s = mp.zeros(m, 1)
    x = mp.zeros(m, n)
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

    # pq = linalg.lstsq(x, s)[0]
    success = True
    solver = 'LU solver'

    try:
        pq = mp.lu_solve(x, s)
        # success = True
    # except ZeroDivisionError as err:
    except (ZeroDivisionError, ValueError):
        # if 'matrix is numerically singular' in err.message:
        try:
            pq = mp.qr_solve(x, s)
            pq = pq[0]
            solver = 'QR solver'
        # success = True
        except ValueError:
            # if 'matrix is numerically singular' in err.message:
            success = False
            pq = 123456.7
            # else:
            #     raise
    if success is True:
        pq.rows += 1
        pq[n, 0] = mp.mpc(1, 0)
    return pq, success, solver


def pade(coef, e):
    # Calculation of analitical function on a arbitrary mesh for a given Pade coefficient
    # e -  energy mesh (can be complex or real)
    # coef - Pade coefficients

    nlines = len(e)
    r = len(coef) / 2
    f = mp.zeros(nlines, 1)
    # pq = mp.ones(r * 2 + 1, 1)
    # for i in range(0, r):
    # pq[i] = coef[i]
    # pq[i + r] = coef[i + r]
    for iw in range(0, nlines):
        p = mp.mpc(0.0)
        q = mp.mpc(0.0)
        for i in range(0, r):
            p += coef[i] * e[iw] ** i
        for i in range(0, r + 1):
            q += coef[i + r] * e[iw] ** i

        # f[iw] = np.divide(p, q)
        f[iw] = fdiv(p, q)
    f = fp.matrix(f)
    return f


def pade_m(coef, e, m):
    # Calculation of analitical function on a arbitrary mesh for a given Pade coefficient
    # and first known momenta of function
    # e -  energy mesh (can be complex or real)
    # coef - Pade coefficients
    # m - first three momenta of function

    nlines = len(e)
    r = len(coef) / 2
    f = mp.zeros(nlines, 1)
    pq = mp.ones(r * 2 + 1, 1)
    # for i in range(0, r):
    # pq[i] = coef[i]
    # pq[i + r] = coef[i + r]
    for iw in range(0, nlines):
        p = mp.mpc(0.0)
        q = mp.mpc(0.0)
        for i in range(0, r):
            p += coef[i] * e[iw] ** i
        for i in range(0, r + 1):
            q += coef[i + r] * e[iw] ** i

        f[iw] = fdiv(p, q)

        # f[iw] = np.divide(p, q)
        f[iw] /= e[iw] ** 3
        f[iw] += m[0] / e[iw] + m[1] / (e[iw] ** 2) + m[2] / (e[iw] ** 3)
    f = fp.matrix(f)
    return f


def make_f_prime(f, e, m):
    # F'(z) = z^3 * [F(z) - p_n/z - p'_n/z^2 - p''_n/z^3]
    r = len(e)
    fn = fp.zeros(r, 1)
    for i in range(0, r):
        fn[i] = (e[i] ** 3) * (f[i] - m[0] / e[i] - m[1] / (e[i] ** 2) - m[2] / (e[i] ** 3))
    return fn


def get_moments(coef):
    # returns moments of function calculated from coefficients of Pade decomposition
    n = len(coef) / 2
    # for i in range(0, n):
    #     p[i] = coef[i]
    #     q[i] = coef[i + n]
    # m[0] = re(p[n - 1])
    # m[1] = re(p[n - 2] - m[0] * q[n - 1])
    # m[2] = re(p[n - 3] - p[n - 1] * q[n - 2] - (p[n - 2] - p[n - 1] * q[n - 1]) * q[n - 1])

    m = [1025.615, 2025.615, 3025.615]
    m[0] = float(re(coef[n - 1]))
    m[1] = float(re(coef[n - 2] - coef[n - 1] * coef[2 * n - 1]))
    m[2] = float(re(
        coef[n - 3] - coef[n - 1] * coef[2 * n - 2] - (coef[n - 2] - coef[n - 1] * coef[2 * n - 1]) * coef[2 * n - 1]))
    # m[1] = float(re(coef[n-2] - m[0] * coef[2*n-1]))

    return m


def usemoments(coef, moments):
    # Subroutine uses known moments of function to change coefficients of Pade decomposition
    # p_{n-1} = m_{0}
    # p_{n-2} = m_{1}+m_{0}*q_{n-1}
    # p_{n-2} = m_{2}+m_{1}*q_{n-1}+m_{0}*q_{n-2}

    n = len(coef) / 2

    coef[n - 1] = moments[0]
    coef[n - 2] = moments[1] + moments[0] * coef[-1]
    coef[n - 3] = moments[2] + moments[1] * coef[-1] + moments[0] * coef[-2]
    return coef


def readsigma(filename):
    # Read sigma from AMULET.

    print('Input file contains:')

    with open(filename, 'r') as f:
        data = f.readlines()

    # Analyze the file.

    # Number of lines.
    nlines = len(data)
    print(" Number of lines: %i " % nlines)

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
        nlinesperblock = (nlines + 1) / ndatasets
    elif ndatasets == 0 and data[-1].split():
        ndatasets += 1
        nlinesperblock = (nlines + 2) / ndatasets
    else:
        nlinesperblock = nlines / ndatasets
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
        e = fp.zeros(nlines, 1)
        z = fp.zeros(nlines, 1)
        for i in range(0, nlines):
            s = data[i].split()
            e[i] = 1j * float(s[0])
            z[i] = float(s[1]) + 1j * float(s[2])
    elif l == 5:
        e = fp.zeros(nlines, 1)
        z = fp.zeros(nlines, 2)
        for i in range(0, nlines):
            s = data[i].split()
            e[i] = 1j * float(s[0])
            z[i, 0] = float(s[1]) + 1j * float(s[2])
            z[i, 1] = float(s[3]) + 1j * float(s[4])
    else:
        print("unknown data format")

    return e, z


def write_g_im(filename, e, sigma):
    nlines = len(e)
    with open(filename, 'w') as f:
        for iw in range(0, nlines):
            s = '{0:18.8f}{1:18.12f}{2:18.12f}\n'. \
                format(float(im(e[iw])), float(re(sigma[iw])), float(im(sigma[iw])))
            f.write(s)


def write_g_re(filename, e, sigma):
    with open(filename, 'w') as f:
        for ene, s in zip(e, sigma):
            f.write('{0:18.8f}{1:18.12f}{2:18.12f}\n'.format(float(re(ene)), float(re(s)), float(im(s))))

    # nlines = len(e)
    # with open(filename, 'w') as f:
    #     for iw in range(0, nlines):
    #         s = '{0:18.8f}{1:18.12f}{2:18.12f}\n'. \
    #             format(float(re(e[iw])), float(re(sigma[iw])), float(im(sigma[iw])))
    #         f.write(s)


def calc_residual(f1, f2):
    l1 = len(f1)
    if l1 != len(f2):
        print('WARNING: calc_residual')
        print('Lengths of f1 and f2 are different!\n')
    d = pow(sum([pow(f1[i] - f2[i], 2) for i in range(l1)]), 0.5)
    d /= l1
    return float(abs(d))

    # d = 0.0
    # f1 -= f2
    # for x in f1:
    #     d += abs(x)
    # d /= l1
    # return d


def calc_mean(m):
    a = mp.dps
    mp.dps = 8
    l = len(m)
    tmp = [0.0, 0.0, 0.0]
    for i in range(0, l):
        for j in range(0, 3):
            tmp[j] += m[i][j]
    for i in range(0, 3):
        tmp[i] /= l
    mp.dps = a
    return tmp


def is_imag_negative(w, f):
    analitic = True
    for x in f:
        if im(x) > 0.0:
            analitic = False
            break
    return analitic


# Begin main
if __name__ == "__main__":
    main()

        # Plotting example
        # nlines = len(e)
        # diff = np.zeros(nlines, dtype=mp.mpc)
        # for i in range(0, len(e)):
        #     diff[i] = float(z.real[i,0]-sigma.real[i]) + 1j * float(z.imag[i,0]-sigma.imag[i])
        # plt.plot(e.imag, diff.real, '-', e.imag, diff.imag, '--')
        # plt.plot(e.imag, z.real[:,0], '-', e.imag, z.imag[:,0], '-', e.imag, sigma.real, '--', e.imag, sigma.imag, '--')
        # plt.show()

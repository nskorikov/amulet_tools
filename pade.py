#!/usr/bin/env python
# coding=utf-8
# Program make analitical continuation of complex function defined on Matsubara frequency
# to real energy using Pade approximant. Universal version. 
import time

def main():
    print('Start at %s ' % time.ctime()) 
    start_time = time.time()   
    inp=pade_input()
    


class pade_input():
    import os.path
    import argparse

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
        self.validate_input()
        self.print_input()


    def validate_input(self):
        if not self.random:
            nrandomcycle = 1
        if self.random and self.nrandomcycle==1:
            print('You set switch "-random"')
            raise ValueError('In such case you should set "-nrandomcycle">1')
        if not self.use_ne:
            self.ne = (0, 1)
        if not os.path.exists(self.infile):
            raise TypeError('File %s not exist' % self.infile)
        if self.emax < self.emin:
            raise ValueError('Incorrect input: emax is less then emin')
        if (self.emax - self.emin)/self.npts != self.de:
            print('Check parameters of real energy')    
            print("Continuation will be performed to interval "
          "[%5.2f,%5.2f] with step %4.3f" % (emin, emax, de))
            self.npts = (self.emax - self.emin)//self.de
        if self.use_moments and self.m == (0.0, 0.0, 0.0):
            print('You set switch "-use_moments"')
            raise ValueError('Values of moments should be defined in commandline')
             
#         global emin, de, npts, use_moments, randomp, ls, npo, use_ne, ne, infile
#         global infile, logfile, m, nrandomcycle

    def print_input(self, direction='sys.stdout'):
        old = sys.stdout
        if direction != 'sys.stdout':
            print('direction=%s' % direction)
            sys.stdout = open(direction, 'a')
        if self.use_ne:
            print("The symmetry of Green function G(z*)=-G*(z) will be accounted")
            print("The number of negative points will be varied in interval: [%4i, %4i] " % 
                  (self.ne[0], self.ne[1]))
        else:
            print("The symmetry of Green function will not be accounted")
        print("The number of positive points will be varied in interval: [%4i, %4i] " % 
              (self.npo[0], self.npo[1]))
        if self.random:
            print("Some random points will be added to sequential set of points")
        if self.use_moments:
            print("Momenta of function will be accounted in continuation: "
                  "%5.2f %5.2f %5.2f" % self.m)
        if self.ls:
            print("Coefficients of Pade polinom will be finded by Least Squares method")
        print("Continuation will be performed to interval "
              "[%5.2f,%5.2f] with step %4.3f" % (self.emin, self.emax, self.de))
        print("Function from %s will be continued to real axis" % self.infile)
        print("Log of execution will be duplicated to %s" % self.logfile)
        sys.stdout.flush()
        sys.stdout = old
    

    def handle_commandline(self):
        """
        Method defines possible commandline keys and them default values, parse commandline and 
        return dictionary 'inputdata'  
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", default="g.dat",
                            help="file with input function "
                            "[default: %(default)s]")
        parser.add_argument("-logfile", default="sets.dat",
                            help="file for log, new data will be appended to existent"
                            "[default: %(default)s]")
        parser.add_argument("-emin", default=-10.0, type=float,
                            help="minimum energy on real axis "
                            "[default: %(default)d]")
        parser.add_argument("-emax", default=10.0, type=float,
                            help="maximum energy on real axis ")
        parser.add_argument("-de", default=0.01, type=float,
                            help="energy step on real axis "
                            "[default: %(default)f]")
        parser.add_argument("-npts", type=int, default=2000,
                            help="number of points on real energy axis"
                            "[default: %(default)i]")
        parser.add_argument("-use_moments", action='store_true',
                            help="Use or not external information about momenta"
                            "[default: %(default)s]")
        parser.add_argument("-m", nargs=3, default=(0.0, 0.0, 0.0), type=float,
                            help="first momenta of function: m0, m1, m2")
        parser.add_argument("-pm", "--print_moments", action='store_true',
                            help="Print or not estimated values of momenta"
                            "[default: %(default)s]")
        parser.add_argument("-random", action='store_true',
                            help="Use or not randomly picked points in input set"
                            "[default: %(default)s]")
        parser.add_argument("-nrandomcycle", type=int, default=200,
                            help="number cycles with random points"
                            "[default: %(default)i]")
        parser.add_argument("-ls", action='store_true',
                            help="Use non-diagonal form of Pade coefficients matrix"
                            "[default: %(default)s]")
        parser.add_argument("-npo", nargs=2, default=(10, 120), type=int,
                            help="number of input iw points"
                            "[default: %(default)s]")
        parser.add_argument("-use_ne", action='store_true',
                            help="use symmetry of input function: G(z^*)=G^*(z)"
                            "[default: %(default)s]")
        parser.add_argument("-ne", nargs=2, default=(0, 5), type=int,
                            help="number of negative iw points"
                            "[default: %(default)s]")
    
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
                     'logfile': args.logfile
                     }
    
        return inputdata
    
    
if __name__ == "__main__":
    main()


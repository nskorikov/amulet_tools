#!/usr/bin/env python
from multiprocessing import Process, Queue
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name, q):
    info('function f')
    print('hello', name)
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    info('main line')
    q = Queue()
    p = Process(target=f, args=('bob', q,))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()

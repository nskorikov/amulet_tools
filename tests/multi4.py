"""
Process class can also be subclassed just like threading.Thread;
Queue works like queue.Queue but for cross-process, not cross-thread
"""

import random
from mpmath import mp
import os, time, queue
from multiprocessing import Process, Queue           # process-safe shared queue
                                                     # queue is a pipe + locks/semas
class Counter(Process):
    label = '  @'
    def __init__(self, start, queue):                # retain state for use in run
        self.state = start
        self.n = start
        self.post  = queue
        Process.__init__(self)

    def run(self):                                   # run in newprocess on start()
        n = self.n
        a = []
        x = []
        random.seed()
        for i in range(0, n):
            aa = [random.uniform(1, 10) * 2 *i + random.uniform(1, 10) * 1j for j in range(0,n)]
            a.append(aa)
            x.append(random.uniform(1, 10) * 3 * i + random.uniform(1, 10) * 1j)
        a = mp.matrix(a)
        x = mp.matrix(x)
        b = mp.zeros(n, 1)
        for j in range(0, n):
            for i in range(0, n):
                b[j] += a[j, i] * x[i]
        pq = mp.qr_solve(a, b)[0]
        diff = []
        for i in range(1, n):
            diff.append(pq[i]-x[i])
        print(self.label ,self.pid, self.state)
        self.post.put([self.pid, self.state, diff])

if __name__ == '__main__':
    print('start', os.getpid())
    expected = 9

    post = Queue()
    ps = []
    for i in range(8):
        p = Counter(50+i, post)
        p.start()
        ps.append(p)
#     q.start( )
#     r.start( )

    while expected:     # parent consumes data on queue
        time.sleep(1)                         # this is essentially like a GUI,
        try:                                    # though GUIs often use threads
            data = post.get(block=False)
        except queue.Empty:
            pass
        else:
            print('posted:', data[0], data[1])
        expected -= 1

#             for d in data[2]:
#                 print('%f %f' % (d.real, d.imag), type(d))
#             print('\n')
    for p in ps:
        p.join()
    print('finish', os.getpid())

#!/usr/bin/python

import numpy as np
from mpmath import mp

m = 30
n = 30

a = []
b = []
x = []
for i in range(0, n):
    aa = [np.random.rand() * 2 *i + np.random.rand() * 1j for j in range(0,m)]
    a.append(aa)
    x.append(np.random.rand() * 3 * i + np.random.rand() * 1j)

for j in range(0, m):
    bb = 0.0
    for i in range(0, n):
        bb +=  a[j][i] * x[i]
    b.append(bb)
        
pq = np.linalg.lstsq(a, b)[0]
print(len(pq))
mp.prec=12
pq1 = mp.qr_solve(a, b)[0]
print(len(pq1), type(pq1))
for i in range(1, n):
    diff = pq[i]-pq1[i]
    print('%f %f' % (diff.real, diff.imag))
print ' '


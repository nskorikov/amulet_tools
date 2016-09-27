#!/usr/bin/env python

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
aa = np.array(a, dtype=np.complex128)
bb = np.array(b, dtype=np.complex128)
pq1 = np.linalg.lstsq(aa, bb)[0]
for i in range(1, n):
    diff = pq[i]-x[i]
    print('%f %f' % (diff.real, diff.imag), type(diff))
print(' ')



mp.prec=128
pq = mp.qr_solve(a, b)[0]

aa = mp.matrix(a)
bb = mp.zeros(m, 1)
for j in range(0, m):
    for i in range(0, n):
        bb[j] += aa[j, i] * x[i]

pq1 = mp.qr_solve(a, b)[0]
for i in range(1, n):
    diff = pq[i]-pq1[i]
    print('%f %f' % (diff.real, diff.imag), type(diff))
print( ' ')


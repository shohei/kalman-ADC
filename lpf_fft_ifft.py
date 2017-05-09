# -*- coding: utf-8 -*-
from pylab import *
import numpy as np
import pdb

fin = open("3_3.csv","r").readlines()
xs = []
ys = []
for line in fin:
    x  = float(line.rstrip().split(',')[0])
    y = float(line.rstrip().split(',')[1])
    xs.append(x)
    ys.append(y)

xs = xs[2:-1]
ys = ys[2:-1]

ave = sum(ys)/len(ys)
aves = [ave for i in range(len(ys))]
subplot(221)
plot(xs,ys,'b-')
plot(xs,aves,'r-',linewidth=3.0)
l = ['raw','average']
legend(l)
title(u'原波形')

Fk = np.fft.fft(ys)
P = Fk*np.conj(Fk)
subplot(222)
semilogy(P)
title(u'パワースペクトル')

subplot(223)
fs = 20
for i in range(len(P)):
    if i>fs:
        P[i]=0
        Fk[i]=0

semilogy(P)
ylim([10**3,10**14])
title(u'フィルタ後パワースペクトル\n第20高調波以上を阻止')

subplot(224)
f = np.fft.ifft(Fk)
plot(f)
title(u'ローパスフィルタをかけた信号波形')

show()

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:05:55 2020

@author: StoyanBoyukliyski
"""

import numpy as np
import matplotlib.pyplot as plt

TotalR = 1000000
TotalS = 1000000

muR = 30*10**6
sigmaR = 4*10**6
 
muE = 20*10**6
sigmaE = 2*10**6


S = np.random.normal(muE, sigmaE, TotalS)
R = np.random.normal(muR, sigmaR, TotalR)

countR, binsR, patches = plt.hist(R, bins = 100, density = True)
plt.plot(binsR, (1/(sigmaR*np.sqrt(2*np.pi)))*np.exp(-(binsR-muR)**2/(2*sigmaR**2)))

countS, binsS, patches = plt.hist(S, 100, density = True)
plt.plot(binsS, (1/(sigmaE*np.sqrt(2*np.pi)))*np.exp(-(binsS-muE)**2/(2*sigmaE**2)))
plt.show()


Pf = 0
for j in range(len(binsR)-2):
    Pfcur = 0
    CountR = 0
    for i in range(len(binsS)-2):
        if binsR[j]<= binsS[i]:
            CountR = (countR[j+1]+countR[j])*(binsR[j+1]-binsR[j])/2
        else:
            pass
        Pfcur = Pfcur + CountR*(countS[i+1]+countS[i])*(binsS[i+1]-binsS[i])/2
    Pf = Pf + Pfcur
    
print(Pf)


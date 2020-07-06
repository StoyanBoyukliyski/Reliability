# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:42:51 2020

@author: StoyanBoyukliyski
"""

import numpy as np
import matplotlib.pyplot as plt

bins = np.linspace(-5, 5, 10000)
'''
These stuff serve for validation, do not remove,
    use in conjunction with previous file for normal distribution!
    
muR = 30*10**6
sigmaR = 4*10**6
 
muE = 20*10**6
sigmaE = 2*10**6
'''

muR = 30*10**6
sigmaR = 2*10**6
 
muE = 20*10**6
sigmaE = 4*10**6

distr = "normal"

if distr == "normal": 
    muG = muR - muE
    sigmaG = np.sqrt(sigmaR**2 + sigmaE**2)
elif distr == "lognormal":
    sigmaRlog = np.log(1+(sigmaR/muR)**2)
    sigmaElog = np.log(1+(sigmaE/muE)**2)
    muRlog = np.log(muR) -sigmaRlog**2/2
    muElog = np.log(muE) -sigmaElog**2/2
    muG = muRlog-muElog
    sigmaG = np.sqrt(sigmaRlog**2 +sigmaElog**2)
else:
    ValueError("Please type in either: 'normal' or 'lognormal'")
def CalculateFORM():
    a = np.array([1, -4, -8])
    mu = np.array([250, 10, 10])
    Cx = np.array([[900, 0, 0],[0, 9, 6],[0, 6, 9]])
    a0 = 0
    
    betac = (a0 + np.matmul(np.transpose(a),mu))/np.sqrt(np.matmul(np.matmul(np.transpose(a),Cx),a))
    print("BETAC = " + str(betac))
    return betac

    
def StandardNormalCDF(beta):
    delF = 0
    for i in range(1, len(bins)):
        if bins[i] < beta:
            delF = delF + (((1/np.sqrt(np.pi*2))*np.exp(-bins[i]**2/2)) + ((1/np.sqrt(np.pi*2))*np.exp(-bins[i-1]**2/2)))*abs(bins[i]-bins[i-1])/2
        else:
            pass
    return delF

def StandardNormalPDF(beta):
    PDF = (1/np.sqrt(np.pi*2))*np.exp(-beta**2/2)    
    return PDF

Value = []
CumProbability = []
DensityProbability = []

for j in np.arange(-5, 5, 0.1):
    CumProbability.append(StandardNormalCDF(j))
    DensityProbability.append(StandardNormalPDF(j))
    Value.append(j)

ValueF = []
CumProbabilityF = []
DensityProbabilityF = []

for j in np.arange(-5, -muG/sigmaG, 0.1):
    CumProbabilityF.append(StandardNormalCDF(j))
    DensityProbabilityF.append(StandardNormalPDF(j))
    ValueF.append(j)
    
def ReverseCDF(Target):
    delF = 0
    i = 1
    while delF < Target:
        delF = delF + (((1/np.sqrt(np.pi*2))*np.exp(-bins[i]**2/2)) + ((1/np.sqrt(np.pi*2))*np.exp(-bins[i-1]**2/2)))*abs(bins[i]-bins[i-1])/2
        output = bins[i]
        i = i + 1
    print("Characteristic beta = " + str(output))
    return output

if distr == "normal": 
    Rk = muR + ReverseCDF(0.05)*sigmaR
    Sk = muE + ReverseCDF(0.95)*sigmaE
    FactorofSafety = Rk/Sk
    print("Characteristic Factor of Safety = " + str(FactorofSafety))
    print("Central Factor of Safety = " + str(muR/muE))
elif distr == "lognormal":
    Rk = muRlog + ReverseCDF(0.05)*sigmaRlog
    Sk = muElog + ReverseCDF(0.95)*sigmaElog
    FactorofSafety = Rk/Sk
    print("Characteristic Factor of Safety = " + str(FactorofSafety))
    print("Central Factor of Safety = " + str(muRlog/muElog))
else:
    ValueError("Characteristic Please type in either: 'normal' or 'lognormal'")



figure, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(Value, DensityProbability, "b")
ax1.fill_between(ValueF, DensityProbabilityF)
ax1.axvline(0)
ax1.axhline(0)
ax2.plot(Value, CumProbability, "b")
ax2.plot([-muG/sigmaG, -muG/sigmaG], [0, StandardNormalCDF(-muG/sigmaG)], "r--")
ax2.plot([0, -muG/sigmaG], [StandardNormalCDF(-muG/sigmaG), StandardNormalCDF(-muG/sigmaG)], "r--")
ax2.axvline(0)
ax2.axhline(0)
print("Ultimate Probability of Failure = " + str(StandardNormalCDF(-muG/sigmaG)))

print("FORM: " + str(StandardNormalCDF(-CalculateFORM())))


        
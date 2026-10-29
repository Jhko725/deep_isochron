from scipy import interpolate
import numpy as np
from auto import *

def createMultSegPeriodCopy2D(rSol,probStr,nCopies):
    solution    = rSol
    PARE={}
    for i in range(1,81):
        aux     = 'PAR(%d)'%(i)
        PARE[i] = solution[aux]
    PARE[14] = PARE[11]*nCopies
    m           = len(solution['U(1)'])
    iniSol      = np.zeros(((m-1)*nCopies+1, 11))
    iniSol[:,0] = np.linspace(0,1,(m-1)*nCopies+1)
    for i in range(1,9):
        tck = interpolate.splrep(solution['t'],solution['U(' + str(i) + ')'], s=0)
        iniSol[:,i] = interpolate.splev(iniSol[:,0], tck, der=0)
    # Create Full Periodic Solution
    auxU   = {}
    if PARE[12]<=0:
        auxT     = solution['t']
        auxU[1]  = solution['U(3)']
        auxU[2]  = solution['U(4)']
        PARE[12] = 0
    elif PARE[13] <= 0:
        auxT     = solution['t']
        auxU[1]  = solution['U(3)']
        auxU[2]  = solution['U(4)']
        PARE[13] = 0
    else:
        auxT   = np.concatenate((solution['t'][0:-1]*PARE[13], PARE[13]+solution['t']*PARE[12]))/PARE[11]
        auxU[1]= np.concatenate((solution['U(3)'][0:-1], solution['U(1)']))
        auxU[2]= np.concatenate((solution['U(4)'][0:-1], solution['U(2)']))
    for i in range(1,3):
        auxVectorT  = auxT[0:-1]
        auxVectorU  = auxU[i][0:-1]
        for j in range(1,nCopies):
            auxVectorT = np.concatenate((auxVectorT, j+auxT[0:-1])) 
            auxVectorU = np.concatenate((auxVectorU,auxU[i][0:-1])) 
        auxVectorT  = np.concatenate((auxVectorT, [nCopies]))
        auxVectorU  = np.concatenate((auxVectorU,[auxU[i][0]])) 
        tck         = interpolate.splrep(auxVectorT/nCopies,auxVectorU, s=0)
        iniSol[:,i+8] = interpolate.splev(iniSol[:,0], tck, der=0)
    rApp = run(np.transpose(iniSol),PAR=PARE,e=probStr+'Iso', c=probStr, IPS  =   4, ICP=[36,34,12,11,72,73,74,75], NMX=1, ISW=1, NDIM = 10, NBC = 13, NINT=0)
    return rApp

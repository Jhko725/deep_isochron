from scipy import interpolate
import numpy as np

def createMultSegPeriodCopy2D(rSol,nCopies):
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

###################################################################################################
######## Corresponding number of the parameter for coupleL.f90
######## 1  :  a
######## 2  :  gamma
######## 3  :  kappa
######## 4  :  delta
######## 5  :  tau
###################################################################################################
probStr    = 'fncMultiSeg'
PARO       = {}
PARO[1]    =  0.7         # a
PARO[2]    =  0.8         # b
PARO[3]    =  3.0         # c
PARO[4]    = -0.4         # z

rPre    = {}

rPre[0] = run([0.9066, -0.2582],e= probStr + 'Iso', c = probStr, PAR=PARO, ICP =  [3,80], DS=-0.01,NPR=3,NMX=100,SP=['HB1']) 
rPre[1] = run(rPre[0]('HB1'), IPS= 2, ISW=-1, ICP =  [3,11,80], DS=0.1,NPR=3,NMX=100,SP=[],UZSTOP={3:3})
rPre[2] = run(rPre[1]('UZ1'), IPS= 4, ISW= 1, NDIM=4, NBC=6, NINT=0, ICP =  [13,12,11], PAR={12:rPre[1]('UZ1')['PAR(11)']}, UZSTOP={13:1.0})
rPre[3] = run(rPre[2]('UZ1'), ICP =  [71,12,11], UZSTOP={71:0.5})
rPre[4] = run(rPre[3]('UZ1'), NDIM=8, NBC=11, ICP =  [34,35,12,11], SP=['BP2'], PAR={34:1.2,35:0}, DS=-0.1,DSMAX=1e-01)
rPre[5] = run(rPre[4]('BP1'), ISW=-1, UZSTOP={35:1.0},SP=['LP0','BP0'])

phaseVec = np.arange(0,1.01,0.01)
rPre[6]  = run(rPre[5]('UZ1'), ICP =  [13,34,12,11], SP=[], DS=-0.01, DSMAX=0.05, NMX=2000,NPR=0,UZR={13:rPre[5]('UZ1')['PAR(11)']*phaseVec},RL0=0, RL1=rPre[5]('UZ1')['PAR(11)']*1.01)
rPre[7]  = run(rPre[5]('UZ1'), ICP =  [13,34,12,11], SP=[], DS=0.01, DSMAX=0.05, NMX=2000,NPR=0,UZR={13:rPre[5]('UZ1')['PAR(11)']*phaseVec},RL0=0, RL1=rPre[5]('UZ1')['PAR(11)']*1.01)
rPre[8]  = rl(merge(rPre[6]+rPre[7]))

numUZR   = len(rPre[8]('UZ'))
for i in range(1,numUZR+1):
    rIso      = {}
    rIso[1]   = createMultSegPeriodCopy2D(rPre[8]('UZ'+str(i)),5)
    rIso[2]   = createMultSegPeriodCopy2D(rPre[8]('UZ'+str(i)),1)
    rIso[3]   = run(rIso[1], DS=1e-05,DSMAX=10,NMX=8000,NPR=200,UZSTOP={36:[1e-06,-1e-6]},SP=['LP0','BP0'])
    rIso[4]   = run(rIso[2], DS=-1e-05,DSMAX=10,NMX=8000,NPR=200,UZSTOP={36:[1e-06,-1e-6]},SP=['LP0','BP0'])
    data      = np.column_stack((np.concatenate((rIso[3]['PAR(72)'][::-1],rIso[4]['PAR(72)'])),np.concatenate((rIso[3]['PAR(73)'][::-1],rIso[4]['PAR(73)'])),np.concatenate((rIso[3]['PAR(74)'][::-1],rIso[4]['PAR(74)'])),np.concatenate((rIso[3]['PAR(75)'][::-1],rIso[4]['PAR(75)']))))
    np.savetxt('./dataIso/' + probStr + '_Iso_' + str(i) +'.csv', data, delimiter='\t')

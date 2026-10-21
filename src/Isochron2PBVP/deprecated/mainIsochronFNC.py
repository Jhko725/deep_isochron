from scipy import interpolate
import numpy as np

def createSecondCopyPeriodicSolution2D(rSol):
    solution = rSol
    m        = len(solution['U(1)'])
    iniSol   = np.zeros((m, 7))
    for i in range(1,5):
        iniSol[:,i] = solution['U(' + str(i) + ')']
    ##
    iniSol[:,0] = solution['t']
    for i in range(5,7):
        iniSol[:,i] = solution['U(' + str(i-4) + ')']
    PARE={}
    for i in range(1,81):
        aux     = 'PAR(%d)'%(i)
        PARE[i] = solution[aux]
    PARE[12] = PARE[11]
    rApp = run(np.transpose(iniSol),PAR=PARE,e=probStr+'Iso', c=probStr, IPS  =   4, ICP=[36,34,11], NMX=1, ISW=1, NDIM = 6, NBC = 7,NINT=1)
    return rApp

def createMultiplePeriodCopyPeriodicSolution2D(rSol,nCopies):
    solution    = rSol
    m           = len(solution['U(1)'])
    iniSol      = np.zeros(((m-1)*nCopies+1, 9))
    iniSol[:,0] = np.linspace(0,1,(m-1)*nCopies+1)
    for i in range(1,5):
        tck = interpolate.splrep(solution['t'],solution['U(' + str(i) + ')'], s=0)
        iniSol[:,i] = interpolate.splev(iniSol[:,0], tck, der=0)
    for i in range(5,7):
        auxVectorT  = solution['t'][0:-1]
        auxVectorU  = solution['U(' + str(i-4) + ')'][0:-1]
        for j in range(1,nCopies):
            auxVectorT = np.concatenate((auxVectorT, j+solution['t'][0:-1])) 
            auxVectorU = np.concatenate((auxVectorU,solution['U(' + str(i-4) + ')'][0:-1])) 
        auxVectorT  = np.concatenate((auxVectorT, [nCopies]))
        auxVectorU  = np.concatenate((auxVectorU,[solution['U(' + str(i-4) + ')'][0]])) 
        tck         = interpolate.splrep(auxVectorT/nCopies,auxVectorU, s=0)
        iniSol[:,i] = interpolate.splev(iniSol[:,0], tck, der=0)
    for i in range(7,9):
        tck = interpolate.splrep(solution['t'],solution['U(' + str(i-6) + ')'], s=0)
        iniSol[:,i] = interpolate.splev(iniSol[:,0], tck, der=0)
    #iniSol[:,7]  = np.ones((m-1)*nCopies+1)*solution['U(1)'][0]
    #iniSol[:,8]  = np.ones((m-1)*nCopies+1)*solution['U(2)'][0]
    PARE={}
    for i in range(1,81):
        aux     = 'PAR(%d)'%(i)
        PARE[i] = solution[aux]
    PARE[12] = PARE[11]*nCopies
    PARE[13] = PARE[11]
    PARE[14] = 1.0
    rApp = run(np.transpose(iniSol),PAR=PARE,e=probStr+'Iso', c=probStr, IPS  =   4, ICP=[36,34,14,70,11], NMX=1, ISW=1, NDIM = 8, NBC = 12, NINT=0)
    return rApp

#rPreCon[1]   = initializeTwoVectorBundleLin13(U,pPer,1,1,'1','2',pPlane=[1.2,0.0,0.0,0.0],nPlane=[1.0,0.0,0.0,0.0],DSMAXaux=1e-02,FloqInit=-1)
###################################################################################################
######## Corresponding number of the parameter for coupleL.f90
######## 1  :  a
######## 2  :  gamma
######## 3  :  kappa
######## 4  :  delta
######## 5  :  tau
###################################################################################################
probStr    = 'fnc'
PARO       = {}
PARO[1]    =  0.7         # a
PARO[2]    =  0.8         # b
PARO[3]    =  3.0         # c
PARO[4]    = -0.4         # z

rPre    = {}

rPre[0] = run([0.9066, -0.2582],e= probStr + 'Iso', c = probStr, PAR=PARO, ICP =  [3,80], DS=-0.01,NPR=3,NMX=100,SP=['HB1']) 
rPre[1] = run(rPre[0]('HB1'), IPS= 2, ISW=-1, ICP =  [3,11,80], DS=0.1,NPR=3,NMX=100,SP=[],UZSTOP={3:3}) 
rPre[2] = run(rPre[1]('UZ1'), IPS= 4, ISW= 1, NDIM=4, NBC=5, NINT=1, ICP =  [34,35,11], SP=['BP2'], PAR={34:1.2,35:0}, DS=-0.1)
rPre[3] = run(rPre[2]('BP1'), ISW=-1, UZSTOP={35:1.0},SP=['LP0','BP0'])
rPre[4] = createMultiplePeriodCopyPeriodicSolution2D(rPre[3]('UZ1'),2)
rPre[5] = run(rPre[4], DS=-1e-05,DSMAX=0.005,NMX=1000,NPR=100,UZSTOP={36:[-1e-5,1e-5]},SP=['LP0','BP0'])

phaseVec = np.arange(0.1,1,0.1)
rPre[6] = run(rPre[4], ICP=[13,14,70,34,11], UZR={14:phaseVec},SP=['LP4'], DS=1e-1,DSMAX=0.1,NMX=200,NPR=5)#,EPSL = 1e-8, EPSU = 1e-8, EPSS = 1e-6,NCOL=6,NTST=400)


UZSTOP = {12:rPre[3]('UZ1')['PAR(11)']}


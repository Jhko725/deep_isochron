from scipy import interpolate
import numpy as np
import sys
from pathlib import Path

parent_folder = Path.cwd().parent
sys.path.insert(0, str(parent_folder))

from isoFunctions import createMultSegPeriodCopy2D

###################################################################################################
######## Corresponding number of the parameter for fncMultiSeg
######## 1  :  a
######## 2  :  b
######## 3  :  c
######## 4  :  z
######## 11 :  Period of periodic orbit
######## 12 :  Period of the first segment of periodic orbit
######## 13 :  Period of the second segment of periodic orbit
######## 34 :  Adjoint Floquet multipliers of the periodic orbit
######## 35 :  L_2-norm of the the first point (second segment) of the Adjoint Floquet Bundle
######## 36 :  Distance away along the linear eigenspace that is normal the Adjoint Floquet Bundle
######## 72:   Point (first coordinate) along the period orbit where the Isochron is being computed
######## 73 :  Point (second coordinate) along the period orbit where the Isochron is being computed
######## 74 :  Point (first coordinate) that makes up the Isochron
######## 75 :  Point (second coordinate) that makes up the Isochron
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
    rIso[1]   = createMultSegPeriodCopy2D(rPre[8]('UZ'+str(i)),probStr,5)
    rIso[3]   = run(rIso[1], DS=1e-05,DSMAX=10,NMX=8000,NPR=200,UZSTOP={36:[1e-06,-1e-6],74:[10,-10],74:[10,-10]},SP=['LP0','BP0'])
    rIso[4]   = run(rIso[1], DS=-1e-05,DSMAX=10,NMX=8000,NPR=200,UZSTOP={36:[1e-06,-1e-6],74:[10,-10],74:[10,-10]},SP=['LP0','BP0'])
    data      = np.column_stack((np.concatenate((rIso[3]['PAR(72)'][::-1],rIso[4]['PAR(72)'])),np.concatenate((rIso[3]['PAR(73)'][::-1],rIso[4]['PAR(73)'])),np.concatenate((rIso[3]['PAR(74)'][::-1],rIso[4]['PAR(74)'])),np.concatenate((rIso[3]['PAR(75)'][::-1],rIso[4]['PAR(75)']))))
    np.savetxt('./dataIso/' + probStr + '_Iso_' + str(i) +'.csv', data, delimiter='\t')

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:30:20 2020

@author: tjcze
"""

import scipy as sc
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, odeint
import decimal
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt 

De = decimal.Decimal


names = ['C_EDC' , 'C_EC' , 'C_HCl' , 'C_Coke' , 'C_CP' , 'C_Di' , 'C_C4H6Cl2' , 'C_C6H6' , 'C_C2H2' , 'C_C11' , 'C_C112' , 'C_R1' , 'C_R2' , 'C_R3' , 'C_R4' , 'C_R5' , 'C_R6' , 'C_VCM' ]
C = [sp.symbols('{}'.format(i)) for i in names]
Easyms = [sp.symbols('Ea_{}'.format(e + 1)) for e in range(31)]
Ksyms = [sp.symbols('K_{}'.format(k + 1)) for k in range(31)]
C0 = [194.01463146683884, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]
# ks = [5.9E+15,1.3E13,1E+12,5E+11,1.2E+13,2E+11,1E+13,1E+13,1.7E+13,1.2E+13,1.7E+13,91000000000,1.2E+14,5E+11,20000000000,3E+11,2.1E+14,5E+14,2E+13,1E+14,1.6E+14]
ks = [5900000000.0, 13000000.0, 1000000.0, 500000.0, 12000000.0, 200000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 91000.0, 120000000.0, 500000.0, 20000.0, 300000.0, 210000000.0, 500000000.0, 20000000.0, 100000000.0, 160000000.0]
# Eas = [342.0,7.0,42.0,45.0,34.0,48.0,13.0,12.0,4.0,6.0,15.0,0.0,56.0,31.0,30.0,61.0,84.0,90.0,70.0,20.0,70.0]
v_0 = 9.50
taut = 1.0 /v_0
T = 773.15
R = 8.314
Eas = [342000.0, 7000.0, 42000.0, 45000.0, 34000.0, 48000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 0.0, 56000.0, 31000.0, 30000.0, 61000.0, 84000.0, 90000.0, 70000.0, 20000.0, 70000.0]
# print(Eas)
Cdict = dict(zip(C,C0))
Kdict = dict(zip(Ksyms, ks))
Edict = dict(zip(Easyms,Eas))
ell = ['Kv','Ev']
EE = ['K','Ea']
SS = [Ksyms, Easyms]
CL = str('{} = {}'.format([*C],'C')).replace('[','').replace(']','')
KL = str(r'{} = {}'.format([*Ksyms],'Kv')).replace('[','').replace(']','')
EL = str(r'{} = {}'.format([*Easyms],'Ev')).replace('[','').replace(']','')

ffpath = r'C:\Users\tjcze\Desktop\Thesis\Python\symbolgen'
with open("{}\RHSLists.txt".format(ffpath),'w') as output:
       for i in range(len(ell)):
              output.write(str('{} = kargs.get({}) \n'.format(EE[i],'{}{}{}'.format("'",EE[i],"'"))))
              output.write('{} = list({}.values()) \n'.format(ell[i],EE[i]))
              output.write(str('{} = {} \n'.format([*SS[i]],'{}'.format(ell[i]))).replace('[','').replace(']',''))
       output.write(str('{} = {} \n'.format([*C],'{}'.format('C'))).replace('[','').replace(']',''))
argsr = {}
argsr['K'] = Kdict
argsr['Ea'] = Edict
def RHS(z, C, T, R, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_10, K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20, K_21, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21):
       C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_VCM = C 

       return [-C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T)) - C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T)) - C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T)) - C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T)) - C_EDC*C_R6*K_6*np.exp(-Ea_6/(R*T)) - C_EDC*K_1*np.exp(-Ea_1/(R*T)),
              -C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T)) + C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T)) + C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T)),
              C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T)) + C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T)) + 2*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T)) + C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T)) + C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T)) + C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T)) + C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T)) + C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T)),
              2*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T)),
              C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T)),
              -C_Di*C_R1*K_19*np.exp(-Ea_19/(R*T)) + C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T)) + C_R6*K_19*np.exp(-Ea_19/(R*T)),
              C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T)),
              0.5*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T)),
              -2*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T)) - 1.0*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T)) - C_C2H2*C_R1*K_18*np.exp(-Ea_18/(R*T)) + C_R5*K_18*np.exp(-Ea_18/(R*T)),
              -C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T)) + C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T)),
              -C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T)) + C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T)),
              -C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T)) - C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T)) + 0.5*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T)) - 2*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T)) - C_C2H2*C_R1*K_18*np.exp(-Ea_18/(R*T)) - C_Di*C_R1*K_19*np.exp(-Ea_19/(R*T)) - C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T)) - C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T)) + C_EDC*K_1*np.exp(-Ea_1/(R*T)) - C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T)) - C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T)) - C_R1*C_VCM*K_12*np.exp(-Ea_12/(R*T)) - C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T)) - C_R1*C_VCM*K_17*np.exp(-Ea_17/(R*T)) + C_R3*K_17*np.exp(-Ea_17/(R*T)) + C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T)) + C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T)) + C_R5*K_18*np.exp(-Ea_18/(R*T)) + C_R6*K_19*np.exp(-Ea_19/(R*T)),
              C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T)) - C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T)) + C_EDC*K_1*np.exp(-Ea_1/(R*T)) - C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T)) - C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T)),
              C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T)) + C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T)) + C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T)) + C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T)) + C_EDC*C_R6*K_6*np.exp(-Ea_6/(R*T)) - C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T)) - C_R3*K_17*np.exp(-Ea_17/(R*T)),
              C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T)) - C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T)) + C_R1*C_VCM*K_12*np.exp(-Ea_12/(R*T)) - C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T)),
              -0.5*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T)) + C_C2H2*C_R1*K_18*np.exp(-Ea_18/(R*T)) - C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T)) + C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T)) + C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T)) - C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T)) - C_R5*K_18*np.exp(-Ea_18/(R*T)),
              C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T)) + C_Di*C_R1*K_19*np.exp(-Ea_19/(R*T)) - C_EDC*C_R6*K_6*np.exp(-Ea_6/(R*T)) - C_R6*K_19*np.exp(-Ea_19/(R*T)),
              C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T)) + C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T)) - C_R1*C_VCM*K_12*np.exp(-Ea_12/(R*T)) - C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T)) - C_R1*C_VCM*K_17*np.exp(-Ea_17/(R*T)) - C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T)) + C_R3*K_17*np.exp(-Ea_17/(R*T)) - C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T)) - C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T))]
                                                 

def jacob(z, C, T, R, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_10, K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20, K_21, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21):
       C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_VCM = C 

       
       JacN = [[-C_R1*K_2*np.exp(-Ea_2/(R*T)) - C_R2*K_3*np.exp(-Ea_3/(R*T)) - C_R4*K_4*np.exp(-Ea_4/(R*T)) - C_R5*K_5*np.exp(-Ea_5/(R*T)) - C_R6*K_6*np.exp(-Ea_6/(R*T)) - K_1*np.exp(-Ea_1/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EDC*K_2*np.exp(-Ea_2/(R*T)), -C_EDC*K_3*np.exp(-Ea_3/(R*T)), 0, -C_EDC*K_4*np.exp(-Ea_4/(R*T)), -C_EDC*K_5*np.exp(-Ea_5/(R*T)), -C_EDC*K_6*np.exp(-Ea_6/(R*T)), 0],
              [C_R2*K_3*np.exp(-Ea_3/(R*T)), -C_R1*K_9*np.exp(-Ea_9/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EC*K_9*np.exp(-Ea_9/(R*T)), C_EDC*K_3*np.exp(-Ea_3/(R*T)) + C_VCM*K_14*np.exp(-Ea_14/(R*T)), 0, 0, 0, 0, C_R2*K_14*np.exp(-Ea_14/(R*T))],
              [C_R1*K_2*np.exp(-Ea_2/(R*T)), C_R1*K_9*np.exp(-Ea_9/(R*T)), 0, 0, 0, 0, 0, 0, 2*C_R1**2*K_21*np.exp(-Ea_21/(R*T)), C_R1*K_10*np.exp(-Ea_10/(R*T)), C_R1*K_11*np.exp(-Ea_11/(R*T)), C_C11*K_10*np.exp(-Ea_10/(R*T)) + C_C112*K_11*np.exp(-Ea_11/(R*T)) + 4*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T)) + C_EC*K_9*np.exp(-Ea_9/(R*T)) + C_EDC*K_2*np.exp(-Ea_2/(R*T)) + C_R2*K_7*np.exp(-Ea_7/(R*T)) + C_R3*K_8*np.exp(-Ea_8/(R*T)) + C_VCM*K_13*np.exp(-Ea_13/(R*T)), C_R1*K_7*np.exp(-Ea_7/(R*T)), C_R1*K_8*np.exp(-Ea_8/(R*T)), 0, 0, 0, C_R1*K_13*np.exp(-Ea_13/(R*T))],
              [0, 0, 0, 0, 0, 0, 0, 0, 2*C_R1**2*K_21*np.exp(-Ea_21/(R*T)), 0, 0, 4*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T)), 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*K_16*np.exp(-Ea_16/(R*T)), 0, C_R5*K_16*np.exp(-Ea_16/(R*T))],
              [0, 0, 0, 0, 0, -C_R1*K_19*np.exp(-Ea_19/(R*T)), 0, 0, 0, 0, 0, -C_Di*K_19*np.exp(-Ea_19/(R*T)) + C_R3*K_8*np.exp(-Ea_8/(R*T)), 0, C_R1*K_8*np.exp(-Ea_8/(R*T)), 0, 0, K_19*np.exp(-Ea_19/(R*T)), 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*K_15*np.exp(-Ea_15/(R*T)), 0, 0, C_R4*K_15*np.exp(-Ea_15/(R*T))],
              [0, 0, 0, 0, 0, 0, 0, 0, 1.0*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T)), 0, 0, 0, 0, 0, 0, 0.5*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T)), 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -4*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T)) - 1.0*C_R1**2*K_21*np.exp(-Ea_21/(R*T)) - C_R1*K_18*np.exp(-Ea_18/(R*T)), 0, 0, -2.0*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T)) - C_C2H2*K_18*np.exp(-Ea_18/(R*T)), 0, 0, 0, -2*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T)) + K_18*np.exp(-Ea_18/(R*T)), 0, 0],
              [C_R4*K_4*np.exp(-Ea_4/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*K_10*np.exp(-Ea_10/(R*T)), 0, -C_C11*K_10*np.exp(-Ea_10/(R*T)), 0, 0, C_EDC*K_4*np.exp(-Ea_4/(R*T)), 0, 0, 0],
              [C_R5*K_5*np.exp(-Ea_5/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*K_11*np.exp(-Ea_11/(R*T)), -C_C112*K_11*np.exp(-Ea_11/(R*T)), 0, 0, 0, C_EDC*K_5*np.exp(-Ea_5/(R*T)), 0, 0],
              [-C_R1*K_2*np.exp(-Ea_2/(R*T)) + K_1*np.exp(-Ea_1/(R*T)), -C_R1*K_9*np.exp(-Ea_9/(R*T)), 0, 0, 0, -C_R1*K_19*np.exp(-Ea_19/(R*T)), 0, 0, 1.0*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T)) - 2*C_R1**2*K_21*np.exp(-Ea_21/(R*T)) - C_R1*K_18*np.exp(-Ea_18/(R*T)), -C_R1*K_10*np.exp(-Ea_10/(R*T)), -C_R1*K_11*np.exp(-Ea_11/(R*T)), -C_C11*K_10*np.exp(-Ea_10/(R*T)) - C_C112*K_11*np.exp(-Ea_11/(R*T)) - 4*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T)) - C_C2H2*K_18*np.exp(-Ea_18/(R*T)) - C_Di*K_19*np.exp(-Ea_19/(R*T)) - C_EC*K_9*np.exp(-Ea_9/(R*T)) - C_EDC*K_2*np.exp(-Ea_2/(R*T)) - C_R2*K_7*np.exp(-Ea_7/(R*T)) - C_R3*K_8*np.exp(-Ea_8/(R*T)) - C_VCM*K_12*np.exp(-Ea_12/(R*T)) - C_VCM*K_13*np.exp(-Ea_13/(R*T)) - C_VCM*K_17*np.exp(-Ea_17/(R*T)), -C_R1*K_7*np.exp(-Ea_7/(R*T)), -C_R1*K_8*np.exp(-Ea_8/(R*T)) + K_17*np.exp(-Ea_17/(R*T)), C_VCM*K_15*np.exp(-Ea_15/(R*T)), 0.5*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T)) + C_VCM*K_16*np.exp(-Ea_16/(R*T)) + K_18*np.exp(-Ea_18/(R*T)), K_19*np.exp(-Ea_19/(R*T)), -C_R1*K_12*np.exp(-Ea_12/(R*T)) - C_R1*K_13*np.exp(-Ea_13/(R*T)) - C_R1*K_17*np.exp(-Ea_17/(R*T)) + C_R4*K_15*np.exp(-Ea_15/(R*T)) + C_R5*K_16*np.exp(-Ea_16/(R*T))],
              [-C_R2*K_3*np.exp(-Ea_3/(R*T)) + K_1*np.exp(-Ea_1/(R*T)), C_R1*K_9*np.exp(-Ea_9/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EC*K_9*np.exp(-Ea_9/(R*T)) - C_R2*K_7*np.exp(-Ea_7/(R*T)), -C_EDC*K_3*np.exp(-Ea_3/(R*T)) - C_R1*K_7*np.exp(-Ea_7/(R*T)) - C_VCM*K_14*np.exp(-Ea_14/(R*T)), 0, 0, 0, 0, -C_R2*K_14*np.exp(-Ea_14/(R*T))],
              [C_R1*K_2*np.exp(-Ea_2/(R*T)) + C_R2*K_3*np.exp(-Ea_3/(R*T)) + C_R4*K_4*np.exp(-Ea_4/(R*T)) + C_R5*K_5*np.exp(-Ea_5/(R*T)) + C_R6*K_6*np.exp(-Ea_6/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EDC*K_2*np.exp(-Ea_2/(R*T)) - C_R3*K_8*np.exp(-Ea_8/(R*T)), C_EDC*K_3*np.exp(-Ea_3/(R*T)), -C_R1*K_8*np.exp(-Ea_8/(R*T)) - K_17*np.exp(-Ea_17/(R*T)), C_EDC*K_4*np.exp(-Ea_4/(R*T)), C_EDC*K_5*np.exp(-Ea_5/(R*T)), C_EDC*K_6*np.exp(-Ea_6/(R*T)), 0],
              [-C_R4*K_4*np.exp(-Ea_4/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, C_R1*K_10*np.exp(-Ea_10/(R*T)), 0, C_C11*K_10*np.exp(-Ea_10/(R*T)) + C_VCM*K_12*np.exp(-Ea_12/(R*T)), 0, 0, -C_EDC*K_4*np.exp(-Ea_4/(R*T)) - C_VCM*K_15*np.exp(-Ea_15/(R*T)), 0, 0, C_R1*K_12*np.exp(-Ea_12/(R*T)) - C_R4*K_15*np.exp(-Ea_15/(R*T))],
              [-C_R5*K_5*np.exp(-Ea_5/(R*T)), 0, 0, 0, 0, 0, 0, 0, -1.0*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T)) + C_R1*K_18*np.exp(-Ea_18/(R*T)), 0, 0, C_C2H2*K_18*np.exp(-Ea_18/(R*T)) + C_VCM*K_13*np.exp(-Ea_13/(R*T)), C_VCM*K_14*np.exp(-Ea_14/(R*T)), 0, 0, -0.5*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T)) - C_EDC*K_5*np.exp(-Ea_5/(R*T)) - C_VCM*K_16*np.exp(-Ea_16/(R*T)) - K_18*np.exp(-Ea_18/(R*T)), 0, C_R1*K_13*np.exp(-Ea_13/(R*T)) + C_R2*K_14*np.exp(-Ea_14/(R*T)) - C_R5*K_16*np.exp(-Ea_16/(R*T))],
              [-C_R6*K_6*np.exp(-Ea_6/(R*T)), 0, 0, 0, 0, C_R1*K_19*np.exp(-Ea_19/(R*T)), 0, 0, 0, 0, C_R1*K_11*np.exp(-Ea_11/(R*T)), C_C112*K_11*np.exp(-Ea_11/(R*T)) + C_Di*K_19*np.exp(-Ea_19/(R*T)), 0, 0, 0, 0, -C_EDC*K_6*np.exp(-Ea_6/(R*T)) - K_19*np.exp(-Ea_19/(R*T)), 0],
              [C_R5*K_5*np.exp(-Ea_5/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R2*K_7*np.exp(-Ea_7/(R*T)) - C_VCM*K_12*np.exp(-Ea_12/(R*T)) - C_VCM*K_13*np.exp(-Ea_13/(R*T)) - C_VCM*K_17*np.exp(-Ea_17/(R*T)), C_R1*K_7*np.exp(-Ea_7/(R*T)) - C_VCM*K_14*np.exp(-Ea_14/(R*T)), K_17*np.exp(-Ea_17/(R*T)), -C_VCM*K_15*np.exp(-Ea_15/(R*T)), C_EDC*K_5*np.exp(-Ea_5/(R*T)) - C_VCM*K_16*np.exp(-Ea_16/(R*T)), 0, -C_R1*K_12*np.exp(-Ea_12/(R*T)) - C_R1*K_13*np.exp(-Ea_13/(R*T)) - C_R1*K_17*np.exp(-Ea_17/(R*T)) - C_R2*K_14*np.exp(-Ea_14/(R*T)) - C_R4*K_15*np.exp(-Ea_15/(R*T)) - C_R5*K_16*np.exp(-Ea_16/(R*T))]]
                                   
       return JacN

tend = 30.0
tdist = tend*v_0
tnum = 100000
inc = tdist/tnum
timee = np.linspace(0,int(tend), num=tnum)
dist = [inc*i for i in range(tnum)]
# print(dist)
Z = sp.symbols('Z')
res = solve_ivp(RHS, [0.0, tdist], C0 , method = 'Radau', t_eval=dist,  args=(773.15, 8.314, 5.9E15, 13000000.0, 1000000.0, 500000.0, 12000000.0, 200000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 91000.0, 120000000.0, 500000.0, 20000.0, 300000.0, 2.1E+8,5E+8,2E+7, 100000000.0, 160000000.0, 342000.0, 7000.0, 42000.0, 45000.0, 34000.0, 48000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 0.0, 56000.0, 31000.0, 30000.0, 61000.0, 84000.0, 90000.0, 70000.0, 20000.0, 70000.0), jac=jacob, rtol=1E-13, atol=1E-13) #  , first_step=1E-2, max_step=1E-3, jac= lambda Z, C: jacob(Z,C, **args), rtol=1E-9, atol=1E-9
Edc = res.y[0]
time = np.array(res.t)
print(Edc[:])
final = Edc[-1]
print('Final conversion = {} %'.format((100.0*((C0[0] - final)/C0[0]))))
fig = plt.figure()
plt1 = plt.plot(dist, Edc, 'b-', label='Edc')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title("EDC")
plt.grid()
plt.show()

       

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:30:20 2020

@author: tjcze
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 


names = ['C_EDC' , 'C_EC' , 'C_HCl' , 'C_Coke' , 'C_CP' , 'C_Di' , 'C_C4H6Cl2' , 'C_C6H6' , 'C_C2H2' , 'C_C11' , 'C_C112' , 'C_R1' , 'C_R2' , 'C_R3' , 'C_R4' , 'C_R5' , 'C_R6' , 'C_VCM' ]
C = [sp.symbols('{}'.format(i)) for i in names]
Temperature = float(500) #[°C]
Temp_K = (Temperature + 273.15) #[K]
Pstart_atm = float(12.0)
PascalP  = Pstart_atm*101325.0 #[Pa]
initedc = (PascalP/(Temp_K*8.314))
CCl4_p = float(0.0002)
EDC_p = 1.0 - CCl4_p
EDC0 = EDC_p*initedc
CCL40 = CCl4_p*initedc
Easyms = [sp.symbols('Ea_{}'.format(e + 1)) for e in range(31)]
Ksyms = [sp.symbols('K_{}'.format(k + 1)) for k in range(31)]
C0 = [EDC0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, CCL40, 0.0, 0.0]
ks = [5900000000000000.0, 2200000.0, 13000000.0, 12000000.0, 1000000.0, 500000.0, 200000.0, 100000.0, 1000000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 17000000.0, 16000000.0, 91000.0, 120000000.0, 300000.0, 20000.0, 500000.0, 210000000000000.0, 500000000000000.0, 20000000000000.0, 25000000000000.0, 1000000.0, 500000.0, 500000.0, 10000000.0, 100000000.0, 160000000.0]
v_0 = 9.50
taut = 1.0 /v_0
T = 773.15
R = 8.314
Eas = [342000.0, 230000.0, 7000.0, 34000.0, 42000.0, 45000.0, 48000.0, 56000.0, 63000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 17000.0, 14000.0, 0.0, 56000.0, 61000.0, 30000.0, 31000.0, 84000.0, 90000.0, 70000.0, 70000.0, 33000.0, 33000.0, 33000.0, 13000.0, 20000.0, 70000.0]

def RHS(z, C, T, R, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_10, K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20, K_21, K_22, K_23, K_24, K_25, K_26, K_27, K_28, K_29, K_30, K_31, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Ea_22, Ea_23, Ea_24, Ea_25, Ea_26, Ea_27, Ea_28, Ea_29, Ea_30, Ea_31):
       C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_Tri, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_C1112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_R7, C_R8, C_CCl4, C_CHCl3, C_VCM = C 

       return [-C_EDC*C_R1*K_3*np.exp(-Ea_3/(R*T)) - C_EDC*C_R2*K_4*np.exp(-Ea_4/(R*T)) - C_EDC*C_R4*K_5*np.exp(-Ea_5/(R*T)) - C_EDC*C_R5*K_6*np.exp(-Ea_6/(R*T)) - C_EDC*C_R6*K_7*np.exp(-Ea_7/(R*T)) - C_EDC*C_R7*K_8*np.exp(-Ea_8/(R*T)) - C_EDC*C_R8*K_9*np.exp(-Ea_9/(R*T)) - C_EDC*K_1*np.exp(-Ea_1/(R*T)),
              -C_EC*C_R1*K_12*np.exp(-Ea_12/(R*T)) + C_EDC*C_R2*K_4*np.exp(-Ea_4/(R*T)) + C_R2*C_VCM*K_19*np.exp(-Ea_19/(R*T)),
              C_C11*C_R1*K_13*np.exp(-Ea_13/(R*T)) + C_C1112*C_R1*K_15*np.exp(-Ea_15/(R*T)) + C_C112*C_R1*K_14*np.exp(-Ea_14/(R*T)) + 2*C_C2H2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)) + C_CHCl3*C_R1*K_16*np.exp(-Ea_16/(R*T)) + C_EC*C_R1*K_12*np.exp(-Ea_12/(R*T)) + C_EDC*C_R1*K_3*np.exp(-Ea_3/(R*T)) + C_R1*C_R2*K_10*np.exp(-Ea_10/(R*T)) + C_R1*C_R3*K_11*np.exp(-Ea_11/(R*T)) + C_R1*C_VCM*K_18*np.exp(-Ea_18/(R*T)),
              2*C_C2H2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)),
              C_R5*C_VCM*K_21*np.exp(-Ea_21/(R*T)),
              C_CCl4*C_R5*K_27*np.exp(-Ea_27/(R*T)) - C_Di*C_R1*K_24*np.exp(-Ea_24/(R*T)) + C_R1*C_R3*K_11*np.exp(-Ea_11/(R*T)) + C_R6*C_R8*K_29*np.exp(-Ea_29/(R*T)) + C_R6*K_24*np.exp(-Ea_24/(R*T)),
              -C_R1*C_Tri*K_25*np.exp(-Ea_25/(R*T)) + C_R7*K_25*np.exp(-Ea_25/(R*T)),
              C_R4*C_VCM*K_20*np.exp(-Ea_20/(R*T)),
              2*C_C2H2**2*C_R5*K_30*np.exp(-Ea_30/(R*T)),
              -2*C_C2H2**2*C_R5*K_30*np.exp(-Ea_30/(R*T)) - 2*C_C2H2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)) - C_C2H2*C_R1*K_23*np.exp(-Ea_23/(R*T)) + C_R5*K_23*np.exp(-Ea_23/(R*T)),
              -C_C11*C_R1*K_13*np.exp(-Ea_13/(R*T)) + C_EDC*C_R4*K_5*np.exp(-Ea_5/(R*T)),
              -C_C112*C_R1*K_14*np.exp(-Ea_14/(R*T)) + C_CCl4*C_R4*K_26*np.exp(-Ea_26/(R*T)) + C_EDC*C_R6*K_7*np.exp(-Ea_7/(R*T)),
              -C_C1112*C_R1*K_15*np.exp(-Ea_15/(R*T)) - C_C1112*C_R8*K_28*np.exp(-Ea_28/(R*T)) + C_CCl4*C_R6*K_28*np.exp(-Ea_28/(R*T)) + C_EDC*C_R7*K_8*np.exp(-Ea_8/(R*T)),
              -C_C11*C_R1*K_13*np.exp(-Ea_13/(R*T)) - C_C1112*C_R1*K_15*np.exp(-Ea_15/(R*T)) - C_C112*C_R1*K_14*np.exp(-Ea_14/(R*T)) + 2*C_C2H2**2*C_R5*K_30*np.exp(-Ea_30/(R*T)) - 2*C_C2H2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)) - C_C2H2*C_R1*K_23*np.exp(-Ea_23/(R*T)) + C_CCl4*K_2*np.exp(-Ea_2/(R*T)) - C_CHCl3*C_R1*K_16*np.exp(-Ea_16/(R*T)) - C_Di*C_R1*K_24*np.exp(-Ea_24/(R*T)) - C_EC*C_R1*K_12*np.exp(-Ea_12/(R*T)) - C_EDC*C_R1*K_3*np.exp(-Ea_3/(R*T)) + C_EDC*K_1*np.exp(-Ea_1/(R*T)) - C_R1*C_R2*K_10*np.exp(-Ea_10/(R*T)) - C_R1*C_R3*K_11*np.exp(-Ea_11/(R*T)) - C_R1*C_Tri*K_25*np.exp(-Ea_25/(R*T)) - C_R1*C_VCM*K_17*np.exp(-Ea_17/(R*T)) - C_R1*C_VCM*K_18*np.exp(-Ea_18/(R*T)) - C_R1*C_VCM*K_22*np.exp(-Ea_22/(R*T)) + C_R3*K_22*np.exp(-Ea_22/(R*T)) + C_R4*C_VCM*K_20*np.exp(-Ea_20/(R*T)) + C_R5*C_VCM*K_21*np.exp(-Ea_21/(R*T)) + C_R5*K_23*np.exp(-Ea_23/(R*T)) + C_R6*K_24*np.exp(-Ea_24/(R*T)) + C_R7*K_25*np.exp(-Ea_25/(R*T)),
              C_EC*C_R1*K_12*np.exp(-Ea_12/(R*T)) - C_EDC*C_R2*K_4*np.exp(-Ea_4/(R*T)) + C_EDC*K_1*np.exp(-Ea_1/(R*T)) - C_R1*C_R2*K_10*np.exp(-Ea_10/(R*T)) - C_R2*C_VCM*K_19*np.exp(-Ea_19/(R*T)),
              C_EDC*C_R1*K_3*np.exp(-Ea_3/(R*T)) + C_EDC*C_R2*K_4*np.exp(-Ea_4/(R*T)) + C_EDC*C_R4*K_5*np.exp(-Ea_5/(R*T)) + C_EDC*C_R5*K_6*np.exp(-Ea_6/(R*T)) + C_EDC*C_R6*K_7*np.exp(-Ea_7/(R*T)) + C_EDC*C_R7*K_8*np.exp(-Ea_8/(R*T)) - C_R1*C_R3*K_11*np.exp(-Ea_11/(R*T)) + C_R1*C_VCM*K_22*np.exp(-Ea_22/(R*T)) - C_R3*K_22*np.exp(-Ea_22/(R*T)),
              C_C11*C_R1*K_13*np.exp(-Ea_13/(R*T)) - C_CCl4*C_R4*K_26*np.exp(-Ea_26/(R*T)) - C_EDC*C_R4*K_5*np.exp(-Ea_5/(R*T)) + C_R1*C_VCM*K_17*np.exp(-Ea_17/(R*T)) - C_R4*C_VCM*K_20*np.exp(-Ea_20/(R*T)),
              -2*C_C2H2**2*C_R5*K_30*np.exp(-Ea_30/(R*T)) + C_C2H2*C_R1*K_23*np.exp(-Ea_23/(R*T)) - C_CCl4*C_R5*K_27*np.exp(-Ea_27/(R*T)) - C_EDC*C_R5*K_6*np.exp(-Ea_6/(R*T)) + C_R1*C_VCM*K_18*np.exp(-Ea_18/(R*T)) + C_R2*C_VCM*K_19*np.exp(-Ea_19/(R*T)) - C_R5*C_VCM*K_21*np.exp(-Ea_21/(R*T)) - C_R5*K_23*np.exp(-Ea_23/(R*T)),
              C_C1112*C_R8*K_28*np.exp(-Ea_28/(R*T)) + C_C112*C_R1*K_14*np.exp(-Ea_14/(R*T)) - C_CCl4*C_R6*K_28*np.exp(-Ea_28/(R*T)) + C_Di*C_R1*K_24*np.exp(-Ea_24/(R*T)) - C_EDC*C_R6*K_7*np.exp(-Ea_7/(R*T)) - C_R6*C_R8*K_29*np.exp(-Ea_29/(R*T)) - C_R6*K_24*np.exp(-Ea_24/(R*T)),
              C_C1112*C_R1*K_15*np.exp(-Ea_15/(R*T)) - C_EDC*C_R7*K_8*np.exp(-Ea_8/(R*T)) + C_R1*C_Tri*K_25*np.exp(-Ea_25/(R*T)) - C_R7*K_25*np.exp(-Ea_25/(R*T)),
              -C_C1112*C_R8*K_28*np.exp(-Ea_28/(R*T)) + C_CCl4*C_R4*K_26*np.exp(-Ea_26/(R*T)) + C_CCl4*C_R5*K_27*np.exp(-Ea_27/(R*T)) + C_CCl4*C_R6*K_28*np.exp(-Ea_28/(R*T)) + C_CCl4*K_2*np.exp(-Ea_2/(R*T)) + C_CHCl3*C_R1*K_16*np.exp(-Ea_16/(R*T)) - C_EDC*C_R8*K_9*np.exp(-Ea_9/(R*T)) - C_R6*C_R8*K_29*np.exp(-Ea_29/(R*T)),
              C_C1112*C_R8*K_28*np.exp(-Ea_28/(R*T)) - C_CCl4*C_R4*K_26*np.exp(-Ea_26/(R*T)) - C_CCl4*C_R5*K_27*np.exp(-Ea_27/(R*T)) - C_CCl4*C_R6*K_28*np.exp(-Ea_28/(R*T)) - C_CCl4*K_2*np.exp(-Ea_2/(R*T)) + C_R6*C_R8*K_29*np.exp(-Ea_29/(R*T)),
              -C_CHCl3*C_R1*K_16*np.exp(-Ea_16/(R*T)) + C_EDC*C_R8*K_9*np.exp(-Ea_9/(R*T)),
              C_EDC*C_R5*K_6*np.exp(-Ea_6/(R*T)) + C_R1*C_R2*K_10*np.exp(-Ea_10/(R*T)) - C_R1*C_VCM*K_17*np.exp(-Ea_17/(R*T)) - C_R1*C_VCM*K_18*np.exp(-Ea_18/(R*T)) - C_R1*C_VCM*K_22*np.exp(-Ea_22/(R*T)) - C_R2*C_VCM*K_19*np.exp(-Ea_19/(R*T)) + C_R3*K_22*np.exp(-Ea_22/(R*T)) - C_R4*C_VCM*K_20*np.exp(-Ea_20/(R*T)) - C_R5*C_VCM*K_21*np.exp(-Ea_21/(R*T))]
                                   

def jacob(z, C, T, R, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_10, K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20, K_21, K_22, K_23, K_24, K_25, K_26, K_27, K_28, K_29, K_30, K_31, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Ea_22, Ea_23, Ea_24, Ea_25, Ea_26, Ea_27, Ea_28, Ea_29, Ea_30, Ea_31):
       C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_Tri, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_C1112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_R7, C_R8, C_CCl4, C_CHCl3, C_VCM = C 

       
       JacN = [[-C_R1*K_3*np.exp(-Ea_3/(R*T)) - C_R2*K_4*np.exp(-Ea_4/(R*T)) - C_R4*K_5*np.exp(-Ea_5/(R*T)) - C_R5*K_6*np.exp(-Ea_6/(R*T)) - C_R6*K_7*np.exp(-Ea_7/(R*T)) - C_R7*K_8*np.exp(-Ea_8/(R*T)) - C_R8*K_9*np.exp(-Ea_9/(R*T)) - K_1*np.exp(-Ea_1/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EDC*K_3*np.exp(-Ea_3/(R*T)), -C_EDC*K_4*np.exp(-Ea_4/(R*T)), 0, -C_EDC*K_5*np.exp(-Ea_5/(R*T)), -C_EDC*K_6*np.exp(-Ea_6/(R*T)), -C_EDC*K_7*np.exp(-Ea_7/(R*T)), -C_EDC*K_8*np.exp(-Ea_8/(R*T)), -C_EDC*K_9*np.exp(-Ea_9/(R*T)), 0, 0, 0],
              [C_R2*K_4*np.exp(-Ea_4/(R*T)), -C_R1*K_12*np.exp(-Ea_12/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EC*K_12*np.exp(-Ea_12/(R*T)), C_EDC*K_4*np.exp(-Ea_4/(R*T)) + C_VCM*K_19*np.exp(-Ea_19/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, C_R2*K_19*np.exp(-Ea_19/(R*T))],
              [C_R1*K_3*np.exp(-Ea_3/(R*T)), C_R1*K_12*np.exp(-Ea_12/(R*T)), 0, 0, 0, 0, 0, 0, 0, 2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)), C_R1*K_13*np.exp(-Ea_13/(R*T)), C_R1*K_14*np.exp(-Ea_14/(R*T)), C_R1*K_15*np.exp(-Ea_15/(R*T)), C_C11*K_13*np.exp(-Ea_13/(R*T)) + C_C1112*K_15*np.exp(-Ea_15/(R*T)) + C_C112*K_14*np.exp(-Ea_14/(R*T)) + 4*C_C2H2*C_R1*K_31*np.exp(-Ea_31/(R*T)) + C_CHCl3*K_16*np.exp(-Ea_16/(R*T)) + C_EC*K_12*np.exp(-Ea_12/(R*T)) + C_EDC*K_3*np.exp(-Ea_3/(R*T)) + C_R2*K_10*np.exp(-Ea_10/(R*T)) + C_R3*K_11*np.exp(-Ea_11/(R*T)) + C_VCM*K_18*np.exp(-Ea_18/(R*T)), C_R1*K_10*np.exp(-Ea_10/(R*T)), C_R1*K_11*np.exp(-Ea_11/(R*T)), 0, 0, 0, 0, 0, 0, C_R1*K_16*np.exp(-Ea_16/(R*T)), C_R1*K_18*np.exp(-Ea_18/(R*T))],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)), 0, 0, 0, 4*C_C2H2*C_R1*K_31*np.exp(-Ea_31/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*K_21*np.exp(-Ea_21/(R*T)), 0, 0, 0, 0, 0, C_R5*K_21*np.exp(-Ea_21/(R*T))],
              [0, 0, 0, 0, 0, -C_R1*K_24*np.exp(-Ea_24/(R*T)), 0, 0, 0, 0, 0, 0, 0, -C_Di*K_24*np.exp(-Ea_24/(R*T)) + C_R3*K_11*np.exp(-Ea_11/(R*T)), 0, C_R1*K_11*np.exp(-Ea_11/(R*T)), 0, C_CCl4*K_27*np.exp(-Ea_27/(R*T)), C_R8*K_29*np.exp(-Ea_29/(R*T)) + K_24*np.exp(-Ea_24/(R*T)), 0, C_R6*K_29*np.exp(-Ea_29/(R*T)), C_R5*K_27*np.exp(-Ea_27/(R*T)), 0, 0],
              [0, 0, 0, 0, 0, 0, -C_R1*K_25*np.exp(-Ea_25/(R*T)), 0, 0, 0, 0, 0, 0, -C_Tri*K_25*np.exp(-Ea_25/(R*T)), 0, 0, 0, 0, 0, K_25*np.exp(-Ea_25/(R*T)), 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*K_20*np.exp(-Ea_20/(R*T)), 0, 0, 0, 0, 0, 0, C_R4*K_20*np.exp(-Ea_20/(R*T))],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 4*C_C2H2*C_R5*K_30*np.exp(-Ea_30/(R*T)), 0, 0, 0, 0, 0, 0, 0, 2*C_C2H2**2*K_30*np.exp(-Ea_30/(R*T)), 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, -4*C_C2H2*C_R5*K_30*np.exp(-Ea_30/(R*T)) - 2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)) - C_R1*K_23*np.exp(-Ea_23/(R*T)), 0, 0, 0, -4*C_C2H2*C_R1*K_31*np.exp(-Ea_31/(R*T)) - C_C2H2*K_23*np.exp(-Ea_23/(R*T)), 0, 0, 0, -2*C_C2H2**2*K_30*np.exp(-Ea_30/(R*T)) + K_23*np.exp(-Ea_23/(R*T)), 0, 0, 0, 0, 0, 0],
              [C_R4*K_5*np.exp(-Ea_5/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*K_13*np.exp(-Ea_13/(R*T)), 0, 0, -C_C11*K_13*np.exp(-Ea_13/(R*T)), 0, 0, C_EDC*K_5*np.exp(-Ea_5/(R*T)), 0, 0, 0, 0, 0, 0, 0],
              [C_R6*K_7*np.exp(-Ea_7/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*K_14*np.exp(-Ea_14/(R*T)), 0, -C_C112*K_14*np.exp(-Ea_14/(R*T)), 0, 0, C_CCl4*K_26*np.exp(-Ea_26/(R*T)), 0, C_EDC*K_7*np.exp(-Ea_7/(R*T)), 0, 0, C_R4*K_26*np.exp(-Ea_26/(R*T)), 0, 0],
              [C_R7*K_8*np.exp(-Ea_8/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*K_15*np.exp(-Ea_15/(R*T)) - C_R8*K_28*np.exp(-Ea_28/(R*T)), -C_C1112*K_15*np.exp(-Ea_15/(R*T)), 0, 0, 0, 0, C_CCl4*K_28*np.exp(-Ea_28/(R*T)), C_EDC*K_8*np.exp(-Ea_8/(R*T)), -C_C1112*K_28*np.exp(-Ea_28/(R*T)), C_R6*K_28*np.exp(-Ea_28/(R*T)), 0, 0],
              [-C_R1*K_3*np.exp(-Ea_3/(R*T)) + K_1*np.exp(-Ea_1/(R*T)), -C_R1*K_12*np.exp(-Ea_12/(R*T)), 0, 0, 0, -C_R1*K_24*np.exp(-Ea_24/(R*T)), -C_R1*K_25*np.exp(-Ea_25/(R*T)), 0, 0, 4*C_C2H2*C_R5*K_30*np.exp(-Ea_30/(R*T)) - 2*C_R1**2*K_31*np.exp(-Ea_31/(R*T)) - C_R1*K_23*np.exp(-Ea_23/(R*T)), -C_R1*K_13*np.exp(-Ea_13/(R*T)), -C_R1*K_14*np.exp(-Ea_14/(R*T)), -C_R1*K_15*np.exp(-Ea_15/(R*T)), -C_C11*K_13*np.exp(-Ea_13/(R*T)) - C_C1112*K_15*np.exp(-Ea_15/(R*T)) - C_C112*K_14*np.exp(-Ea_14/(R*T)) - 4*C_C2H2*C_R1*K_31*np.exp(-Ea_31/(R*T)) - C_C2H2*K_23*np.exp(-Ea_23/(R*T)) - C_CHCl3*K_16*np.exp(-Ea_16/(R*T)) - C_Di*K_24*np.exp(-Ea_24/(R*T)) - C_EC*K_12*np.exp(-Ea_12/(R*T)) - C_EDC*K_3*np.exp(-Ea_3/(R*T)) - C_R2*K_10*np.exp(-Ea_10/(R*T)) - C_R3*K_11*np.exp(-Ea_11/(R*T)) - C_Tri*K_25*np.exp(-Ea_25/(R*T)) - C_VCM*K_17*np.exp(-Ea_17/(R*T)) - C_VCM*K_18*np.exp(-Ea_18/(R*T)) - C_VCM*K_22*np.exp(-Ea_22/(R*T)), -C_R1*K_10*np.exp(-Ea_10/(R*T)), -C_R1*K_11*np.exp(-Ea_11/(R*T)) + K_22*np.exp(-Ea_22/(R*T)), C_VCM*K_20*np.exp(-Ea_20/(R*T)), 2*C_C2H2**2*K_30*np.exp(-Ea_30/(R*T)) + C_VCM*K_21*np.exp(-Ea_21/(R*T)) + K_23*np.exp(-Ea_23/(R*T)), K_24*np.exp(-Ea_24/(R*T)), K_25*np.exp(-Ea_25/(R*T)), 0, K_2*np.exp(-Ea_2/(R*T)), -C_R1*K_16*np.exp(-Ea_16/(R*T)), -C_R1*K_17*np.exp(-Ea_17/(R*T)) - C_R1*K_18*np.exp(-Ea_18/(R*T)) - C_R1*K_22*np.exp(-Ea_22/(R*T)) + C_R4*K_20*np.exp(-Ea_20/(R*T)) + C_R5*K_21*np.exp(-Ea_21/(R*T))],
              [-C_R2*K_4*np.exp(-Ea_4/(R*T)) + K_1*np.exp(-Ea_1/(R*T)), C_R1*K_12*np.exp(-Ea_12/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EC*K_12*np.exp(-Ea_12/(R*T)) - C_R2*K_10*np.exp(-Ea_10/(R*T)), -C_EDC*K_4*np.exp(-Ea_4/(R*T)) - C_R1*K_10*np.exp(-Ea_10/(R*T)) - C_VCM*K_19*np.exp(-Ea_19/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, -C_R2*K_19*np.exp(-Ea_19/(R*T))],
              [C_R1*K_3*np.exp(-Ea_3/(R*T)) + C_R2*K_4*np.exp(-Ea_4/(R*T)) + C_R4*K_5*np.exp(-Ea_5/(R*T)) + C_R5*K_6*np.exp(-Ea_6/(R*T)) + C_R6*K_7*np.exp(-Ea_7/(R*T)) + C_R7*K_8*np.exp(-Ea_8/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EDC*K_3*np.exp(-Ea_3/(R*T)) - C_R3*K_11*np.exp(-Ea_11/(R*T)) + C_VCM*K_22*np.exp(-Ea_22/(R*T)), C_EDC*K_4*np.exp(-Ea_4/(R*T)), -C_R1*K_11*np.exp(-Ea_11/(R*T)) - K_22*np.exp(-Ea_22/(R*T)), C_EDC*K_5*np.exp(-Ea_5/(R*T)), C_EDC*K_6*np.exp(-Ea_6/(R*T)), C_EDC*K_7*np.exp(-Ea_7/(R*T)), C_EDC*K_8*np.exp(-Ea_8/(R*T)), 0, 0, 0, C_R1*K_22*np.exp(-Ea_22/(R*T))],
              [-C_R4*K_5*np.exp(-Ea_5/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R1*K_13*np.exp(-Ea_13/(R*T)), 0, 0, C_C11*K_13*np.exp(-Ea_13/(R*T)) + C_VCM*K_17*np.exp(-Ea_17/(R*T)), 0, 0, -C_CCl4*K_26*np.exp(-Ea_26/(R*T)) - C_EDC*K_5*np.exp(-Ea_5/(R*T)) - C_VCM*K_20*np.exp(-Ea_20/(R*T)), 0, 0, 0, 0, -C_R4*K_26*np.exp(-Ea_26/(R*T)), 0, C_R1*K_17*np.exp(-Ea_17/(R*T)) - C_R4*K_20*np.exp(-Ea_20/(R*T))],
              [-C_R5*K_6*np.exp(-Ea_6/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, -4*C_C2H2*C_R5*K_30*np.exp(-Ea_30/(R*T)) + C_R1*K_23*np.exp(-Ea_23/(R*T)), 0, 0, 0, C_C2H2*K_23*np.exp(-Ea_23/(R*T)) + C_VCM*K_18*np.exp(-Ea_18/(R*T)), C_VCM*K_19*np.exp(-Ea_19/(R*T)), 0, 0, -2*C_C2H2**2*K_30*np.exp(-Ea_30/(R*T)) - C_CCl4*K_27*np.exp(-Ea_27/(R*T)) - C_EDC*K_6*np.exp(-Ea_6/(R*T)) - C_VCM*K_21*np.exp(-Ea_21/(R*T)) - K_23*np.exp(-Ea_23/(R*T)), 0, 0, 0, -C_R5*K_27*np.exp(-Ea_27/(R*T)), 0, C_R1*K_18*np.exp(-Ea_18/(R*T)) + C_R2*K_19*np.exp(-Ea_19/(R*T)) - C_R5*K_21*np.exp(-Ea_21/(R*T))],
              [-C_R6*K_7*np.exp(-Ea_7/(R*T)), 0, 0, 0, 0, C_R1*K_24*np.exp(-Ea_24/(R*T)), 0, 0, 0, 0, 0, C_R1*K_14*np.exp(-Ea_14/(R*T)), C_R8*K_28*np.exp(-Ea_28/(R*T)), C_C112*K_14*np.exp(-Ea_14/(R*T)) + C_Di*K_24*np.exp(-Ea_24/(R*T)), 0, 0, 0, 0, -C_CCl4*K_28*np.exp(-Ea_28/(R*T)) - C_EDC*K_7*np.exp(-Ea_7/(R*T)) - C_R8*K_29*np.exp(-Ea_29/(R*T)) - K_24*np.exp(-Ea_24/(R*T)), 0, C_C1112*K_28*np.exp(-Ea_28/(R*T)) - C_R6*K_29*np.exp(-Ea_29/(R*T)), -C_R6*K_28*np.exp(-Ea_28/(R*T)), 0, 0],
              [-C_R7*K_8*np.exp(-Ea_8/(R*T)), 0, 0, 0, 0, 0, C_R1*K_25*np.exp(-Ea_25/(R*T)), 0, 0, 0, 0, 0, C_R1*K_15*np.exp(-Ea_15/(R*T)), C_C1112*K_15*np.exp(-Ea_15/(R*T)) + C_Tri*K_25*np.exp(-Ea_25/(R*T)), 0, 0, 0, 0, 0, -C_EDC*K_8*np.exp(-Ea_8/(R*T)) - K_25*np.exp(-Ea_25/(R*T)), 0, 0, 0, 0],
              [-C_R8*K_9*np.exp(-Ea_9/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R8*K_28*np.exp(-Ea_28/(R*T)), C_CHCl3*K_16*np.exp(-Ea_16/(R*T)), 0, 0, C_CCl4*K_26*np.exp(-Ea_26/(R*T)), C_CCl4*K_27*np.exp(-Ea_27/(R*T)), C_CCl4*K_28*np.exp(-Ea_28/(R*T)) - C_R8*K_29*np.exp(-Ea_29/(R*T)), 0, -C_C1112*K_28*np.exp(-Ea_28/(R*T)) - C_EDC*K_9*np.exp(-Ea_9/(R*T)) - C_R6*K_29*np.exp(-Ea_29/(R*T)), C_R4*K_26*np.exp(-Ea_26/(R*T)) + C_R5*K_27*np.exp(-Ea_27/(R*T)) + C_R6*K_28*np.exp(-Ea_28/(R*T)) + K_2*np.exp(-Ea_2/(R*T)), C_R1*K_16*np.exp(-Ea_16/(R*T)), 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R8*K_28*np.exp(-Ea_28/(R*T)), 0, 0, 0, -C_CCl4*K_26*np.exp(-Ea_26/(R*T)), -C_CCl4*K_27*np.exp(-Ea_27/(R*T)), -C_CCl4*K_28*np.exp(-Ea_28/(R*T)) + C_R8*K_29*np.exp(-Ea_29/(R*T)), 0, C_C1112*K_28*np.exp(-Ea_28/(R*T)) + C_R6*K_29*np.exp(-Ea_29/(R*T)), -C_R4*K_26*np.exp(-Ea_26/(R*T)) - C_R5*K_27*np.exp(-Ea_27/(R*T)) - C_R6*K_28*np.exp(-Ea_28/(R*T)) - K_2*np.exp(-Ea_2/(R*T)), 0, 0],
              [C_R8*K_9*np.exp(-Ea_9/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_CHCl3*K_16*np.exp(-Ea_16/(R*T)), 0, 0, 0, 0, 0, 0, C_EDC*K_9*np.exp(-Ea_9/(R*T)), 0, -C_R1*K_16*np.exp(-Ea_16/(R*T)), 0],
              [C_R5*K_6*np.exp(-Ea_6/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R2*K_10*np.exp(-Ea_10/(R*T)) - C_VCM*K_17*np.exp(-Ea_17/(R*T)) - C_VCM*K_18*np.exp(-Ea_18/(R*T)) - C_VCM*K_22*np.exp(-Ea_22/(R*T)), C_R1*K_10*np.exp(-Ea_10/(R*T)) - C_VCM*K_19*np.exp(-Ea_19/(R*T)), K_22*np.exp(-Ea_22/(R*T)), -C_VCM*K_20*np.exp(-Ea_20/(R*T)), C_EDC*K_6*np.exp(-Ea_6/(R*T)) - C_VCM*K_21*np.exp(-Ea_21/(R*T)), 0, 0, 0, 0, 0, -C_R1*K_17*np.exp(-Ea_17/(R*T)) - C_R1*K_18*np.exp(-Ea_18/(R*T)) - C_R1*K_22*np.exp(-Ea_22/(R*T)) - C_R2*K_19*np.exp(-Ea_19/(R*T)) - C_R4*K_20*np.exp(-Ea_20/(R*T)) - C_R5*K_21*np.exp(-Ea_21/(R*T))]]
                                   
       return JacN

tend = 30.0
tdist = tend*v_0
tnum = 100000
inc = tdist/tnum
timee = np.linspace(0,int(tend), num=tnum)
dist = [inc*i for i in range(tnum)]
Z = sp.symbols('Z')
res = solve_ivp(RHS, [0.0, tdist], C0 , method = 'Radau', t_eval=dist,  args=(773.15, 8.314, 5900000000000000.0, 2200000.0, 13000000.0, 12000000.0, 1000000.0, 500000.0, 200000.0, 100000.0, 1000000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 17000000.0, 16000000.0, 91000.0, 120000000.0, 300000.0, 20000.0, 500000.0, 210000000.0, 500000000.0, 20000000.0, 25000000000000.0, 1000000.0, 500000.0, 500000.0, 10000000.0, 100000000.0, 160000000.0, 342000.0, 230000.0, 7000.0, 34000.0, 42000.0, 45000.0, 48000.0, 56000.0, 63000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 17000.0, 14000.0, 0.0, 56000.0, 61000.0, 30000.0, 31000.0, 84000.0, 90000.0, 70000.0, 70000.0, 33000.0, 33000.0, 33000.0, 13000.0, 20000.0, 70000.0), jac=jacob, rtol=1E-13, atol=1E-13) #  , first_step=1E-2, max_step=1E-3, jac= lambda Z, C: jacob(Z,C, **args), rtol=1E-9, atol=1E-9
Edc = res.y[0]
HCl = res.y[2]
VCM = res.y[-1]
final = Edc[-1]
print('Final conversion = {} %'.format((100.0*((EDC0 - final)/EDC0))))
time = np.array(res.t)
fig = plt.figure()
plt1 = plt.plot(dist, Edc, 'b-', label='EDC')
plt2 = plt.plot(dist, HCl, 'r-', label='HCl')
plt3 = plt.plot(dist, VCM, 'g-', label='VCM')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title("EDC")
plt.legend()
plt.grid()
plt.show()

       

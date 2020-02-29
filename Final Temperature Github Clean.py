# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:15:07 2019
@author: Travis Czechorski 
         tjcze01@gmail.com
"""



from __future__ import division
import time
import math
import matplotlib.pyplot as plt 
import numpy as np
import sympy as sp
import mpmath as mp
from tqdm import tqdm
from sympy import *
import thermo as tc
from operator import mul
from scipy.integrate import solve_ivp
from functools import reduce
import pandas as pd
import decimal
from decimal import Decimal, getcontext
import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
sp.init_session(use_latex=False,quiet=True)
plt.ioff()
getcontext().prec = 50

#Data for inital kinetics
#https://doi.org/10.1021/ie8006903
#Ignore inital overflow errors for the first 5-10ish iterations, unless you can fix it  :)

    
mp.pretty = True
# Dictionary for all relevent forward and reverse reactions
Initreactions = [
{"Ea" : 1, "K_Value": 1,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 2, "K_Value": 2,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 3, "K_Value": 3,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  1, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 4, "K_Value": 4,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 1, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 5, "K_Value": 5,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}},
{"Ea" : 6, "K_Value": 6,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  1, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 1, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 7, "K_Value": 7,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}},
{"Ea" : 8, "K_Value": 8,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 1, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 9, "K_Value": 9,
 "Reactants" : {'EDC':  0, 'EC':  1, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 10, "K_Value": 10,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 1, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 11, "K_Value": 11,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 1, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  1, 'VCM':  0}},
{"Ea" : 12, "K_Value": 12,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 13, "K_Value": 13,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}},
{"Ea" : 14, "K_Value": 14,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  1, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}},
{"Ea" : 15, "K_Value": 15,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  1, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 16, "K_Value": 16,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  1,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 17, "K_Value": 17,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}},
{"Ea" : 18, "K_Value": 18,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  1,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 19, "K_Value": 19,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  1, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 1, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 20, "K_Value": 20,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  2,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  1, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 21, "K_Value": 21,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  1,'C11' : 0, 'C112' : 0, 'R1': 2, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  2, 'Coke': 2,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
]

Eqlist = [
{"Name" : 'EDC' ,
 "Reactions" : {"Reaction 1" : -1 ,"Reaction 2" : -1,"Reaction 3" : -1,"Reaction 4" : -1,"Reaction 5" : -1,"Reaction 6" : -1, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}}, 
{"Name": 'EC',
  "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 1,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : -1, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 1, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
  "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'HCl',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 1,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" : 1, "Reaction 8" : 1, "Reaction 9" : 1, "Reaction 10" : 1, "Reaction 11" :1, "Reaction 12" :0, "Reaction 13" :1, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" : 1}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'Coke',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" : 1}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'CP',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" : 1, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'Di',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" : 1, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" : 1, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :-1, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C4H6Cl2',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" : 1, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C6H6',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" : 1, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C2H2',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :1, "Reaction 19" :0, "Reaction 20" : -1, "Reaction 21" :-1}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :-1, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C11',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 1,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" : -1, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C112',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 1,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" : 0, "Reaction 11" :-1, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R1',
 "Reactions" : {"Reaction 1" : 1 ,"Reaction 2" : -1,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" : -1, "Reaction 8" : -1, "Reaction 9" : -1, "Reaction 10" : -1, "Reaction 11" : -1, "Reaction 12" : -1, "Reaction 13" : -1, "Reaction 14" : 0, "Reaction 15" : 1, "Reaction 16" : 1, "Reaction 17" :1, "Reaction 18" :1, "Reaction 19" : 1, "Reaction 20" : 1, "Reaction 21" : -1}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :-1, "Reaction 18" :-1, "Reaction 19" :-1, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R2',
 "Reactions" : {"Reaction 1" : 1 ,"Reaction 2" : 0,"Reaction 3" : -1,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" : -1, "Reaction 8" :0, "Reaction 9" : 1, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : -1, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R3',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 1,"Reaction 3" : 1,"Reaction 4" : 1,"Reaction 5" : 1,"Reaction 6" : 1, "Reaction 7" :0, "Reaction 8" : -1, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" : -1, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R4',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : -1,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" : 1, "Reaction 11" :0, "Reaction 12" : 1, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" : -1, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R5',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : -1,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" : 1, "Reaction 14" : 1, "Reaction 15" :0, "Reaction 16" :-1, "Reaction 17" :0, "Reaction 18" :-1, "Reaction 19" :0, "Reaction 20" :-1, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :1, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R6',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : -1, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" : 1, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :-1, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :1, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'VCM',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 1,"Reaction 6" : 0, "Reaction 7" :1, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" : -1, "Reaction 13" : -1, "Reaction 14" : -1, "Reaction 15" : -1, "Reaction 16" : -1, "Reaction 17" : 1, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :-1, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}}, 
]

#Frequency factor ko (Initial k)

De = decimal.Decimal

names2 = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM']
temps2 = ['0s','1s']

namesf = ['EDC','EC','HCl','Coke', 'CP','Di','Tri','C4H6Cl2','C6H6','C2H2','C11','C112','C1112','R1','R2','R3','R4','R5','R6','R7','R8','CCl4','CHCl3','VCM']
tempsf = ['0','1']

#namespd = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM','T0','T1','Pure','S_VCM','S_HCl','Yield']
namespdf = ['EDC','EC','HCl','Coke', 'CP','Di','Tri','C4H6Cl2','C6H6','C2H2','C11','C112','C1112','R1','R2','R3','R4','R5','R6','R7','R8','CCl4','CHCl3','VCM','T0','T1','Pure','S_VCM','S_HCl','Yield']

namesj = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM']
tempsj = ['0','1']

Ea = [342.0,7.0,42.0,45.0,34.0,48.0,13.0,12.0,4.0,6.0,15.0,0.0,56.0,31.0,30.0,61.0,84.0,90.0,70.0,20.0,70.0] #[kJ/mol]
Eab = [De(x*1000) for x in Ea] #J/mol
ks = [5.9E+15,1.3E13,1E+12,5E+11,1.2E+13,2E+11,1E+13,1E+13,1.7E+13,1.2E+13,1.7E+13,91000000000,1.2E+14,5E+11,20000000000,3E+11,2.1E+14,5E+14,2E+13,1E+14,1.6E+14]
n = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2]
k_0b = [i/(1E6**(j-1)) for i,j in zip(ks,n)]
k_0 = [De(i) for i in k_0b]

newk = [i/(1E6**(j-1)) for i,j in zip(ks,n)]

k_0g = [5.9E15, 1.3E13, 1.0E12, 5.0E11, 1.2E13, 2.0E11, 1.0E13, 1.0E13, 1.7E13, 1.2E13, 1.7E13, 9.1E10, 1.2E14, 5.0E11, 2.0E10, 3.0E11, 2.1E14, 5.0E14, 2.0E13, 1.0E14, 1.6E14] #If first order: [1/s]. If second order: [cm^3/mol*s]
k_0s = [1.3E13, 1.0E12, 5.0E11, 1.2E13, 2.0E11, 1.0E13, 1.0E13, 1.7E13, 1.2E13, 1.7E13, 9.1E10, 1.2E14, 5.0E11, 2.0E10, 3.0E11, 2.1E14, 5.0E14, 2.0E13, 1.0E14, 1.6E14]
k_0m = [i/1E6 for i in k_0s]
k_0ff = [5900000000000000.0, 13000000000000.0, 1000000000000.0, 500000000000.0, 12000000000000.0, 200000000000.0, 10000000000000.0, 10000000000000.0, 17000000000000.0, 12000000000000.0, 17000000000000.0, 91000000000.0, 120000000000000.0, 500000000000.0, 20000000000.0, 300000000000.0, 210000000000000.0, 500000000000000.0, 20000000000000.0, 100000000000000.0, 160000000000000.0]
for i,j in enumerate(k_0s):
    newval = k_0s[i]
    k_0ff.append(newval)
k_0ffD = [De(i) for i in k_0ff]
k_0b = [i/(1E6**(j-1)) for i,j in zip(ks,n)]
#k_0b.insert(0,5.9E15) 

k_0 = [De(i) for i in k_0b]
k_0f = [5.90E15,	2.20E12,	1.30E13,	1.20E13,	1.00E12,	5.00E11,	2.00E11,	1.00E11,	1.00E12,	1.00E13,	1.00E13,	1.70E13,	1.20E13,	1.70E13,	1.70E13,	1.60E13,	9.10E10,	1.20E14,	3.00E11,	2.00E10,	5.00E11,	2.10E14,	5.00E14,	2.00E13,	2.50E13,	1.00E12, 5.00E11,	5.00E11,	1.00E13,	1.00E14,	1.60E14]
k_0fb = [De(x) for x in k_0f]
Eaf = [342,230,7,34,42,45,48,56,63,13,12,4,6,15,17,14,0,56,61,30,31,84,90,70,70,33,33,33,13,20,70]
Eafb = [De(x*1000) for x in Eaf]

Rvalc = 8.314 # [J/mol*K]
temp_c = 500 # [°C]
temp_k = temp_c + 273.15 # [K]
Temp_K = De(temp_k)
Twallv = temp_c + 273.15


elist = ['C_EDC',' C_EC',' C_HCl',' C_Coke',' C_CP',' C_Di',' C_Tri',' C_C4H6Cl2',' C_C6H6',' C_C2H2',' C_C11',' C_C112',' C_C1112',' C_R1',' C_R2',' C_R3',' C_R4',' C_R5',' C_R6',' C_R7',' C_R8',' C_CCl4',' C_CHCl3',' C_VCM',' T0',' T1']
consf = ['1','2','3']
cons = ['1','2','3']

con1 = sp.symbols('con1')
con2 = sp.symbols('con2')
con3 = sp.symbols('con3')
#cons = [5.228861948970217258E06, 4.573981356040483661E04, 2.392702360656382751E11]

elist = ['C_EDC',' C_EC',' C_HCl',' C_Coke',' C_CP',' C_Di',' C_Tri',' C_C4H6Cl2',' C_C6H6',' C_C2H2',' C_C11',' C_C112',' C_C1112',' C_R1',' C_R2',' C_R3',' C_R4',' C_R5',' C_R6',' C_R7',' C_R8',' C_CCl4',' C_CHCl3',' C_VCM',' T0',' T1']

# These are mostly a group of strings used to name the chemicals 
eqsnum = 21
reacteqs = []
prodeqs = []
reacteqs2 = []
prodeqs2 = []
namesb = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM']
namesb2 = ['1,2-dichloroethane','Ethylchloride','Hydrogen Chloride','Coke', '1-/2-chloroprene','1,1-/cis-/trans-dichloroethylene',r'$C_{4}H_{6}Cl_{2}$',r'$C_{6}H_{6}$',r'$C_{2}H_{2}$','1,1-dichloroethane','1,1,2-trichloroethane',r'$R_{1}',r'$R_{2}',r'$R_{3}',r'$R_{4}',r'$R_{5}',r'$R_{6}','Vinyl Chloride Monomer']
namespd = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM','T0','T1','Pure','Selectivity VCM','Selectivity HCl','Yield VCM','Constant 1','Constant 2','Constant 3','h Coefficient','U Coefficient','k value','Reynolds']
namesdiff = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM','dEDCdz','dECdz','dHCldz','dCokedz','dCPdz','dDidz','dC4H6Cl2dz','dC6H6dz','dC2H2dz','dC11dz','dC112dz','dR1dz','dR2dz','dR3dz','dR4dz','dR5dz','dR6dz','dVCMdz']
tempsb = ['0','1']
tdiff = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM','t'] #
tdiff2 = ['t']
consb = ['1','2','3']
con1 = sp.symbols('con1')
con2 = sp.symbols('con2')
con3 = sp.symbols('con3')
namest = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM']
namesc = ['1,2-bis(chloranyl)ethane', 'chloranylethane', 'chlorane', '2-chloranylbuta-1,3-diene', '1,2-bis(chloranyl)ethene', '3,4-bis(chloranyl)but-1-ene', 'benzene', 'ethyne', '1,1-bis(chloranyl)ethane', '1,1,2-tris(chloranyl)ethane', 'chlorine', '1,1-bis(chloranyl)ethene', 'bis(chloranyl)-fluoranyl-methane', '1,3-dioxolan-2-one', '1,1-bis(fluoranyl)ethene', '1,1,2-tris(chloranyl)ethene', 'chloranylethene'] 
namesd = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM','T','T1']
namesdb = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM','T','T1','Pure']
tempst2 = ['0','1']
tempst = ['0','1']
tempst3 = ['4','5']
#Frequency factor ko (Initial k)
Twall_c = 500.0 #[°C]
Twall = Twall_c + 273.15
def prod(seq):
    return reduce(mul, seq) if seq else 1

Tc = 500.0
Tk = Tc + 273.15


#Cas ['107-06-2','75-00-3','7647-01-0','126-99-8','1,2-dichloroethylene','760-23-6','71-43-2','74-86-2','75-34-3','79-00-5','Cl','75-35-4','75-43-4','96-49-1','75-38-7','79-01-6','75-01-4']

start = time.time() #Real time when the program starts to run

   
  

mp.autoprec(50)

def KtoC(K):
    Cval = K - 273.15
    return Cval

def CtoK(C):
    Kval = C + 273.15
    return Kval



def symfunc(names, rxnum):
              Csyms = [sp.symbols(r'C_{}'.format('{}'.format(i))) for i in names]
              Ksyms = [sp.symbols(r'K_{}'.format(j)) for j in range(rxnum)]
              EAsyms = [sp.symbols(r'Ea_{}'.format(k)) for k in range(rxnum)]
              Tsyms = [sp.symbols('T0'), sp.symbols('T1')]
              
              return Csyms, Ksyms, EAsyms, Tsyms




#These are mostly functions used to calculate the unused diffusion coefficients
def Dab(T,P,M1,M2,sa,sb,col):
    C1 = 1.883E-20
    T1 = T**(3.0/2.0)
    ma = 1.0/M1
    mb = 1.0/M2
    mf = (ma+mb)**(0.5)
    sab = (sa + sb)/2.0
    top = C1*T1*mf
    bottom = P*(sab**2.0)*col
    final = top/bottom
    return final

def colint(Tstar):
       A = 1.06036
       B = 0.15610
       C = 0.19300
       D = 0.47635
       E = 1.03587
       F = 1.52996
       G = 1.76474
       H = 3.89411
       colint = A/(Tstar**B) + C/math.exp(D*Tstar) + E/math.exp(F*Tstar) + G/math.exp(H*Tstar)
       return colint


def DI(Ys,Dijs):
        flist = []
        for i in range(0,len(Ys),1):
            Yi = Ys[i]
            Di = Dijs[i]
            Yj = [g for q,g in enumerate(Ys) if q!=Yi] 
            Djb = [w for j,w in enumerate(Dijs) if j!=Di]
            Dif =  lambda l: [item for sublist in l for item in sublist]
            Dj = Dif(Djb)
            YjDj = [x/y for x,y in zip(Yj,Dj)]
            YjDjt = sum(YjDj)
            top = 1 - Yi
            Df = top/YjDjt
            flist.append(Df)
            Yj.clear()
            Dj.clear()
            YjDj.clear()
        return flist
                  


def getnu(velocity,rho,distance,diameter,k,viscosity,vs,Twall,Tgas,pr,re):
        Nu = [0.0]
        if re <= 3000.0:
                        
                        term1a = lambda re,pr,distance,diameter,viscosity,vs:  ((re*pr*(distance/diameter))**(1/3) * (viscosity/vs)**0.14)
                        term1 = term1a(re,pr,distance,diameter,viscosity,vs)
                        if term1 <=2:
                            Nu[0] = 3.66
                            
                        if term1 > 2:
                            f1 = lambda re,pr,distance,diameter,viscosity,vs : (1.86*((re*pr)/(distance/diameter))**(1/3) *(viscosity/vs)**0.14)
                            Nu[0] = f1(re,pr,distance,diameter,viscosity,vs)
        elif re > 3000.0 and  re < 10000.0:
                        fval = lambda re,pr,distance,diameter,viscosity,vs:  ((0.79*math.log(re) - 1.64)**-2)
                        f = fval(re,pr,distance,diameter,viscosity,vs)
                        f2 = lambda fval,re,pr,distance,diameter,viscosity,vs : ((f/8.0)*(re-1000.0)*pr)/(1 + 12.7*(f**(1/2))*(pr**(2/3)-1))
                        Nu[0] = f2(re,pr,distance,diameter,viscosity,vs)
        elif re >= 10000.0:
                        if Twall <= Tgas:
                            f3 = lambda re,pr,distance,diameter,viscosity,vs : 0.023*(re**(4/5))*(pr**0.3)
                            Nu[0] = f3(re,pr,distance,diameter,viscosity,vs)
                            
                        if Twall > Tgas:
#                            f4 = lambda re,pr,distance,diameter,viscosity,vs : 0.023*(re**(4/5))*(pr**0.4)
                            f5 = lambda re,pr,distance,diameter,viscosity,vs : 0.027*(re**(4.0/5.0))*((pr**0.4)**(1.0/3.0))*((viscosity/vs)**0.14)
#                            Nu[0] = f4(re,pr,distance,diameter,viscosity,vs)
                            Nu[0] = f5(re,pr,distance,diameter,viscosity,vs)
        
        Nuv = Nu[0]
        
        if (distance/diameter) < 60:
            Nu.clear()
            f5 = lambda Nuv, distance,diameter : (Nuv)/(1 + (1/((distance/diameter)**(2/3)))) 
            Nu.append(f5(Nuv, distance,diameter))
        else: 
                pass
        return float(Nu[0])   
    
def reynolds(rho,velocity,distance,viscosity):
        p = rho
        v = velocity
        x = distance 
        u = viscosity
        Re = (p*x*v)/u
        return Re

def hterm(Nu,distance,k):
        h = (k*Nu)/distance
        return h

def Pr(cp,viscosity,k):
    # cp = [J/kg-K]
    # u = [Pa*s] = [N*s/m^2]
    # k = [W/m-K]
        Cp = cp
        µ = viscosity
        pr = (Cp*µ)/k
        return pr

def Nus(velocity,rho,distance,diameter,k,viscosity,vs,Twall,Tgas,pr,re):
    Nuv = getnu(velocity,rho,distance,diameter,k,viscosity,vs,Twall,Tgas,pr,re)
    
    Nu = Nuv
    
    return Nu



def Uvalue(di,do,hi,ho,kpipe):

    Uval = 1.0/((do/di)*(1.0/hi) + ((do*math.log(do/di)/(2.0*kpipe))) + (1.0/ho))
    return Uval

def rhomix(conc,molar_mass):
    rhomix = [i*j for i,j in zip(conc,molar_mass)]
    rhomixanswer = sum(rhomix)
    return rhomixanswer

def cpmix(cp,conc):
    contot = sum(conc)
    Ci = [i/contot for i in conc]
    cpmix = [i*j for i,j in zip(Ci,cp)]
    cpmixanswer = sum(cpmix)
    return cpmixanswer
        
def chemicals(CAS,temp):
        return tc.Chemical('{}'.format(CAS), T=temp)
    
    
def formula(string):
        return tc.serialize_formula('{}'.format(string))
        
        
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)       
        
def cp(alist):
            c2alist = []
            for i in alist:
                
                chemicalt = getattr(i,'Cpgm') # [J/mol/K] 
                c2alist.append(chemicalt)
            return c2alist
        
        
def mw(alist): 
            c2blist = []
            for i in alist:
                
                chemicalt2 = getattr(i,'MW')  # [g/mol]
                c2blist.append(chemicalt2)
            return c2blist
        
def name(namelist):
            nlist = namelist 
            names = []
            for i in nlist:
                chemicaltn = getattr(i,"IUPAC_name") 
                names.append(chemicaltn)
            return [nlist]
        
def cpsum(alist):
            chemlist = alist
            c2alist2 = []
            for i in chemlist:
                
                chemicalta = getattr(i,'Cpgm') # [J/mol/K] 
                c2alist2.append(chemicalta)
            return sum(c2alist2)
        
def mwsum(alist):
            chemlist = alist
            #c2alist2 = []
            for i in chemlist:
                c2clist2 = []
                chemicaltc = getattr(i,'MW') # [g/m^3] 
                c2clist2.append(chemicaltc)
            return sum(c2clist2)
        
def mixprop(IDs,mwmix,T,P):
            mwmix = tc.Mixture(IDs = IDs, mwmix=mwmix,T=T,  P=P)
            return mwmix
        
def mixpropcpg(IDs,mwmix,T,P):
            mwmix = tc.Mixture(IDs = IDs, mwmix=mwmix,T=T,  P=P)
            return mwmix.Cpg
        
def mixproprho(IDs,mwmix,T,P): #[kg/m^3]
            mwmix = tc.Mixture(IDs = IDs, mwmix=mwmix,T=T,  P=P)
            return mwmix.rhog 
        
def mixpropkmix(IDs,mwmix,T,P): #[Pa*s]
            mwmix = tc.Mixture(IDs = IDs, mwmix=mwmix,T=T,  P=P)
            return mwmix.kg 
        
def mixpropvmix(IDs,mwmix,T,P): #[Pa*s]
            mwmix = tc.Mixture(IDs = IDs, mwmix=mwmix,T=T,  P=P)
            return mwmix.mugs

def Si(tb):
    si = 1.5*tb
    return si

def Sij(si,sj):
    Si = si
    Sj = sj
    sij = math.sqrt(Si*Sj)*0.733
    return sij

def Aij(T,Ta,Tbb,ua,ub,Ma,Mb,ka,kb,cpa,cpb,cva,cvb):
    S1 = Si(Ta)
    S2 = Si(Tbb)
    S12 = Sij(S1,S2)
    Mab = (Mb/Ma)**(3.0/4.0)
    uab = Uab(ka,kb,cpa,cpb,cva,cvb)
    S1t = S1/T
    S2t = S2/T
    S12t = S12/T
    S1T = 1.0 + S1t
    S2T = 1.0 + S2t
    S12T = 1.0 + S12t
    sf1 = S1T/S2T
    sf12 = S12T/S1T
    brackets = math.sqrt(Mab*uab*sf1)
    curlybrackets = (1.0 + brackets)**2
    aijval = (curlybrackets*sf12)*(1.0/4.0)
    return aijval

def Aijlist(T,Tbs,u,Mws,Ks,Cp,Cv):
    lengthq = len(Mws)
    Aijf = []
    for i in range(0,lengthq,1):
        aijs = []
        for j in range(0,lengthq,1):
            Ta = Tbs[i]
            Tbval = Tbs[j]
            ua = u[i]
            ub = u[j]
            ka = Ks[i]
            kb = Ks[j]
            cpa = Cp[i]
            cpb = Cp[j]
            cva = Cv[i]
            cvb = Cv[j]
            Ma = Mws[i]
            Mb = Mws[j]
            aij1 = Aij(T,Ta,Tbval,ua,ub,Ma,Mb,ka,kb,cpa,cpb,cva,cvb)
            aijs.append(aij1)
        Aijf.append(aijs[:])
        aijs.clear()
    return Aijf

def kmix(T,yi,ks,u,Tbl,Mws,Cp,Cv):
    lengthf = len(yi)
    aijs2 = Aijlist(T,Tbl,u,Mws,ks,Cp,Cv)
    klist = []
    for i in range(0,lengthf,1):
        bottom = []
        for j in range(0,lengthf,1):
            aij = aijs2[i][j]
            aijfrac = aij*yi[j]
            bottom.append(aijfrac)
        bval = float(sum(bottom))*float(1.0/yi[i])
        top = ks[i]
        ki = top/bval
        klist.append(ki)
        bottom.clear()
    kmixval = sum(klist)
    return kmixval
        

def Uab(ka,kb,cpa,cpb,cva,cvb):
    y1 = cpa/cva
    y2 = cpb/cvb
    kk = ka/kb
    cab = cpb/cpa
    r1 = (9.0 -(5.0/y2))
    r2 = (9.0 -(5.0/y1))
    rr = r1/r2
    uab = float(kk*cab*rr)
    return uab
        
            
    


#RHS (right hand side) is a function that returns the system of differential equations as a list/array to be used with the solve_ivp method
#There are 20 equations in total, one for each compound (listed in the order from the list excel/word file, 18), and two equations for temperature (T,dT)
#Steady-State 1-D Heat Balance: (d^2T)/(dz^2) = (U_coeff*α*(T_wall - T) + ∑r_i*∆H_rxn)/(k_mix); α = (Surface Area / Volume) or (2*π*r*L/π*r^2*L) 
#Each equation is used to solve for the concentration of each compound and it is the exact same order that the compound lists are defined shortly below


def RHS(z, C, R, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_10, K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20, K_21, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Constant_1, Constant_2, Constant_3):
       C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_VCM, T0, T1 = C 

       return [-C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T0)) - C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T0)) - C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T0)) - C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T0)) - C_EDC*C_R6*K_6*np.exp(-Ea_6/(R*T0)) - C_EDC*K_1*np.exp(-Ea_1/(R*T0)),
              -C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T0)) + C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T0)) + C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T0)),
              C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T0)) + C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T0)) + 2*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)) + C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T0)) + C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T0)) + C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T0)) + C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T0)) + C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T0)),
              2*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)),
              C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T0)),
              -C_Di*C_R1*K_19*np.exp(-Ea_19/(R*T0)) + C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T0)) + C_R6*K_19*np.exp(-Ea_19/(R*T0)),
              C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T0)),
              0.5*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T0)),
              -2*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T0)) - 1.0*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)) - C_C2H2*C_R1*K_18*np.exp(-Ea_18/(R*T0)) + C_R5*K_18*np.exp(-Ea_18/(R*T0)),
              -C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T0)) + C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T0)),
              -C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T0)) + C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T0)),
              -C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T0)) - C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T0)) + 0.5*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T0)) - 2*C_C2H2*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)) - C_C2H2*C_R1*K_18*np.exp(-Ea_18/(R*T0)) - C_Di*C_R1*K_19*np.exp(-Ea_19/(R*T0)) - C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T0)) - C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T0)) + C_EDC*K_1*np.exp(-Ea_1/(R*T0)) - C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T0)) - C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T0)) - C_R1*C_VCM*K_12*np.exp(-Ea_12/(R*T0)) - C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T0)) - C_R1*C_VCM*K_17*np.exp(-Ea_17/(R*T0)) + C_R3*K_17*np.exp(-Ea_17/(R*T0)) + C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T0)) + C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T0)) + C_R5*K_18*np.exp(-Ea_18/(R*T0)) + C_R6*K_19*np.exp(-Ea_19/(R*T0)),
              C_EC*C_R1*K_9*np.exp(-Ea_9/(R*T0)) - C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T0)) + C_EDC*K_1*np.exp(-Ea_1/(R*T0)) - C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T0)) - C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T0)),
              C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T0)) + C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T0)) + C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T0)) + C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T0)) + C_EDC*C_R6*K_6*np.exp(-Ea_6/(R*T0)) - C_R1*C_R3*K_8*np.exp(-Ea_8/(R*T0)) - C_R3*K_17*np.exp(-Ea_17/(R*T0)),
              C_C11*C_R1*K_10*np.exp(-Ea_10/(R*T0)) - C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T0)) + C_R1*C_VCM*K_12*np.exp(-Ea_12/(R*T0)) - C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T0)),
              -0.5*C_C2H2**2*C_R5*K_20*np.exp(-Ea_20/(R*T0)) + C_C2H2*C_R1*K_18*np.exp(-Ea_18/(R*T0)) - C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T0)) + C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T0)) + C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T0)) - C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T0)) - C_R5*K_18*np.exp(-Ea_18/(R*T0)),
              C_C112*C_R1*K_11*np.exp(-Ea_11/(R*T0)) + C_Di*C_R1*K_19*np.exp(-Ea_19/(R*T0)) - C_EDC*C_R6*K_6*np.exp(-Ea_6/(R*T0)) - C_R6*K_19*np.exp(-Ea_19/(R*T0)),
              C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T0)) + C_R1*C_R2*K_7*np.exp(-Ea_7/(R*T0)) - C_R1*C_VCM*K_12*np.exp(-Ea_12/(R*T0)) - C_R1*C_VCM*K_13*np.exp(-Ea_13/(R*T0)) - C_R1*C_VCM*K_17*np.exp(-Ea_17/(R*T0)) - C_R2*C_VCM*K_14*np.exp(-Ea_14/(R*T0)) + C_R3*K_17*np.exp(-Ea_17/(R*T0)) - C_R4*C_VCM*K_15*np.exp(-Ea_15/(R*T0)) - C_R5*C_VCM*K_16*np.exp(-Ea_16/(R*T0)),
              T1,
              Constant_1*T1 - Constant_2*(-T0 + Twall) - Constant_3*(-C_EDC*C_R1*K_2*np.exp(-Ea_2/(R*T0)) - C_EDC*C_R2*K_3*np.exp(-Ea_3/(R*T0)) - C_EDC*C_R4*K_4*np.exp(-Ea_4/(R*T0)) - C_EDC*C_R5*K_5*np.exp(-Ea_5/(R*T0)) - C_EDC*C_R6*K_6*np.exp(-Ea_6/(R*T0)) - C_EDC*K_1*np.exp(-Ea_1/(R*T0)))]

def jacob(z, C, R, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_10, K_11, K_12, K_13, K_14, K_15, K_16, K_17, K_18, K_19, K_20, K_21, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Constant_1, Constant_2, Constant_3):
       C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_VCM, T0, T1 = C 
       
       JacT = [[-C_R1*K_2*np.exp(-Ea_2/(R*T0)) - C_R2*K_3*np.exp(-Ea_3/(R*T0)) - C_R4*K_4*np.exp(-Ea_4/(R*T0)) - C_R5*K_5*np.exp(-Ea_5/(R*T0)) - C_R6*K_6*np.exp(-Ea_6/(R*T0)) - K_1*np.exp(-Ea_1/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EDC*K_2*np.exp(-Ea_2/(R*T0)), -C_EDC*K_3*np.exp(-Ea_3/(R*T0)), 0, -C_EDC*K_4*np.exp(-Ea_4/(R*T0)), -C_EDC*K_5*np.exp(-Ea_5/(R*T0)), -C_EDC*K_6*np.exp(-Ea_6/(R*T0)), 0, 0, 0],
              [C_R2*K_3*np.exp(-Ea_3/(R*T0)), -C_R1*K_9*np.exp(-Ea_9/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EC*K_9*np.exp(-Ea_9/(R*T0)), C_EDC*K_3*np.exp(-Ea_3/(R*T0)) + C_VCM*K_14*np.exp(-Ea_14/(R*T0)), 0, 0, 0, 0, C_R2*K_14*np.exp(-Ea_14/(R*T0)), 0, 0],
              [C_R1*K_2*np.exp(-Ea_2/(R*T0)), C_R1*K_9*np.exp(-Ea_9/(R*T0)), 0, 0, 0, 0, 0, 0, 2*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)), C_R1*K_10*np.exp(-Ea_10/(R*T0)), C_R1*K_11*np.exp(-Ea_11/(R*T0)), C_C11*K_10*np.exp(-Ea_10/(R*T0)) + C_C112*K_11*np.exp(-Ea_11/(R*T0)) + 4*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T0)) + C_EC*K_9*np.exp(-Ea_9/(R*T0)) + C_EDC*K_2*np.exp(-Ea_2/(R*T0)) + C_R2*K_7*np.exp(-Ea_7/(R*T0)) + C_R3*K_8*np.exp(-Ea_8/(R*T0)) + C_VCM*K_13*np.exp(-Ea_13/(R*T0)), C_R1*K_7*np.exp(-Ea_7/(R*T0)), C_R1*K_8*np.exp(-Ea_8/(R*T0)), 0, 0, 0, C_R1*K_13*np.exp(-Ea_13/(R*T0)), 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 2*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)), 0, 0, 4*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*K_16*np.exp(-Ea_16/(R*T0)), 0, C_R5*K_16*np.exp(-Ea_16/(R*T0)), 0, 0],
              [0, 0, 0, 0, 0, -C_R1*K_19*np.exp(-Ea_19/(R*T0)), 0, 0, 0, 0, 0, -C_Di*K_19*np.exp(-Ea_19/(R*T0)) + C_R3*K_8*np.exp(-Ea_8/(R*T0)), 0, C_R1*K_8*np.exp(-Ea_8/(R*T0)), 0, 0, K_19*np.exp(-Ea_19/(R*T0)), 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*K_15*np.exp(-Ea_15/(R*T0)), 0, 0, C_R4*K_15*np.exp(-Ea_15/(R*T0)), 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1.0*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T0)), 0, 0, 0, 0, 0, 0, 0.5*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T0)), 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, -4*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T0)) - 1.0*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)) - C_R1*K_18*np.exp(-Ea_18/(R*T0)), 0, 0, -2.0*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T0)) - C_C2H2*K_18*np.exp(-Ea_18/(R*T0)), 0, 0, 0, -2*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T0)) + K_18*np.exp(-Ea_18/(R*T0)), 0, 0, 0, 0],
              [C_R4*K_4*np.exp(-Ea_4/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*K_10*np.exp(-Ea_10/(R*T0)), 0, -C_C11*K_10*np.exp(-Ea_10/(R*T0)), 0, 0, C_EDC*K_4*np.exp(-Ea_4/(R*T0)), 0, 0, 0, 0, 0],
              [C_R5*K_5*np.exp(-Ea_5/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*K_11*np.exp(-Ea_11/(R*T0)), -C_C112*K_11*np.exp(-Ea_11/(R*T0)), 0, 0, 0, C_EDC*K_5*np.exp(-Ea_5/(R*T0)), 0, 0, 0, 0],
              [-C_R1*K_2*np.exp(-Ea_2/(R*T0)) + K_1*np.exp(-Ea_1/(R*T0)), -C_R1*K_9*np.exp(-Ea_9/(R*T0)), 0, 0, 0, -C_R1*K_19*np.exp(-Ea_19/(R*T0)), 0, 0, 1.0*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T0)) - 2*C_R1**2*K_21*np.exp(-Ea_21/(R*T0)) - C_R1*K_18*np.exp(-Ea_18/(R*T0)), -C_R1*K_10*np.exp(-Ea_10/(R*T0)), -C_R1*K_11*np.exp(-Ea_11/(R*T0)), -C_C11*K_10*np.exp(-Ea_10/(R*T0)) - C_C112*K_11*np.exp(-Ea_11/(R*T0)) - 4*C_C2H2*C_R1*K_21*np.exp(-Ea_21/(R*T0)) - C_C2H2*K_18*np.exp(-Ea_18/(R*T0)) - C_Di*K_19*np.exp(-Ea_19/(R*T0)) - C_EC*K_9*np.exp(-Ea_9/(R*T0)) - C_EDC*K_2*np.exp(-Ea_2/(R*T0)) - C_R2*K_7*np.exp(-Ea_7/(R*T0)) - C_R3*K_8*np.exp(-Ea_8/(R*T0)) - C_VCM*K_12*np.exp(-Ea_12/(R*T0)) - C_VCM*K_13*np.exp(-Ea_13/(R*T0)) - C_VCM*K_17*np.exp(-Ea_17/(R*T0)), -C_R1*K_7*np.exp(-Ea_7/(R*T0)), -C_R1*K_8*np.exp(-Ea_8/(R*T0)) + K_17*np.exp(-Ea_17/(R*T0)), C_VCM*K_15*np.exp(-Ea_15/(R*T0)), 0.5*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T0)) + C_VCM*K_16*np.exp(-Ea_16/(R*T0)) + K_18*np.exp(-Ea_18/(R*T0)), K_19*np.exp(-Ea_19/(R*T0)), -C_R1*K_12*np.exp(-Ea_12/(R*T0)) - C_R1*K_13*np.exp(-Ea_13/(R*T0)) - C_R1*K_17*np.exp(-Ea_17/(R*T0)) + C_R4*K_15*np.exp(-Ea_15/(R*T0)) + C_R5*K_16*np.exp(-Ea_16/(R*T0)), 0, 0],
              [-C_R2*K_3*np.exp(-Ea_3/(R*T0)) + K_1*np.exp(-Ea_1/(R*T0)), C_R1*K_9*np.exp(-Ea_9/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EC*K_9*np.exp(-Ea_9/(R*T0)) - C_R2*K_7*np.exp(-Ea_7/(R*T0)), -C_EDC*K_3*np.exp(-Ea_3/(R*T0)) - C_R1*K_7*np.exp(-Ea_7/(R*T0)) - C_VCM*K_14*np.exp(-Ea_14/(R*T0)), 0, 0, 0, 0, -C_R2*K_14*np.exp(-Ea_14/(R*T0)), 0, 0],
              [C_R1*K_2*np.exp(-Ea_2/(R*T0)) + C_R2*K_3*np.exp(-Ea_3/(R*T0)) + C_R4*K_4*np.exp(-Ea_4/(R*T0)) + C_R5*K_5*np.exp(-Ea_5/(R*T0)) + C_R6*K_6*np.exp(-Ea_6/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EDC*K_2*np.exp(-Ea_2/(R*T0)) - C_R3*K_8*np.exp(-Ea_8/(R*T0)), C_EDC*K_3*np.exp(-Ea_3/(R*T0)), -C_R1*K_8*np.exp(-Ea_8/(R*T0)) - K_17*np.exp(-Ea_17/(R*T0)), C_EDC*K_4*np.exp(-Ea_4/(R*T0)), C_EDC*K_5*np.exp(-Ea_5/(R*T0)), C_EDC*K_6*np.exp(-Ea_6/(R*T0)), 0, 0, 0],
              [-C_R4*K_4*np.exp(-Ea_4/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, C_R1*K_10*np.exp(-Ea_10/(R*T0)), 0, C_C11*K_10*np.exp(-Ea_10/(R*T0)) + C_VCM*K_12*np.exp(-Ea_12/(R*T0)), 0, 0, -C_EDC*K_4*np.exp(-Ea_4/(R*T0)) - C_VCM*K_15*np.exp(-Ea_15/(R*T0)), 0, 0, C_R1*K_12*np.exp(-Ea_12/(R*T0)) - C_R4*K_15*np.exp(-Ea_15/(R*T0)), 0, 0],
              [-C_R5*K_5*np.exp(-Ea_5/(R*T0)), 0, 0, 0, 0, 0, 0, 0, -1.0*C_C2H2*C_R5*K_20*np.exp(-Ea_20/(R*T0)) + C_R1*K_18*np.exp(-Ea_18/(R*T0)), 0, 0, C_C2H2*K_18*np.exp(-Ea_18/(R*T0)) + C_VCM*K_13*np.exp(-Ea_13/(R*T0)), C_VCM*K_14*np.exp(-Ea_14/(R*T0)), 0, 0, -0.5*C_C2H2**2*K_20*np.exp(-Ea_20/(R*T0)) - C_EDC*K_5*np.exp(-Ea_5/(R*T0)) - C_VCM*K_16*np.exp(-Ea_16/(R*T0)) - K_18*np.exp(-Ea_18/(R*T0)), 0, C_R1*K_13*np.exp(-Ea_13/(R*T0)) + C_R2*K_14*np.exp(-Ea_14/(R*T0)) - C_R5*K_16*np.exp(-Ea_16/(R*T0)), 0, 0],
              [-C_R6*K_6*np.exp(-Ea_6/(R*T0)), 0, 0, 0, 0, C_R1*K_19*np.exp(-Ea_19/(R*T0)), 0, 0, 0, 0, C_R1*K_11*np.exp(-Ea_11/(R*T0)), C_C112*K_11*np.exp(-Ea_11/(R*T0)) + C_Di*K_19*np.exp(-Ea_19/(R*T0)), 0, 0, 0, 0, -C_EDC*K_6*np.exp(-Ea_6/(R*T0)) - K_19*np.exp(-Ea_19/(R*T0)), 0, 0, 0],
              [C_R5*K_5*np.exp(-Ea_5/(R*T0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R2*K_7*np.exp(-Ea_7/(R*T0)) - C_VCM*K_12*np.exp(-Ea_12/(R*T0)) - C_VCM*K_13*np.exp(-Ea_13/(R*T0)) - C_VCM*K_17*np.exp(-Ea_17/(R*T0)), C_R1*K_7*np.exp(-Ea_7/(R*T0)) - C_VCM*K_14*np.exp(-Ea_14/(R*T0)), K_17*np.exp(-Ea_17/(R*T0)), -C_VCM*K_15*np.exp(-Ea_15/(R*T0)), C_EDC*K_5*np.exp(-Ea_5/(R*T0)) - C_VCM*K_16*np.exp(-Ea_16/(R*T0)), 0, -C_R1*K_12*np.exp(-Ea_12/(R*T0)) - C_R1*K_13*np.exp(-Ea_13/(R*T0)) - C_R1*K_17*np.exp(-Ea_17/(R*T0)) - C_R2*K_14*np.exp(-Ea_14/(R*T0)) - C_R4*K_15*np.exp(-Ea_15/(R*T0)) - C_R5*K_16*np.exp(-Ea_16/(R*T0)), 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [-Constant_3*(-C_R1*K_2*np.exp(-Ea_2/(R*T0)) - C_R2*K_3*np.exp(-Ea_3/(R*T0)) - C_R4*K_4*np.exp(-Ea_4/(R*T0)) - C_R5*K_5*np.exp(-Ea_5/(R*T0)) - C_R6*K_6*np.exp(-Ea_6/(R*T0)) - K_1*np.exp(-Ea_1/(R*T0))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EDC*Constant_3*K_2*np.exp(-Ea_2/(R*T0)), C_EDC*Constant_3*K_3*np.exp(-Ea_3/(R*T0)), 0, C_EDC*Constant_3*K_4*np.exp(-Ea_4/(R*T0)), C_EDC*Constant_3*K_5*np.exp(-Ea_5/(R*T0)), C_EDC*Constant_3*K_6*np.exp(-Ea_6/(R*T0)), 0, Constant_2, Constant_1]]

       return JacT

def Bviral(T,Tc,pc,omega):
       Tr = T/Tc
       B0 = 0.1445 - 0.33/Tr - 0.1385/(Tr**2) - 0.0121/(Tr**3)
       B1 = 0.073 + 0.46/Tr - 0.5/(Tr**2) - 0.097/(Tr**3) - 0.0073/(Tr**8)
       Br = B0 + omega*B1
       B = Br*8.314*(Tc/pc)
       return B

def Bviral2(T,Tc,pc,omega):
       Tr = T/Tc
       B0 = 0.083 - 0.422/(Tr**1.6)
       B1 = 0.139 - 0.172/(Tr**4.2)
       Br = B0 + omega*B1
       B = Br*8.314*(Tc/pc)
       return B

def Bviral3(T,Tc,pc,omega):
       Tr = T/Tc
       B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3 - 0.000607/Tr**8
       B1 = 0.0637 + 0.331/Tr**2 - 0.423/Tr**3 - 0.008/Tr**8
       Br = B0 + omega*B1
       B = Br*8.314*(Tc/pc)
       return B

def Bviral4(T,Tc,pc,omega,a,b,dipole):
       if dipole != 0:
              a = -2.188E-4*(dipole**4) - 7.831E-21*(dipole**8)
              b = 0
       else:
              a = 0
              b = 0
       Tr = T/Tc
       B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3 - 0.000607/Tr**8
       B1 = 0.0637 + 0.331/Tr**2 - 0.423/Tr**3 - 0.008/Tr**8
       B2 = 1/(Tr**6)
       B3 = -1./(Tr**8)
       Br = B0 + omega*B1 + a*B2 + b*B3
       B = Br*8.314*(Tc/pc)
       return B

def btoz(B,T,P):
       Z = 1.0 + (B*P)/(T*8.314)
       return Z
 
def densitytovm(p,MW):
       vmval = 1/((1E3*p)/(MW))
       return vmval




def alistfun(Temp):
    EDCp = tc.Chemical('107-06-2', T=Temp,P=PascalP)
    ECp = tc.Chemical('75-00-3', T=Temp,P=PascalP)
    HClp = tc.Chemical('7647-01-0', T=Temp,P=PascalP)
    Cokep = tc.Chemical('Activated charcoal', T=Temp,P=PascalP)
    CPp = tc.Chemical('126-99-8', T=Temp,P=PascalP)
    Dip = tc.Chemical('126-99-8', T=Temp,P=PascalP)
    C4H6Cl2p = tc.Chemical('760-23-6', T=Temp,P=PascalP)
    C6H6p = tc.Chemical('71-43-2', T=Temp,P=PascalP) 
    C2H2p = tc.Chemical('74-86-2', T=Temp,P=PascalP) 
    C11p = tc.Chemical('75-34-3', T=Temp,P=PascalP)
    C112p = tc.Chemical('79-00-5', T=Temp,P=PascalP)
    R1p = tc.Chemical('7647-01-0', T=Temp, P=PascalP)
    R2p = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    R3p = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    R4p = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    R5p = tc.Chemical('107-06-2', T=Temp,P=PascalP)
    R6p = tc.Chemical('107-06-2', T=Temp,P=PascalP)
    VCMp = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    alist = [EDCp,ECp,HClp,Cokep,CPp,Dip,C4H6Cl2p,C6H6p,C2H2p,C11p,C112p,R1p,R2p,R3p,R4p,R5p,R6p,VCMp]
    return alist

def alistfun2(Temp):
    EDCp = tc.Chemical('107-06-2', T=Temp,P=PascalP)
    ECp = tc.Chemical('75-00-3', T=Temp,P=PascalP)
    HClp = tc.Chemical('7647-01-0', T=Temp,P=PascalP)
    Cokep = tc.Chemical('Activated charcoal', T=Temp,P=PascalP)
    CPp = tc.Chemical('126-99-8', T=Temp,P=PascalP)
    Dip = tc.Chemical('126-99-8', T=Temp,P=PascalP)
    C4H6Cl2p = tc.Chemical('760-23-6', T=Temp,P=PascalP)
    C6H6p = tc.Chemical('71-43-2', T=Temp,P=PascalP) 
    C2H2p = tc.Chemical('74-86-2', T=Temp,P=PascalP) 
    C11p = tc.Chemical('75-34-3', T=Temp,P=PascalP)
    C112p = tc.Chemical('79-00-5', T=Temp,P=PascalP)
    R1p = tc.Chemical('7647-01-0', T=Temp, P=PascalP)
    R2p = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    R3p = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    R4p = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    R5p = tc.Chemical('107-06-2', T=Temp,P=PascalP)
    R6p = tc.Chemical('107-06-2', T=Temp,P=PascalP)
    VCMp = tc.Chemical('75-01-4', T=Temp,P=PascalP)
    alist = [EDCp,ECp,HClp,Cokep,CPp,Dip,C4H6Cl2p,C6H6p,C2H2p,C11p,C112p,R1p,R2p,R3p,R4p,R5p,R6p,VCMp]
    return alist



names = []
C1b = sp.symbols('C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7], C[8], C[9], C[10], C[11], C[12], C[13], C[14], C[15], C[16], C[17], C[18], C[19]')

#Fluid properties
omegaedc = 0.28600000000000003
Tcedc = 563
Pcedc = 5380*1E3
dipoleedc = 1.572
density = (1E6*1.253)/1E3 #kg/m^3
mwedc = 98.95 # g/mol
mwedckg = 98.95/1E3 # kg/mol
delhm = 71000 #[J/mol]
Temperature = float(input('Enter starting temperature [°C] --> ')) #[°C]
Temp_K = CtoK(Temperature) #[K]
Twall_c = Temperature #[°C]
Twall = Twall_c + 273.15
Pbar = 12.0 #[bar]
Pstart_atm = float(input("Enter starting pressure [atm] --> "))
PascalP  = Pstart_atm*101325.0 #[Pa]
Patm = PascalP/101325
volume_flow = float(input("Enter volumetric flow rate [m^3/s] --> ")) #[m^3/s]
volume_flowcm = volume_flow*1E6 #[cm^3/s]
pipe_thickness = float(input("Enter pipe thickness [m] --> ")) #0.025 #[m]
do = float(input("Enter total pipe diameter [m] --> "))# 0.5 #Outer diameter [m]
ro = do/2.0  #Outer radius [m]
di = do - 2*pipe_thickness #Inner diameter [m]
ri = di/2.0 #Inner radius [m]
cross_area = math.pi*(ri**2)
Ac = math.pi*(ri**2)
u_z = volume_flow/cross_area #[m/s]
CCl4_p = 0
CCl4_p = 0 #Percentage of CCl4
EDC_p = 1-CCl4_p
begmix = tc.Mixture(IDs=['107-06-2'], zs=[EDC_p],T=Temp_K, P=PascalP)
edcmw = begmix.MW
Rvalc = begmix.R_specific
rhoino = begmix.rho #[kg/m^3]
rhoin = rhoino*(1E3)*(1E-6)
rhoin2 = rhoin/edcmw
rhoing = begmix.rhogm #[mol/m^3]
rhoingb = begmix.rhogm/1E6 #[mol/m^3]
Temp_vals = []
mws = begmix.MW # g/mol
mwskg = mws/1E3 # kg/mol
Rval = Rvalc*mwskg
rhoc = rhoin/mws
rhoc2 = (PascalP/(Temp_K*8.314))/1E6 #[mol/m^3]
inedc = rhoing/1E6
initedc = rhoing
desired_time = int(input("Enter total reaction time [s] --> ")) #[s]
end_dist = float(u_z*desired_time)
dist_neat = round(end_dist,1)
F_in = volume_flow*rhoc2 #[mol/s]
L = int(u_z*desired_time) # [m]
Surface_area = math.pi*2.0*ri*L
alpha = (math.pi*2.0*ri*L)/(Ac*L)
patm = PascalP/101325 #[atm] 
Ratm = 8.20573660809596E-5 #[m^3*atm/K*mol]
Rkcal = 1.98720425864083E-3 #[kcal/K*mol]
segment_second = int(input('Enter iterations per second --> '))
gnodes = 100
gnodes2 = 5
chngamnt = float(input('Enter temperature change per iteration [°C] --> '))
iternum = int(input('Enter total iterations --> '))
graph_num = 1
segment_num = desired_time*segment_second #Segments/Second
time_nodes = int(iternum/gnodes)
graph_nodes = [int(i*time_nodes) for i in range(0,gnodes,1)]
time_nodesb = int(iternum/gnodes2)
graph_nodes2 = [int(i*time_nodesb) for i in range(0,gnodes2,1)]
endnum = int(L)
distance = L
segmentlength = dist_int = L/int(segment_num) #[m]
total_distance = np.linspace(0,L,num=(segment_num))
reaction_time = L/u_z #reaction_time = L/u_z #[s]
time_per_step = segmentlength/u_z #[s]
time_int = 1/segment_second
alistb = alistfun(Temp_K)
ksteel = 16.3 #[W/m*K]
time_vals = [i*time_int for i in range(0,segment_num,1)]
vflow_cmf = volume_flowcm/segment_second #[cm^3/s]
vflows = volume_flow/segment_second
time_valsb = [i*time_int for i in range(0,segment_num+1,1)]
tmvpd = np.asarray(time_valsb)
dist_c = [i*u_z for i in time_vals]
dist_cm = [i*u_z for i in time_vals]
dist_b = [i*u_z for i in time_valsb]
dstpd = np.asarray(dist_b)
ffd = int(dist_c[-1])
Ls = segmentlength #m
vintflow = Ls*(math.pi)*(ri**2) #[m^3/s]
vintflowcm = (Ls*100)*(math.pi)*((ri*100)**2) #[m^3/s]
dist_int = Ls
volume_flowcmf = (volume_flowcm)/segment_second #[m^3/s]
vintflowf = vintflow/segment_second
tn = np.linspace(0,L,desired_time)

#Concentrations
EDC = [Decimal(initedc)]
EC = [Decimal(0.0)]
HCl = [Decimal(0.0)] 
Coke = [Decimal(0.0)] 
CP = [Decimal(0.0)]
Di = [Decimal(0.0)]
C4H6Cl2 = [Decimal(0.0)]
C6H6 = [Decimal(0.0)]
C2H2 = [Decimal(0.0)]
C11 = [Decimal(0.0)]
C112 = [Decimal(0.0)]
R1 = [Decimal(0.0)]
R2 = [Decimal(0.0)]
R3 = [Decimal(0.0)]
R4 = [Decimal(0.0)]
R5 = [Decimal(0.0)]
R6 = [Decimal(0.0)]
VCM = [Decimal(0.0)]
T0 = [Decimal(Temp_K)]
T1 = [Decimal(0.0)]
EDCj = [Decimal(initedc)]
ECj = [Decimal(0.0)]
HClj = [Decimal(0.0)] 
Cokej = [Decimal(0.0)] 
CPj = [Decimal(0.0)]
Dij = [Decimal(0.0)]
C4H6Cl2j = [Decimal(0.0)]
C6H6j = [Decimal(0.0)]
C2H2j = [Decimal(0.0)]
C11j = [Decimal(0.0)]
C112j = [Decimal(0.0)]
R1j = [Decimal(0.0)]
R2j = [Decimal(0.0)]
R3j = [Decimal(0.0)]
R4j = [Decimal(0.0)]
R5j = [Decimal(0.0)]
R6j = [Decimal(0.0)]
VCMj = [Decimal(0.0)]
T0j = [Decimal(Temp_K)]
T1j = [Decimal(0.0)]
EDCl = []
ECl = []
HCll = [] 
Cokel = [] 
CPl = []
Dil = []
C4H6Cl2l = []
C6H6l = []
C2H2l = []
C11l = []
C112l = []
R1l = []
R2l = []
R3l = []
R4l = []
R5l = []
R6l = []
VCMl = []
T0l = []
T1l = []
pr1 = [100.0]
EDClj = []
EClj = []
HCllj = [] 
Cokelj = [] 
CPlj = []
Dilj = []
C4H6Cl2lj = []
C6H6lj = []
C2H2lj = []
C11lj = []
C112lj = []
R1lj = []
R2lj = []
R3lj = []
R4lj = []
R5lj = []
R6lj = []
VCMlj = []
T0lj = []
T1lj = []
pr1j = [100.0]

selectivity = [1.0]
selectivityl = []

selectivity2 = [1.0]
selectivity2l = []

selectivityj = [1.0]
selectivityjl = []

selectivity2j = [1.0]
selectivity2jl = []

yield_vcm = [1.0]
yield_vcml = []
yield_vcmj = [1.0]
yield_vcmjl = []

y0 = [Decimal(initedc), EC[-1],HCl[-1],Coke[-1],CP[-1],Di[-1],C4H6Cl2[-1],C6H6[-1],C2H2[-1],C11[-1],C112[-1],R1[-1],R2[-1],R3[-1],R4[-1],R5[-1],R6[-1],VCM[-1], T0[-1], T1[-1]]
C_Total = [sum(y0)]
C_Totalj = [sum(y0)]
initial_edc = Decimal(initedc)
conversion = [0.0]
RE_vals = []
U_coeffs = []
h_vals = []
c1_vals = []
c2_vals = []
c3_vals = []
kmix_vals = []
RE_valsj = []
U_coeffsj = []
h_valsj = []
c1_valsj = []
c2_valsj = []
c3_valsj = []
kmix_valsj = []

RE_valsl = []
U_coeffsl = []
h_valsl = []
c1_valsl = []
c2_valsl = []
c3_valsl = []
kmix_valsl = []
RE_valsjl = []
U_coeffsjl = []
h_valsjl = []
c1_valsjl = []
c2_valsjl = []
c3_valsjl = []
kmix_valsjl = []

alistb = alistfun(Temp_K)
conversion_EDC = []
conversion_EDCb = []
conversion_EDCb = [0.0]
conversion_EDCj = []
conversion_EDCjb = [0.0]
products = []
productsj = []
dproducts = []
prods1 = []
prods1j = []
dprods1 = []
pure = []
purej = []
dpure = []


for i in alistb:
    chemicalt = getattr(i,'IUPAC_name') 
    names.append(chemicalt)

dist_end = endnum #Final position 
J_eval = [0]
J_evalt = [0]
dist_value= [0.0]
Tbl = []
CAS = []
prhos = []
mws = []
pvis = []
kks = []
for i in alistb:
            Casnum = getattr(i,'CAS') 
            CAS.append(Casnum) 
            tb = getattr(i,'Tb')
            Tbl.append(float(tb))
            chemicalpi = getattr(i,'rhogm')
            prhos.append(float(chemicalpi))
            chemicalmwi = getattr(i,'MW')
            mws.append(float(chemicalmwi/1000.0))
            chemicalmug = getattr(i,'mug') 
            pvis.append(float(chemicalmug))
            chemicalkl = getattr(i,'kg')
            kks.append(float(chemicalkl))
prhot = sum(prhos)
mfracs = [i/prhot for i in prhos]
ys = tc.utils.zs_to_ws(mfracs,mws)
zvar = np.linspace(0,L,int(L/Ls))
cons1a = [5.228861948970217258E06, 4.573981356040483661E04, 2.392702360656382751E11]
z1 = sp.symbols('z1')
t0 = 0
rxnnum = len(Ea)
ysv, Ks, EAs, Ts = symfunc(namesb, rxnnum)
ysv.append(sp.symbols('T0'))
ysv.append(sp.symbols('T1'))
yblank = []
Temp_vals2 = []
Twalls = []
iterlist = [i for i in range(0,iternum,1)]

rtolval = 1E-7
rtolvalj = 1E-7
atolval = 1E-7
atolvalj = 1E-7
firststepval = 1000.0
Ls2 = firststepval
Rr = 8.314462618
Tcedc = 563.0
Pcedc = 5380*1E3
omegaedc = 0.28600000000000003
intvol = math.pi*(ri**2)*Ls
intvols = math.pi*(ri**2)*Ls*segment_second
totalvol = math.pi*(ri**2)*L
tau = totalvol/volume_flow
tauinv = tau**-1
tau3 = (totalvol/u_z)/segment_second
tau2 = intvol/u_z
tau4 = intvols/u_z
taui = 1.0/u_z
taui2 = 1.0
for il in tqdm(range(iternum)):
#for il in range(iternum):
    
    change = il*chngamnt
    amount_new = Temp_K - change
    bedc = Bviral3(amount_new,Tcedc,Pcedc,omegaedc)
    zedc = btoz(bedc,amount_new,PascalP)
    Temp_vals.append(amount_new)
    Temp_vals2.insert(0,amount_new)
    Tnew = Temp_vals[-1]
    T0[0] = Decimal(amount_new)
    T0j[0] = Decimal(amount_new)
    dt = [Tnew]
    diffT0 = [Decimal(Tnew)]
    Twall = amount_new
    Twalls.append(amount_new)
    initedc = PascalP/(amount_new*Rr*zedc) #rhoinit
    EDC = [Decimal(initedc)]
    EDCj = [Decimal(initedc)]
    Eabb = [342,7,42,45,34,48,13,12,4,6,15,0,56,31,30,61,84,90,70,20,70] #[kJ/mol]
    Eab = [x*1000.0 for x in Eabb]
    Ea = [Decimal(x) for x in Eab] #kJ/mol
    ks = [5.9E+15,1.3E13,1E+12,5E+11,1.2E+13,2E+11,1E+13,1E+13,1.7E+13,1.2E+13,1.7E+13,91000000000,1.2E+14,5E+11,20000000000,3E+11,2.1E+14,5E+14,2E+13,1E+14,1.6E+14]
#    n = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,2]
    n = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    k_0b = [i/((1E6)**(j-1)) for i,j in zip(ks,n)]
    k_0 = [De(i) for i in k_0b]

    for x2 in dist_c:
#    for x2 in tqdm(dist_c):
        cross_area = math.pi*(ri**2)
        alistb2 = alistfun(float(T0[-1]))
        Y0b = [EDC[-1],EC[-1],HCl[-1],Coke[-1],CP[-1],Di[-1],C4H6Cl2[-1],C6H6[-1],C2H2[-1],C11[-1],C112[-1],R1[-1],R2[-1],R3[-1],R4[-1],R5[-1],R6[-1],VCM[-1]]
        Y0bfa = [float(i) for i in Y0b]
        aaij = [x*0.0 for x in range(0,len(alistb2),1)]
        Aaij = [aaij[:] for x in range(0,len(alistb2),1)]
        Ctotal = sum(Y0b) 
        C_ii = [float(i)/float(Ctotal) for i in Y0b]
        C_im = tc.utils.zs_to_ws(C_ii,mws)
        mfracnf = [i for i in C_im if i != 0.0]
        molfrac = [i for i in C_ii if i != 0.0]
        MWis1 = mw(alistb2)
        MWis2 = [float(i/1000) for i in MWis1]
        Vm2 = []
        Maverage = mean(MWis2)
        names2b = []
        Ca_i = float(EDC[-1])
        Cpi_list2 = []
        mfrac = []
        MWi_list2 = []
        MWgi_list2 = []
        Rhoi_list2 = []
        mu_i2 = []
        klval_i2 = []
        TB_i2 = []
        C_i2 = []
        cass = []
        TG = []
        VS = []
        for i in  Y0bfa:
            if i > 0.0:
                    k = Y0bfa.index(i)
                    i2 = alistb2[k]
                    C_i2.append(float(i)/float(Ctotal))
                    chemicalcpi = getattr(i2,'Cpgm') #[J/mol*K] 
                    Cpi_list2.append(float(chemicalcpi))
                    chemicalmwi = getattr(i2,'MW') #[g/mol] 
                    MWgi_list2.append(float(chemicalmwi)) #[g/mol]
                    MWi_list2.append(float(chemicalmwi/1000.0)) #[kg/mol]
                    chemicalpi = getattr(i2,'rhogm') #[mol/m^3] 
                    Rhoi_list2.append(float(chemicalpi))
                    chemicalkl = getattr(i2,'kg') #[mol/m^3] 
                    klval_i2.append(float(chemicalkl))
                    chemicalmug= getattr(i2,'mu') #  [Pa*s] 
                    mu_i2.append(float(chemicalmug))
                    chemicaltb= getattr(i2,'Tb') # [K]
                    TB_i2.append(float(chemicaltb))
                    chemicalvm = getattr(i2,'Vmg')
                    Vm2.append(float(chemicalvm/1E6))
                    chemicalna = getattr(i2,'IUPAC_name')
                    names2b.append(chemicalna)
                    chemicalcas = getattr(i2,'CAS')
                    cass.append(chemicalcas)
                    thermo = tc.chemical.Chemical(chemicalcas, T=float(T0[-1]), P=PascalP)
                    TG.append(thermo.ThermalConductivityGas)
                    thermov = tc.chemical.Chemical(chemicalcas, T=float(T0[-1]), P=PascalP)
                    VS.append(thermov.ViscosityGas)
            else:
                pass
        
        cpavg = mean(Cpi_list2)
        rhoavg = mean(Rhoi_list2)
        rhovm = [i*j for i,j in zip(Vm2,Rhoi_list2)]
        rhoend = float(PascalP/(Rval*float(T0[-1])))
        rhotot2 = sum(Rhoi_list2)
        Massfrac2b = [float(i/rhotot2) for i in Rhoi_list2]
        Massfrac = [float(i*j) for i,j in zip(C_i2,MWgi_list2)]
        Mavg = sum(Massfrac)
        wsl = [float((i*j)/Mavg) for i,j in zip(C_i2,MWgi_list2)]
        wsl2 = tc.utils.zs_to_ws(C_i2,MWgi_list2)
        Rho_Cp = [i*j for i,j in zip(Cpi_list2,Rhoi_list2)]
        RCPavg = mean(Rho_Cp)
        
        Cpglist = []
        gmix = tc.Mixture(IDs = names2b, zs=C_i2,  T=float(T0[-1]), P=PascalP)
        viss = gmix.mugs
        tbss = gmix.Tbs
        mwss = gmix.MWs
        ksss = gmix.kgs
        CPs = gmix.Cpgms
        CVs = gmix.Cvgms
        kval2 = kmix(float(T0[-1]),C_i2,ksss, viss, tbss, mwss,CPs,CVs) #[W/m*K]
        Molew = gmix.MWs
        sigma = gmix.molecular_diameters
        stemix = gmix.Stockmayers
        cpmmix = gmix.Cpgm # [J/mol*K]
        cpkgmix = gmix.Cpg # [J/kg*K]
        viscosityb = gmix.mug # [Pa*s] or [kg/m*s^2]
        vislist = gmix.mugs
        viscosityc = viscosityb # [kg/m*s^2]
        viscosity = tc.viscosity.Brokaw(T=float(T0[-1]),ys=C_i2,mus=viss,MWs=mwss,molecular_diameters=sigma,Stockmayers=stemix) # [Pa*s]
        rhob = gmix.rhog #[mol/m^3]
        rhoc = gmix.rhogm #[mol/m^3]
        rhocpgm1 = rhoc*cpmmix # [J/m^3*K]
        distancesb = Ls #m
        distances = Ls # m
        velocityb = u_z #m/s
        velocity = velocityb #m/s
        Tgas = float(T0[-1]) #T = K
        diameter = di #cm
        tclist = gmix.Tcs
        Vclist = gmix.Vcs
        Zcmix = gmix.Zg
        TC = tc.critical.modified_Wilson_Tc(zs=C_i2, Tcs=tclist,Aijs=Aaij)
        TCb = tc.critical.Li(zs=C_i2, Tcs=tclist, Vcs=Vclist)
        ek = tc.lennard_jones.epsilon_Tee_Gotoh_Steward_1(TC)
        ek2 = tc.lennard_jones.epsilon_Bird_Stewart_Lightfoot_critical(TCb)
        ek3 = tc.lennard_jones.epsilon_Flynn(TCb)
        ek4 = tc.lennard_jones.epsilon_Stiel_Thodos(TCb, Zcmix)
        Tstar = tc.lennard_jones.Tstar(T=float(T0[-1]), epsilon_k=ek, epsilon=None)
        Tstar2 = tc.lennard_jones.Tstar(T=float(T0[-1]), epsilon_k=ek2, epsilon=None)
        Tstar3 = tc.lennard_jones.Tstar(T=float(T0[-1]), epsilon_k=ek3, epsilon=None)
        Tstar4 = tc.lennard_jones.Tstar(T=float(T0[-1]), epsilon_k=ek4, epsilon=None)
        colint1 = tc.lennard_jones.collision_integral_Kim_Monroe(Tstar, l=1, s=1)
        gmixw2 = tc.Mixture(IDs = names2b, zs=C_i2, T=float(Twalls[-1]), P=PascalP)
        vsb = gmixw2.mug # [Pa*s]
        vslist = gmixw2.mugs # [Pa*s]
        vs = tc.viscosity.Brokaw(T=float(Twalls[-1]),ys=C_i2,mus=viss,MWs=mwss,molecular_diameters=sigma,Stockmayers=stemix) # [Pa*S]
        pr = Pr(cpkgmix,viscosity,kval2)
        re = reynolds(rhob,u_z,Ls,viscosity)
        nuval = Nus(velocity,rhoc,Ls,diameter,kval2,viscosity,vs,float(Twalls[-1]),float(T0[-1]),pr,re)
        gash = hterm(nuval,Ls,kval2)
        h20 = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20cp = h20.Cpgm
        h20rhob = h20.rhogm #[mol/m^3]
        h20rho = h20rhob #[mol/m^3]
        h20kb = h20.kg # [W/(m*K)]
        h20k = h20kb # [W/m*k]
        h20visb = h20.mug # [Pa*s]
        h20vis = h20visb # [g/m*s^2]
        prh20 = Pr(h20cp,h20vis,h20k)
        reh20 = reynolds(h20rho,velocity,Ls,h20vis)
        h20wall = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20vsb = h20wall.mug # [Pa*s]
        h20vs = h20vsb # [g/m*s^2]
        hnuval = Nus(velocity,h20rho,Ls,diameter,h20k,h20vis,h20vs,float(Twalls[-1]),float(Twalls[-1]),prh20,reh20)
        h20h = hterm(hnuval,Ls,h20k)
        U_coeffb = Uvalue(diameter,do,gash,h20h,ksteel) # [W/m^2*k]
        U_coeff = U_coeffb # [W/m^2*k]
        delhm = 71000.0 #J/mol
        con1b = RCPavg/kval2
        con1 = (u_z*rhocpgm1)/kval2 # [1/m] -> [1/cm]
        con2 = (alpha*U_coeff)/kval2 # [m] (Sa) 
        con3 = delhm/kval2 # -> [m] 
        RE_vals.append(re)
        U_coeffs.append(U_coeff)
        h_vals.append(gash)
        kmix_vals.append(kval2)
        c1_vals.append(con1)
        c2_vals.append(con2)
        c3_vals.append(con3)
        C_i2.clear()
        klval_i2.clear()
        mu_i2.clear()
        TB_i2.clear()
        MWgi_list2.clear()
        Massfrac.clear()
        mfracnf.clear()
        wsl.clear()
        names2b.clear()
        Y0bj = [EDCj[-1],ECj[-1],HClj[-1],Cokej[-1],CPj[-1],Dij[-1],C4H6Cl2j[-1],C6H6j[-1],C2H2j[-1],C11j[-1],C112j[-1],R1j[-1],R2j[-1],R3j[-1],R4j[-1],R5j[-1],R6j[-1],VCMj[-1]]
        Y0bjf = [float(i) for i in Y0bj]
        Ctotalj = sum(Y0bjf)
        C_iij = [float(jj)/float(Ctotalj) for jj in Y0bj]
        alistb2j = alistfun(float(T0j[-1]))
        MWis1j = mw(alistb2j)
        MWis2j = [float(j/1000) for j in MWis1j]
        Vm2j = []
        Maveragej = mean(MWis2j)
        names2bj = []
        Ca_ij = float(EDCj[-1])
        Cpi_list2j = []
        mfracj = []
        MWi_list2j = []
        MWgi_list2j = []
        Rhoi_list2j = []
        mu_i2j = []
        klval_i2j = []
        TB_i2j = []
        C_i2j = []
        for ij in  Y0bjf:
            if ij > 0.0:
                    kj = Y0bjf.index(ij)
                    i2j = alistb2j[kj]
                    C_i2j.append(float(ij)/float(Ctotalj))
                    chemicalcpij = getattr(i2j,'Cpgm') #[J/mol*K] 
                    Cpi_list2j.append(chemicalcpij)
                    chemicalmwij = getattr(i2j,'MW') #[g/mol] 
                    MWgi_list2j.append(chemicalmwij) #[g/mol]
                    MWi_list2j.append(float(chemicalmwij/1000)) #[kg/mol]
                    chemicalpij = getattr(i2j,'rhogm') #[mol/m^3] 
                    Rhoi_list2j.append(chemicalpij) #[mol/cm^3]
                    chemicalklj = getattr(i2j,'kg') #[W/m*K] 
                    klval_i2j.append(chemicalklj) #[W/cm*K]
                    chemicalmugj= getattr(i2j,'mug') #[Pa*s] 
                    mu_i2j.append(chemicalmugj)
                    chemicaltbj= getattr(i2j,'Tb') #[K]
                    TB_i2j.append(chemicaltbj)
                    chemicalvmj = getattr(i2j,'Vmg') #[mol/m^3]
                    Vm2j.append(chemicalvmj) #[mol/cm^3]
                    chemicalnaj = getattr(i2j,'IUPAC_name')
                    names2bj.append(chemicalnaj)
            else:
                pass
        cpavgj = mean(Cpi_list2j)
        rhoavgj = mean(Rhoi_list2j)
        rhovmj = [ij*jj for ij,jj in zip(Vm2j,Rhoi_list2j)]
        rhoendj = float(PascalP/(Rval*float(T0j[-1])))
        rhotot2j = sum(Rhoi_list2j)
        Massfrac2j = [float(ij/rhotot2j) for ij in Rhoi_list2j]
        Rho_Cpj = [i*j for i,j in zip(Cpi_list2j,Rhoi_list2j)]
        RCPavgj = mean(Rho_Cpj)
        Cpglistj = []
        Massfracj = [float(i*j) for i,j in zip(C_i2j,MWgi_list2j)]
        Mavgj = sum(Massfracj)
        wslj = [float((i*j)/Mavgj) for i,j in zip(C_i2j,MWgi_list2j)]
        gmixj = tc.Mixture(IDs = names2bj, zs=C_i2j, T=float(T0j[-1]), P=PascalP)
        Rho_Cp = [i*j for i,j in zip(Cpi_list2,Rhoi_list2)]
        RCPavg = mean(Rho_Cp)
        Cpglist = []
        vissj = gmixj.mugs
        tbssj = gmixj.Tbs
        mwssj = gmixj.MWs
        ksssj = gmixj.kgs
        Molewj = gmixj.MWs
        CPsj = gmixj.Cpgms
        CVsj = gmixj.Cvgms
        kval2j = kmix(float(T0j[-1]),C_i2j,ksssj, vissj, tbssj, mwssj,CPsj,CVsj) #[W/m*K]
        sigmaj = gmixj.molecular_diameters
        stemixj = gmixj.Stockmayers
        cpgmmixj = gmixj.Cpgm  #[J/mol*K]
        viscosityj = gmixj.mug # [g/cm*s^2]
        vislistj= gmixj.mugs # [g/cm*s^2]
        vs2 = tc.viscosity.Brokaw(T=float(T0j[-1]),ys=C_i2j,mus=vissj,MWs=mwssj,molecular_diameters=sigmaj,Stockmayers=stemixj)
        rhoj = gmixj.rhogm #[mol/cm^3]
        rhocpgm2 = rhoj*cpgmmixj # [J/cm^3*K]
        surface_area = math.pi*2*ri*Ls
        distances = Ls #cm
        velocityj = u_z #cm/s
        total_volume = math.pi*Ls*(ri**2)
        alpha = surface_area/total_volume
        Tgasj = float(T0j[-1]) #T = K
        diameterj = di #cm
        tclistj = gmixj.Tcs
        Vclistj = gmixj.Vcs
        Zcmixj = gmixj.Zg
        TCj = tc.critical.modified_Wilson_Tc(zs=C_i2j, Tcs=tclistj,Aijs=Aaij)
        TCbj = tc.critical.Li(zs=C_i2j, Tcs=tclistj, Vcs=Vclistj)
        ekj = tc.lennard_jones.epsilon_Tee_Gotoh_Steward_1(TCj)
        ek1j = tc.lennard_jones.epsilon_Tee_Gotoh_Steward_1(TCbj)
        ek2j = tc.lennard_jones.epsilon_Bird_Stewart_Lightfoot_critical(TCbj)
        ek3j = tc.lennard_jones.epsilon_Flynn(TCbj)
        ek4j = tc.lennard_jones.epsilon_Stiel_Thodos(TCbj, Zcmix)
        Tstarj = tc.lennard_jones.Tstar(T=float(T0j[-1]), epsilon_k=ekj, epsilon=None)
        Tstar1j = tc.lennard_jones.Tstar(T=float(T0j[-1]), epsilon_k=ek1j, epsilon=None)
        Tstar2j = tc.lennard_jones.Tstar(T=float(T0j[-1]), epsilon_k=ek2j, epsilon=None)
        Tstar3j = tc.lennard_jones.Tstar(T=float(T0j[-1]), epsilon_k=ek3j, epsilon=None)
        Tstar4j = tc.lennard_jones.Tstar(T=float(T0j[-1]), epsilon_k=ek4j, epsilon=None)
        colintj = tc.lennard_jones.collision_integral_Kim_Monroe(Tstarj, l=1, s=1)
        colint1j = tc.lennard_jones.collision_integral_Kim_Monroe(Tstar1j, l=1, s=1)
        Tstarj = tc.lennard_jones.Tstar(T=float(T0j[-1]), epsilon_k=ekj, epsilon=None)
        colintj = tc.lennard_jones.collision_integral_Kim_Monroe(Tstarj, l=1, s=1)
        gmixw2j = tc.Mixture(IDs = names2bj, zs=C_i2j, T=float(Twalls[-1]), P=PascalP)
        vsj = gmixw2j.mug # [g/cm*s^2]
        vs2j = tc.viscosity.Brokaw(T=float(Twalls[-1]),ys=C_i2j,mus=vissj,MWs=mwssj,molecular_diameters=sigmaj,Stockmayers=stemixj)
        prj = Pr(cpgmmixj,vsj,kval2j)
        rej = reynolds(rhoj,velocityj,distances,vsj)
        nuvalj = Nus(velocity,rhoj,distances,diameter,kval2j,vsj,vs2j,float(Twalls[-1]),T0j[-1],prj,rej)
        gashj = hterm(nuvalj,distances,kval2j)
        h20j = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20cpj = h20j.Cpgm
        h20rhoj = h20j.rhogm #[mol/m^3]
        h20kj = h20j.kg # [W/(m*K)]
        h20visj = h20j.mug # [g/cm*s^2]
        prh20j = Pr(h20cpj,h20visj,h20kj)
        reh20j = reynolds(h20rhoj,velocityj,distances,h20visj)
        h20wallj = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20vsj = h20wallj.mug # [g/cm*s^2]
        hnuvalj = Nus(velocityj,h20rhoj,distances,diameter,h20kj,h20visj,h20vsj,float(Twalls[-1]),float(Twalls[-1]),prh20j,reh20j)
        h20hj = hterm(hnuvalj,distances,h20kj)
        U_coeffj = Uvalue(diameter,do,gashj,h20hj,ksteel) #(W/(m2•K))
        delhm = 71000.0 #J/mol
        con1j = (u_z*rhocpgm2)/kval2j
        con2j = (alpha*U_coeffj)/kval2j
        con3j = delhm/kval2j
        RE_valsj.append(rej)
        U_coeffsj.append(U_coeffj)
        h_valsj.append(gashj)
        kmix_valsj.append(kval2j)
        c1_valsj.append(con1j)
        c2_valsj.append(con2j)
        c3_valsj.append(con3j)
        z = sp.symbols('z')
        t_start = x2
        t_end = x2 + Ls
        z2 = sp.symbols('z')
        Y02b = [EDC[-1],EC[-1],HCl[-1],Coke[-1],CP[-1],Di[-1],C4H6Cl2[-1],C6H6[-1],C2H2[-1],C11[-1],C112[-1],R1[-1],R2[-1],R3[-1],R4[-1],R5[-1],R6[-1],VCM[-1]]
        Y02 = [EDC[-1],EC[-1],HCl[-1],Coke[-1],CP[-1],Di[-1],C4H6Cl2[-1],C6H6[-1],C2H2[-1],C11[-1],C112[-1],R1[-1],R2[-1],R3[-1],R4[-1],R5[-1],R6[-1],VCM[-1],T0[-1],T1[-1]]
        Y02j = [EDCj[-1],ECj[-1],HClj[-1],Cokej[-1],CPj[-1],Dij[-1],C4H6Cl2j[-1],C6H6j[-1],C2H2j[-1],C11j[-1],C112j[-1],R1j[-1],R2j[-1],R3j[-1],R4j[-1],R5j[-1],R6j[-1],VCMj[-1],T0j[-1],T1j[-1]]
        Y02jb = [EDCj[-1],ECj[-1],HClj[-1],Cokej[-1],CPj[-1],Dij[-1],C4H6Cl2j[-1],C6H6j[-1],C2H2j[-1],C11j[-1],C112j[-1],R1j[-1],R2j[-1],R3j[-1],R4j[-1],R5j[-1],R6j[-1],VCMj[-1]]
        Y0D = [De(x) for x in Y02]
        Y0Db = [De(x) for x in Y02b]
        Y0Dj = [De(x) for x in Y02j]
        Y0Dj2 = [De(x) for x in Y02jb]
        args1 = [De(c1_vals[-1]),De(c2_vals[-1]),De(c3_vals[-1]),k_0, De(Twalls[-1])]
        argsj = [float(c1_valsj[-1]),float(c2_valsj[-1]),float(c3_valsj[-1]),k_0b, float(Twalls[-1])]
        argsjD = [De(c1_valsj[-1]),De(c2_valsj[-1]),De(c3_valsj[-1]),k_0, De(Twalls[-1])]
        getcontext().prec = 25
        precision = 25
        res = solve_ivp(RHS, [0.0,Ls], Y02 , method = 'Radau',  args=(8.314, 5.9E15, 13000000.0, 1000000.0, 500000.0, 12000000.0, 200000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 91000.0, 120000000.0, 500000.0, 20000.0, 300000.0, 2.1E+8,5E+8,2E+7, 100000000.0, 160000000.0, 342000.0, 7000.0, 42000.0, 45000.0, 34000.0, 48000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 0.0, 56000.0, 31000.0, 30000.0, 61000.0, 84000.0, 90000.0, 70000.0, 20000.0, 70000.0, float(c1_valsj[-1]),float(c2_valsj[-1]),float(c3_valsj[-1])), jac=jacob)
        resj = solve_ivp(RHS, [0.0,Ls], Y02j , method = 'Radau',  args=(8.314, 5.9E15, 13000000.0, 1000000.0, 500000.0, 12000000.0, 200000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 91000.0, 120000000.0, 500000.0, 20000.0, 300000.0, 2.1E+8,5E+8,2E+7, 100000000.0, 160000000.0, 342000.0, 7000.0, 42000.0, 45000.0, 34000.0, 48000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 0.0, 56000.0, 31000.0, 30000.0, 61000.0, 84000.0, 90000.0, 70000.0, 20000.0, 70000.0, float(c1_valsj[-1]),float(c2_valsj[-1]),float(c3_valsj[-1])), jac=jacob)
        Ls2 = firststepval
        edcint = initedc - res.y[0][-1]
        edcintj = initedc - resj.y[0][-1]
        yield1 = res.y[17][-1]/edcint
        yield1j = resj.y[17][-1]/edcintj
        yield_vcm.append(yield1)
        yield_vcmj.append(yield1j)
        selectivity_val = (res.y[17][-1]/res.y[2][-1])
        selectivity.append(selectivity_val)
        selectivity_val2 = (res.y[2][-1]/res.y[17][-1])
        selectivity2.append(selectivity_val2)
        selectivity_valj = (resj.y[17][-1]/resj.y[2][-1])
        selectivityj.append(selectivity_valj)
        selectivity_val2j = (resj.y[2][-1]/resj.y[17][-1])
        selectivity2j.append(selectivity_val2j)
        conversionedc = 100.0*(1.0 - res.y[0][-1]/initedc)
        conversionedcj = 100.0*(1.0 - resj.y[0][-1]/initedc)
        conversion_EDCb.append(conversionedc)
        conversion_EDCjb.append(conversionedcj)
        EDC.append(Decimal(res.y[0][-1]))
        EC.append(Decimal(res.y[1][-1]))
        HCl.append(Decimal(res.y[2][-1]))
        Coke.append(Decimal(res.y[3][-1]))
        CP.append(Decimal(res.y[4][-1]))
        Di.append(Decimal(res.y[5][-1]))
        C4H6Cl2.append(Decimal(res.y[6][-1]))
        C6H6.append(Decimal(res.y[7][-1]))
        C2H2.append(Decimal(res.y[8][-1]))
        C11.append(Decimal(res.y[9][-1]))
        C112.append(Decimal(res.y[10][-1]))
        R1.append(Decimal(res.y[11][-1]))
        R2.append(Decimal(res.y[12][-1]))
        R3.append(Decimal(res.y[13][-1]))
        R4.append(Decimal(res.y[14][-1]))
        R5.append(Decimal(res.y[15][-1]))
        R6.append(Decimal(res.y[16][-1]))
        VCM.append(Decimal(res.y[17][-1]))
        T0.append(Decimal(res.y[18][-1]))
        T1.append(Decimal(res.y[19][-1]))
        EDCj.append(Decimal(resj.y[0][-1]))
        ECj.append(Decimal(resj.y[1][-1]))
        HClj.append(Decimal(resj.y[2][-1]))
        Cokej.append(Decimal(resj.y[3][-1]))
        CPj.append(Decimal(resj.y[4][-1]))
        Dij.append(Decimal(resj.y[5][-1]))
        C4H6Cl2j.append(Decimal(resj.y[6][-1]))
        C6H6j.append(Decimal(resj.y[7][-1]))
        C2H2j.append(Decimal(resj.y[8][-1]))
        C11j.append(Decimal(resj.y[9][-1]))
        C112j.append(Decimal(resj.y[10][-1]))
        R1j.append(Decimal(resj.y[11][-1]))
        R2j.append(Decimal(resj.y[12][-1]))
        R3j.append(Decimal(resj.y[13][-1]))
        R4j.append(Decimal(resj.y[14][-1]))
        R5j.append(Decimal(resj.y[15][-1]))
        R6j.append(Decimal(resj.y[16][-1]))
        VCMj.append(Decimal(resj.y[17][-1]))
        T0j.append(Decimal(resj.y[18][-1]))
        T1j.append(Decimal(resj.y[19][-1]))
        Y1 = [res.y[0][-1], res.y[1][-1],res.y[2][-1],res.y[3][-1], res.y[4][-1], res.y[5][-1], res.y[6][-1], res.y[7][-1], res.y[8][-1], res.y[9][-1], res.y[10][-1], res.y[11][-1],res.y[12][-1], res.y[13][-1], res.y[14][-1], res.y[15][-1], res.y[16][-1], res.y[17][-1]]
        Y1j = [resj.y[0][-1], resj.y[1][-1],resj.y[2][-1],resj.y[3][-1], resj.y[4][-1], resj.y[5][-1], resj.y[6][-1], resj.y[7][-1], resj.y[8][-1], resj.y[9][-1], resj.y[10][-1], resj.y[11][-1],resj.y[12][-1], resj.y[13][-1], resj.y[14][-1], resj.y[15][-1], resj.y[16][-1], resj.y[17][-1]]
        P1 = [res.y[1][-1],res.y[2][-1],res.y[3][-1], res.y[4][-1], res.y[5][-1], res.y[6][-1], res.y[7][-1], res.y[8][-1], res.y[9][-1], res.y[10][-1], res.y[11][-1],res.y[12][-1], res.y[13][-1], res.y[14][-1], res.y[15][-1], res.y[16][-1], res.y[17][-1]]
        P1j = [resj.y[1][-1],resj.y[2][-1],resj.y[3][-1], resj.y[4][-1], resj.y[5][-1], resj.y[6][-1], resj.y[7][-1], resj.y[8][-1], resj.y[9][-1], resj.y[10][-1], resj.y[11][-1],resj.y[12][-1], resj.y[13][-1], resj.y[14][-1], resj.y[15][-1], resj.y[16][-1], resj.y[17][-1]]
        D1 = [res.y[2][-1],res.y[17][-1]]
        D1j = [resj.y[2][-1],resj.y[17][-1]]
        C_T0 = sum(Y1)
        C_Total.append(C_T0)
        C_T0j = sum(Y1j)
        C_Totalj.append(C_T0j)
        prod1 = sum(P1)
        prod1j = sum(P1j)
        des1 = sum(D1)
        des1j = sum(D1j)
        pur1 = (sum(D1)/sum(P1)) * 100
        pr1.append(pur1)
        pur1j = (sum(D1j)/sum(P1j)) * 100 
        pr1j.append(pur1j)
        prev_eval = J_eval[0] #This loads the previous number of jacobian calculations
        j_eval = resj.nfev + prev_eval #This adds the previous to the most recent amount
        J_eval[0] = j_eval #This stores them in the initial list from above
    EDCl.append([i for i in EDC])
    ECl.append([i for i in EC])
    HCll.append([i for i in HCl])
    Cokel.append([i for i in Coke])
    CPl.append([i for i in CP])
    Dil.append([i for i in Di])
    C4H6Cl2l.append([i for i in C4H6Cl2])
    C6H6l.append([i for i in C6H6])
    C2H2l.append([i for i in C2H2])
    C11l.append([i for i in C11])
    C112l.append([i for i in C112])
    R1l.append([i for i in R1])
    R2l.append([i for i in R2])
    R3l.append([i for i in R3])
    R4l.append([i for i in R4])
    R5l.append([i for i in R5])
    R6l.append([i for i in R6])
    VCMl.append([i for i in VCM])
    T0l.append([i for i in T0])
    T1l.append([i for i in T1])
    selectivityl.append([i for i in selectivity])
    selectivity2l.append([i for i in selectivity2])
    selectivityjl.append([i for i in selectivityj])
    selectivity2jl.append([i for i in selectivity2j])
    yield_vcml.append([i*100 for i in yield_vcm])
    yield_vcmjl.append([i*100 for i in yield_vcmj])
    EDClj.append([i for i in EDCj])
    EClj.append([i for i in ECj])
    HCllj.append([i for i in HClj])
    Cokelj.append([i for i in Cokej])
    CPlj.append([i for i in CPj])
    Dilj.append([i for i in Dij])
    C4H6Cl2lj.append([i for i in C4H6Cl2j])
    C6H6lj.append([i for i in C6H6j])
    C2H2lj.append([i for i in C2H2j])
    C11lj.append([i for i in C11j])
    C112lj.append([i for i in C112j])
    R1lj.append([i for i in R1j])
    R2lj.append([i for i in R2j])
    R3lj.append([i for i in R3j])
    R4lj.append([i for i in R4j])
    R5lj.append([i for i in R5j])
    R6lj.append([i for i in R6j])
    VCMlj.append([i for i in VCMj])
    T0lj.append([i for i in T0j])
    T1lj.append([i for i in T1j])
    RE_valsl.append([De(i) for i in RE_vals])
    U_coeffsl.append([De(i) for i in U_coeffs])
    h_valsl.append([De(i) for i in h_vals])
    kmix_valsl.append([De(i) for i in kmix_vals])
    c1_valsl.append([De(i) for i in c1_vals])
    c2_valsl.append([De(i) for i in c2_vals])
    c3_valsl.append([De(i) for i in c3_vals])
    RE_valsjl.append([De(i) for i in RE_valsj])
    U_coeffsjl.append([De(i) for i in U_coeffsj])
    h_valsjl.append([De(i) for i in h_valsj])
    kmix_valsjl.append([De(i) for i in kmix_valsj])
    c1_valsjl.append([De(i) for i in c1_valsj])
    c2_valsjl.append([De(i) for i in c2_valsj])
    c3_valsjl.append([De(i) for i in c3_valsj])
    conversion_EDC.append([De(i) for i in conversion_EDCb])
    conversion_EDCj.append([De(i) for i in conversion_EDCjb]) 
    pure.append(pr1[:])
    purej.append(pr1j[:])
    RE_vals.clear()
    RE_valsj.clear()
    c1_vals.clear()
    c2_vals.clear()
    c3_vals.clear()
    c1_valsj.clear()
    c2_valsj.clear()
    c3_valsj.clear()
    U_coeffs.clear()
    U_coeffsj.clear()
    h_vals.clear()
    h_valsj.clear()
    kmix_vals.clear()
    kmix_valsj.clear()
    EDC.clear()
    EC.clear()
    HCl.clear()
    Coke.clear()    
    CP.clear()
    Di.clear()
    C4H6Cl2.clear()
    C6H6.clear()
    C2H2.clear()
    C11.clear()
    C112.clear()
    R1.clear()
    R2.clear()
    R3.clear()
    R4.clear()
    R5.clear()
    R6.clear()
    VCM.clear()
    T0.clear()
    T1.clear()
    selectivity.clear()
    selectivity2.clear()
    selectivityj.clear()
    selectivity2j.clear()
    yield_vcm.clear()
    yield_vcmj.clear()
    EDCj.clear()
    ECj.clear()
    HClj.clear()
    Cokej.clear()    
    CPj.clear()
    Dij.clear()
    C4H6Cl2j.clear()
    C6H6j.clear()
    C2H2j.clear()
    C11j.clear()
    C112j.clear()
    R1j.clear()
    R2j.clear()
    R3j.clear()
    R4j.clear()
    R5j.clear()
    R6j.clear()
    VCMj.clear()
    T0j.clear()
    T1j.clear()
#    EDC = [Decimal(initedc)]
    EC = [Decimal(0.0)]
    HCl = [Decimal(0.0)] 
    Coke = [Decimal(0.0)] 
    CP = [Decimal(0.0)]
    Di = [Decimal(0.0)]
    C4H6Cl2 = [Decimal(0.0)]
    C6H6 = [Decimal(0.0)]
    C2H2 = [Decimal(0.0)]
    C11 = [Decimal(0.0)]
    C112 = [Decimal(0.0)]
    R1 = [Decimal(0.0)]
    R2 = [Decimal(0.0)]
    R3 = [Decimal(0.0)]
    R4 = [Decimal(0.0)]
    R5 = [Decimal(0.0)]
    R6 = [Decimal(0.0)]
    VCM = [Decimal(0.0)]
    T0 = [Decimal(Temp_K)]
    T1 = [Decimal(0.0)]
    selectivity = [1.0]
    selectivity2 = [1.0]
    selectivityj = [1.0]
    selectivity2j = [1.0]
    yield_vcm = [1.0]
    yield_vcmj = [1.0]
#    EDCj = [Decimal(initedc)]
    ECj = [Decimal(0.0)]
    HClj = [Decimal(0.0)] 
    Cokej = [Decimal(0.0)] 
    CPj = [Decimal(0.0)]
    Dij = [Decimal(0.0)]
    C4H6Cl2j = [Decimal(0.0)]
    C6H6j = [Decimal(0.0)]
    C2H2j = [Decimal(0.0)]
    C11j = [Decimal(0.0)]
    C112j = [Decimal(0.0)]
    R1j = [Decimal(0.0)]
    R2j = [Decimal(0.0)]
    R3j = [Decimal(0.0)]
    R4j = [Decimal(0.0)]
    R5j = [Decimal(0.0)]
    R6j = [Decimal(0.0)]
    VCMj = [Decimal(0.0)]
    T0j = [Decimal(Temp_K)]
    T1j = [Decimal(0.0)]
    Eab.clear()
    Ea.clear()
    conversion_EDCb.clear()
    conversion_EDCjb.clear()
    conversion_EDCb = [0.0]
    conversion_EDCjb = [0.0]
    pr1.clear()
    pr1j.clear()
#    dpr1.clear()
    pr1.append(100.0)
    pr1j.append(100.0)
    rtolval = 1E-7
    rtolvalj = 1E-7
    atolval = 1E-7
    atolvalj = 1E-7
    Ls2 = firststepval
    
eaconvf = []
eaconvjf = []

for i,j in enumerate(conversion_EDC):
    val1 = conversion_EDC[i]
    val2 = val1[-1]
    eaconvf.append(val2)
    
for i2,j2 in enumerate(conversion_EDCj):
    val1j = conversion_EDCj[i2]
    val2j = val1j[-1]
    eaconvjf.append(val2j)    
    
totallist1 = [EDCl,ECl,HCll,Cokel,CPl,Dil,C4H6Cl2l,C6H6l,C2H2l,C11l,C112l,R1l,R2l,R3l,R4l,R5l,R6l,VCMl,T0l,T1l,pure,selectivityl,selectivity2l,yield_vcml,c1_valsl,c2_valsl,c3_valsl,h_valsl,U_coeffsl,kmix_valsl,RE_valsl]
totallist2 = [EDClj,EClj,HCllj,Cokelj,CPlj,Dilj,C4H6Cl2lj,C6H6lj,C2H2lj,C11lj,C112lj,R1lj,R2lj,R3lj,R4lj,R5lj,R6lj,VCMlj,T0lj,T1lj,purej,selectivityjl,selectivity2jl,yield_vcmjl,c1_valsjl,c2_valsjl,c3_valsjl,h_valsjl,U_coeffsjl,kmix_valsjl,RE_valsjl]
for i,j in enumerate(totallist1):
    namel = namespd[i]
    a1 = np.asarray(totallist1[i])
    a1j = np.asarray(totallist2[i])
    a2 = a1.astype(np.longdouble)
    a2j = a1j.astype(np.longdouble)
    lista = totallist1[i]
    listb = totallist2[i]
    ab = []
    abj = []
    abd = []
    att = []
    attj = []
    attd = []
    at = [time_valsb]
    atj = [time_valsb]
    at2 = [time_valsb]
    ad = [dist_b]
    adj = [dist_b]
    ad2 = [dist_b]
    for j,k in enumerate(lista):
        ll = np.asarray(lista[j])
        llf = ll.astype(np.longdouble)
        ad.append(llf)
        at.append(llf)
    for j,k in enumerate(listb):
        llj = np.asarray(listb[j])
        llfj = llj.astype(np.longdouble)
        adj.append(llfj)
        atj.append(llfj)
    eea = Temp_vals[:]
    eet = Temp_vals[:]
    eead = Temp_vals[:]
    eav = np.asarray(eea)
    eas = ["Temperature Value: {} [K]".format(i) for i in Temp_vals]
    eaf = ['Temperatures Values']
    eaf.append(eas)
    ttv = ["Time [s]"]
    ttv2 = ["Time Interval: {} [s]".format(time_int)]
    ddv = ["Distance [m]"]
    ddv2 = ["Velocity: {} [m/s]".format(u_z)]
    ddvd = ["Distance [m]"]
    ddv2d = ["Velocity: {} [m/s]".format(u_z)]
    for i,j in enumerate(Temp_vals):
        eaf.append(eas[i])
        ttv.append(eas[i])
        ddv.append(eas[i])
        ddvd.append(eas[i])
        ttv2.append(Temp_vals[i])
        ddv2.append(Temp_vals[i])
        ddv2d.append(Temp_vals[i])
    indexd = pd.Index(ddv,name=['Units'])
    indext = pd.Index(ttv,name=['Units'])
    indexdd = pd.Index(ddvd,name=['Units'])
    eat = np.asarray(eet)
    df = pd.DataFrame(ad,index=[*ddv])
    df.index.name = 'Units'
    edv = pd.DataFrame(ddv2,index=[*ddv])
    edv.index.name = 'Units'
    frames = [edv,df]
    result1 = pd.concat(frames,join='outer',axis=1, ignore_index=True)
    dt = pd.DataFrame(at,index=[*ttv])
    dt.index.name = 'Units'
    etv = pd.DataFrame(ttv2,index=[*ttv])
    etv.index.name = 'Units'
    framest = [etv,dt]
    result2 = pd.concat(framest,join='outer',axis=1, ignore_index=True)
    dfD = pd.DataFrame(ad2,index=[*ddv])
    dfD.index.name = 'Units'
    edvd = pd.DataFrame(ddv2d,index=[*ddvd])
    edvd.index.name = 'Units'
    result1.to_excel(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{} Data.xlsx".format(namel), sheet_name="{} Data.xlsx".format(namel))
    result2.to_excel(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{}-Ti Data.xlsx".format(namel), sheet_name="{}-Ti Data.xlsx".format(namel))
    dfj = pd.DataFrame(adj,index=[*ddv])
    dtj = pd.DataFrame(atj,index=[*ttv])
    dfDd = pd.DataFrame(ad2,index=[*ddvd])
    edvj = pd.DataFrame(ddv2,index=[*ddv])
    edvj.index.name = 'Units'
    etvj = pd.DataFrame(ttv2,index=[*ttv])
    etvj.index.name = 'Units'
    framesj = [edvj,dfj]
    framestj = [etvj,dtj]
    ans1j = pd.concat(framesj,join='outer',axis=1, ignore_index=True)
    ans2j = pd.concat(framestj,join='outer',axis=1, ignore_index=True)
    ans1j.to_excel(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{}j Data.xlsx".format(namel), sheet_name="{}-J Data.xlsx".format(namel))
    ans2j.to_excel(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{}j-Ti Data.xlsx".format(namel), sheet_name="{}-J-Ti Data.xlsx".format(namel))


xD = np.linspace(0,u_z*desired_time,segment_num+1)
timeD = np.linspace(0,desired_time,segment_num+1)


edc = np.array(EDCl)
np.savetxt(r"{}\EDC.txt".format(dir_path),edc)
ec = np.array(ECl)
np.savetxt(r"{}\EC.txt".format(dir_path),ec)
hcl = np.array(HCll)
np.savetxt(r"{}\HCl.txt".format(dir_path),hcl)
cc = np.array(Cokel)
np.savetxt(r"{}\Coke.txt".format(dir_path),cc)   
cp1 = np.array(CPl)
np.savetxt(r"{}\cpb.txt".format(dir_path),cp1)
di = np.array(Dil)
np.savetxt(r"{}\Di.txt".format(dir_path),di)
c4h6cl2 = np.array(C4H6Cl2l)
np.savetxt(r"{}\C4H6Cl2.txt".format(dir_path),c4h6cl2) 
c6h6 = np.array(C6H6l)
np.savetxt(r"{}\C6H6.txt".format(dir_path),c6h6) 
c2h2 = np.array(C2H2l)
np.savetxt(r"{}\C2H2.txt".format(dir_path),c2h2)
c11 = np.array(C11l)
np.savetxt(r"{}\C11.txt".format(dir_path),c11)
c112 = np.array(C112l)
np.savetxt(r"{}\C112.txt".format(dir_path),c112)
r1 = np.array(R1l)
np.savetxt(r"{}\R1.txt".format(dir_path),r1) 
r2 = np.array(R2l)
np.savetxt(r"{}\R2.txt".format(dir_path),r2)
r3 = np.array(R3l) 
np.savetxt(r"{}\R3.txt".format(dir_path),r3)
r4 = np.array(R4l)
np.savetxt(r"{}\R4.txt".format(dir_path),r4) 
r5 = np.array(R5l)
np.savetxt(r"{}\R5.txt".format(dir_path),r5)
r6 = np.array(R6l)
np.savetxt(r"{}\R6.txt".format(dir_path),r6)
vcm = np.array(VCMl)
np.savetxt(r"{}\VCM.txt".format(dir_path),vcm)

convedc = np.array(conversion_EDC)
np.savetxt(r"{}\Conversion EDC.txt".format(dir_path),convedc)
edcj = np.array(EDClj)
np.savetxt(r"{}\EDCj.txt".format(dir_path),edcj)
ecj = np.array(EClj)
np.savetxt(r"{}\ECj.txt".format(dir_path),ecj)
hclj = np.array(HCllj)
np.savetxt(r"{}\HClj.txt".format(dir_path),hclj)
ccj = np.array(Cokelj)
np.savetxt(r"{}\Cokej.txt".format(dir_path),ccj)   
cp1j = np.array(CPlj)
np.savetxt(r"{}\cpbj.txt".format(dir_path),cp1j)
dij = np.array(Dilj)
np.savetxt(r"{}\Dij.txt".format(dir_path),dij)
c4h6cl2j = np.array(C4H6Cl2lj)
np.savetxt(r"{}\C4H6Cl2j.txt".format(dir_path),c4h6cl2j) 
c6h6j = np.array(C6H6lj)
np.savetxt(r"{}\C6H6j.txt".format(dir_path),c6h6j) 
c2h2j = np.array(C2H2lj)
np.savetxt(r"{}\C2H2j.txt".format(dir_path),c2h2j)
c11j = np.array(C11lj)
np.savetxt(r"{}\C11j.txt".format(dir_path),c11j)
c112j = np.array(C112lj)
np.savetxt(r"{}\C112j.txt".format(dir_path),c112j)
r1j = np.array(R1lj)
np.savetxt(r"{}\R1j.txt".format(dir_path),r1j) 
r2j = np.array(R2lj)
np.savetxt(r"{}\R2j.txt".format(dir_path),r2j)
r3j = np.array(R3lj) 
np.savetxt(r"{}\R3j.txt".format(dir_path),r3j)
r4j = np.array(R4lj)
np.savetxt(r"{}\R4j.txt".format(dir_path),r4j) 
r5j = np.array(R5lj)
np.savetxt(r"{}\R5j.txt".format(dir_path),r5j)
r6j = np.array(R6lj)
np.savetxt(r"{}\R6j.txt".format(dir_path),r6j)
vcmj = np.array(VCMlj)
np.savetxt(r"{}\VCMj.txt".format(dir_path),vcmj)
convedcj = np.array(conversion_EDCj)
np.savetxt(r"{}\Conversion_EDCJac.txt".format(dir_path),convedcj)
total = np.array(C_Total,ndmin=1)
np.savetxt(r"{}\Total.txt".format(dir_path),total)
purel = np.array(pure,ndmin=1)
np.savetxt(r"{}\Purity.txt".format(dir_path),purel)
distance = np.array(xD)
np.savetxt(r"{}\Distance.txt".format(dir_path),distance)   
t0 = np.array(T0l,ndmin=1)
np.savetxt(r"{}\Temperature.txt".format(dir_path),t0)
dT = np.array(T1l,ndmin=1)
np.savetxt(r"{}\TemperatureDifferential.txt".format(dir_path),dT) 
revals = np.array(RE_valsl,ndmin=1)
np.savetxt(r"{}\Reynolds.txt".format(dir_path),revals)
uvals = np.array(U_coeffsl,ndmin=1)
np.savetxt(r"{}\Uvals.txt".format(dir_path),uvals)
hvals = np.array(h_valsl,ndmin=1)
np.savetxt(r"{}\Hvals.txt".format(dir_path),hvals)
kvals = np.array(kmix_valsl,ndmin=1)
np.savetxt(r"{}\Kvals.txt".format(dir_path),kvals)
c1vals = np.array(c1_valsl,ndmin=1)
np.savetxt(r"{}\Constant1.txt".format(dir_path),c1vals) 
c2vals = np.array(c2_valsl,ndmin=1)
np.savetxt(r"{}\Constant2.txt".format(dir_path),c2vals)
c3vals = np.array(c3_valsl,ndmin=1)
np.savetxt(r"{}\Constant3.txt".format(dir_path),c3vals) 
revalsj = np.array(RE_valsjl,ndmin=1)
np.savetxt(r"{}\ReynoldsJ.txt".format(dir_path),revalsj)
uvalsj = np.array(U_coeffsjl,ndmin=1)
np.savetxt(r"{}\Uvalsj.txt".format(dir_path),uvalsj)
hvalsj = np.array(h_valsjl,ndmin=1)
np.savetxt(r"{}\Hvalsj.txt".format(dir_path),hvalsj)
kvalsj = np.array(kmix_valsjl,ndmin=1)
np.savetxt(r"{}\Kvalsj.txt".format(dir_path),kvalsj)
c1valsj = np.array(c1_valsjl,ndmin=1)
np.savetxt(r"{}\Constant1j.txt".format(dir_path),c1valsj) 
c2valsj = np.array(c2_valsjl,ndmin=1)
np.savetxt(r"{}\Constant2j.txt".format(dir_path),c2valsj)
c3valsj = np.array(c3_valsjl,ndmin=1)
np.savetxt(r"{}\Constant3j.txt".format(dir_path),c3valsj) 
totalj = np.array(C_Totalj,ndmin=1)
np.savetxt(r"{}\Totalj.txt".format(dir_path),totalj)
purejl = np.array(purej,ndmin=1)
np.savetxt(r"{}\Purityj.txt".format(dir_path),purejl)
t0j = np.array(T0lj,ndmin=1)
np.savetxt(r"{}\TemperatureJac.txt".format(dir_path),t0j)
dTj = np.array(T1lj,ndmin=1)
np.savetxt(r"{}\TemperatureDifferentialJac.txt".format(dir_path),dTj) 
tvals = np.array(Temp_vals,ndmin=1)
np.savetxt(r"{}\Temperaturevals.txt".format(dir_path),tvals)
tvals2 = np.array(Temp_vals2,ndmin=1)
np.savetxt(r"{}\Temperaturevals2.txt".format(dir_path),tvals2) 
eaendc = np.array(eaconvf,ndmin=1)
np.savetxt(r"{}\FinalConversion.txt".format(dir_path),eaendc) 
eaendcj = np.array(eaconvjf,ndmin=1)
np.savetxt(r"{}\FinalConversionJ.txt".format(dir_path),eaendcj) 
d2tdz2 = np.array(T1lj,ndmin=1)
np.savetxt(r"{}\TemperatureDifferentialJac.txt".format(dir_path),dTj)

selectvcm = np.array(selectivityl,ndmin=1)
np.savetxt(r"{}\selectivityvcm.txt".format(dir_path),selectvcm) 
selecthcl = np.array(selectivity2l,ndmin=1)
np.savetxt(r"{}\selectivityhcl.txt".format(dir_path),selecthcl) 

selectvcmj = np.array(selectivityjl,ndmin=1)
np.savetxt(r"{}\selectivityvcmj.txt".format(dir_path),selectvcmj) 
selecthclj = np.array(selectivity2jl,ndmin=1)
np.savetxt(r"{}\selectivityhclj.txt".format(dir_path),selecthclj) 

yieldvcm = np.array(yield_vcml,ndmin=1)
np.savetxt(r"{}\selectivityvcmj.txt".format(dir_path),yieldvcm) 
yieldvcmj = np.array(yield_vcmjl,ndmin=1)
np.savetxt(r"{}\selectivityvcmj.txt".format(dir_path),yieldvcmj)

#Temperature = Temp_K - 273.15

#viridis = cm.get_cmap('viridis', iternum+1)
#
#concl = [edc,ec,hcl,cc,cp1,di,c4h6cl2,c6h6,c2h2,c11,c112,r1,r2,r3,r4,r5,r6,vcm]
#concjl = [edcj,ecj,hclj,ccj,cp1j,dij,c4h6cl2j,c6h6j,c2h2j,c11j,c112j,r1j,r2j,r3j,r4j,r5j,r6j,vcmj]
#
#
#for i, j in enumerate(concl):
#    cur_list = concl[i]
#    nameD = namesb[i]
#    nameD2 = namesb2[i]
#    for ii,jj in enumerate(graph_nodes):
#        fig = plt.figure()
#        cur_list2 = concl[i][jj]
#        eavalg = str(Temp_vals[jj])
#        index_jj = int(jj)
#        
#        
#        plt.plot(xD, cur_list2, color=viridis.colors[index_jj,:], label=r"Temperature {}K".format(eavalg))
#        plt.legend(loc='best')
#        plt.grid()
#        plt.xlabel(r'Distance [$m$]')
#        plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#        plt.title('{} Concentration'.format(nameD2))
#        fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{}-Concentration NoJ-{}.pdf".format(nameD,eavalg))
#        fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{}-Concentration NoJ-{}.svg".format(nameD,eavalg))
#
#for i, j in enumerate(concjl):
#    cur_list = concjl[i]
#    nameD = namesb[i]
#    nameD2 = namesb2[i]
#    for ii,jj in enumerate(graph_nodes):
#        fig = plt.figure()
#        cur_list2 = concjl[i][jj]
#        eavalg = str(Temp_vals[jj])
#        index_jj = int(jj)
#        
#        
#        plt.plot(xD, cur_list2, color=viridis.colors[index_jj,:], label=r"Temperature {}K".format(eavalg))
#        plt.legend(loc='best')
#        plt.grid()
#        plt.xlabel(r'Distance [$m$]')
#        plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#        plt.title('{} Concentration'.format(nameD2))
#        fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{}-Concentration J-{}.pdf".format(nameD,eavalg))
#        fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\{}-Concentration J-{}.svg".format(nameD,eavalg))
#
#
#
#font = {'family': 'serif',
#        'color':  'black',
#        'weight': 'normal',
#        'size': 16,
#        }
#
#font2 = {'family': 'serif',
#        'color':  'black',
#        'weight': 'normal',
#        'size': 30,
#        'ha': 'center',
#        'va': 'bottom'
#        }
#
#font3 = {'family': 'serif',
#        'color':  'black',
#        'weight': 'normal',
#        'size': 18,
#        'ha': 'center',
#        'va': 'center',
#        'bbox': dict(boxstyle="square", fc="white", ec="black", pad=0.1)
#        }
#
##fig, ax = plt.subplots()
##ax.plot(xD, edc[0,:]) 
##ax.set(xlabel=r'Distance [$m$]', ylabel=r'Concentration [$\frac{mol}{m^3}$]', 
##      title="EDC Concentration Profile")
##ax.grid()
#fig = plt.figure()
#plt1 = plt.plot(xD, edc[0,:])
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.xlabel(r'Distance [$m$]')
#plt.title("EDC Concentration Profile")
#plt.grid()
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDCconcentration.png") #Saves the graph
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDCconcentration.svg")
#
#fig, ax = plt.subplots()
#ax.plot(tvals, eaendc, 'b-', label='Conversion')
#ax.plot(tvals, eaendcj, 'r-', label='Conversion With Jacobian')
#ax.set(xlabel="Initial Temperature [K]", ylabel='Conversion [%]', title="Final Conversion")
#ax.legend()
#ax.grid()
#ax.invert_xaxis()
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Final Conversionback.pdf", bbox_inches='tight')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Final Conversionback.svg", bbox_inches='tight')
#
#fig = plt.figure()
#plt1e = plt.plot(tvals, eaendc, 'b-')
#plt2e = plt.plot(tvals, eaendcj, 'g-')
#plt.ylabel('Conversion')
#plt.xlabel("Initial Temperature [K]")
#plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#plt.title("Final Conversion")
#plt.legend(['Conversion','Conversion With Jacobian'],loc="best")
#plt.grid()
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Final Conversion.png") #Saves the graph
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Final Conversion.svg")
#
#fig = plt.figure()
#for ii,i in enumerate(graph_nodes2):
#    index_i = int(i) 
#    
#    
#    plt.plot(xD, edc[i, :], color=viridis.colors[index_i, :], label="Temperature Value: {} [K]".format(Temp_vals[i]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('EDC Concentration')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC-NoJ.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC-NoJ.svg")
#
#fig = plt.figure()
#for ii,i in enumerate(graph_nodes2):
#    index_i = int(i) 
#    
#    
#    plt.plot(xD, edcj[i, :], color=viridis.colors[index_i, :], label="Temperature Value: {} [K]".format(Temp_vals[i]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('EDC Concentration')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC-J.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC-J.svg")
#
#
#for jj,j in enumerate(graph_nodes):
#       fig = plt.figure()
#       index_j = int(j)
#       
#       
#       plt.plot(xD, edcj[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#       TemperatureC = KtoC(Temp_K)
#       plt.legend(loc='best')
#       plt.grid()
#       plt.xlabel(r'Distance [$m$]')
#       plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#       plt.title('EDC With Jacobian')
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC.pdf")
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC.svg")
#
#
#
#
#
#for jj,j in enumerate(graph_nodes):
#       fig = plt.figure()
#       
#       
#       index_k = int(j)
#       plt.plot(timeD, convedcj[j, :], color=viridis.colors[index_k,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#       TemperatureC = KtoC(Temp_K)
#       plt.legend(loc='best')
#       plt.axhline(y=100, color='k', linestyle='--')
#       plt.grid()
#       plt.xlabel('Times [s]')
#       plt.ylabel('Conversion')
#       plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#       plt.title('EDC Jacobian Conversion')
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC-conv.pdf")
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EDC-conv.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, ecj[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('Ethylchloride With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EC.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\EC.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#       index_j = int(j)
#       
#       
#       plt.plot(xD, hclj[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#       TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('Hydrogen Chloride With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\HCl.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\HCl.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, ccj[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('Coke With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Coke.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Coke.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, cp1j[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('1-/2-chloroprene With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\CP.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\CP.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, dij[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('Di With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Di.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Di.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, c4h6cl2j[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title(r'$C_{4}H_{6}Cl_{2}$ With Jacobian', fontdict=font2)
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C4H6Cl2.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C4H6Cl2.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, c6h6j[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title(r'$C_{6}H_{6}$ With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C6H6.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C6H6.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, c2h2j[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title(r'$C_{2}H_{2}$ With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C2H2.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C2H2.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, c11j[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('1,1-dichloroethane With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C11.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C11.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#    index_j = int(j)
#    
#    
#    plt.plot(xD, c112j[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#    TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('1,1,2-trichloroethane With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C112.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\C112.svg")
#
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#       index_j = int(j)
#       
#       
#       plt.plot(xD, vcmj[j, :], color=viridis.colors[index_j,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#       TemperatureC = KtoC(Temp_K)
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#plt.title('VCM With Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\VCM.pdf")
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\VCM.svg")
#
#
#
#for jj,j in enumerate(graph_nodes):
#       fig = plt.figure()
#       
#       
#       index_t = int(j)
#       plt.plot(xD, t0[j, :], color=viridis.colors[index_t,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#       plt.legend(loc='best')
#       plt.axhline(y=t0[j, 0], color='k', linestyle='--')
#       plt.grid()
#       plt.xlabel(r'Distance [$m$]')
#       plt.ylabel('Temperature [K]')
#       plt.title('Temperature Profile')
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\T-Profile{}.pdf".format(Temp_vals[j]))
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\T-Profile{}.svg".format(Temp_vals[j]))
#
#for jj,j in enumerate(graph_nodes):
#       fig = plt.figure()
#       
#       
#       index_t = int(j)
#       plt.plot(xD, t0j[j, :], color=viridis.colors[index_t,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
#       plt.legend(loc='best')
#       plt.axhline(y=t0j[j, 0], color='k', linestyle='--')
#       plt.grid()
#       plt.xlabel(r'Distance [$m$]')
#       plt.ylabel('Temperature [K]')
#       plt.title('Temperature Profile')
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\T-Profile-J{}.pdf".format(Temp_vals[j]))
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\T-Profile-J{}.svg".format(Temp_vals[j]))
#
##for jj,j in enumerate(graph_nodes):
##       fig = plt.figure()
##       
##       
##       index_t = int(j)
##       plt.plot(xD, dt0[j, :], color=viridis.colors[index_t,:], label="Temperature Value: {} [K]".format(Temp_vals[j]))
##       plt.legend(loc='best')
##       plt.axhline(y=Twalls[jj], color='k', linestyle='--')
##       plt.grid()
##       plt.xlabel(r'Distance [$m$]')
##       plt.ylabel('Temperature [K]')
##       plt.title('Temperature Profile')
##       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\T-Profile Diffusion{}.pdf".format(Temp_vals[j]))
##       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\T-Profile Diffusion{}.svg".format(Temp_vals[j]))
#
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    
#    plt.plot(xD, r1j[j, :], 'b--', label=r'$R_{1}$')
#    plt.plot(xD, r2j[j, :], 'g--', label=r'$R_{2}$')
#    plt.plot(xD, r3j[j, :], 'r--', label=r'$R_{3}$')
#    plt.plot(xD, r4j[j, :], 'c--', label=r'$R_{4}$')
#    plt.plot(xD, r5j[j, :], 'y--', label=r'$R_{5}$')
#    plt.plot(xD, r6j[j, :], 'm--', label=r'$R_{6}$')
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#    plt.legend(loc='best')
#    plt.title(r"Radicals Concentration at {} K".format(Temp_vals[j]))
#    plt.grid()
#     
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\RadicalT{}.pdf".format(Temp_vals[j]), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\RadicalT{}.svg".format(Temp_vals[j]), bbox_inches='tight')
#
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#
#    index_j = int(j)
#    TemperatureC = KtoC(Temp_K)       
#    plt.plot(xD, ecj[j, :], 'g-', label='Ethylchloride')
#    plt.plot(xD, ccj[j, :], 'r-', label='Soot/Coke')
#    plt.plot(xD, cp1j[j, :], 'b-', label='1-/2-chloroprene')
#    plt.plot(xD, dij[j, :], 'y-', label='1,1-dichloroethylene')
#    plt.plot(xD, c4h6cl2j[j, :], 'o-', label=r'$C_{4}H_{6}Cl_{2}$') #C4H6Cl2
#    plt.plot(xD, c6h6j[j, :], 'r-', label=r'$C_{6}H_{6}$')
#    plt.plot(xD, c2h2j[j, :], 'k-', label=r'$C_{2}H_{2}$')
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#    plt.legend(loc='best')
#    plt.title(r"By-Product Concentration at {} K".format(Temp_vals[j]))
#    plt.grid()
#     
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\By-ProductT{}.pdf".format(Temp_vals[j]), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\By-ProductT{}.svg".format(Temp_vals[j]), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    TemperatureC = KtoC(Temp_K)
#    index_j = int(j)
#    plt.plot(xD, edc[j, :], 'b-', label='EDC')
#    plt.plot(xD, hcl[j, :], 'r-', label='HCl')
#    plt.plot(xD, vcm[j, :], 'g-', label='VCM')
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#    plt.legend(loc='best')
#    plt.title("Temperature {} K".format(Temp_vals[j]))
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ProductsNoJT{}.pdf".format(Temp_vals[j]), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ProductsNoJT{}.svg".format(Temp_vals[j]), bbox_inches='tight')
#    
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    TemperatureC = KtoC(Temp_vals[j])
#    index_j = int(j)
#    plt.plot(xD, edcj[j, :], 'b-', label='EDC')
#    plt.plot(xD, hclj[j, :], 'r-', label='HCl')
#    plt.plot(xD, vcmj[j, :], 'g-', label='VCM')
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
#    plt.legend(loc='best')
#    plt.title("Temperature {} K".format(Temp_vals[j]))
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ProductsJT{}.pdf".format(Temp_vals[j]), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ProductsJT{}.svg".format(Temp_vals[j]), bbox_inches='tight') 
#    
#   
#
#for jj,j in enumerate(graph_nodes):
#       fig = plt.figure()
#       
#       
#       index_t = int(j)
#       plt.plot(xD, purejl[j, :], color=viridis.colors[index_t,:], label="Temperature {} K".format(Temp_vals[j]))
#       plt.legend(loc='best')
#       plt.grid()
#       plt.xlabel(r'Distance [$m$]')
#       plt.ylabel('Product Purity')
#       plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#       plt.title('Product Purity - Jacobian')
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Temp_vals-PurityJ{}.pdf".format(Temp_vals[j]))
#       fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Temp_vals-PurityJ{}.svg".format(Temp_vals[j]))
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#       index_t = int(j)
#       plt.plot(xD, purejl[j, :], color=viridis.colors[index_t,:], label="Temperature {} K".format(Temp_vals[j]))
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel('Product Purity')
#plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#plt.title('Product Purity - Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\OverallTemp_vals-PurityJ{}.pdf".format(Temp_vals[j]))
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\OverallTemp_vals-PurityJ{}.svg".format(Temp_vals[j]))
#
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    
#    
#    index_t = int(j)
#    plt.plot(xD, purel[j, :], color=viridis.colors[index_t,:], label="Temperature {} K".format(Temp_vals[j]))
#    plt.legend(loc='best')
#    plt.grid()
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel('Product Purity')
#    plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#    plt.title('Product Purity')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Temp_vals-Purity{}.pdf".format(Temp_vals[j]))
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Temp_vals-Purity{}.svg".format(Temp_vals[j]))
#
#fig = plt.figure()
#for jj,j in enumerate(graph_nodes2):
#       index_t = int(j)
#       plt.plot(xD, purel[j, :], color=viridis.colors[index_t,:], label="Temperature {} K".format(Temp_vals[j]))
#plt.legend(loc='best')
#plt.grid()
#plt.xlabel(r'Distance [$m$]')
#plt.ylabel('Product Purity')
#plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#plt.title('Product Purity - Jacobian')
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\OverallTemp_vals-Purity{}.pdf".format(Temp_vals[j]))
#fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\OverallTemp_vals-Purity{}.svg".format(Temp_vals[j]))
#
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#
#    index_j = int(j)
#    plt.plot(xD, convedc[j, :], 'b-', label="Temperature {} K".format(Temp_vals[j]))
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel('Conversion')
#    plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#    plt.legend(loc='best')
#    plt.title("Conversion of EDC")
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ConversionT{}.pdf".format(Temp_vals[j]), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ConversionT{}.svg".format(Temp_vals[j]), bbox_inches='tight')
#    
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt.plot(xD, convedcj[j, :], 'b-', label="Temperature {} K".format(Temp_vals[j]))
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel('Conversion')
#    plt.gca().set_yticklabels(['{:d}%'.format(int(x)) for x in plt.gca().get_yticks()]) 
#    plt.legend(loc='best')
#    plt.title("Conversion of EDC")
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ConversionJT{}.pdf".format(eavalg), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\ConversionJT{}.svg".format(eavalg), bbox_inches='tight')
#    
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt.plot(xD[0:-1], kvalsj[j, :], 'b-', label="Temperature {} K".format(Temp_vals[j]))
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel(r'Thermal Conductivity [$\frac{W}{m * k}$]')
#    plt.legend(loc='best')
#    plt.title('Mixture Thermal Conductivity ($\kappa_{mix}$)')
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\KvalsJT{}.pdf".format(eavalg), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\KvalsJT{}.svg".format(eavalg), bbox_inches='tight')
#    
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt.plot(xD[0:-1], uvalsj[j, :], 'b-', label="Temperature {} K".format(Temp_vals[j]))
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel(r'U [$\frac{W}{m^2 \cdot K}$]')
#    plt.legend(loc='best')
#    plt.title('Overall Heat Transfer Coefficient ($\it{U}$)')
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\UvalsJT{}.pdf".format(eavalg), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\UvalsJT{}.svg".format(eavalg), bbox_inches='tight')
#    
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt.plot(xD[0:-1], hvalsj[j, :], 'b-', label="Temperature {} K".format(Temp_vals[j]))
#    plt.xlabel(r'Distance [$m$]')
#    plt.ylabel(r'h [$\frac{W}{m^2 \cdot K}$]')
#    plt.legend(loc='best')
#    plt.title('Heat Transfer Coefficient ($\it{h}$)')
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\HvalsJT{}.pdf".format(eavalg), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\HvalsJT{}.svg".format(eavalg), bbox_inches='tight')
#    
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt1 = plt.plot(xD[0:-1], c1valsj[j][:], 'b-')
#    Cval = KtoC(Temp_vals[j])
#    plt1 = plt.plot(xD[0:-1], c1valsj[j][:], 'b-')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Constant 1 [$\frac{}{}$]'.format(str('{1}'),str('{m}')), fontdict=font)
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$\frac{\it{u_{z}}*\rho*C_{p}}{\kappa}$"],loc='best',fontsize='large')
#    plt.title('Constant 1', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Constant1JT{}.pdf".format(eavalg))
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Constant1JT{}.svg".format(eavalg))
##    plt.draw()
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt2 = plt.plot(xD[0:-1], c2valsj[j][:], 'b-', label='Constant 2')
#    Cval = KtoC(Temp_vals[j])
#    plt2 = plt.plot(xD[0:-1], c2valsj[j][:], 'b-', label='Constant 2')
#    plt.xlabel(r'Distance [$\frac{1}{m^2}$]', fontdict=font)
#    plt.ylabel(r'Constant 2 [$\frac{K}{m^2}$]', fontdict=font)
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$\frac{\it{U}*\alpha}{\kappa}$"],loc='best',fontsize='large')
#    plt.title('Constant 2', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Constant2JT{}.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Constant2JT{}.svg".format(float(eavalg)), bbox_inches='tight')
#    
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt3 = plt.plot(xD[0:-1], c3valsj[j][:], 'b-', label='Constant 3')
#    Cval = KtoC(Temp_vals[j])
#    plt3 = plt.plot(xD[0:-1], c3valsj[j][:], 'b-', label='Constant 3')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Constant 3 [$\frac{s*m*K}{mol}$]', fontdict=font)
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$\frac{\Delta{H_{rxn}}}{\kappa}$"],loc='best',fontsize='large')
#    plt.title('Constant 3', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Constant3JT{}.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Constant3JT{}.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt3 = plt.plot(xD[0:-1], revals[j][:], 'b-', label='Reynolds')
#    Cval = KtoC(Temp_vals[j])
#    plt3 = plt.plot(xD[0:-1], revals[j][:], 'b-', label='Reynolds')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Reynolds Number $R_{e}$', fontdict=font)
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$\frac{\it{\rho}*\it{u_{z}}*L_{s}}{\it{\mu}}$"],loc='best',fontsize='large')
#    plt.title('Reynolds Number', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Reynolds Number {}K.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Reynolds Number {}K.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt3 = plt.plot(xD[0:-1], revalsj[j][:], 'b-', label='Reynolds')
#    Cval = KtoC(Temp_vals[j])
#    plt3 = plt.plot(xD[0:-1], revalsj[j][:], 'b-', label='Reynolds')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Reynolds Number [$R_{e}$]', fontdict=font)
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$\frac{\it{\rho}*\it{u_{z}}*L_{s}}{\it{\mu}}$"],loc='best',fontsize='large')
#    plt.title('Reynolds Number Jacobian', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Reynolds Number J {}K.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Reynolds Number J {}K.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt4 = plt.plot(xD, selectvcm[j][:], 'b-', label='Selectivity')
#    Cval = KtoC(Temp_vals[j])
#    plt4 = plt.plot(xD, selectvcm[j][:], 'b-', label='Selectivity')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Selectivity [$S_{VCM}$]', fontdict=font)
#    plt.axhline(y=float(1.0), color='k', linestyle='--')
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$S_{\frac{VCM}{HCl}}$"],loc='best',fontsize='large')
#    plt.title('Selectivity of Vinyl Chloride Monomer', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityVCM-T-{}.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityVCM-T-{}.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt5 = plt.plot(xD, selecthcl[j][:], 'b-', label='Selectivity')
#    Cval = KtoC(Temp_vals[j])
#    plt5 = plt.plot(xD, selecthcl[j][:], 'b-', label='Selectivity')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Selectivity [$S_{HCl}$]', fontdict=font)
#    plt.axhline(y=float(1.0), color='k', linestyle='--')
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$S_{\frac{HCl}{VCM}}$"],loc='best',fontsize='large')
#    plt.title('Selectivity of Hydrogen Chloride', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityHCl-T-{}.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityHCl-T-{}.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt4 = plt.plot(xD, selectvcmj[j][:], 'b-', label='Selectivity')
#    Cval = KtoC(Temp_vals[j])
#    plt4 = plt.plot(xD, selectvcmj[j][:], 'b-', label='Selectivity')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Selectivity [$S_{VCM}$]', fontdict=font)
#    plt.axhline(y=float(1.0), color='k', linestyle='--')
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$S_{\frac{VCM}{HCl}}$"],loc='best',fontsize='large')
#    plt.title('Selectivity of Vinyl Chloride Monomer Jacobian', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityVCMJ-T-{}.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityVCMJ-T-{}.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt5 = plt.plot(xD, selecthclj[j][:], 'b-', label='Selectivity')
#    Cval = KtoC(Temp_vals[j])
#    plt5 = plt.plot(xD, selecthclj[j][:], 'b-', label='Selectivity')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Selectivity [$S_{HCl}$]', fontdict=font)
#    plt.axhline(y=float(1.0), color='k', linestyle='--')
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$S_{\frac{HCl}{VCM}}$"],loc='best',fontsize='large')
#    plt.title('Selectivity of Hydrogen Chloride Jacobian', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityHClJ-T-{}.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\SelectivityHClJ-T-{}.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt4 = plt.plot(xD, yieldvcm[j][:], 'b-', label='Yield')
#    Cval = KtoC(Temp_vals[j])
#    plt4 = plt.plot(xD, yieldvcm[j][:], 'b-', label='Yield')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Yield [$Y_{VCM}$]', fontdict=font)
#    plt.axhline(y=float(100.0), color='k', linestyle='--')
#    plt.gca().set_yticklabels(['{:.0f}%'.format(float(x)) for x in plt.gca().get_yticks()]) 
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$Y_{VCM}$"],loc='best',fontsize='large')
#    plt.title('Yield of Vinyl Chloride Monomer', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Yield VCM {}K.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Yield VCM {}K.svg".format(float(eavalg)), bbox_inches='tight')
#
#for jj,j in enumerate(graph_nodes):
#    fig = plt.figure()
#    index_j = int(j)
#    eavalg = str(Temp_vals[j])
#    plt4 = plt.plot(xD, yieldvcmj[j][:], 'b-', label='Yield')
#    Cval = KtoC(Temp_vals[j])
#    plt4 = plt.plot(xD, yieldvcmj[j][:], 'b-', label='Yield')
#    plt.xlabel(r'Distance [$m$]', fontdict=font)
#    plt.ylabel(r'Yield [$Y_{VCM}$]', fontdict=font)
#    plt.axhline(y=float(100.0), color='k', linestyle='--')
#    plt.gca().set_yticklabels(['{:.0f}%'.format(float(x)) for x in plt.gca().get_yticks()]) 
#    plt.legend([r"Temperature: {}$^\circ$C".format(Cval),r"$Y_{VCM}$"],loc='best',fontsize='large')
#    plt.title('Yield of Vinyl Chloride Monomer Jacobian', fontdict=font)
#    plt.grid()
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Yield VCM J {}K.pdf".format(float(eavalg)), bbox_inches='tight')
#    fig.savefig(r"C:\Users\tjcze\Desktop\Thesis\Python\Simple\Temperature-Values\Yield VCM J {}K.svg".format(float(eavalg)), bbox_inches='tight')
#
#
#print("The Jacobian was evaluated %d times." % J_eval[0])
end = time.time() #Time when it finishes, this is real time

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Completion Time: {} Hours {} Minutes {} Seconds".format(int(hours),int(minutes),int(seconds)))

timer(start,end) # Prints the amount of time passed since starting the program
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 04:10:35 2020

@author: tjcze
"""
import sympy as sp
from sympy import init_printing, var
from sympy.physics.vector import vlatex
import matplotlib.pyplot as plt
import IPython 
from IPython.display import display, Latex, Math, Image, DisplayObject
import pickle
import pprint as pp
pprint = pp.pprint
# sp.init_printing(use_latex=True)
ip = IPython.core.getipython.get_ipython()
# ip.display_formatter.formatters['text/latex'].enabled = True


Initreactions = [
{"Ea" : 1, "K_Value": 1, 'Reverse': 0,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 2, "K_Value": 2, 'Reverse': 0,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 3, "K_Value": 3, 'Reverse': 0,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  1, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 4, "K_Value": 4, 'Reverse': 0,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 1, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 5, "K_Value": 5, 'Reverse': 0,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}},
{"Ea" : 6, "K_Value": 6, 'Reverse': 0,
 "Reactants" : {'EDC':  1, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  1, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 1, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 7, "K_Value": 7, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}},
{"Ea" : 8, "K_Value": 8, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 1, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 9, "K_Value": 9, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  1, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 10, "K_Value": 10, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 1, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 11, "K_Value": 11, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 1, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  1, 'VCM':  0}},
{"Ea" : 12, "K_Value": 12, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 13, "K_Value": 13, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  1, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}},
{"Ea" : 14, "K_Value": 14, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  1, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  1, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}},
{"Ea" : 15, "K_Value": 15, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  1, 'R5':  0, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  1, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 16, "K_Value": 16, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  1}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  1,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 17, "K_Value": 17, 'Reverse': 1,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  1, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  1}},
{"Ea" : 18, "K_Value": 18, 'Reverse': 1,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  1,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 19, "K_Value": 19, 'Reverse': 1,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  1, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 1, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 20, "K_Value": 20, 'Reverse': 0,
 "Reactants" : {'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  0, 'C2H2':  2,'C11' : 0, 'C112' : 0, 'R1': 0, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  1, 'R6':  0, 'VCM':  0}, 
 "Products" :{'EDC':  0, 'EC':  0, 'HCl':  0, 'Coke': 0,'CP':  0,'Di': 0, 'C4H6Cl2':  0, 'C6H6':  1, 'C2H2':  0,'C11' : 0, 'C112' : 0, 'R1': 1, 'R2':  0, 'R3':  0, 'R4':  0, 'R5':  0, 'R6':  0, 'VCM':  0}},
{"Ea" : 21, "K_Value": 21, 'Reverse': 0,
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
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" : 1/4, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C2H2',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :1, "Reaction 19" :0, "Reaction 20" : -1, "Reaction 21" :-1/2 }, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :-1, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C11',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 1,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" : -1, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'C112',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 1,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" : 0, "Reaction 11" :-1, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R1',
 "Reactions" : {"Reaction 1" : 1 ,"Reaction 2" : -1,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" : -1, "Reaction 8" : -1, "Reaction 9" : -1, "Reaction 10" : -1, "Reaction 11" : -1, "Reaction 12" : -1, "Reaction 13" : -1, "Reaction 14" : 0, "Reaction 15" : 1, "Reaction 16" : 1, "Reaction 17" :1, "Reaction 18" :1, "Reaction 19" : 1, "Reaction 20" : 1/4, "Reaction 21" : -1}, 
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
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : -1,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" : 1, "Reaction 14" : 1, "Reaction 15" :0, "Reaction 16" :-1, "Reaction 17" :0, "Reaction 18" :-1, "Reaction 19" :0, "Reaction 20" :-1/4, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :1, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'R6',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : -1, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" : 1, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" :0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :-1, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :0, "Reaction 18" :0, "Reaction 19" :1, "Reaction 20" :0, "Reaction 21" :0}},
{"Name" : 'VCM',
 "Reactions" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 1,"Reaction 6" : 0, "Reaction 7" :1, "Reaction 8" :0, "Reaction 9" :0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" : -1, "Reaction 13" : -1, "Reaction 14" : -1, "Reaction 15" : -1, "Reaction 16" : -1, "Reaction 17" : 1, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}, 
 "Reverse" : {"Reaction 1" : 0 ,"Reaction 2" : 0,"Reaction 3" : 0,"Reaction 4" : 0,"Reaction 5" : 0,"Reaction 6" : 0, "Reaction 7" :0, "Reaction 8" :0, "Reaction 9" : 0, "Reaction 10" :0, "Reaction 11" :0, "Reaction 12" :0, "Reaction 13" :0, "Reaction 14" : 0, "Reaction 15" :0, "Reaction 16" :0, "Reaction 17" :-1, "Reaction 18" :0, "Reaction 19" :0, "Reaction 20" :0, "Reaction 21" :0}}, 
]

class symbolgen:
       
       def __init__(self, nlist,  Initlist, EQlist):
              self.nameslist = nlist
              self.rxnnum = len(self.nameslist)
              self.initlist = Initlist
              self.Eqlist = EQlist
       
       def initl(self):
               return self.initlist
        
       def latexin(self):
              latexs = self.eqlist(Eqlist, self.reactants, self.products)[1]
              return latexs
       
       def symsinit(self):
               return self.symfunc(self.nameslist,self.rxnnum)[0]
       
       def rinit(self):
               return self.initfunc(self.initreactions, self.C)[0]
       
       def pinit(self):
               return self.initfunc(self.initreactions, self.C)[1]
       
       initreactions = property(initl)
       C = property(symsinit, symsinit)
       reactants = property(rinit)
       products = property(pinit)
       latexs = property(latexin)
        
       def symfunc(names, rxnum):
              Csyms = [sp.symbols(r'C_{}'.format('{}'.format(i))) for i in names]
              Ksyms = [sp.symbols(r'K_{}'.format(j)) for j in range(rxnum)]
              EAsyms = [sp.symbols(r'Ea_{}'.format(k)) for k in range(rxnum)]
              Tsyms = [sp.symbols('T0'), sp.symbols('T1')]
              
              return Csyms, Ksyms, EAsyms, Tsyms
       
       def numfunc(Cs):
              cl = len(Cs)
              fcs = []
              for i in range(cl):
                     As = []
                     Ns = []
                     NNs = []
                     val3 = Cs[i]
                     se = list(val3)
                     count = 0
                     sb = list(val3)
                     SG = []
                     fnum = len(se) - 1
                     fend = len(se) - 1
                     for sv in range(len(sb)):
                            fend = len(se)
                            vv = sb[sv]
                            N = vv.isnumeric()
                            A = vv.isalpha()
                            ff = fend - sv
                            if A == True and count == 0:
                                   As.append(vv)
                                   SG.append(vv)
                                   count = 0
                                   fnum -= 1
                            elif A == True and count > 0:
                                   
                                   NNa = "".join(Ns)
                                   SG.append(NNa)
                                   SG.append(vv)
                                   Ns.clear()
                                   count = 0
                                   fnum -= 1
                            elif A == True and count >= 2:
                                  
                                   NNa = "".join(Ns)
                                   NNs.append(NNa)
                                   Ns.clear()
                                   SG.append(NNa)
                                   SG.append(vv)
                                   count = 0
                                   fnum -= 1
                                   
                            elif N == True and ff > 1:
                                   Ns.append(vv)
                                   count += 1
                            elif N == True and ff <= 1:
                                   Ns.append(vv)
                                   
                                   if len(Ns) >= 2:
                                          NNa = "".join(Ns)
                                          NNs.append(NNa)
                                          SG.append(NNa)
                                                 
                                   else:
                                          SG.append(vv)
                                   
                     count = 0
                     Ns.clear()
                     As.clear()
                     val2 = str(Cs[i])
                     s = list(val2)
                     for j in range(len(SG)):
                            charv = SG[j]
                            try:
                                    charvi = int(SG[j])
                                    SG[j] = charv.replace('{}'.format(charvi), ('_{' + '{}'.format(charvi) + '}')) 
                            except:
                                    pass                                                 
                     ss = "".join(SG)
                     s.clear()
                     fcs.append(ss)
              return fcs
                            
                     
       
       def rterm(Ci, a):
              termi = sp.Mul(a,sp.Pow(Ci, abs(int(a))))
              return termi
       
       def rprod(Ci, a, Cj, b):
              term1 = symbolgen.rterm(Ci, a)
              term2 = symbolgen.rterm(Cj, b)
              term3 = sp.Mul(term1,term2)
              return term3
       
       def initfunc(initlist, C):
              reactants = []
              products = []
              for i, j in enumerate(initlist):
                     Reactants = initlist[i]['Reactants']
                     Products = initlist[i]['Products']
                     Rvals = list(Reactants.values())
                     Pvals = list(Products.values())
                     Ks = sp.symbols('K_{}'.format(i+1))
                     Eas = sp.symbols('Ea_{}'.format(i+1))
                     ee = sp.exp(sp.Mul(-Eas,sp.Pow(sp.Mul(sp.Symbol('R'),sp.Symbol('T')), sp.Integer(-1))))
                     rterms = []
                     pterms = []
                     rtotal = sp.Integer(1)
                     ptotal = sp.Integer(1)
                     for k, l in zip(C,Rvals):
                            if l != 0:
                                   term = symbolgen.rterm(k, l)
                                   rterms.append(term)
                     for t in rterms:
                            rtotal = sp.Mul(rtotal,t)
                     for m, n in zip(C,Pvals):
                            if n != 0:
                                   pterm = symbolgen.rterm(m, n)
                                   pterms.append(pterm)
                     for tt in pterms:
                            ptotal = sp.Mul(ptotal,tt)
                     reactants.append(sp.Mul(Ks,sp.Mul(rtotal,ee)))
                     products.append(sp.Mul(Ks,sp.Mul(ptotal,ee)))
              return [reactants, products]
       
       def eqlist(eqlistl, R, P):
              reactants = R
              products = P
              EQS = []
              leqns = []
              for i ,j in enumerate(eqlistl):
                     Reactions = eqlistl[i]['Reactions']
                     Reverse = eqlistl[i]['Reverse']
                     Rxn = list(Reactions.values())
                     RxnR = list(Reverse.values())
                     eqn = []
                     Reacts = [i*j for i, j in zip(Rxn, reactants) if i != 0]
                     Prods = [i*j for i, j in zip(RxnR, products) if i != 0]
                     if not Prods:
                            eee= sum(Reacts)
                            rlatex = sp.latex(eee)
                            leqns.append(rlatex)
                            EQS.append(eee)
                     else:
                            eqn = sum(Reacts)
                            peqn = sum(Prods)
                            eeqn = sp.Add(eqn,peqn)
                            rlatex = sp.latex(eeqn)
                            leqns.append(rlatex)
                            EQS.append(eeqn)
              return [EQS, leqns]
       
       def dislat(lnames, latexs, indvar):
              Latexs = []
              Displays = []
              for i in range(len(latexs)):
                     dd = '{d'+ '{}'.format(sp.symbols(lnames[i]))+ '}'
                     dt = '{d'+ '{}'.format(sp.symbols(indvar))+ '}'
                     dde = r'$\dfrac{}{}'.format(dd,dt) + ' = ' + '{}$'.format(latexs[i])
                     ddg = Latex(dde)
                     Latexs.append(dde)
                     Displays.append(ddg)
              return Displays, Latexs
       
       def chemeq(Cs, rxn, inits):
              ceqs = []
              ceqsD = []
              ceqsw = []
              for i in range(rxn):
                     Reactants = inits[i]['Reactants']
                     Products = inits[i]['Products']
                     Reverse = inits[i]['Reverse']
                     Rvals = list(Reactants.values())
                     rvals = [Rvals[kk] for kk in range(len(Rvals)) if Rvals[kk] != 0]
                     Rname = symbolgen.numfunc(list(Reactants.keys()))
                     rname = [sp.symbols('{}'.format(Rname[h])) for h in range(len(Rname)) if Rvals[h] !=0]
                     Pvals = list(Products.values())
                     pvals = [Pvals[kk] for kk in range(len(Pvals)) if Pvals[kk] != 0]
                     Pname = symbolgen.numfunc(list(Products.keys()))
                     pname = [sp.symbols('{}'.format(Pname[h])) for h in range(len(Pname)) if Pvals[h] !=0]
                     CRvals = sum([sp.Mul(sp.Integer(ii),jj) for ii, jj in zip(rvals,rname) if ii != 0])
                     CPvals = sum([sp.Mul(sp.Integer(ii),jj) for ii, jj in zip(pvals,pname) if ii != 0])
                     if Reverse == 0:
                            cheme = r'${} \longrightarrow {}$'.format(CRvals, CPvals) 
                     if Reverse == 1:
                            cheme = r'${} \rightleftharpoons {}$'.format(CRvals, CPvals) 
                     ceqsD.append(Latex(cheme))
                     ceqs.append(cheme)
                     if Reverse == 0:
                            chemw = r'{} \\longrightarrow {}'.format(CRvals, CPvals) 
                     if Reverse == 1:
                            chemw = r'{} \\rightleftharpoons {}'.format(CRvals, CPvals)
                     ceqsw.append(chemw)
              return ceqs, ceqsD, ceqsw
       
       def rhseqs(equations, kk, ea, r):
              EQLIST = []
              EQLISTF = []
              EQLISTT = []
              Tsyms = [sp.symbols('T0'), sp.symbols('T1')]
              Twalls = sp.symbols('Twall')
              Consyms = [sp.symbols('Constant_{}'.format(i)) for i in range(1,4,1)]
              for ind, e in enumerate(equations):
                     eqn = [r'{}'.format(e).replace('{','').replace('}','')] 
                     Ksyms = [sp.symbols('K_{}'.format(i+1)) for i in range(len(kk))]
                     EAsyms = [sp.symbols('-Ea_{}'.format(i+1)) for i in range(len(ea))]
                     kdictionary = dict(zip(Ksyms, kk))
                     eadictionary = dict(zip(EAsyms, ea))
                     eqn3 = e.subs(kdictionary)
                     eqn4 = eqn3.subs(eadictionary)
                     eqn5 = eqn4.subs({'R' : 8.31446261815324})
                     eqn6 = str(r'{}'.format(eqn5.subs({'*exp' : '*sp.exp'})))
                     eqn6b = eqn5.subs({'*exp' : '*sp.exp'})
                     EQLISTF.append(eqn6b)
                     EQLIST.append(eqn[0])
                     EQLISTT.append(eqn[0])
              EQLISTT.append(sp.symbols('T1'))
              Teq = Consyms[0]*Tsyms[1] - (Consyms[0]*(Twalls - Tsyms[0]) + Consyms[2]*sp.sympify(EQLIST[0]))
              EQLISTT.append(Teq)
              
                     
              return EQLIST, EQLISTF, EQLISTT
       
       def jacobian(rhs, y):
              funcs = [i for i in rhs]
              eqnl = len(funcs)
              cl = len(y)
              J  = [[i for i in range(cl)] for j in range(eqnl)]
              Jn  = [[i for i in range(cl)] for j in range(eqnl)]
              mfunc = lambda i, j: sp.diff(funcs[i], y[j])
              Jjj = sp.Matrix(eqnl, cl, lambda i, j: mfunc(i, j))
              Jj = sp.matrix2numpy(Jjj)
              for i in range(eqnl):
                      for j in range(cl):
                             J[i][j] = str('{}'.format('{}'.format(mfunc(i, j)).replace('*exp',  '*sp.exp')))
              for i in range(eqnl):
                      for j in range(cl):
                             Jn[i][j] = str('{}'.format('{}'.format(mfunc(i, j)).replace('*exp',  '*np.exp')))
              MatrixJ = sp.simplify(sp.Matrix(Jj))
              LatexMatrix = sp.latex(sp.matrix2numpy(sp.Matrix(Jj)))
              lm = sp.latex(MatrixJ, mode='inline', itex=True, mat_delim="(", mat_str='array')
              return J, Jn,  MatrixJ, lm, LatexMatrix
       
       def sysgen(self):
              equations, latexs = self.eqlist(self.Eqlist, self.reactants, self.products)
              return equations
       
       def sysdis(self):
              equations, latexs = self.eqlist(self.Eqlist, self.reactants, self.products)
              slatex, dlatex = self.dislat(self.nameslist, self.latexs, self.indvar)
              return dlatex
       
       def dis(self):
              slatex, dlatex = self.dislat(self.nameslist, self.latexs, self.indvar)
              for i in slatex:
                     display(i)
                     
       def gen(names, rxn, inits, eqs, intz):
              Cs, Ks, EAs, Ts = symbolgen.symfunc(names, rxn)
              reacts, prods = symbolgen.initfunc(inits, Cs)
              equats, latexss = symbolgen.eqlist(eqs, reacts, prods)
              slat, dlat = symbolgen.dislat(names, latexss, intz)
              Chem, ChemD, ChemW = symbolgen.chemeq(Cs, rxn, inits)
              return Cs, reacts, prods, equats, slat, dlat, Chem, ChemD, ChemW
       
       def fullgen(names, rxn, inits, eqs, intz, filepathf, kk, ea, r):
              Cs, Ks, EAs, Ts = symbolgen.symfunc(names, rxn)
              CsT = Cs.copy()
              CsT.append(sp.symbols('T0'))
              CsT.append(sp.symbols('T1'))
              reacts, prods = symbolgen.initfunc(inits, Cs)
              equats, latexss = symbolgen.eqlist(eqs, reacts, prods)
              slat, dlat = symbolgen.dislat(names, latexss, intz)
              Chem, ChemD, ChemW = symbolgen.chemeq(Cs, rxn, inits)
              RHS, RHSf, RHST = symbolgen.rhseqs(equats, kk, ea, r)
              Jac, JacN, Jacm, lm, latexmatrix = symbolgen.jacobian(RHS, Cs)
              JacT, JacNT, JacmT, lmT, latexmatrixT = symbolgen.jacobian(RHST, CsT)
              fp = symbolgen.csave(Chem, filepathf)
              fc = symbolgen.psave(names, dlat, filepathf)
              ff = symbolgen.fsave(filepathf, equats, dlat, Chem, ChemW, RHS, RHSf, RHST, Jac, JacN, Jacm, lm, latexmatrix, JacT, JacNT)
              return Cs, reacts, prods, equats, slat, dlat, Chem, ChemD, ChemW, RHS, RHSf, Jac, JacN, Jacm, lm, latexmatrix, JacT, JacNT, JacmT, lmT, latexmatrixT
       
       
       def psave(nameslist, LATEXD, fpath):
              filename = fpath
              for s,k in enumerate(LATEXD):
                     fig = plt.figure()
                     ax = fig.add_axes([0,0,1,1])
                     left, width = .25, .5
                     bottom, height = .25, .5
                     right = left + width
                     top = bottom + height
                     ax.set_axis_off()
                     text = ax.text(0.5*(left+right), 0.5*(bottom+top),k , va= 'center',ha= 'center', bbox= dict(boxstyle="round", fc="white", alpha= 0.3, ec="black", pad=0.2))
                     fig.savefig(r'{}\{}.svg'.format(filename,nameslist[s]), bbox_inches='tight')
                     fig.savefig(r'{}\{}.pdf'.format(filename,nameslist[s]), bbox_inches='tight')
                     
       def csave(LATEXC, fpath):
              filename = fpath
              for s,k in enumerate(LATEXC):
                     fig = plt.figure()
                     ax = fig.add_axes([0,0,1,1])
                     left, width = .25, .5
                     bottom, height = .25, .5
                     right = left + width
                     top = bottom + height
                     ax.set_axis_off()
                     text = ax.text(0.5*(left+right), 0.5*(bottom+top),k , va= 'center',ha= 'center', bbox= dict(boxstyle="round", fc="white", alpha= 0.3, ec="black", pad=0.2))
                     fig.savefig(r'{}\Reaction {}.svg'.format(filename, s), bbox_inches='tight')
                     fig.savefig(r'{}\Reaction {}.pdf'.format(filename, s), bbox_inches='tight')
                     
       def fsave(ffpath, eqns, eqnslat, crxns, crxnsw, rhseq, rhseqf, RHST, Jac, JacN, JacM, lm, latexmatrix, JacT, JacNT):
              with open(r"{}\Equations.txt".format(ffpath), "w") as output:
                     output.write("[")
                     el = len(eqns)
                     eel = 0
                     for eqn in eqns:
                            eel += 1
                            if eel < el:
                                   output.write('{},\n'.format(str(eqn)))
                            if eel >= el:
                                   output.write('{}]'.format(str(eqn)))
                  
              with open(r"{}\EquationsLatex.txt".format(ffpath), "w") as output:
                     for eqnlat in eqnslat:
                            output.write('{}\n'.format(str(eqnlat)))
                  
              with open(r"{}\ReactionsLatex.txt".format(ffpath), "w") as output:
                     for crxn in crxns:
                            output.write('{}\n'.format(str(crxn)))
                            
              with open(r"{}\ReactionsLatexWord.txt".format(ffpath), "w") as output:
                     for crxnw in crxnsw:
                            output.write('{}\n'.format(str(crxnw)))
                            
              with open(r"{}\RHSsymbols.txt".format(ffpath), "w") as output:
                     output.write("[")
                     ll = len(rhseq)
                     lr = 0
                     for rhs in rhseq:
                            lr += 1
                            
                            if lr < ll:
                                   output.write('{},\n'.format(rhs))
                            elif lr >= ll:
                                   output.write('{}]'.format(rhs))
                     # output.write("]")
              
              with open(r"{}\RHST.txt".format(ffpath), "w") as output:
                     output.write("[")
                     ll = len(RHST)
                     lr = 0
                     for rhst in RHST:
                            lr += 1
                            
                            if lr < ll:
                                   output.write('{},\n'.format(rhst))
                            elif lr >= ll:
                                   output.write('{}]'.format(rhst))
              
              with open(r"{}\RHS.txt".format(ffpath), "w") as output:
                     output.write("[")
                     ll = len(rhseqf)
                     lr = 0
                     for rhsff in rhseqf:
                            lr += 1
                            if lr < ll:
                                   output.write('{},\n'.format(rhsff))
                            elif lr >= ll:
                                   output.write("{}".format(rhsff))
                     output.write("]")
                            
              with open(r"{}\Jacobun.txt".format(ffpath), "w") as output:
                     output.write("[")
                     jj = len(Jac)
                     jjj = 0
                     for i in range(len(Jac)):
                             jjj += 1
                             Jrow = Jac[i][:]
                             if jjj < jj:
                                    output.write('{},\n'.format(Jrow))
                             elif jjj >= jj:
                                    output.write('{}'.format(Jrow))
                     output.write("]")
              
              with open(r"{}\JacobunT.txt".format(ffpath), "w") as output:
                                   output.write("[")
                                   jj = len(JacT)
                                   jjj = 0
                                   for i in range(len(JacT)):
                                           jjj += 1
                                           Jrowt = JacT[i][:]
                                           if jjj < jj:
                                                  output.write('{},\n'.format(Jrowt))
                                           elif jjj >= jj:
                                                  output.write('{}'.format(Jrowt))
                                   output.write("]")
                      
              with open(r"{}\JacobNumpy.txt".format(ffpath), "w") as output:
                     output.write("[")
                     jjb = len(JacN)
                     jjn = 0
                     for i in range(len(JacN)):
                             jjn += 1
                             Jrown = JacN[i][:]
                             if jjn < jjb:
                                    output.write('{},\n'.format(Jrown))
                             elif jjn >= jjb:
                                    output.write('{}'.format(Jrown))
                     output.write("]")
              
              with open("{}\JacobianMatrix.txt".format(ffpath),'w') as output:
                            output.write('{}'.format(JacM))
                            
              with open(r"{}\JacobianLatex.txt".format(ffpath), "w") as output:
                            output.write('{}'.format(lm))
                            
              with open("{}\Jacobun.txt".format(ffpath)) as filein, open("{}\Jacobian.txt".format(ffpath),'w') as fileout:
                     for line in filein:
                         line=line.replace("'","")
                         fileout.write(line)
                     
              with open("{}\JacobunT.txt".format(ffpath)) as filein, open("{}\JacobianTemp.txt".format(ffpath),'w') as fileout:
                     for line in filein:
                         line=line.replace("'","")
                         fileout.write(line)
              
              with open("{}\JacobNumpy.txt".format(ffpath)) as filein, open("{}\JacobianNumpy.txt".format(ffpath),'w') as fileout:
                     for line in filein:
                         line=line.replace("'","")
                         fileout.write(line)
                     
                         
              with open("{}\RHS.txt".format(ffpath)) as filein, open("{}\RightHandSide.txt".format(ffpath),'w') as fileout:
                            fileinl = filein.readlines()       
                            lfia = len(fileinl)
                            lffb = 0
                            for line in fileinl:
                                lffb += 1
                                line=line.replace("'","")
                                line=line.replace("exp","sp.exp")
                                if lffb < lfia:
                                          fileout.write('{}'.format(line))
                                elif lffb >= lfia:
                                          fileout.write('{}'.format(line))
              
              with open("{}\RHST.txt".format(ffpath)) as filein, open("{}\RightHandSideTemp.txt".format(ffpath),'w') as fileout:
                            fileinl = filein.readlines()       
                            lfia = len(fileinl)
                            lffb = 0
                            for line in fileinl:
                                lffb += 1
                                line=line.replace("'","")
                                line=line.replace("exp","sp.exp")
                                if lffb < lfia:
                                          fileout.write('{}'.format(line))
                                elif lffb >= lfia:
                                          fileout.write('{}'.format(line))
              
              with open("{}\RHSsymbols.txt".format(ffpath)) as filein, open("{}\RightHandSideSymbols.txt".format(ffpath),'w') as fileout:
                            fileinl = filein.readlines()        
                            lfi = len(fileinl)
                            lff = 0
                            for line in fileinl:
                                line=line.replace("'","")
                                line=line.replace("exp","sp.exp")
                                lff += 1
                                if lff < lfi:
                                          fileout.write('{}'.format(line))
                                elif lff >= lfi:
                                       
                                          fileout.write('{}'.format(line))
                            
                                          
                     
              pickle.dumps(JacM)      
              with open('{}\JacobianMatrixPickle.txt'.format(ffpath),'wb') as f:
                      pickle.dump(JacM, f)
                      
       def lsave(klist, elist, Clist, ffpath):
              kkl = len(klist)
              Easyms = [sp.symbols('-Ea_{}'.format(ea + 1)) for ea in range(kkl)]
              Ksyms = [sp.symbols('K_{}'.format(kk + 1)) for kk in range(kkl)]
              vlist = ['Kv','Ev']
              dlist = ['K','Ea']
              Vlist = [Ksyms, Easyms]
              with open("{}\RHSLists.txt".format(ffpath),'w') as output:
                     for i in range(len(vlist)):
                            output.write(str('{} = kargs.get({}) \n'.format(dlist[i],'{}{}{}'.format("'",dlist[i],"'"))))
                            output.write('{} = list({}.values()) \n'.format(vlist[i],dlist[i]))
                            output.write(str('{} = {} \n'.format([*Vlist[i]],'{}'.format(vlist[i]))).replace('[','').replace(']',''))
                     output.write(str('{} = {} \n'.format([*C],'{}'.format('C'))).replace('[','').replace(']',''))
              
       def argsf(ks, eas):
              lll = len(ks)
              Easyms = [sp.symbols('Ea_{}'.format(ea + 1)) for ea in range(lll)]
              Ksyms = [sp.symbols('K_{}'.format(kk + 1)) for kk in range(lll)]
              Kdict = dict(zip(Ksyms, ks))
              Edict = dict(zip(Easyms,eas))
              args = {}
              args['K'] = Kdict
              args['Ea'] = Edict
              
              return args

              


names = ['EDC','EC','HCl','Coke', 'CP','Di','C4H6Cl2','C6H6','C2H2','C11','C112','R1','R2','R3','R4','R5','R6','VCM']
namesf = ['EDC','EC','HCl','Coke', 'CP','Di','Tri','C4H6Cl2','C6H6','C2H2','C11','C112','C1112','R1','R2','R3','R4','R5','R6','R7','R8','CCl4','CHCl3','VCM']
indfvar = 'z'
kk = [5.9E+15,1.3E13,1E+12,5E+11,1.2E+13,2E+11,1E+13,1E+13,1.7E+13,1.2E+13,1.7E+13,91000000000,1.2E+14,5E+11,20000000000,3E+11,2.1E+14,5E+14,2E+13,1E+14,1.6E+14]
eaf = [342.0,7.0,42.0,45.0,34.0,48.0,13.0,12.0,4.0,6.0,15.0,0.0,56.0,31.0,30.0,61.0,84.0,90.0,70.0,20.0,70.0]
ea = [i*1000.0 for i in eaf]
r = 8.314
ffpath = r'C:\Users\tjcze\Desktop\Thesis\Python\symbolgen'
C, reacts, prods, equations, slat, dlat, chem, chemD, chemw, rhs, rhsf, jac, jacn, jacM, lm, latexmatrix, JacT, JacNT, JacmT, lmT, latexmatrixT = symbolgen.fullgen(names, 21, Initreactions, Eqlist, indfvar, ffpath, kk, ea, r)
symbolgen.lsave(kk, ea, C, ffpath)
kargs = symbolgen.argsf(kk, ea)
symbolgen.fsave(ffpath, equations, slat, chem, chemw, rhs, rhsf, JacT, JacNT, JacmT, lmT, latexmatrixT)
print(kargs)


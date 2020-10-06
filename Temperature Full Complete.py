# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:26:23 2020

@author: tjcze
"""
from __future__ import division
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import sympy as sp
import mpmath as mp
from matplotlib import cm
import thermo as tc
from tqdm import tqdm
from scipy.integrate import solve_ivp
import pandas as pd
import os

clear = os.system('cls')
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
path_fol = r"{}\Temperature - Full".format(dir_path)
# print("\n")
# print("Current working directory:\n")
# print("{}\n".format(cwd))
try:
    os.mkdir(path_fol)
    # print("New Folder was created\n")
    # print("Current working directory - Created Folder Path:\n")
    # print("{}\n".format(path_fol))
except Exception:
    # print("Current working directory - Current Folder Path:\n")
    # print("{}\n".format(path_fol))
    pass
sp.init_session(use_latex=False, quiet=True)
plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 10})
# Data for inital kinetics
# https://doi.org/10.1021/ie8006903
# Thesis: https://ir.library.louisville.edu/etd/3359/

R_gas = 8.31446261815324

# Dictionary for all relevent forward and reverse reactions
Initreactionsf = [
    {"Ea": 1, "K_Value": 1,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 1, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 2, "K_Value": 2,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 1, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 1, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 3, "K_Value": 3,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 4, "K_Value": 4,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 1, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 1, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 5, "K_Value": 5,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 1, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 1, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 6, "K_Value": 6,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1}},
    {"Ea": 7, "K_Value": 7,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 1, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 1, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 8, "K_Value": 8,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 1, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 1, 'R1': 0, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 9, "K_Value": 9,
     "Reactants": {'EDC': 1, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 1, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 1, 'VCM': 0}},
    {"Ea": 10, "K_Value": 10,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 1, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1}},
    {"Ea": 11, "K_Value": 11,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 1, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 12, "K_Value": 12,
     "Reactants": {'EDC': 0, 'EC': 1, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 1, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 13, "K_Value": 13,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 1, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 1, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 14, "K_Value": 14,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 1, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 1, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 15, "K_Value": 15,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 1, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 1, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 16, "K_Value": 16,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 1, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 1, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 17, "K_Value": 17,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 1, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 18, "K_Value": 18,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 1, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 19, "K_Value": 19,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 1, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1},
     "Products": {'EDC': 0, 'EC': 1, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 20, "K_Value": 20,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 1, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 1, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 21, "K_Value": 21,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 1, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 22, "K_Value": 22,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 1, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 1}},
    {"Ea": 23, "K_Value": 23,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 1, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 24, "K_Value": 24,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 1, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 1, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 25, "K_Value": 25,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 1, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 1, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 26, "K_Value": 26,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 1, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 1, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 1, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 1, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 27, "K_Value": 27,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 1, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 1, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 1, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 28, "K_Value": 28,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 1, 'R7': 0, 'R8': 0, 'CCl4': 1, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 1, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 1, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 29, "K_Value": 29,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 1, 'R7': 0, 'R8': 1, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 1, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 1, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 30, "K_Value": 30,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 2, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 1, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 1, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 1, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}},
    {"Ea": 31, "K_Value": 31,
     "Reactants": {'EDC': 0, 'EC': 0, 'HCl': 0, 'Coke': 0, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 1, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 2, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0},
     "Products": {'EDC': 0, 'EC': 0, 'HCl': 2, 'Coke': 2, 'CP': 0, 'Di': 0, 'Tri': 0, 'C4H6Cl2': 0, 'C6H6': 0, 'C2H2': 0, 'C11': 0, 'C112': 0, 'C1112': 0, 'R1': 0, 'R2': 0, 'R3': 0, 'R4': 0, 'R5': 0, 'R6': 0, 'R7': 0, 'R8': 0, 'CCl4': 0, 'CHCl3': 0, 'VCM': 0}}]

Eqlistf = [
    {"Name": 'EDC',
     "Reactions": {"Reaction 1": -1, "Reaction 2": 0, "Reaction 3": -1, "Reaction 4": -1, "Reaction 5": -1, "Reaction 6": -1, "Reaction 7": -1, "Reaction 8": -1, "Reaction 9": -1, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'EC',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 1, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": -1, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 1, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'HCl',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 1, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 1, "Reaction 11": 1, "Reaction 12": 1, "Reaction 13": 1, "Reaction 14": 1, "Reaction 15": 1, "Reaction 16": 1, "Reaction 17": 0, "Reaction 18": 1, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 1},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'Coke',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 1},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'CP',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 1, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'Di',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 1, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 1, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 1, "Reaction 28": 0, "Reaction 29": 1, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": -1, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'Tri',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 1, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": -1, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'C4H6Cl2',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 1, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'C6H6',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 1, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'C2H2',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 1, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": -1, "Reaction 31": -1},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": -1, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'C11',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 1, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": -1, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'C112',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 1, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": -1, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 1, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'C1112',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 1, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": -1, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 1, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": -1, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R1',
     "Reactions": {"Reaction 1": 1, "Reaction 2": 1, "Reaction 3": -1, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": -1, "Reaction 11": -1, "Reaction 12": -1, "Reaction 13": -1, "Reaction 14": -1, "Reaction 15": -1, "Reaction 16": -1, "Reaction 17": -1, "Reaction 18": -1, "Reaction 19": 0, "Reaction 20": 1, "Reaction 21": 1, "Reaction 22": 1, "Reaction 23": 1, "Reaction 24": 1, "Reaction 25": 1, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 1, "Reaction 31": -1},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": -1, "Reaction 23": -1, "Reaction 24": -1, "Reaction 25": -1, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R2',
     "Reactions": {"Reaction 1": 1, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": -1, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": -1, "Reaction 11": 0, "Reaction 12": 1, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": -1, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R3',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 1, "Reaction 4": 1, "Reaction 5": 1, "Reaction 6": 1, "Reaction 7": 1, "Reaction 8": 1, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": -1, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": -1, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 1, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R4',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": -1, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 1, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 1, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": -1, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": -1, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R5',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": -1, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 1, "Reaction 19": 1, "Reaction 20": 0, "Reaction 21": -1, "Reaction 22": 0, "Reaction 23": -1, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": -1, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": -1, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 1, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R6',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": -1, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 1, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": -1, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": -1, "Reaction 29": -1, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 1, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 1, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R7',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": -1, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 1, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": -1, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 1, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'R8',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 1, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": -1, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 1, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 1, "Reaction 27": 1, "Reaction 28": 1, "Reaction 29": -1, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": -1, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'CCl4',
     "Reactions": {"Reaction 1": 0, "Reaction 2": -1, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": -1, "Reaction 27": -1, "Reaction 28": -1, "Reaction 29": 1, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 1, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'CHCl3',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 1, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": -1, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": 0, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}},
    {"Name": 'VCM',
     "Reactions": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 1, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 1, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": -1, "Reaction 18": -1, "Reaction 19": -1, "Reaction 20": -1, "Reaction 21": -1, "Reaction 22": 1, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0},
     "Reverse": {"Reaction 1": 0, "Reaction 2": 0, "Reaction 3": 0, "Reaction 4": 0, "Reaction 5": 0, "Reaction 6": 0, "Reaction 7": 0, "Reaction 8": 0, "Reaction 9": 0, "Reaction 10": 0, "Reaction 11": 0, "Reaction 12": 0, "Reaction 13": 0, "Reaction 14": 0, "Reaction 15": 0, "Reaction 16": 0, "Reaction 17": 0, "Reaction 18": 0, "Reaction 19": 0, "Reaction 20": 0, "Reaction 21": 0, "Reaction 22": -1, "Reaction 23": 0, "Reaction 24": 0, "Reaction 25": 0, "Reaction 26": 0, "Reaction 27": 0, "Reaction 28": 0, "Reaction 29": 0, "Reaction 30": 0, "Reaction 31": 0}}]

# List of names and acronyms

namesf = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', 'C4H6Cl2', 'C6H6', 'C2H2', 'C11', 'C112', 'C1112', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'CCl4', 'CHCl3', 'VCM', 'T', 'dT/dz']
namespdf = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', 'C4H6Cl2', 'C6H6', 'C2H2', 'C11', 'C112', 'C1112', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'CCl4', 'CHCl3', 'VCM', 'T0', 'T1', 'Pure', 'S_VCM', 'S_HCl', 'Yield']
namesj = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', 'C4H6Cl2', 'C6H6', 'C2H2', 'C11', 'C112', 'C1112', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'CCl4', 'CHCl3', 'VCM']

# These are mostly a group of strings used to name the chemicals
eqsnum = len(Initreactionsf)
reacteqs = []
prodeqs = []
reacteqs2 = []
prodeqs2 = []
namesb = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', 'C4H6Cl2', 'C6H6', 'C2H2', 'C11', 'C112', 'C1112', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'CCl4', 'CHCl3', 'VCM']
namesb2 = ['1, 2-dichloroethane', 'Ethylchloride', 'Hydrogen Chloride', 'Coke', '1-/2-chloroprene', '1, 1-/cis-/trans-dichloroethylene', 'trichloroethylene', r'$C_{4}H_{6}Cl_{2}$', r'$C_{6}H_{6}$', r'$C_{2}H_{2}$', '1, 1-dichloroethane', '1, 1, 2-trichloroethane', '1, 1, 1, 2-/1, 1, 2, 2-tetrachloroethane', r'$R_{1}$', r'$R_{2}$', r'$R_{3}$', r'$R_{4}$', r'$R_{5}$', r'$R_{6}$', r'$R_{7}$', r'$R_{8}$', 'tetrachloromethane', 'trichloromethane', 'Vinyl Chloride Monomer']
namespd = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', 'C4H6Cl2', 'C6H6', 'C2H2', 'C11', 'C112', 'C1112', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'VCM', 'T0', 'T1', 'Pure', 'Selectivity VCM', 'Selectivity HCl', 'Yield VCM', 'Constant 1', 'Constant 2', 'Constant 3', 'h Coefficient', 'U Coefficient', 'k value', 'Reynolds']
namesb3 = ['1,2-dichloroethane', 'Ethylchloride', 'Hydrogen Chloride', 'Coke', '1-/2-chloroprene', '1,1-/cis-/trans-dichloroethylene', 'trichloroethylene', r'$C_{4}H_{6}Cl_{2}$', r'$C_{6}H_{6}$', r'$C_{2}H_{2}$', '1,1-dichloroethane', '1,1,2-trichloroethane', '1,1,1,2-/1,1,2,2-tetrachloroethane', r'$R_{1}$', r'$R_{2}$', r'$R_{3}$', r'$R_{4}$', r'$R_{5}$', r'$R_{6}$', r'$R_{7}$', r'$R_{8}$', 'tetrachloromethane', 'trichloromethane', 'Vinyl Chloride Monomer']


# Cas ['107-06-2','75-00-3','7647-01-0','126-99-8','1,2-dichloroethylene','760-23-6','71-43-2','74-86-2','75-34-3','79-00-5','Cl','75-35-4','75-43-4','96-49-1','75-38-7','79-01-6','75-01-4']

start = time.time() #Real time when the program starts to run

mp.autoprec(50)


def flatten(l_l):
    flat_list = [item for sublist in l_l for item in sublist]
    return flat_list


def KtoC(K):
    Cval = K - 273.15
    return Cval


def CtoK(C):
    Kval = C + 273.15
    return Kval

#This function generates the sympy symbols


def symfunc(names, rxnum):
    Csyms = [sp.symbols(r'C_{}'.format('{}'.format(i))) for i in names]
    Ksyms = [sp.symbols(r'K_{}'.format(j + 1)) for j in range(rxnum)]
    EAsyms = [sp.symbols(r'Ea_{}'.format(k + 1)) for k in range(rxnum)]
    Tsyms = [sp.symbols('T'), sp.symbols('T1')]

    return Csyms, Ksyms, EAsyms, Tsyms

#These are mostly functions used to calculate the unused diffusion coefficients


def Dab(T, P, M1, M2, sa, sb, col):
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


def DI(Ys, Dijs):
    flist = []
    for i in range(0, len(Ys), 1):
        Yi = Ys[i]
        Di = Dijs[i]
        Yj = [g for q, g in enumerate(Ys) if q != Yi]
        Djb = [w for j, w in enumerate(Dijs) if j != Di]
        Dj = flatten(Djb)
        YjDj = [x/y for x, y in zip(Yj, Dj)]
        YjDjt = sum(YjDj)
        top = 1 - Yi
        Df = top/YjDjt
        flist.append(Df)
        Yj.clear()
        Dj.clear()
        YjDj.clear()
    return flist

# Calculates the Nusselt number based on given input


def getnu(velocity, rho, distance, diameter, k, viscosity, vs, Twall, Tgas, pr, re):
    Nu = [0.0]
    if re <= 3000.0:

        def term1a(re, pr, distance, diameter, viscosity, vs):
            return ((re * pr * (distance / diameter))**(1 / 3) * (viscosity / vs)**0.14)
        term1 = term1a(re, pr, distance, diameter, viscosity, vs)
        if term1 <= 2:
            Nu[0] = 3.66

        if term1 > 2:

            def f1(re, pr, distance, diameter, viscosity, vs):
                return (1.86 * ((re * pr) / (distance / diameter))**(1 / 3) * (viscosity / vs)**0.14)
            Nu[0] = f1(re, pr, distance, diameter, viscosity, vs)
    elif re > 3000.0 and re < 10000.0:

        def f2(fval, re, pr, distance, diameter, viscosity, vs):
            def fval(re, pr, distance, diameter, viscosity, vs):
                return ((0.79 * math.log(re) - 1.64)**-2)
            f = fval(re, pr, distance, diameter, viscosity, vs)
            return ((f / 8.0) * (re - 1000.0) * pr) / (1 + 12.7 * (f**(1 / 2)) * (pr**(2 / 3) - 1))
        Nu[0] = f2(re, pr, distance, diameter, viscosity, vs)
    elif re >= 10000.0:
        if Twall <= Tgas:

            def f3(re, pr, distance, diameter, viscosity, vs):
                return 0.023 * (re**(4 / 5)) * (pr**0.3)
            Nu[0] = f3(re, pr, distance, diameter, viscosity, vs)

        if Twall > Tgas:

            def f4(re, pr, distance, diameter, viscosity, vs):
                return 0.027 * (re**(4.0 / 5.0)) * ((pr**0.4)**(1.0 / 3.0)) * ((viscosity / vs)**0.14)
            Nu[0] = f4(re, pr, distance, diameter, viscosity, vs)

    Nuv = Nu[0]

    if distance / diameter < 60 or distance / diameter < 60.0:
        Nu.clear()

        def f5(Nuv, distance, diameter):
            return (Nuv) / (1 + (1 / ((distance / diameter)**(2 / 3))))
        Nu.append(f5(Nuv, distance, diameter))
    else:
        pass
    return float(Nu[0])


#Calcluates Reynolds number


def reynolds(rho, velocity, distance, viscosity):
    p = rho
    v = velocity
    x = distance
    u = viscosity
    Re = (p*x*v)/u
    return Re

#Solves for the thermal convection term (h)


def hterm(Nu, distance, k):
    h = (k*Nu)/distance
    return h

#Calculates the Prandtl number


def Pr(cp, viscosity, k):
    # Cp = [J/kg-K]
    # µ = [Pa*s] = [N*s/m^2]
    # k = [W/m-K]
    return (cp*viscosity)/k


def Nus(velocity, rho, distance, diameter, k, viscosity, vs, Twall, Tgas, pr, re):
    Nuv = getnu(velocity, rho, distance, diameter, k, viscosity, vs, Twall, Tgas, pr, re)
    Nu = Nuv
    return Nu

#Calculates the overall heat transfer coefficient (U)


def Uvalue(di, do, hi, ho, kpipe):

    Uval = 1.0/((do/di)*(1.0/hi) + ((do*math.log(do/di)/(2.0*kpipe))) + (1.0/ho))
    return Uval

#Calculates the density of a gas mixture


def rhomix(conc, molar_mass):
    rhomix = [i*j for i, j in zip(conc, molar_mass)]
    rhomixanswer = sum(rhomix)
    return rhomixanswer

#Calculates the Specific heat capacity (constant pressure) of a gas mixture


def cpmix(cp, conc):
    contot = sum(conc)
    Ci = [i/contot for i in conc]
    cpmix = [i*j for i, j in zip(Ci, cp)]
    cpmixanswer = sum(cpmix)
    return cpmixanswer


def chemicals(CAS, temp):
    return tc.Chemical('{}'.format(CAS), T=temp)


def formula(string):
    return tc.serialize_formula('{}'.format(string))


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def cp(alist):
    c2alist = []
    for i in alist:
        chemicalt = getattr(i, 'Cpgm')  # [J/mol/K]
        c2alist.append(chemicalt)
    return c2alist


def mw(alist):
    c2blist = []
    for i in alist:
        chemicalt2 = getattr(i, 'MW')  # [g/mol]
        c2blist.append(chemicalt2)
    return c2blist


def name(namelist):
    nlist = namelist
    names = []
    for i in nlist:
        chemicaltn = getattr(i, "IUPAC_name")
        names.append(chemicaltn)
    return [nlist]


def cpsum(alist):
    chemlist = alist
    c2alist2 = []
    for i in chemlist:
        chemicalta = getattr(i, 'Cpgm')  # [J/mol/K]
        c2alist2.append(chemicalta)
    return sum(c2alist2)


def mwsum(alist):
    chemlist = alist
    for i in chemlist:
        c2clist2 = []
        chemicaltc = getattr(i, 'MW')  # [g/m^3]
        c2clist2.append(chemicaltc)
    return sum(c2clist2)


def mixprop(IDsv, mwmixv, Tv, Pv):
    mwmix = tc.Mixture(IDsv, mwmixv, Tv, Pv)
    return mwmix


def mixpropcpg(IDsv, mwmixv, Tv, Pv):
    mwmix = tc.Mixture(IDsv, mwmixv, Tv, Pv)
    return mwmix.Cpg


def mixproprho(IDsv, mwmixv, Tv, Pv):  # [kg/m^3]
    mwmix = tc.Mixture(IDsv, mwmixv, Tv, Pv)
    return mwmix.rhog


def mixpropkmix(IDsv, mwmixv, Tv, Pv):  # [Pa*s]
    mwmix = tc.Mixture(IDsv, mwmixv, Tv, Pv)
    return mwmix.kg


def mixpropvmix(IDsv, mwmixv, Tv, Pv):  # [Pa*s]
    mwmix = tc.Mixture(IDsv, mwmixv, Tv, Pv)
    return mwmix.mugs

#These are mostly functions used to calculate the unused diffusion coefficients


def Si(tb):
    si = 1.5*tb
    return si


def Sij(si, sj):
    Si = si
    Sj = sj
    sij = math.sqrt(Si*Sj)*0.733
    return sij


def Aij(T, Ta, Tbb, ua, ub, Ma, Mb, ka, kb, cpa, cpb, cva, cvb):
    S1 = Si(Ta)
    S2 = Si(Tbb)
    S12 = Sij(S1, S2)
    Mab = (Mb/Ma)**(3.0/4.0)
    uab = Uab(ka, kb, cpa, cpb, cva, cvb)
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


def Aijlist(T, Tbs, u, Mws, Ks, Cp, Cv):
    lengthq = len(Mws)
    Aijf = []
    for i in range(0, lengthq, 1):
        aijs = []
        for j in range(0, lengthq, 1):
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
            aij1 = Aij(T, Ta, Tbval, ua, ub, Ma, Mb, ka, kb, cpa, cpb, cva, cvb)
            aijs.append(aij1)
        Aijf.append(aijs[:])
        aijs.clear()
    return Aijf

#Calculates the thermal conductivity constant (k) of a mixture


def kmix(T, yi, ks, u, Tbl, Mws, Cp, Cv):
    lengthf = len(yi)
    aijs2 = Aijlist(T, Tbl, u, Mws, ks, Cp, Cv)
    klist = []
    for i in range(0, lengthf, 1):
        bottom = []
        for j in range(0, lengthf, 1):
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


def Uab(ka, kb, cpa, cpb, cva, cvb):
    y1 = cpa/cva
    y2 = cpb/cvb
    kk = ka/kb
    cab = cpb/cpa
    r1 = (9.0 - (5.0/y2))
    r2 = (9.0 - (5.0/y1))
    rr = r1/r2
    uab = float(kk*cab*rr)
    return uab

# RHS (right hand side) is a function that returns the system of differential equations as a list/array to be used with the solve_ivp method
# There are 20 equations in total, one for each compound (listed in the order from the list excel/word file, 18), and two equations for temperature (T, dT)
# Steady-State 1-D Heat Balance: (d^2T)/(dz^2) = (U_coeff*α*(T_wall - T) + ∑r_i*∆H_rxn)/(k_mix); α = (Surface Area / Volume) or (2*π*r*L/π*r^2*L)
# Each equation is used to solve for the concentration of each compound and it is the exact same order that the compound lists are defined shortly below
# There is a program that calculates both the Jacobian and RHS equation systems (both symbolically and with certain numerical values plugged in) and saves them in the exact necessary format at in a txt file at https://github.com/tjczec01/chemicalsystem
# It will also outpus the necessary Latex equations for printing purposes


def RHS(t, y, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13, k_14, k_15, k_16, k_17, k_18, k_19, k_20, k_21, k_22, k_23, k_24, k_25, k_26, k_27, k_28, k_29, k_30, k_31, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Ea_22, Ea_23, Ea_24, Ea_25, Ea_26, Ea_27, Ea_28, Ea_29, Ea_30, Ea_31, R, Constant_1, Constant_2, Constant_3, Twallrhs, Taurhs):
    # k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13, k_14, k_15, k_16, k_17, k_18, k_19, k_20, k_21, k_22, k_23, k_24, k_25, k_26, k_27, k_28, k_29, k_30, k_31, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Ea_22, Ea_23, Ea_24, Ea_25, Ea_26, Ea_27, Ea_28, Ea_29, Ea_30, Ea_31, R, Constant_1, Constant_2, Constant_3, Twalls, Tau = args
    try:
        C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_Tri, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_C1112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_R7, C_R8, C_CCl4, C_CHCl3, C_VCM, T, T1 = y
        EQ_EDC = -C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) - C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) - C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) - C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T)) - C_EDC*k_1*math.exp(-Ea_1/(R*T))
        EQ_EC = -C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) + C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T))
        EQ_HCl = C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) + C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) + C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) + 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) + C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) + C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) + C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) + C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) + C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T))
        EQ_Coke = 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T))
        EQ_CP = C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T))
        EQ_Di = C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) - C_Di*C_R1*k_24*math.exp(-Ea_24/(R*T)) + C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) + C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T)) + C_R6*k_24*math.exp(-Ea_24/(R*T))
        EQ_Tri = -C_R1*C_Tri*k_25*math.exp(-Ea_25/(R*T)) + C_R7*k_25*math.exp(-Ea_25/(R*T))
        EQ_C4H6Cl2 = C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T))
        EQ_C6H6 = 2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T))
        EQ_C2H2 = -2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) - C_C2H2*C_R1*k_23*math.exp(-Ea_23/(R*T)) + C_R5*k_23*math.exp(-Ea_23/(R*T))
        EQ_C11 = -C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) + C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T))
        EQ_C112 = -C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) + C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) + C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T))
        EQ_C1112 = -C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) - C_C1112*C_R8*k_28*math.exp(-Ea_28/(R*T)) + C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) + C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T))
        EQ_R1 = -C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) - C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) - C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) + 2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) - C_C2H2*C_R1*k_23*math.exp(-Ea_23/(R*T)) + C_CCl4*k_2*math.exp(-Ea_2/(R*T)) - C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) - C_Di*C_R1*k_24*math.exp(-Ea_24/(R*T)) - C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) - C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) + C_EDC*k_1*math.exp(-Ea_1/(R*T)) - C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) - C_R1*C_Tri*k_25*math.exp(-Ea_25/(R*T)) - C_R1*C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T)) - C_R1*C_VCM*k_22*math.exp(-Ea_22/(R*T)) + C_R3*k_22*math.exp(-Ea_22/(R*T)) + C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T)) + C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T)) + C_R5*k_23*math.exp(-Ea_23/(R*T)) + C_R6*k_24*math.exp(-Ea_24/(R*T)) + C_R7*k_25*math.exp(-Ea_25/(R*T))
        EQ_R2 = C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) + C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) + C_EDC*k_1*math.exp(-Ea_1/(R*T)) - C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T))
        EQ_R3 = -C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) - C_R3*k_22*math.exp(-Ea_22/(R*T))
        EQ_R4 = C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) - C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) - C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) + C_R1*C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T))
        EQ_R5 = -2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) - C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) + C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T)) + C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T)) - C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T)) - C_R5*k_23*math.exp(-Ea_23/(R*T))
        EQ_R6 = C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) - C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) - C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T)) - C_R6*k_24*math.exp(-Ea_24/(R*T))
        EQ_R7 = C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) - C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_R7*k_25*math.exp(-Ea_25/(R*T))
        EQ_R8 = -C_C1112*C_R8*k_28*math.exp(-Ea_28/(R*T)) + C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) + C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) + C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) + C_CCl4*k_2*math.exp(-Ea_2/(R*T)) + C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) + C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) + C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) + C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T)) - C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T))
        EQ_CCl4 = -C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) - C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) - C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) - C_CCl4*k_2*math.exp(-Ea_2/(R*T)) + C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T))
        EQ_CHCl3 = -C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) + C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T))
        EQ_VCM = C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) + C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_R1*C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T)) - C_R1*C_VCM*k_22*math.exp(-Ea_22/(R*T)) - C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T)) + C_R3*k_22*math.exp(-Ea_22/(R*T)) - C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T)) - C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T))
        EQ_T = T1
        EQ_dT = Constant_1*T1 - Constant_2*(-T + Twallrhs) - Constant_3*(-C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) - C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) - C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) - C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T)) - C_EDC*k_1*math.exp(-Ea_1/(R*T)))
        return [Taurhs*EQ_EDC, Taurhs*EQ_EC, Taurhs*EQ_HCl, Taurhs*EQ_Coke, Taurhs*EQ_CP, Taurhs*EQ_Di, Taurhs*EQ_Tri, Taurhs*EQ_C4H6Cl2, Taurhs*EQ_C6H6, Taurhs*EQ_C2H2, Taurhs*EQ_C11, Taurhs*EQ_C112, Taurhs*EQ_C1112, Taurhs*EQ_R1, Taurhs*EQ_R2, Taurhs*EQ_R3, Taurhs*EQ_R4, Taurhs*EQ_R5, Taurhs*EQ_R6, Taurhs*EQ_R7, Taurhs*EQ_R8, Taurhs*EQ_CCl4, Taurhs*EQ_CHCl3, Taurhs*EQ_VCM, EQ_T, EQ_dT]
    
    except Exception:
        C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_Tri, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_C1112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_R7, C_R8, C_CCl4, C_CHCl3, C_VCM, T, T1 = y
        print(R, T, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Ea_22, Ea_23, Ea_24, Ea_25, Ea_26, Ea_27, Ea_28, Ea_29, Ea_30, Ea_31)
        EQ_EDC = -C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) - C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) - C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) - C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T)) - C_EDC*k_1*math.exp(-Ea_1/(R*T))
        EQ_EC = -C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) + C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T))
        EQ_HCl = C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) + C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) + C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) + 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) + C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) + C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) + C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) + C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) + C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T))
        EQ_Coke = 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T))
        EQ_CP = C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T))
        EQ_Di = C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) - C_Di*C_R1*k_24*math.exp(-Ea_24/(R*T)) + C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) + C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T)) + C_R6*k_24*math.exp(-Ea_24/(R*T))
        EQ_Tri = -C_R1*C_Tri*k_25*math.exp(-Ea_25/(R*T)) + C_R7*k_25*math.exp(-Ea_25/(R*T))
        EQ_C4H6Cl2 = C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T))
        EQ_C6H6 = 2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T))
        EQ_C2H2 = -2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) - C_C2H2*C_R1*k_23*math.exp(-Ea_23/(R*T)) + C_R5*k_23*math.exp(-Ea_23/(R*T))
        EQ_C11 = -C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) + C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T))
        EQ_C112 = -C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) + C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) + C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T))
        EQ_C1112 = -C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) - C_C1112*C_R8*k_28*math.exp(-Ea_28/(R*T)) + C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) + C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T))
        EQ_R1 = -C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) - C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) - C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) + 2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - 2*C_C2H2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) - C_C2H2*C_R1*k_23*math.exp(-Ea_23/(R*T)) + C_CCl4*k_2*math.exp(-Ea_2/(R*T)) - C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) - C_Di*C_R1*k_24*math.exp(-Ea_24/(R*T)) - C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) - C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) + C_EDC*k_1*math.exp(-Ea_1/(R*T)) - C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) - C_R1*C_Tri*k_25*math.exp(-Ea_25/(R*T)) - C_R1*C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T)) - C_R1*C_VCM*k_22*math.exp(-Ea_22/(R*T)) + C_R3*k_22*math.exp(-Ea_22/(R*T)) + C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T)) + C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T)) + C_R5*k_23*math.exp(-Ea_23/(R*T)) + C_R6*k_24*math.exp(-Ea_24/(R*T)) + C_R7*k_25*math.exp(-Ea_25/(R*T))
        EQ_R2 = C_EC*C_R1*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) + C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) + C_EDC*k_1*math.exp(-Ea_1/(R*T)) - C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T))
        EQ_R3 = -C_R1*C_R3*k_11*math.exp(-Ea_11/(R*T)) - C_R3*k_22*math.exp(-Ea_22/(R*T))
        EQ_R4 = C_C11*C_R1*k_13*math.exp(-Ea_13/(R*T)) - C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) - C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) + C_R1*C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T))
        EQ_R5 = -2*C_C2H2**2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) - C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) + C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T)) + C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T)) - C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T)) - C_R5*k_23*math.exp(-Ea_23/(R*T))
        EQ_R6 = C_C112*C_R1*k_14*math.exp(-Ea_14/(R*T)) - C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) - C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T)) - C_R6*k_24*math.exp(-Ea_24/(R*T))
        EQ_R7 = C_C1112*C_R1*k_15*math.exp(-Ea_15/(R*T)) - C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_R7*k_25*math.exp(-Ea_25/(R*T))
        EQ_R8 = -C_C1112*C_R8*k_28*math.exp(-Ea_28/(R*T)) + C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) + C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) + C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) + C_CCl4*k_2*math.exp(-Ea_2/(R*T)) + C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) + C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) + C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) + C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T)) - C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T))
        EQ_CCl4 = -C_CCl4*C_R4*k_26*math.exp(-Ea_26/(R*T)) - C_CCl4*C_R5*k_27*math.exp(-Ea_27/(R*T)) - C_CCl4*C_R6*k_28*math.exp(-Ea_28/(R*T)) - C_CCl4*k_2*math.exp(-Ea_2/(R*T)) + C_R6*C_R8*k_29*math.exp(-Ea_29/(R*T))
        EQ_CHCl3 = -C_CHCl3*C_R1*k_16*math.exp(-Ea_16/(R*T)) + C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T))
        EQ_VCM = C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) + C_R1*C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_R1*C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_R1*C_VCM*k_18*math.exp(-Ea_18/(R*T)) - C_R1*C_VCM*k_22*math.exp(-Ea_22/(R*T)) - C_R2*C_VCM*k_19*math.exp(-Ea_19/(R*T)) + C_R3*k_22*math.exp(-Ea_22/(R*T)) - C_R4*C_VCM*k_20*math.exp(-Ea_20/(R*T)) - C_R5*C_VCM*k_21*math.exp(-Ea_21/(R*T))
        EQ_T = T1
        EQ_dT = Constant_1*T1 - Constant_2*(-T + Twallrhs) - Constant_3*(-C_EDC*C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_EDC*C_R2*k_5*math.exp(-Ea_5/(R*T)) - C_EDC*C_R4*k_6*math.exp(-Ea_6/(R*T)) - C_EDC*C_R5*k_4*math.exp(-Ea_4/(R*T)) - C_EDC*C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_EDC*C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_EDC*C_R8*k_9*math.exp(-Ea_9/(R*T)) - C_EDC*k_1*math.exp(-Ea_1/(R*T)))
        return [Taurhs*EQ_EDC, Taurhs*EQ_EC, Taurhs*EQ_HCl, Taurhs*EQ_Coke, Taurhs*EQ_CP, Taurhs*EQ_Di, Taurhs*EQ_Tri, Taurhs*EQ_C4H6Cl2, Taurhs*EQ_C6H6, Taurhs*EQ_C2H2, Taurhs*EQ_C11, Taurhs*EQ_C112, Taurhs*EQ_C1112, Taurhs*EQ_R1, Taurhs*EQ_R2, Taurhs*EQ_R3, Taurhs*EQ_R4, Taurhs*EQ_R5, Taurhs*EQ_R6, Taurhs*EQ_R7, Taurhs*EQ_R8, Taurhs*EQ_CCl4, Taurhs*EQ_CHCl3, Taurhs*EQ_VCM, EQ_T, EQ_dT]  # return [Taurhs*EQ1, Taurhs*EQ2, Taurhs*EQ3, Taurhs*EQ4, Taurhs*EQ5, Taurhs*EQ6, Taurhs*EQ7, Taurhs*EQ8, Taurhs*EQ9, Taurhs*EQ10, Taurhs*EQ11, Taurhs*EQ12, Taurhs*EQ13, Taurhs*EQ14, Taurhs*EQ15, Taurhs*EQ16, Taurhs*EQ17, Taurhs*EQ18, Taurhs*EQ19, Taurhs*EQ20, Taurhs*EQ21, Taurhs*EQ22, Taurhs*EQ23, Taurhs*EQ24, EQ25, EQ26]


def Jacob(t, y, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13, k_14, k_15, k_16, k_17, k_18, k_19, k_20, k_21, k_22, k_23, k_24, k_25, k_26, k_27, k_28, k_29, k_30, k_31, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Ea_22, Ea_23, Ea_24, Ea_25, Ea_26, Ea_27, Ea_28, Ea_29, Ea_30, Ea_31, R, Constant_1, Constant_2, Constant_3, Twallsj, Tau):
    # k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13, k_14, k_15, k_16, k_17, k_18, k_19, k_20, k_21, k_22, k_23, k_24, k_25, k_26, k_27, k_28, k_29, k_30, k_31, Ea_1, Ea_2, Ea_3, Ea_4, Ea_5, Ea_6, Ea_7, Ea_8, Ea_9, Ea_10, Ea_11, Ea_12, Ea_13, Ea_14, Ea_15, Ea_16, Ea_17, Ea_18, Ea_19, Ea_20, Ea_21, Ea_22, Ea_23, Ea_24, Ea_25, Ea_26, Ea_27, Ea_28, Ea_29, Ea_30, Ea_31, R, Constant_1, Constant_2, Constant_3, Twalls, Tau = args
    C_EDC, C_EC, C_HCl, C_Coke, C_CP, C_Di, C_Tri, C_C4H6Cl2, C_C6H6, C_C2H2, C_C11, C_C112, C_C1112, C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_R7, C_R8, C_CCl4, C_CHCl3, C_VCM, T, T1 = y
    Jac = [[-C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_R2*k_5*math.exp(-Ea_5/(R*T)) - C_R4*k_6*math.exp(-Ea_6/(R*T)) - C_R5*k_4*math.exp(-Ea_4/(R*T)) - C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_R8*k_9*math.exp(-Ea_9/(R*T)) - k_1*math.exp(-Ea_1/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EDC*k_3*math.exp(-Ea_3/(R*T)), -C_EDC*k_5*math.exp(-Ea_5/(R*T)), 0, -C_EDC*k_6*math.exp(-Ea_6/(R*T)), -C_EDC*k_4*math.exp(-Ea_4/(R*T)), -C_EDC*k_7*math.exp(-Ea_7/(R*T)), -C_EDC*k_8*math.exp(-Ea_8/(R*T)), -C_EDC*k_9*math.exp(-Ea_9/(R*T)), 0, 0, 0, -C_EDC*C_R1*Ea_3*k_3*math.exp(-Ea_3/(R*T))/(R*T**2) - C_EDC*C_R2*Ea_5*k_5*math.exp(-Ea_5/(R*T))/(R*T**2) - C_EDC*C_R4*Ea_6*k_6*math.exp(-Ea_6/(R*T))/(R*T**2) - C_EDC*C_R5*Ea_4*k_4*math.exp(-Ea_4/(R*T))/(R*T**2) - C_EDC*C_R6*Ea_7*k_7*math.exp(-Ea_7/(R*T))/(R*T**2) - C_EDC*C_R7*Ea_8*k_8*math.exp(-Ea_8/(R*T))/(R*T**2) - C_EDC*C_R8*Ea_9*k_9*math.exp(-Ea_9/(R*T))/(R*T**2) - C_EDC*Ea_1*k_1*math.exp(-Ea_1/(R*T))/(R*T**2), 0],
           [C_R2*k_5*math.exp(-Ea_5/(R*T)), -C_R1*k_12*math.exp(-Ea_12/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_EC*k_12*math.exp(-Ea_12/(R*T)), C_EDC*k_5*math.exp(-Ea_5/(R*T)) + C_VCM*k_19*math.exp(-Ea_19/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, C_R2*k_19*math.exp(-Ea_19/(R*T)), -C_EC*C_R1*Ea_12*k_12*math.exp(-Ea_12/(R*T))/(R*T**2) + C_EDC*C_R2*Ea_5*k_5*math.exp(-Ea_5/(R*T))/(R*T**2) + C_R2*C_VCM*Ea_19*k_19*math.exp(-Ea_19/(R*T))/(R*T**2), 0],
           [C_R1*k_3*math.exp(-Ea_3/(R*T)), C_R1*k_12*math.exp(-Ea_12/(R*T)), 0, 0, 0, 0, 0, 0, 0, 2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)), C_R1*k_13*math.exp(-Ea_13/(R*T)), C_R1*k_14*math.exp(-Ea_14/(R*T)), C_R1*k_15*math.exp(-Ea_15/(R*T)), C_C11*k_13*math.exp(-Ea_13/(R*T)) + C_C1112*k_15*math.exp(-Ea_15/(R*T)) + C_C112*k_14*math.exp(-Ea_14/(R*T)) + 4*C_C2H2*C_R1*k_31*math.exp(-Ea_31/(R*T)) + C_CHCl3*k_16*math.exp(-Ea_16/(R*T)) + C_EC*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*k_3*math.exp(-Ea_3/(R*T)) + C_R2*k_10*math.exp(-Ea_10/(R*T)) + C_R3*k_11*math.exp(-Ea_11/(R*T)) + C_VCM*k_18*math.exp(-Ea_18/(R*T)), C_R1*k_10*math.exp(-Ea_10/(R*T)), C_R1*k_11*math.exp(-Ea_11/(R*T)), 0, 0, 0, 0, 0, 0, C_R1*k_16*math.exp(-Ea_16/(R*T)), C_R1*k_18*math.exp(-Ea_18/(R*T)), C_C11*C_R1*Ea_13*k_13*math.exp(-Ea_13/(R*T))/(R*T**2) + C_C1112*C_R1*Ea_15*k_15*math.exp(-Ea_15/(R*T))/(R*T**2) + C_C112*C_R1*Ea_14*k_14*math.exp(-Ea_14/(R*T))/(R*T**2) + 2*C_C2H2*C_R1**2*Ea_31*k_31*math.exp(-Ea_31/(R*T))/(R*T**2) + C_CHCl3*C_R1*Ea_16*k_16*math.exp(-Ea_16/(R*T))/(R*T**2) + C_EC*C_R1*Ea_12*k_12*math.exp(-Ea_12/(R*T))/(R*T**2) + C_EDC*C_R1*Ea_3*k_3*math.exp(-Ea_3/(R*T))/(R*T**2) + C_R1*C_R2*Ea_10*k_10*math.exp(-Ea_10/(R*T))/(R*T**2) + C_R1*C_R3*Ea_11*k_11*math.exp(-Ea_11/(R*T))/(R*T**2) + C_R1*C_VCM*Ea_18*k_18*math.exp(-Ea_18/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)), 0, 0, 0, 4*C_C2H2*C_R1*k_31*math.exp(-Ea_31/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*C_C2H2*C_R1**2*Ea_31*k_31*math.exp(-Ea_31/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*k_21*math.exp(-Ea_21/(R*T)), 0, 0, 0, 0, 0, C_R5*k_21*math.exp(-Ea_21/(R*T)), C_R5*C_VCM*Ea_21*k_21*math.exp(-Ea_21/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, -C_R1*k_24*math.exp(-Ea_24/(R*T)), 0, 0, 0, 0, 0, 0, 0, -C_Di*k_24*math.exp(-Ea_24/(R*T)) + C_R3*k_11*math.exp(-Ea_11/(R*T)), 0, C_R1*k_11*math.exp(-Ea_11/(R*T)), 0, C_CCl4*k_27*math.exp(-Ea_27/(R*T)), C_R8*k_29*math.exp(-Ea_29/(R*T)) + k_24*math.exp(-Ea_24/(R*T)), 0, C_R6*k_29*math.exp(-Ea_29/(R*T)), C_R5*k_27*math.exp(-Ea_27/(R*T)), 0, 0, C_CCl4*C_R5*Ea_27*k_27*math.exp(-Ea_27/(R*T))/(R*T**2) - C_Di*C_R1*Ea_24*k_24*math.exp(-Ea_24/(R*T))/(R*T**2) + C_R1*C_R3*Ea_11*k_11*math.exp(-Ea_11/(R*T))/(R*T**2) + C_R6*C_R8*Ea_29*k_29*math.exp(-Ea_29/(R*T))/(R*T**2) + C_R6*Ea_24*k_24*math.exp(-Ea_24/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, -C_R1*k_25*math.exp(-Ea_25/(R*T)), 0, 0, 0, 0, 0, 0, -C_Tri*k_25*math.exp(-Ea_25/(R*T)), 0, 0, 0, 0, 0, k_25*math.exp(-Ea_25/(R*T)), 0, 0, 0, 0, -C_R1*C_Tri*Ea_25*k_25*math.exp(-Ea_25/(R*T))/(R*T**2) + C_R7*Ea_25*k_25*math.exp(-Ea_25/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_VCM*k_20*math.exp(-Ea_20/(R*T)), 0, 0, 0, 0, 0, 0, C_R4*k_20*math.exp(-Ea_20/(R*T)), C_R4*C_VCM*Ea_20*k_20*math.exp(-Ea_20/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 4*C_C2H2*C_R5*k_30*math.exp(-Ea_30/(R*T)), 0, 0, 0, 0, 0, 0, 0, 2*C_C2H2**2*k_30*math.exp(-Ea_30/(R*T)), 0, 0, 0, 0, 0, 0, 2*C_C2H2**2*C_R5*Ea_30*k_30*math.exp(-Ea_30/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, -4*C_C2H2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - 2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) - C_R1*k_23*math.exp(-Ea_23/(R*T)), 0, 0, 0, -4*C_C2H2*C_R1*k_31*math.exp(-Ea_31/(R*T)) - C_C2H2*k_23*math.exp(-Ea_23/(R*T)), 0, 0, 0, -2*C_C2H2**2*k_30*math.exp(-Ea_30/(R*T)) + k_23*math.exp(-Ea_23/(R*T)), 0, 0, 0, 0, 0, 0, -2*C_C2H2**2*C_R5*Ea_30*k_30*math.exp(-Ea_30/(R*T))/(R*T**2) - 2*C_C2H2*C_R1**2*Ea_31*k_31*math.exp(-Ea_31/(R*T))/(R*T**2) - C_C2H2*C_R1*Ea_23*k_23*math.exp(-Ea_23/(R*T))/(R*T**2) + C_R5*Ea_23*k_23*math.exp(-Ea_23/(R*T))/(R*T**2), 0],
           [C_R4*k_6*math.exp(-Ea_6/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*k_13*math.exp(-Ea_13/(R*T)), 0, 0, -C_C11*k_13*math.exp(-Ea_13/(R*T)), 0, 0, C_EDC*k_6*math.exp(-Ea_6/(R*T)), 0, 0, 0, 0, 0, 0, 0, -C_C11*C_R1*Ea_13*k_13*math.exp(-Ea_13/(R*T))/(R*T**2) + C_EDC*C_R4*Ea_6*k_6*math.exp(-Ea_6/(R*T))/(R*T**2), 0],
           [C_R6*k_7*math.exp(-Ea_7/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*k_14*math.exp(-Ea_14/(R*T)), 0, -C_C112*k_14*math.exp(-Ea_14/(R*T)), 0, 0, C_CCl4*k_26*math.exp(-Ea_26/(R*T)), 0, C_EDC*k_7*math.exp(-Ea_7/(R*T)), 0, 0, C_R4*k_26*math.exp(-Ea_26/(R*T)), 0, 0, -C_C112*C_R1*Ea_14*k_14*math.exp(-Ea_14/(R*T))/(R*T**2) + C_CCl4*C_R4*Ea_26*k_26*math.exp(-Ea_26/(R*T))/(R*T**2) + C_EDC*C_R6*Ea_7*k_7*math.exp(-Ea_7/(R*T))/(R*T**2), 0],
           [C_R7*k_8*math.exp(-Ea_8/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*k_15*math.exp(-Ea_15/(R*T)) - C_R8*k_28*math.exp(-Ea_28/(R*T)), -C_C1112*k_15*math.exp(-Ea_15/(R*T)), 0, 0, 0, 0, C_CCl4*k_28*math.exp(-Ea_28/(R*T)), C_EDC*k_8*math.exp(-Ea_8/(R*T)), -C_C1112*k_28*math.exp(-Ea_28/(R*T)), C_R6*k_28*math.exp(-Ea_28/(R*T)), 0, 0, -C_C1112*C_R1*Ea_15*k_15*math.exp(-Ea_15/(R*T))/(R*T**2) - C_C1112*C_R8*Ea_28*k_28*math.exp(-Ea_28/(R*T))/(R*T**2) + C_CCl4*C_R6*Ea_28*k_28*math.exp(-Ea_28/(R*T))/(R*T**2) + C_EDC*C_R7*Ea_8*k_8*math.exp(-Ea_8/(R*T))/(R*T**2), 0],
           [-C_R1*k_3*math.exp(-Ea_3/(R*T)) + k_1*math.exp(-Ea_1/(R*T)), -C_R1*k_12*math.exp(-Ea_12/(R*T)), 0, 0, 0, -C_R1*k_24*math.exp(-Ea_24/(R*T)), -C_R1*k_25*math.exp(-Ea_25/(R*T)), 0, 0, 4*C_C2H2*C_R5*k_30*math.exp(-Ea_30/(R*T)) - 2*C_R1**2*k_31*math.exp(-Ea_31/(R*T)) - C_R1*k_23*math.exp(-Ea_23/(R*T)), -C_R1*k_13*math.exp(-Ea_13/(R*T)), -C_R1*k_14*math.exp(-Ea_14/(R*T)), -C_R1*k_15*math.exp(-Ea_15/(R*T)), -C_C11*k_13*math.exp(-Ea_13/(R*T)) - C_C1112*k_15*math.exp(-Ea_15/(R*T)) - C_C112*k_14*math.exp(-Ea_14/(R*T)) - 4*C_C2H2*C_R1*k_31*math.exp(-Ea_31/(R*T)) - C_C2H2*k_23*math.exp(-Ea_23/(R*T)) - C_CHCl3*k_16*math.exp(-Ea_16/(R*T)) - C_Di*k_24*math.exp(-Ea_24/(R*T)) - C_EC*k_12*math.exp(-Ea_12/(R*T)) - C_EDC*k_3*math.exp(-Ea_3/(R*T)) - C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_R3*k_11*math.exp(-Ea_11/(R*T)) - C_Tri*k_25*math.exp(-Ea_25/(R*T)) - C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_VCM*k_18*math.exp(-Ea_18/(R*T)) - C_VCM*k_22*math.exp(-Ea_22/(R*T)), -C_R1*k_10*math.exp(-Ea_10/(R*T)), -C_R1*k_11*math.exp(-Ea_11/(R*T)) + k_22*math.exp(-Ea_22/(R*T)), C_VCM*k_20*math.exp(-Ea_20/(R*T)), 2*C_C2H2**2*k_30*math.exp(-Ea_30/(R*T)) + C_VCM*k_21*math.exp(-Ea_21/(R*T)) + k_23*math.exp(-Ea_23/(R*T)), k_24*math.exp(-Ea_24/(R*T)), k_25*math.exp(-Ea_25/(R*T)), 0, k_2*math.exp(-Ea_2/(R*T)), -C_R1*k_16*math.exp(-Ea_16/(R*T)), -C_R1*k_17*math.exp(-Ea_17/(R*T)) - C_R1*k_18*math.exp(-Ea_18/(R*T)) - C_R1*k_22*math.exp(-Ea_22/(R*T)) + C_R4*k_20*math.exp(-Ea_20/(R*T)) + C_R5*k_21*math.exp(-Ea_21/(R*T)), -C_C11*C_R1*Ea_13*k_13*math.exp(-Ea_13/(R*T))/(R*T**2) - C_C1112*C_R1*Ea_15*k_15*math.exp(-Ea_15/(R*T))/(R*T**2) - C_C112*C_R1*Ea_14*k_14*math.exp(-Ea_14/(R*T))/(R*T**2) + 2*C_C2H2**2*C_R5*Ea_30*k_30*math.exp(-Ea_30/(R*T))/(R*T**2) - 2*C_C2H2*C_R1**2*Ea_31*k_31*math.exp(-Ea_31/(R*T))/(R*T**2) - C_C2H2*C_R1*Ea_23*k_23*math.exp(-Ea_23/(R*T))/(R*T**2) + C_CCl4*Ea_2*k_2*math.exp(-Ea_2/(R*T))/(R*T**2) - C_CHCl3*C_R1*Ea_16*k_16*math.exp(-Ea_16/(R*T))/(R*T**2) - C_Di*C_R1*Ea_24*k_24*math.exp(-Ea_24/(R*T))/(R*T**2) - C_EC*C_R1*Ea_12*k_12*math.exp(-Ea_12/(R*T))/(R*T**2) - C_EDC*C_R1*Ea_3*k_3*math.exp(-Ea_3/(R*T))/(R*T**2) + C_EDC*Ea_1*k_1*math.exp(-Ea_1/(R*T))/(R*T**2) - C_R1*C_R2*Ea_10*k_10*math.exp(-Ea_10/(R*T))/(R*T**2) - C_R1*C_R3*Ea_11*k_11*math.exp(-Ea_11/(R*T))/(R*T**2) - C_R1*C_Tri*Ea_25*k_25*math.exp(-Ea_25/(R*T))/(R*T**2) - C_R1*C_VCM*Ea_17*k_17*math.exp(-Ea_17/(R*T))/(R*T**2) - C_R1*C_VCM*Ea_18*k_18*math.exp(-Ea_18/(R*T))/(R*T**2) - C_R1*C_VCM*Ea_22*k_22*math.exp(-Ea_22/(R*T))/(R*T**2) + C_R3*Ea_22*k_22*math.exp(-Ea_22/(R*T))/(R*T**2) + C_R4*C_VCM*Ea_20*k_20*math.exp(-Ea_20/(R*T))/(R*T**2) + C_R5*C_VCM*Ea_21*k_21*math.exp(-Ea_21/(R*T))/(R*T**2) + C_R5*Ea_23*k_23*math.exp(-Ea_23/(R*T))/(R*T**2) + C_R6*Ea_24*k_24*math.exp(-Ea_24/(R*T))/(R*T**2) + C_R7*Ea_25*k_25*math.exp(-Ea_25/(R*T))/(R*T**2), 0],
           [C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_R2*k_5*math.exp(-Ea_5/(R*T)) + C_R6*k_7*math.exp(-Ea_7/(R*T)) + k_1*math.exp(-Ea_1/(R*T)), C_R1*k_12*math.exp(-Ea_12/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EC*k_12*math.exp(-Ea_12/(R*T)) + C_EDC*k_3*math.exp(-Ea_3/(R*T)) - C_R2*k_10*math.exp(-Ea_10/(R*T)), -C_EDC*k_5*math.exp(-Ea_5/(R*T)) - C_R1*k_10*math.exp(-Ea_10/(R*T)) - C_VCM*k_19*math.exp(-Ea_19/(R*T)), 0, 0, 0, C_EDC*k_7*math.exp(-Ea_7/(R*T)), 0, 0, 0, 0, -C_R2*k_19*math.exp(-Ea_19/(R*T)), C_EC*C_R1*Ea_12*k_12*math.exp(-Ea_12/(R*T))/(R*T**2) + C_EDC*C_R1*Ea_3*k_3*math.exp(-Ea_3/(R*T))/(R*T**2) - C_EDC*C_R2*Ea_5*k_5*math.exp(-Ea_5/(R*T))/(R*T**2) + C_EDC*C_R6*Ea_7*k_7*math.exp(-Ea_7/(R*T))/(R*T**2) + C_EDC*Ea_1*k_1*math.exp(-Ea_1/(R*T))/(R*T**2) - C_R1*C_R2*Ea_10*k_10*math.exp(-Ea_10/(R*T))/(R*T**2) - C_R2*C_VCM*Ea_19*k_19*math.exp(-Ea_19/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R3*k_11*math.exp(-Ea_11/(R*T)), 0, -C_R1*k_11*math.exp(-Ea_11/(R*T)) - k_22*math.exp(-Ea_22/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, -C_R1*C_R3*Ea_11*k_11*math.exp(-Ea_11/(R*T))/(R*T**2) - C_R3*Ea_22*k_22*math.exp(-Ea_22/(R*T))/(R*T**2), 0],
           [-C_R4*k_6*math.exp(-Ea_6/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R1*k_13*math.exp(-Ea_13/(R*T)), 0, 0, C_C11*k_13*math.exp(-Ea_13/(R*T)) + C_VCM*k_17*math.exp(-Ea_17/(R*T)), 0, 0, -C_CCl4*k_26*math.exp(-Ea_26/(R*T)) - C_EDC*k_6*math.exp(-Ea_6/(R*T)) - C_VCM*k_20*math.exp(-Ea_20/(R*T)), 0, 0, 0, 0, -C_R4*k_26*math.exp(-Ea_26/(R*T)), 0, C_R1*k_17*math.exp(-Ea_17/(R*T)) - C_R4*k_20*math.exp(-Ea_20/(R*T)), C_C11*C_R1*Ea_13*k_13*math.exp(-Ea_13/(R*T))/(R*T**2) - C_CCl4*C_R4*Ea_26*k_26*math.exp(-Ea_26/(R*T))/(R*T**2) - C_EDC*C_R4*Ea_6*k_6*math.exp(-Ea_6/(R*T))/(R*T**2) + C_R1*C_VCM*Ea_17*k_17*math.exp(-Ea_17/(R*T))/(R*T**2) - C_R4*C_VCM*Ea_20*k_20*math.exp(-Ea_20/(R*T))/(R*T**2), 0],
           [-C_R5*k_4*math.exp(-Ea_4/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, -4*C_C2H2*C_R5*k_30*math.exp(-Ea_30/(R*T)), 0, 0, 0, C_VCM*k_18*math.exp(-Ea_18/(R*T)), C_VCM*k_19*math.exp(-Ea_19/(R*T)), 0, 0, -2*C_C2H2**2*k_30*math.exp(-Ea_30/(R*T)) - C_CCl4*k_27*math.exp(-Ea_27/(R*T)) - C_EDC*k_4*math.exp(-Ea_4/(R*T)) - C_VCM*k_21*math.exp(-Ea_21/(R*T)) - k_23*math.exp(-Ea_23/(R*T)), 0, 0, 0, -C_R5*k_27*math.exp(-Ea_27/(R*T)), 0, C_R1*k_18*math.exp(-Ea_18/(R*T)) + C_R2*k_19*math.exp(-Ea_19/(R*T)) - C_R5*k_21*math.exp(-Ea_21/(R*T)), -2*C_C2H2**2*C_R5*Ea_30*k_30*math.exp(-Ea_30/(R*T))/(R*T**2) - C_CCl4*C_R5*Ea_27*k_27*math.exp(-Ea_27/(R*T))/(R*T**2) - C_EDC*C_R5*Ea_4*k_4*math.exp(-Ea_4/(R*T))/(R*T**2) + C_R1*C_VCM*Ea_18*k_18*math.exp(-Ea_18/(R*T))/(R*T**2) + C_R2*C_VCM*Ea_19*k_19*math.exp(-Ea_19/(R*T))/(R*T**2) - C_R5*C_VCM*Ea_21*k_21*math.exp(-Ea_21/(R*T))/(R*T**2) - C_R5*Ea_23*k_23*math.exp(-Ea_23/(R*T))/(R*T**2), 0],
           [-C_R6*k_7*math.exp(-Ea_7/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R1*k_14*math.exp(-Ea_14/(R*T)), 0, C_C112*k_14*math.exp(-Ea_14/(R*T)), 0, 0, 0, 0, -C_CCl4*k_28*math.exp(-Ea_28/(R*T)) - C_EDC*k_7*math.exp(-Ea_7/(R*T)) - C_R8*k_29*math.exp(-Ea_29/(R*T)) - k_24*math.exp(-Ea_24/(R*T)), 0, -C_R6*k_29*math.exp(-Ea_29/(R*T)), -C_R6*k_28*math.exp(-Ea_28/(R*T)), 0, 0, C_C112*C_R1*Ea_14*k_14*math.exp(-Ea_14/(R*T))/(R*T**2) - C_CCl4*C_R6*Ea_28*k_28*math.exp(-Ea_28/(R*T))/(R*T**2) - C_EDC*C_R6*Ea_7*k_7*math.exp(-Ea_7/(R*T))/(R*T**2) - C_R6*C_R8*Ea_29*k_29*math.exp(-Ea_29/(R*T))/(R*T**2) - C_R6*Ea_24*k_24*math.exp(-Ea_24/(R*T))/(R*T**2), 0],
           [-C_R7*k_8*math.exp(-Ea_8/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R1*k_15*math.exp(-Ea_15/(R*T)), C_C1112*k_15*math.exp(-Ea_15/(R*T)), 0, 0, 0, 0, 0, -C_EDC*k_8*math.exp(-Ea_8/(R*T)) - k_25*math.exp(-Ea_25/(R*T)), 0, 0, 0, 0, C_C1112*C_R1*Ea_15*k_15*math.exp(-Ea_15/(R*T))/(R*T**2) - C_EDC*C_R7*Ea_8*k_8*math.exp(-Ea_8/(R*T))/(R*T**2) - C_R7*Ea_25*k_25*math.exp(-Ea_25/(R*T))/(R*T**2), 0],
           [C_R4*k_6*math.exp(-Ea_6/(R*T)) + C_R5*k_4*math.exp(-Ea_4/(R*T)) + C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_R8*k_9*math.exp(-Ea_9/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_R8*k_28*math.exp(-Ea_28/(R*T)), C_CHCl3*k_16*math.exp(-Ea_16/(R*T)), 0, 0, C_CCl4*k_26*math.exp(-Ea_26/(R*T)) + C_EDC*k_6*math.exp(-Ea_6/(R*T)), C_CCl4*k_27*math.exp(-Ea_27/(R*T)) + C_EDC*k_4*math.exp(-Ea_4/(R*T)), C_CCl4*k_28*math.exp(-Ea_28/(R*T)) - C_R8*k_29*math.exp(-Ea_29/(R*T)), C_EDC*k_8*math.exp(-Ea_8/(R*T)), -C_C1112*k_28*math.exp(-Ea_28/(R*T)) - C_EDC*k_9*math.exp(-Ea_9/(R*T)) - C_R6*k_29*math.exp(-Ea_29/(R*T)), C_R4*k_26*math.exp(-Ea_26/(R*T)) + C_R5*k_27*math.exp(-Ea_27/(R*T)) + C_R6*k_28*math.exp(-Ea_28/(R*T)) + k_2*math.exp(-Ea_2/(R*T)), C_R1*k_16*math.exp(-Ea_16/(R*T)), 0, -C_C1112*C_R8*Ea_28*k_28*math.exp(-Ea_28/(R*T))/(R*T**2) + C_CCl4*C_R4*Ea_26*k_26*math.exp(-Ea_26/(R*T))/(R*T**2) + C_CCl4*C_R5*Ea_27*k_27*math.exp(-Ea_27/(R*T))/(R*T**2) + C_CCl4*C_R6*Ea_28*k_28*math.exp(-Ea_28/(R*T))/(R*T**2) + C_CCl4*Ea_2*k_2*math.exp(-Ea_2/(R*T))/(R*T**2) + C_CHCl3*C_R1*Ea_16*k_16*math.exp(-Ea_16/(R*T))/(R*T**2) + C_EDC*C_R4*Ea_6*k_6*math.exp(-Ea_6/(R*T))/(R*T**2) + C_EDC*C_R5*Ea_4*k_4*math.exp(-Ea_4/(R*T))/(R*T**2) + C_EDC*C_R7*Ea_8*k_8*math.exp(-Ea_8/(R*T))/(R*T**2) - C_EDC*C_R8*Ea_9*k_9*math.exp(-Ea_9/(R*T))/(R*T**2) - C_R6*C_R8*Ea_29*k_29*math.exp(-Ea_29/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_CCl4*k_26*math.exp(-Ea_26/(R*T)), -C_CCl4*k_27*math.exp(-Ea_27/(R*T)), -C_CCl4*k_28*math.exp(-Ea_28/(R*T)) + C_R8*k_29*math.exp(-Ea_29/(R*T)), 0, C_R6*k_29*math.exp(-Ea_29/(R*T)), -C_R4*k_26*math.exp(-Ea_26/(R*T)) - C_R5*k_27*math.exp(-Ea_27/(R*T)) - C_R6*k_28*math.exp(-Ea_28/(R*T)) - k_2*math.exp(-Ea_2/(R*T)), 0, 0, -C_CCl4*C_R4*Ea_26*k_26*math.exp(-Ea_26/(R*T))/(R*T**2) - C_CCl4*C_R5*Ea_27*k_27*math.exp(-Ea_27/(R*T))/(R*T**2) - C_CCl4*C_R6*Ea_28*k_28*math.exp(-Ea_28/(R*T))/(R*T**2) - C_CCl4*Ea_2*k_2*math.exp(-Ea_2/(R*T))/(R*T**2) + C_R6*C_R8*Ea_29*k_29*math.exp(-Ea_29/(R*T))/(R*T**2), 0],
           [C_R8*k_9*math.exp(-Ea_9/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -C_CHCl3*k_16*math.exp(-Ea_16/(R*T)), 0, 0, 0, 0, 0, 0, C_EDC*k_9*math.exp(-Ea_9/(R*T)), 0, -C_R1*k_16*math.exp(-Ea_16/(R*T)), 0, -C_CHCl3*C_R1*Ea_16*k_16*math.exp(-Ea_16/(R*T))/(R*T**2) + C_EDC*C_R8*Ea_9*k_9*math.exp(-Ea_9/(R*T))/(R*T**2), 0],
           [C_R5*k_4*math.exp(-Ea_4/(R*T)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_R2*k_10*math.exp(-Ea_10/(R*T)) - C_VCM*k_17*math.exp(-Ea_17/(R*T)) - C_VCM*k_18*math.exp(-Ea_18/(R*T)) - C_VCM*k_22*math.exp(-Ea_22/(R*T)), C_R1*k_10*math.exp(-Ea_10/(R*T)) - C_VCM*k_19*math.exp(-Ea_19/(R*T)), k_22*math.exp(-Ea_22/(R*T)), -C_VCM*k_20*math.exp(-Ea_20/(R*T)), C_EDC*k_4*math.exp(-Ea_4/(R*T)) - C_VCM*k_21*math.exp(-Ea_21/(R*T)), 0, 0, 0, 0, 0, -C_R1*k_17*math.exp(-Ea_17/(R*T)) - C_R1*k_18*math.exp(-Ea_18/(R*T)) - C_R1*k_22*math.exp(-Ea_22/(R*T)) - C_R2*k_19*math.exp(-Ea_19/(R*T)) - C_R4*k_20*math.exp(-Ea_20/(R*T)) - C_R5*k_21*math.exp(-Ea_21/(R*T)), C_EDC*C_R5*Ea_4*k_4*math.exp(-Ea_4/(R*T))/(R*T**2) + C_R1*C_R2*Ea_10*k_10*math.exp(-Ea_10/(R*T))/(R*T**2) - C_R1*C_VCM*Ea_17*k_17*math.exp(-Ea_17/(R*T))/(R*T**2) - C_R1*C_VCM*Ea_18*k_18*math.exp(-Ea_18/(R*T))/(R*T**2) - C_R1*C_VCM*Ea_22*k_22*math.exp(-Ea_22/(R*T))/(R*T**2) - C_R2*C_VCM*Ea_19*k_19*math.exp(-Ea_19/(R*T))/(R*T**2) + C_R3*Ea_22*k_22*math.exp(-Ea_22/(R*T))/(R*T**2) - C_R4*C_VCM*Ea_20*k_20*math.exp(-Ea_20/(R*T))/(R*T**2) - C_R5*C_VCM*Ea_21*k_21*math.exp(-Ea_21/(R*T))/(R*T**2), 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [-Constant_3*(-C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_R2*k_5*math.exp(-Ea_5/(R*T)) - C_R4*k_6*math.exp(-Ea_6/(R*T)) - C_R5*k_4*math.exp(-Ea_4/(R*T)) - C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_R8*k_9*math.exp(-Ea_9/(R*T)) - k_1*math.exp(-Ea_1/(R*T))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EDC*Constant_3*k_3*math.exp(-Ea_3/(R*T)), C_EDC*Constant_3*k_5*math.exp(-Ea_5/(R*T)), 0, C_EDC*Constant_3*k_6*math.exp(-Ea_6/(R*T)), C_EDC*Constant_3*k_4*math.exp(-Ea_4/(R*T)), C_EDC*Constant_3*k_7*math.exp(-Ea_7/(R*T)), C_EDC*Constant_3*k_8*math.exp(-Ea_8/(R*T)), C_EDC*Constant_3*k_9*math.exp(-Ea_9/(R*T)), 0, 0, 0, Constant_2 - Constant_3*(-C_EDC*C_R1*Ea_3*k_3*math.exp(-Ea_3/(R*T))/(R*T**2) - C_EDC*C_R2*Ea_5*k_5*math.exp(-Ea_5/(R*T))/(R*T**2) - C_EDC*C_R4*Ea_6*k_6*math.exp(-Ea_6/(R*T))/(R*T**2) - C_EDC*C_R5*Ea_4*k_4*math.exp(-Ea_4/(R*T))/(R*T**2) - C_EDC*C_R6*Ea_7*k_7*math.exp(-Ea_7/(R*T))/(R*T**2) - C_EDC*C_R7*Ea_8*k_8*math.exp(-Ea_8/(R*T))/(R*T**2) - C_EDC*C_R8*Ea_9*k_9*math.exp(-Ea_9/(R*T))/(R*T**2) - C_EDC*Ea_1*k_1*math.exp(-Ea_1/(R*T))/(R*T**2)), Constant_1]]
    JacF = [[float(Jac[i][j]*Tau) for j in range(len(Jac[0]))] for i in range(len(Jac) - 2)]
    JacF.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    JacF.append([-Constant_3*(-C_R1*k_3*math.exp(-Ea_3/(R*T)) - C_R2*k_5*math.exp(-Ea_5/(R*T)) - C_R4*k_6*math.exp(-Ea_6/(R*T)) - C_R5*k_4*math.exp(-Ea_4/(R*T)) - C_R6*k_7*math.exp(-Ea_7/(R*T)) - C_R7*k_8*math.exp(-Ea_8/(R*T)) - C_R8*k_9*math.exp(-Ea_9/(R*T)) - k_1*math.exp(-Ea_1/(R*T))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C_EDC*Constant_3*k_3*math.exp(-Ea_3/(R*T)), C_EDC*Constant_3*k_5*math.exp(-Ea_5/(R*T)), 0, C_EDC*Constant_3*k_6*math.exp(-Ea_6/(R*T)), C_EDC*Constant_3*k_4*math.exp(-Ea_4/(R*T)), C_EDC*Constant_3*k_7*math.exp(-Ea_7/(R*T)), C_EDC*Constant_3*k_8*math.exp(-Ea_8/(R*T)), C_EDC*Constant_3*k_9*math.exp(-Ea_9/(R*T)), 0, 0, 0, Constant_2 - Constant_3*(C_EDC*C_R1*-Ea_3*k_3*math.exp(-Ea_3/(R*T))/(R*T**2) + C_EDC*C_R2*-Ea_5*k_5*math.exp(-Ea_5/(R*T))/(R*T**2) + C_EDC*C_R4*-Ea_6*k_6*math.exp(-Ea_6/(R*T))/(R*T**2) + C_EDC*C_R5*-Ea_4*k_4*math.exp(-Ea_4/(R*T))/(R*T**2) + C_EDC*C_R6*-Ea_7*k_7*math.exp(-Ea_7/(R*T))/(R*T**2) + C_EDC*C_R7*-Ea_8*k_8*math.exp(-Ea_8/(R*T))/(R*T**2) + C_EDC*C_R8*-Ea_9*k_9*math.exp(-Ea_9/(R*T))/(R*T**2) + C_EDC*-Ea_1*k_1*math.exp(-Ea_1/(R*T))/(R*T**2)), Constant_1])
    return JacF

# These are mostly functions used to calculate the unused diffusion coefficients


def Bviral(T, Tc, pc, omega):
    Tr = T/Tc
    B0 = 0.1445 - 0.33/Tr - 0.1385/(Tr**2) - 0.0121/(Tr**3)
    B1 = 0.073 + 0.46/Tr - 0.5/(Tr**2) - 0.097/(Tr**3) - 0.0073/(Tr**8)
    Br = B0 + omega*B1
    B = Br*R_gas*(Tc/pc)
    return B


def Bviral2(T, Tc, pc, omega):
    Tr = T/Tc
    B0 = 0.083 - 0.422/(Tr**1.6)
    B1 = 0.139 - 0.172/(Tr**4.2)
    Br = B0 + omega*B1
    B = Br*R_gas*(Tc/pc)
    return B


def Bviral3(T, Tc, pc, omega):
    Tr = T/Tc
    B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3 - 0.000607/Tr**8
    B1 = 0.0637 + 0.331/Tr**2 - 0.423/Tr**3 - 0.008/Tr**8
    Br = B0 + omega*B1
    B = Br*R_gas*(Tc/pc)
    return B


def Bviral4(T, Tc, pc, omega, a, b, dipole):
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
    B = Br*R_gas*(Tc/pc)
    return B


def btoz(B, T, P):
    Z = 1.0 + (B * P) / (T * R_gas)
    return Z


def densitytovm(p, MW):
    vmval = 1 / ((1E3 * p) / (MW))
    return vmval


def alistfun(Temp, PascalP):
    EDCp = tc.Chemical('107-06-2', T=Temp, P=PascalP)
    ECp = tc.Chemical('75-00-3', T=Temp, P=PascalP)
    HClp = tc.Chemical('7647-01-0', T=Temp, P=PascalP)
    Cokep = tc.Chemical('Activated charcoal', T=Temp, P=PascalP)
    CPp = tc.Chemical('126-99-8', T=Temp, P=PascalP)
    Dip = tc.Chemical('126-99-8', T=Temp, P=PascalP)
    Trip = tc.Chemical('79-01-6', T=Temp, P=PascalP)
    C4H6Cl2p = tc.Chemical('760-23-6', T=Temp, P=PascalP)
    C6H6p = tc.Chemical('71-43-2', T=Temp, P=PascalP)
    C2H2p = tc.Chemical('74-86-2', T=Temp, P=PascalP)
    C11p = tc.Chemical('75-34-3', T=Temp, P=PascalP)
    C112p = tc.Chemical('79-00-5', T=Temp, P=PascalP)
    C1112p = tc.Chemical('630-20-6', T=Temp, P=PascalP)
    R1p = tc.Chemical('7647-01-0', T=Temp, P=PascalP)
    R2p = tc.Chemical('75-35-4', T=Temp, P=PascalP)
    R3p = tc.Chemical('75-43-4', T=Temp, P=PascalP)
    R4p = tc.Chemical('96-49-1', T=Temp, P=PascalP)
    R5p = tc.Chemical('75-38-7', T=Temp, P=PascalP)
    R6p = tc.Chemical('79-01-6', T=Temp, P=PascalP)
    R7p = tc.Chemical('630-20-6', T=Temp, P=PascalP)
    R8p = tc.Chemical('67-66-3', T=Temp, P=PascalP)
    CCl4p = tc.Chemical('56-23-5', T=Temp, P=PascalP)
    CHCl3p = tc.Chemical('67-66-3', T=Temp, P=PascalP)
    VCMp = tc.Chemical('75-01-4', T=Temp, P=PascalP)
    alist = [EDCp, ECp, HClp, Cokep, CPp, Dip, Trip, C4H6Cl2p, C6H6p, C2H2p, C11p, C112p, C1112p, R1p, R2p, R3p, R4p, R5p, R6p, R7p, R8p, CCl4p, CHCl3p, VCMp]
    return alist


def alistfun2(Temp, PascalP):
    EDCp = tc.Chemical('107-06-2', T=Temp, P=PascalP)
    ECp = tc.Chemical('75-00-3', T=Temp, P=PascalP)
    HClp = tc.Chemical('7647-01-0', T=Temp, P=PascalP)
    Cokep = tc.Chemical('Activated charcoal', T=Temp, P=PascalP)
    CPp = tc.Chemical('126-99-8', T=Temp, P=PascalP)
    Dip = tc.Chemical('126-99-8', T=Temp, P=PascalP)
    Trip = tc.Chemical('79-01-6', T=Temp, P=PascalP)
    C4H6Cl2p = tc.Chemical('760-23-6', T=Temp, P=PascalP)
    C6H6p = tc.Chemical('71-43-2', T=Temp, P=PascalP)
    C2H2p = tc.Chemical('74-86-2', T=Temp, P=PascalP)
    C11p = tc.Chemical('75-34-3', T=Temp, P=PascalP)
    C112p = tc.Chemical('79-00-5', T=Temp, P=PascalP)
    C1112p = tc.Chemical('630-20-6', T=Temp, P=PascalP)
    R1p = tc.Chemical('7647-01-0', T=Temp, P=PascalP)
    R2p = tc.Chemical('75-35-4', T=Temp, P=PascalP)
    R3p = tc.Chemical('75-43-4', T=Temp, P=PascalP)
    R4p = tc.Chemical('96-49-1', T=Temp, P=PascalP)
    R5p = tc.Chemical('75-38-7', T=Temp, P=PascalP)
    R6p = tc.Chemical('79-01-6', T=Temp, P=PascalP)
    R7p = tc.Chemical('630-20-6', T=Temp, P=PascalP)
    R8p = tc.Chemical('67-66-3', T=Temp, P=PascalP)
    CCl4p = tc.Chemical('56-23-5', T=Temp, P=PascalP)
    CHCl3p = tc.Chemical('67-66-3', T=Temp, P=PascalP)
    VCMp = tc.Chemical('75-01-4', T=Temp, P=PascalP)
    alist = [EDCp, ECp, HClp, Cokep, CPp, Dip, Trip, C4H6Cl2p, C6H6p, C2H2p, C11p, C112p, C1112p, R1p, R2p, R3p, R4p, R5p, R6p, R7p, R8p, CCl4p, CHCl3p, VCMp]
    return alist


Eab = [342, 230, 7, 34, 42, 45, 48, 56, 63, 13, 12, 4, 6, 15, 17, 14, 0, 56, 61, 30, 31, 84, 90, 70, 70, 33, 33, 33, 13, 20, 70]  # [kJ/mol]
Ea = [float(x * 1000.0) for x in Eab]  # [J/mol]

names = []

# Fluid physical and thermal properties
Tc = 650.0  # [°C]
Tk = CtoK(Tc)  # [K]
omegaedc = 0.28600000000000003
Tcedc = 563
Pcedc = 5380 * 1E3
dipoleedc = 1.572
density = (1E6*1.253)/1E3  # [kg/m^3]
mwedc = 98.95 # [g/mol]
mwedckg = 98.95/1E3 # kg/mol
delhm = 71000  # [J/mol]
Temperature = 650.0  # float(input('Enter starting temperature [°C] --> '))  # [°C]
Temp_K = CtoK(Temperature)  # [K]
Twall_c = CtoK(Temperature)  # [K]
Twallsi = Twall_c
Pbar = 12.159  # [bar]
Pstart_atm = 12.0 #float(input("Enter starting pressure [atm] --> "))
PascalP = Pstart_atm*101325.0  # [Pa]
Patm = PascalP/101325.0
volume_flow = 1.5 #float(input("Enter volumetric flow rate [m^3/s] --> ")) #1.5 [m^3/s]
volume_flowcm = volume_flow*1E6  # [cm^3/s]
pipe_thickness = 0.025 #float(input("Enter pipe thickness [m] --> ")) #0.025  # [m]
do = 0.5 #float(input("Enter total pipe diameter [m] --> "))# 0.5 #Outer diameter [m]
ro = do/2.0  # Outer radius [m]
di = do - 2.0*pipe_thickness #Inner diameter [m]
ri = di/2.0  # Inner radius [m]
cross_area = math.pi*(ri**2)
Ac = math.pi*(ri**2)
u_z = volume_flow/cross_area  # [m/s]
initedcb = (PascalP/(Temp_K*R_gas))
CCl4_p = float(0.0002)
EDC_p = 1.0 - CCl4_p
EDC0 = EDC_p*initedcb
CCL40 = CCl4_p*initedcb
begmix = tc.Mixture(IDs=['107-06-2', '56-23-5'], zs=[EDC_p, CCl4_p], T=Temp_K, P=PascalP)
edcmw = begmix.MW
Rvalc = begmix.R_specific
rhoino = begmix.rho  # [kg/m^3]
rhoin = rhoino*(1E3)*(1E-6)
rhoin2 = rhoin/edcmw
rhoing = begmix.rhogm  # [mol/m^3]
rhoingb = begmix.rhogm/1E6  # [mol/cm^3]
Temp_vals = []
Temp_vals2 = []
mws = begmix.MW  # [g/mol]
mwskg = mws/1E3  # [kg/mol]
Rval = Rvalc*mwskg
rhoc = rhoin/mws
rhoc2 = (PascalP/(Temp_K*R_gas))/1E6  # [mol/m^3]
inedc = rhoing/1E6
initedc = rhoing
initccl4 = initedc*CCl4_p
desired_time = 30 #int(input("Enter total reaction time [s] --> "))  # [s]
end_dist = float(u_z*desired_time)
dist_neat = round(end_dist, 1)
F_in = volume_flow*rhoc2  # [mol/s]
L = int(u_z*desired_time) # [m]
Surface_area = math.pi*2.0*ri*L
alpha = (math.pi*2.0*ri*L)/(Ac*L)
patm = PascalP/101325.0  # [atm]
Ratm = 8.20573660809596E-5  # [m^3*atm/K*mol]
Rkcal = 1.98720425864083E-3  # [kcal/K*mol]
segment_second = 10 #int(input('Enter iterations per second --> '))
gnodes = 10  # Divide iternum by this to get the interval over which the graphs will be saved. i.e. 250 / 10 = 25 iterations or 25 Kelvin. i.e. 25 individual graphs
gnodes2 = 10  # Same but this variable controls how many lines will be on a single graph i.e. 10 different temperature will be graphed for a single species
chngamnt = 1.0  # float(input('Enter Activation Energy change per iteration [J / mol] --> '))
iternum = 250  # int(input('Enter total iterations --> '))
segment_num = desired_time*segment_second  # Segments/Second
time_nodes = int(iternum/gnodes)
graph_nodes = [int(i*time_nodes) for i in range(0, gnodes, 1)]
time_nodesb = int(iternum/gnodes2)
graph_nodes2 = [int(i*time_nodesb) for i in range(0, gnodes2, 1)]
endnum = int(L)
distance = L
segmentlength = dist_int = L/int(segment_num)  # [m]
total_distance = np.linspace(0, L, num=(segment_num))
reaction_time = L/u_z  # reaction_time = L/u_z  # [s]
time_per_step = segmentlength/u_z  # [s]
time_int = 1/segment_second
alistb = alistfun(Temp_K, PascalP)
ksteel = 16.3  # [W/m*K]
time_vals = [i*time_int for i in range(0, segment_num, 1)]
vflow_cmf = volume_flowcm/segment_second  # [cm^3/s]
vflows = volume_flow/segment_second
time_valsb = [i*time_int for i in range(0, segment_num+1, 1)]
tmvpd = np.asarray(time_valsb)
dist_c = [(i/segment_second)*u_z for i in range(int(desired_time*segment_second))]
dist_cm = [i*u_z for i in time_vals]
dist_b = [i*u_z for i in time_valsb]
dstpd = np.asarray(dist_b)
ffd = int(dist_c[-1])
Ls = segmentlength  # [m]
vintflow = Ls*(math.pi)*(ri**2)  # [m^3/s]
vintflowcm = (Ls*100)*(math.pi)*((ri*100)**2)  # [cm^3/s]
dist_int = Ls
volume_flowcmf = (volume_flowcm)/segment_second  # [cm^3/s]
vintflowf = vintflow/segment_second
tn = np.linspace(0, L, desired_time)

# rtolval = 1E-3
# rtolvalj = 1E-3
# atolval = 1E-6
# atolvalj = 1E-6


# Concentrations
EDC = [float(initedcb)]
EC = [float(0.0)]
HCl = [float(0.0)]
Coke = [float(0.0)]  # Coke is written as simply C in the paper
CP = [float(0.0)]
Di = [float(0.0)]
Tri = [float(0.0)]
C4H6Cl2 = [float(0.0)]
C6H6 = [float(0.0)]
C2H2 = [float(0.0)]
C11 = [float(0.0)]
C112 = [float(0.0)]
C1112 = [float(0.0)]
R1 = [float(0.0)]
R2 = [float(0.0)]
R3 = [float(0.0)]
R4 = [float(0.0)]
R5 = [float(0.0)]
R6 = [float(0.0)]
R7 = [float(0.0)]
R8 = [float(0.0)]
CCl4 = [float(initccl4)]
CHCl3 = [float(0.0)]
VCM = [float(0.0)]
T0 = [float(Temp_K)]
T1 = [float(0.0)]
Ctotal = float(EDC[-1]) + float(EC[-1]) + float(HCl[-1]) + float(Coke[-1]) + float(CP[-1]) + float(Di[-1]) + float(Tri[-1]) + float(C4H6Cl2[-1]) + float(C6H6[-1]) + float(C2H2[-1]) + float(C11[-1]) + float(C112[-1]) + float(C1112[-1]) + float(R1[-1]) + float(R2[-1]) + float(R3[-1]) + float(R4[-1]) + float(R5[-1]) + float(R6[-1]) + float(R7[-1]) + float(R8[-1]) + float(CCl4[-1]) + float(CHCl3[-1]) + float(VCM[-1])
J_eval = [0]
J_evalt = [0]
EDCj = [float(initedc)]
ECj = [float(0.0)]
HClj = [float(0.0)]
Cokej = [float(0.0)]  # Coke is written as simply C in the paper
CPj = [float(0.0)]
Dij = [float(0.0)]
Trij = [float(0.0)]
C4H6Cl2j = [float(0.0)]
C6H6j = [float(0.0)]
C2H2j = [float(0.0)]
C11j = [float(0.0)]
C112j = [float(0.0)]
C1112j = [float(0.0)]
R1j = [float(0.0)]
R2j = [float(0.0)]
R3j = [float(0.0)]
R4j = [float(0.0)]
R5j = [float(0.0)]
R6j = [float(0.0)]
R7j = [float(0.0)]
R8j = [float(0.0)]
CCl4j = [float(initccl4)]
CHCl3j = [float(0.0)]
VCMj = [float(0.0)]
T0j = [float(Temp_K)]
T1j = [float(0.0)]
Ctotalj = float(EDCj[-1]) + float(ECj[-1]) + float(HClj[-1]) + float(Cokej[-1]) + float(CPj[-1]) + float(Dij[-1]) + float(Trij[-1]) + float(C4H6Cl2j[-1]) + float(C6H6j[-1]) + float(C2H2j[-1]) + float(C11j[-1]) + float(C112j[-1]) + float(C1112j[-1]) + float(R1j[-1]) + float(R2j[-1]) + float(R3j[-1]) + float(R4j[-1]) + float(R5j[-1]) + float(R6j[-1]) + float(R7j[-1]) + float(R8j[-1]) + float(CCl4j[-1]) + float(CHCl3j[-1]) + float(VCMj[-1])
C_Total = [Ctotal]
C_Totalj = [Ctotalj]
initial_edc = float(initedc)
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
kmix_valsj = []
EDCl = []
ECl = []
HCll = []
Cokel = []  # Coke is written as simply C in the paper
CPl = []
Dil = []
Tril = []
C4H6Cl2l = []
C6H6l = []
C2H2l = []
C11l = []
C112l = []
C1112l = []
R1l = []
R2l = []
R3l = []
R4l = []
R5l = []
R6l = []
R7l = []
R8l = []
CCl4l = []
CHCl3l = []
VCMl = []
T0l = []
T1l = []
EDClj = []
EClj = []
HCllj = []
Cokelj = []  # Coke is written as simpljy C in the paper
CPlj = []
Dilj = []
Trilj = []
C4H6Cl2lj = []
C6H6lj = []
C2H2lj = []
C11lj = []
C112lj = []
C1112lj = []
R1lj = []
R2lj = []
R3lj = []
R4lj = []
R5lj = []
R6lj = []
R7lj = []
R8lj = []
CCl4lj = []
CHCl3lj = []
VCMlj = []
T0lj = []
T1lj = []
conversion_EDCf = []
conversion_EDCfj = []
conversion_CCl4 = []
conversion_CCl4j = []
pr1 = [100.0]
pr1j = [100.0]
conversionf = [0.0]
productsf = []
productsfj = []
prods1f = []
prods1fj = []
puref = []
purefj = []

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
y0 = [EDC[-1], EC[-1], HCl[-1], Coke[-1], CP[-1], Di[-1], Tri[-1], C4H6Cl2[-1], C6H6[-1], C2H2[-1], C11[-1], C112[-1], C1112[-1], R1[-1], R2[-1], R3[-1], R4[-1], R5[-1], R6[-1], R7[-1], R8[-1], CCl4[-1], CHCl3[-1], VCM[-1]]
C_Total = [sum(y0)]
C_Totalj = [sum(y0)]
initial_edc = float(initedc)
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

alistb = alistfun(Temp_K, PascalP)
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
    chemicalt = getattr(i, 'IUPAC_name')
    names.append(chemicalt)

dist_end = endnum  # Final position
J_eval = [0]
J_evalt = [0]
dist_value = [0.0]
Tbl = []
CAS = []
prhos = []
mws = []
pvis = []
kks = []
for i in alistb:
    Casnum = getattr(i, 'CAS')
    CAS.append(Casnum)
    tb = getattr(i, 'Tb')
    Tbl.append(float(tb))
    chemicalpi = getattr(i, 'rhogm')
    prhos.append(float(chemicalpi))
    chemicalmwi = getattr(i, 'MW')
    mws.append(float(chemicalmwi/1000.0))
    chemicalmug = getattr(i, 'mug')
    pvis.append(float(chemicalmug))
    chemicalkl = getattr(i, 'kg')
    kks.append(float(chemicalkl))
prhot = sum(prhos)
mfracs = [i/prhot for i in prhos]
ys = tc.utils.zs_to_ws(mfracs, mws)
zvar = np.linspace(0, L, int(L/Ls))
cons1a = [5.228861948970217258E06, 4.573981356040483661E04, 2.392702360656382751E11]
z1 = sp.symbols('z1')
t0 = 0
rxnnum = len(Ea)
ysv, Ks, EAs, Ts = symfunc(namesj, rxnnum)
ysv.append(sp.symbols('T0'))
ysv.append(sp.symbols('T1'))
yblank = []
Twalls = [Twallsi]
iterlist = [i for i in range(0, iternum, 1)]
firststepval = 1000.0
Ls2 = firststepval
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
Taun = 1.0/(volume_flow/cross_area)  # [s/m]

iterslist = [int(i) for i in range(0, iternum, 1)]
# for il in iterslist:
for il in tqdm(iterslist):
    change = il * chngamnt
    amount_new = Temp_K - change
    Temp_vals.append(amount_new)
    Twall = amount_new
    Twalls.append(amount_new)
    EDC = [float(initedc)]
    EDCj = [float(initedc)]
    Eabb = [342, 230, 7, 34, 42, 45, 48, 56, 63, 13, 12, 4, 6, 15, 17, 14, 0, 56, 61, 30, 31, 84, 90, 70, 70, 33, 33, 33, 13, 20, 70]  # [kJ/mol]
    Eab = [x * 1000.0 for x in Eabb]
    Ea = [float(x) for x in Eab]  # kJ/mol
    ks = [5.9E15, 2.2E+12, 1.3E+13, 1.2E+13, 1E+12, 5E+11, 2E+11, 1E+11, 1E+12, 1E+13, 1E+13, 1.7E+13, 1.2E+13, 1.7E+13, 1.7E+13, 1.6E+13, 91000000000, 1.2E+14, 3E+11, 20000000000, 5E+11, 2.1E+14, 5E+14, 2E+13, 2.5E+13, 1E+12, 5E+11, 5E+11, 1E+13, 1E+14, 1.6E+14]
    n = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    k_0 = [float(i/((1E6)**(j - 1))) for i, j in zip(ks, n)]
    T0 = [float(amount_new)]
    T0j = [float(amount_new)]
    # print(k_0)
    # for x2 in tqdm(dist_c):
    for x2 in dist_c:
        cross_area = math.pi*(ri**2)
        alistb2 = alistfun(float(T0[-1]), float(PascalP))
        Y0b = [EDC[-1], EC[-1], HCl[-1], Coke[-1], CP[-1], Di[-1], Tri[-1], C4H6Cl2[-1], C6H6[-1], C2H2[-1], C11[-1], C112[-1], C1112[-1], R1[-1], R2[-1], R3[-1], R4[-1], R5[-1], R6[-1], R7[-1], R8[-1], CCl4[-1], CHCl3[-1], VCM[-1]]
        Y0bfa = [float(i) for i in Y0b]
        aaij = [x*0.0 for x in range(0, len(alistb2), 1)]
        Aaij = [aaij[:] for x in range(0, len(alistb2), 1)]
        Ctotal = sum(Y0b)
        C_ii = [float(i)/float(Ctotal) for i in Y0b]
        C_im = tc.utils.zs_to_ws(C_ii, mws)
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
        for i in Y0bfa:
            if i > 0.0:
                k = Y0bfa.index(i)
                i2 = alistb2[k]
                C_i2.append(float(i)/float(Ctotal))
                chemicalcpi = getattr(i2, 'Cpgm')  # [J/mol*K]
                Cpi_list2.append(float(chemicalcpi))
                chemicalmwi = getattr(i2, 'MW')  # [g/mol]
                MWgi_list2.append(float(chemicalmwi))  # [g/mol]
                MWi_list2.append(float(chemicalmwi/1000.0))  # [kg/mol]
                chemicalpi = getattr(i2, 'rhogm')  # [mol/m^3]
                Rhoi_list2.append(float(chemicalpi))
                chemicalkl = getattr(i2, 'kg')  # [mol/m^3]
                klval_i2.append(float(chemicalkl))
                chemicalmug = getattr(i2, 'mu')  #  [Pa*s]
                mu_i2.append(float(chemicalmug))
                chemicaltb = getattr(i2, 'Tb')  # [K]
                TB_i2.append(float(chemicaltb))
                chemicalvm = getattr(i2, 'Vmg')
                Vm2.append(float(chemicalvm/1E6))
                chemicalna = getattr(i2, 'IUPAC_name')
                names2b.append(chemicalna)
                chemicalcas = getattr(i2, 'CAS')
                cass.append(chemicalcas)
                thermo = tc.chemical.Chemical(chemicalcas, T=float(T0[-1]), P=PascalP)
                TG.append(thermo.ThermalConductivityGas)
                thermov = tc.chemical.Chemical(chemicalcas, T=float(T0[-1]), P=PascalP)
                VS.append(thermov.ViscosityGas)
            else:
                pass
        cpavg = mean(Cpi_list2)
        rhoavg = mean(Rhoi_list2)
        rhovm = [i*j for i, j in zip(Vm2, Rhoi_list2)]
        rhoend = float(PascalP/(Rval*float(T0[-1])))
        rhotot2 = sum(Rhoi_list2)
        Massfrac2b = [float(i/rhotot2) for i in Rhoi_list2]
        Massfrac = [float(i*j) for i, j in zip(C_i2, MWgi_list2)]
        Mavg = sum(Massfrac)
        wsl = [float((i*j)/Mavg) for i, j in zip(C_i2, MWgi_list2)]
        wsl2 = tc.utils.zs_to_ws(C_i2, MWgi_list2)
        Rho_Cp = [i*j for i, j in zip(Cpi_list2, Rhoi_list2)]
        RCPavg = mean(Rho_Cp)
        Cpglist = []
        gmix = tc.Mixture(IDs=names2b, zs=C_i2, T=float(T0[-1]), P=PascalP)
        viss = gmix.mugs
        tbss = gmix.Tbs
        mwss = gmix.MWs
        ksss = gmix.kgs
        CPs = gmix.Cpgms
        CVs = gmix.Cvgms
        Kval1a = gmix.kg
        kval2 = kmix(float(T0[-1]), C_i2, ksss, viss, tbss, mwss, CPs, CVs)  # [W/m*K]
        Molew = gmix.MWs
        sigma = gmix.molecular_diameters
        stemix = gmix.Stockmayers
        cpmmix = gmix.Cpgm  # [J/mol*K]
        cpkgmix = gmix.Cpg  # [J/kg*K]
        viscosityb = gmix.mug  # [Pa*s] or [kg/m*s^2]
        vislist = gmix.mugs
        viscosityc = viscosityb  # [kg/m*s^2]
        viscosity = tc.viscosity.Brokaw(T=float(T0[-1]), ys=C_i2, mus=viss, MWs=mwss, molecular_diameters=sigma, Stockmayers=stemix) # [Pa*s]
        rhob = gmix.rhog  # [mol/m^3]
        rhoc = gmix.rhogm  # [mol/m^3]
        rhocpgm1 = rhoc*cpmmix  # [J/m^3*K]
#        surface_area = math.pi*2*ri*Ls
        distancesb = Ls  # [m]
        distances = Ls  # [m]
        velocityb = u_z  # [m/s]
        velocity = velocityb # [m/s]
#        total_volume = math.pi*Ls*(ri**2)
#        alpha = surface_area/total_volume
        Tgas = float(T0[-1])  # T = K
        diameter = di  # [cm]
        tclist = gmix.Tcs
        Vclist = gmix.Vcs
        Zcmix = gmix.Zg
        TC = tc.critical.modified_Wilson_Tc(zs=C_i2, Tcs=tclist, Aijs=Aaij)
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
        gmixw2 = tc.Mixture(IDs=names2b, zs=C_i2, T=float(Twalls[-1]), P=PascalP)
        vsb = gmixw2.mug  # [Pa*s]
        vslist = gmixw2.mugs  # [Pa*s]
        vs = tc.viscosity.Brokaw(T=float(Twalls[-1]), ys=C_i2, mus=viss, MWs=mwss, molecular_diameters=sigma, Stockmayers=stemix)  # [Pa*S]
        pr = Pr(cpkgmix, viscosity, kval2)
        re = reynolds(rhob, u_z, Ls, viscosity)
        nuval = Nus(velocity, rhoc, Ls, diameter, kval2, viscosity, vs, float(Twalls[-1]), float(T0[-1]), pr, re)
        gash = hterm(nuval, Ls, kval2)
        h20 = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20cp = h20.Cpgm
        h20rhob = h20.rhogm  # [mol/m^3]
        h20rho = h20rhob  # [mol/m^3]
        h20kb = h20.kg  # [W/(m*K)]
        h20k = h20kb  # [W/m*k]
        h20visb = h20.mug  # [Pa*s]
        h20vis = h20visb  # [g/m*s^2]
        prh20 = Pr(h20cp, h20vis, h20k)
        reh20 = reynolds(h20rho, velocity, Ls, h20vis)
        h20wall = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20vsb = h20wall.mug  # [Pa*s]
        h20vs = h20vsb  # [g/m*s^2]
        hnuval = Nus(velocity, h20rho, Ls, diameter, h20k, h20vis, h20vs, float(Twalls[-1]), float(Twalls[-1]), prh20, reh20)
        h20h = hterm(hnuval, Ls, h20k)
        U_coeffb = Uvalue(diameter, do, gash, h20h, ksteel)  # [W/m^2*k]
        U_coeff = U_coeffb  # [W/m^2*k]
        delhm = 71000.0  # J/mol
        con1b = RCPavg/kval2
        con1 = (u_z*rhocpgm1)/kval2  # [1/m] -> [1/cm]
        con2 = (alpha*U_coeff)/kval2  # [m] (Sa)
        con3 = delhm/(u_z*kval2)  # -> [m]
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
        Y0bj = [EDCj[-1], ECj[-1], HClj[-1], Cokej[-1], CPj[-1], Dij[-1], Trij[-1], C4H6Cl2j[-1], C6H6j[-1], C2H2j[-1], C11j[-1], C112j[-1], C1112j[-1], R1j[-1], R2j[-1], R3j[-1], R4j[-1], R5j[-1], R6j[-1], R7j[-1], R8j[-1], CCl4j[-1], CHCl3j[-1], VCMj[-1]]
        Y0bjf = [float(i) for i in Y0bj]
        Ctotalj = sum(Y0bjf)
        C_iij = [float(jj)/float(Ctotalj) for jj in Y0bj]
        alistb2j = alistfun(float(T0j[-1]), float(PascalP))
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
        for ij in Y0bjf:
            if ij > 0.0:
                kj = Y0bjf.index(ij)
                i2j = alistb2j[kj]
                C_i2j.append(float(ij)/float(Ctotalj))
                chemicalcpij = getattr(i2j, 'Cpgm')  # [J/mol*K]
                Cpi_list2j.append(chemicalcpij)
                chemicalmwij = getattr(i2j, 'MW')  # [g/mol]
                MWgi_list2j.append(chemicalmwij)  # [g/mol]
                MWi_list2j.append(float(chemicalmwij/1000))  # [kg/mol]
                chemicalpij = getattr(i2j, 'rhogm')  # [mol/m^3]
                Rhoi_list2j.append(chemicalpij)  # [mol/cm^3]
                chemicalklj = getattr(i2j, 'kg')  # [W/m*K]
                klval_i2j.append(chemicalklj)  # [W/cm*K]
                chemicalmugj = getattr(i2j, 'mug')  # [Pa*s]
                mu_i2j.append(chemicalmugj)
                chemicaltbj = getattr(i2j, 'Tb')  # [K]
                TB_i2j.append(chemicaltbj)
                chemicalvmj = getattr(i2j, 'Vmg')  # [mol/m^3]
                Vm2j.append(chemicalvmj)  # [mol/cm^3]
                chemicalnaj = getattr(i2j, 'IUPAC_name')
                names2bj.append(chemicalnaj)
            else:
                pass
        cpavgj = mean(Cpi_list2j)
        rhoavgj = mean(Rhoi_list2j)
        rhovmj = [ij*jj for ij, jj in zip(Vm2j, Rhoi_list2j)]
        rhoendj = float(PascalP/(Rval*float(T0j[-1])))
        rhotot2j = sum(Rhoi_list2j)
        Massfrac2j = [float(ij/rhotot2j) for ij in Rhoi_list2j]
        Rho_Cpj = [i*j for i, j in zip(Cpi_list2j, Rhoi_list2j)]
        RCPavgj = mean(Rho_Cpj)
        Cpglistj = []
        Massfracj = [float(i*j) for i, j in zip(C_i2j, MWgi_list2j)]
        Mavgj = sum(Massfracj)
        wslj = [float((i*j)/Mavgj) for i, j in zip(C_i2j, MWgi_list2j)]
        gmixj = tc.Mixture(IDs=names2bj, zs=C_i2j, T=float(T0j[-1]), P=PascalP)
        Rho_Cp = [i*j for i, j in zip(Cpi_list2, Rhoi_list2)]
        RCPavg = mean(Rho_Cp)
        Cpglist = []
        vissj = gmixj.mugs
        tbssj = gmixj.Tbs
        mwssj = gmixj.MWs
        ksssj = gmixj.kgs
        Molewj = gmixj.MWs
        CPsj = gmixj.Cpgms
        CVsj = gmixj.Cvgms
        kval2j = kmix(float(T0j[-1]), C_i2j, ksssj, vissj, tbssj, mwssj, CPsj, CVsj)  # [W/m*K]
        sigmaj = gmixj.molecular_diameters
        stemixj = gmixj.Stockmayers
        cpgmmixj = gmixj.Cpgm  # [J/mol*K]
        viscosityj = gmixj.mug  # [g/cm*s^2]
        vislistj = gmixj.mugs  # [g/cm*s^2]
        vs2 = tc.viscosity.Brokaw(T=float(T0j[-1]), ys=C_i2j, mus=vissj, MWs=mwssj, molecular_diameters=sigmaj, Stockmayers=stemixj)
        rhoj = gmixj.rhogm  # [mol/cm^3]
        rhocpgm2 = rhoj*cpgmmixj  # [J/cm^3*K]
        surface_area = math.pi*2*ri*Ls
        distances = Ls  # m
        velocityj = u_z  # [m/s]
        total_volume = math.pi*Ls*(ri**2)
        alpha = surface_area/total_volume
        Tgasj = float(T0j[-1])  # T = K
        diameterj = di  # [m]
        tclistj = gmixj.Tcs
        Vclistj = gmixj.Vcs
        Zcmixj = gmixj.Zg
        TCj = tc.critical.modified_Wilson_Tc(zs=C_i2j, Tcs=tclistj, Aijs=Aaij)
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
        gmixw2j = tc.Mixture(IDs=names2bj, zs=C_i2j, T=float(Twalls[-1]), P=PascalP)
        vsj = gmixw2j.mug  # [g/cm*s^2]
        vs2j = tc.viscosity.Brokaw(T=float(Twalls[-1]), ys=C_i2j, mus=vissj, MWs=mwssj, molecular_diameters=sigmaj, Stockmayers=stemixj)
        prj = Pr(cpgmmixj, vsj, kval2j)
        rej = reynolds(rhoj, velocityj, distances, vsj)
        nuvalj = Nus(velocity, rhoj, distances, diameter, kval2j, vsj, vs2j, float(Twalls[-1]), T0j[-1], prj, rej)
        gashj = hterm(nuvalj, distances, kval2j)
        h20j = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20cpj = h20j.Cpgm
        h20rhoj = h20j.rhogm  # [mol/m^3]
        h20kj = h20j.kg  # [W/(m*K)]
        h20visj = h20j.mug  # [g/cm*s^2]
        prh20j = Pr(h20cpj, h20visj, h20kj)
        reh20j = reynolds(h20rhoj, velocityj, distances, h20visj)
        h20wallj = tc.Chemical('7732-18-5', T=float(Twalls[-1]), P=PascalP)
        h20vsj = h20wallj.mug  # [g/cm*s^2]
        hnuvalj = Nus(velocityj, h20rhoj, distances, diameter, h20kj, h20visj, h20vsj, float(Twalls[-1]), float(Twalls[-1]), prh20j, reh20j)
        h20hj = hterm(hnuvalj, distances, h20kj)
        U_coeffj = Uvalue(diameter, do, gashj, h20hj, ksteel)  # (W/(m2•K))
        delhm = 71000.0  # J/mol
        con1j = (u_z*rhocpgm2)/kval2j
        con2j = (alpha*U_coeffj)/kval2j
        con3j = delhm/(u_z*kval2j)
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
        # 0.0, Ls
        Y02b = [EDC[-1], EC[-1], HCl[-1], Coke[-1], CP[-1], Di[-1], Tri[-1], C4H6Cl2[-1], C6H6[-1], C2H2[-1], C11[-1], C112[-1], C1112[-1], R1[-1], R2[-1], R3[-1], R4[-1], R5[-1], R6[-1], R7[-1], R8[-1], CCl4[-1], CHCl3[-1], VCM[-1]]
        Y0 = [EDC[-1], EC[-1], HCl[-1], Coke[-1], CP[-1], Di[-1], Tri[-1], C4H6Cl2[-1], C6H6[-1], C2H2[-1], C11[-1], C112[-1], C1112[-1], R1[-1], R2[-1], R3[-1], R4[-1], R5[-1], R6[-1], R7[-1], R8[-1], CCl4[-1], CHCl3[-1], VCM[-1], T0[-1], T1[-1]]
        Y0j = [EDCj[-1], ECj[-1], HClj[-1], Cokej[-1], CPj[-1], Dij[-1], Trij[-1], C4H6Cl2j[-1], C6H6j[-1], C2H2j[-1], C11j[-1], C112j[-1], C1112j[-1], R1j[-1], R2j[-1], R3j[-1], R4j[-1], R5j[-1], R6j[-1], R7j[-1], R8j[-1], CCl4j[-1], CHCl3j[-1], VCMj[-1], T0j[-1], T1j[-1]]
        resa = solve_ivp(RHS, [0.0, Ls], Y0, method='Radau', args=(5900000000000000.0, 2200000000000.0, 13000000.0, 12000000.0, 1000000.0, 500000.0, 200000.0, 100000.0, 1000000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 17000000.0, 16000000.0, 91000.0, 120000000.0, 300000.0, 20000.0, 500000.0, 210000000000000.0, 500000000000000.0, 20000000000000.0, 25000000000000.0, 1000000.0, 500000.0, 500000.0, 10000000.0, 100000000.0, 160000000.0, 342.0 * 1000.0, 230000.0, 7000.0, 34000.0, 42000.0, 45000.0, 48000.0, 56000.0, 63000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 17000.0, 14000.0, 0.0, 56000.0, 61000.0, 30000.0, 31000.0, 84000.0, 90000.0, 70000.0, 70000.0, 33000.0, 33000.0, 33000.0, 13000.0, 20000.0, 70000.0, 8.31446261815324, float(c1_vals[-1]), float(c2_vals[-1]), float(c3_vals[-1]), float(amount_new), float(taui)), first_step=1E-1, max_step=10 / segment_second)  # rtol=1E-3, atol=1E-6  , first_step=1E-2, max_step=1E-3, jac= lambda Z, C: jacob(Z,C, **args), rtol=1E-9, atol=1E-9        Ls2 = firststepval
        resb = solve_ivp(RHS, [0.0, Ls], Y0j, method='Radau', args=(5900000000000000.0, 2200000000000.0, 13000000.0, 12000000.0, 1000000.0, 500000.0, 200000.0, 100000.0, 1000000.0, 10000000.0, 10000000.0, 17000000.0, 12000000.0, 17000000.0, 17000000.0, 16000000.0, 91000.0, 120000000.0, 300000.0, 20000.0, 500000.0, 210000000000000.0, 500000000000000.0, 20000000000000.0, 25000000000000.0, 1000000.0, 500000.0, 500000.0, 10000000.0, 100000000.0, 160000000.0, 342.0 * 1000.0, 230000.0, 7000.0, 34000.0, 42000.0, 45000.0, 48000.0, 56000.0, 63000.0, 13000.0, 12000.0, 4000.0, 6000.0, 15000.0, 17000.0, 14000.0, 0.0, 56000.0, 61000.0, 30000.0, 31000.0, 84000.0, 90000.0, 70000.0, 70000.0, 33000.0, 33000.0, 33000.0, 13000.0, 20000.0, 70000.0, 8.31446261815324, float(c1_valsj[-1]), float(c2_valsj[-1]), float(c3_valsj[-1]), float(amount_new), float(taui)), jac=Jacob, first_step=1E-1, max_step=10 / segment_second)  # first_step=1E-2, max_step=1E-3, jac= lambda Z, C: jacob(Z,C, **args), rtol=1E-9, atol=1E-9        Ls2 = firststepval
        edcint = initedc - resa.y[0][-1]
        edcintj = initedc - resb.y[0][-1]
        yield1 = resa.y[23][-1]/edcint
        yield1j = resb.y[23][-1]/edcintj
        yield_vcm.append(yield1)
        yield_vcmj.append(yield1j)
        selectivity_val = (resa.y[23][-1]/resa.y[2][-1])
        selectivity.append(selectivity_val)
        selectivity_val2 = (resa.y[2][-1]/resa.y[23][-1])
        selectivity2.append(selectivity_val2)
        selectivity_valj = (resb.y[23][-1]/resb.y[2][-1])
        selectivityj.append(selectivity_valj)
        selectivity_val2j = (resb.y[2][-1]/resb.y[23][-1])
        selectivity2j.append(selectivity_val2j)
        conversionedc = 100.0*(1.0 - resa.y[0][-1]/initedc)
        conversionedcj = 100.0*(1.0 - resb.y[0][-1]/initedc)
        conversion_EDCb.append(conversionedc)
        conversion_EDCjb.append(conversionedcj)
        EDC.append(float(resa.y[0][-1]))  # This is where all of the lists are appended with the last calculated value for each compound
        EC.append(float(resa.y[1][-1]))
        HCl.append(float(resa.y[2][-1]))
        Coke.append(float(resa.y[3][-1]))
        CP.append(float(resa.y[4][-1]))
        Di.append(float(resa.y[5][-1]))
        Tri.append(float(resa.y[6][-1]))
        C4H6Cl2.append(float(resa.y[7][-1]))
        C6H6.append(float(resa.y[8][-1]))
        C2H2.append(float(resa.y[9][-1]))
        C11.append(float(resa.y[10][-1]))
        C112.append(float(resa.y[11][-1]))
        C1112.append(float(resa.y[12][-1]))
        R1.append(float(resa.y[13][-1]))
        R2.append(float(resa.y[14][-1]))
        R3.append(float(resa.y[15][-1]))
        R4.append(float(resa.y[16][-1]))
        R5.append(float(resa.y[17][-1]))
        R6.append(float(resa.y[18][-1]))
        R7.append(float(resa.y[19][-1]))
        R8.append(float(resa.y[20][-1]))
        CCl4.append(float(resa.y[21][-1]))
        CHCl3.append(float(resa.y[22][-1]))
        VCM.append(float(resa.y[23][-1]))
        T0.append(float(resa.y[24][-1]))
        # print(resa.y[24][-1], resa.y[25][-1])
        # print(resb.y[24][-1], resb.y[25][-1])
        T1.append(float(resa.y[25][-1]))
        EDCj.append(float(resb.y[0][-1]))  # This is where all of the lists are appended with the last calculated value for each compound
        ECj.append(float(resb.y[1][-1]))
        HClj.append(float(resb.y[2][-1]))
        Cokej.append(float(resb.y[3][-1]))
        CPj.append(float(resb.y[4][-1]))
        Dij.append(float(resb.y[5][-1]))
        Trij.append(float(resb.y[6][-1]))
        C4H6Cl2j.append(float(resb.y[7][-1]))
        C6H6j.append(float(resb.y[8][-1]))
        C2H2j.append(float(resb.y[9][-1]))
        C11j.append(float(resb.y[10][-1]))
        C112j.append(float(resb.y[11][-1]))
        C1112j.append(float(resb.y[12][-1]))
        R1j.append(float(resb.y[13][-1]))
        R2j.append(float(resb.y[14][-1]))
        R3j.append(float(resb.y[15][-1]))
        R4j.append(float(resb.y[16][-1]))
        R5j.append(float(resb.y[17][-1]))
        R6j.append(float(resb.y[18][-1]))
        R7j.append(float(resb.y[19][-1]))
        R8j.append(float(resb.y[20][-1]))
        CCl4j.append(float(resb.y[21][-1]))
        CHCl3j.append(float(resb.y[22][-1]))
        VCMj.append(float(resb.y[23][-1]))
        T0j.append(float(resb.y[24][-1]))
        T1j.append(float(resb.y[25][-1]))
        Y1f = [resa.y[0][-1], resa.y[1][-1], resa.y[2][-1], resa.y[3][-1], resa.y[4][-1], resa.y[5][-1], resa.y[6][-1], resa.y[7][-1], resa.y[8][-1], resa.y[9][-1], resa.y[10][-1], resa.y[11][-1], resa.y[12][-1], resa.y[13][-1], resa.y[14][-1], resa.y[15][-1], resa.y[16][-1], resa.y[17][-1], resa.y[18][-1], resa.y[19][-1], resa.y[20][-1], resa.y[21][-1], resa.y[22][-1], resa.y[23][-1]]
        sumf = sum(Y1f)
        C_Total.append(sumf)
        Y1fj = [resb.y[0][-1], resb.y[1][-1], resb.y[2][-1], resb.y[3][-1], resb.y[4][-1], resb.y[5][-1], resb.y[6][-1], resb.y[7][-1], resb.y[8][-1], resb.y[9][-1], resb.y[10][-1], resb.y[11][-1], resb.y[12][-1], resb.y[13][-1], resb.y[14][-1], resb.y[15][-1], resb.y[16][-1], resb.y[17][-1], resb.y[18][-1], resb.y[19][-1], resb.y[20][-1], resb.y[21][-1], resb.y[22][-1], resb.y[23][-1]]
        sumfj = sum(Y1fj)
        C_Totalj.append(sumfj)
        P1f = [resa.y[1][-1], resa.y[2][-1], resa.y[3][-1], resa.y[4][-1], resa.y[5][-1], resa.y[6][-1], resa.y[7][-1], resa.y[8][-1], resa.y[9][-1], resa.y[10][-1], resa.y[11][-1], resa.y[12][-1], resa.y[13][-1], resa.y[14][-1], resa.y[15][-1], resa.y[16][-1], resa.y[17][-1], resa.y[18][-1], resa.y[19][-1], resa.y[20][-1], resa.y[21][-1], resa.y[22][-1], resa.y[23][-1]]
        P1fj = [resb.y[1][-1], resb.y[2][-1], resb.y[3][-1], resb.y[4][-1], resb.y[5][-1], resb.y[6][-1], resb.y[7][-1], resb.y[8][-1], resb.y[9][-1], resb.y[10][-1], resb.y[11][-1], resb.y[12][-1], resb.y[13][-1], resb.y[14][-1], resb.y[15][-1], resb.y[16][-1], resb.y[17][-1], resb.y[18][-1], resb.y[19][-1], resb.y[20][-1], resb.y[21][-1], resb.y[22][-1], resb.y[23][-1]]
        D1f = [resa.y[2][-1], resa.y[23][-1]]
        D1fj = [resb.y[2][-1], resb.y[23][-1]]
        prod1f = sum(P1f)
        prod1fj = sum(P1fj)
        des1f = sum(D1f)
        des1fj = sum(D1fj)
        pur1f = (des1f/prod1f) * 100
        pr1.append(pur1f)
        pur1jf = (des1fj/prod1fj) * 100
        pr1j.append(pur1jf)
        prev_eval = J_eval[0]  # This loads the previous number of jacobian calculations
        j_eval = resb.nfev + prev_eval  # This adds the previous to the most recent amount
        J_eval[0] = j_eval  # This stores them in the initial list from above
    EDCl.append([i for i in EDC])
    ECl.append([i for i in EC])
    HCll.append([i for i in HCl])
    Cokel.append([i for i in Coke])
    CPl.append([i for i in CP])
    Dil.append([i for i in Di])
    Tril.append([i for i in Tri])
    C4H6Cl2l.append([i for i in C4H6Cl2])
    C6H6l.append([i for i in C6H6])
    C2H2l.append([i for i in C2H2])
    C11l.append([i for i in C11])
    C112l.append([i for i in C112])
    C1112l.append([i for i in C1112])
    R1l.append([i for i in R1])
    R2l.append([i for i in R2])
    R3l.append([i for i in R3])
    R4l.append([i for i in R4])
    R5l.append([i for i in R5])
    R6l.append([i for i in R6])
    R7l.append([i for i in R7])
    R8l.append([i for i in R8])
    CCl4l.append([i for i in CCl4])
    CHCl3l.append([i for i in CHCl3])
    VCMl.append([i for i in VCM])
    T0l.append([i*float(1) for i in T0])
    T1l.append([i*float(1) for i in T1])
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
    Trilj.append([i for i in Trij])
    C4H6Cl2lj.append([i for i in C4H6Cl2j])
    C6H6lj.append([i for i in C6H6j])
    C2H2lj.append([i for i in C2H2j])
    C11lj.append([i for i in C11j])
    C112lj.append([i for i in C112j])
    C1112lj.append([i for i in C1112j])
    R1lj.append([i for i in R1j])
    R2lj.append([i for i in R2j])
    R3lj.append([i for i in R3j])
    R4lj.append([i for i in R4j])
    R5lj.append([i for i in R5j])
    R6lj.append([i for i in R6j])
    R7lj.append([i for i in R7j])
    R8lj.append([i for i in R8j])
    CCl4lj.append([i for i in CCl4j])
    CHCl3lj.append([i for i in CHCl3j])
    VCMlj.append([i for i in VCMj])
    T0lj.append([i*float(1) for i in T0j])
    T1lj.append([i*float(1) for i in T1j])
    RE_valsl.append([float(i) for i in RE_vals])
    U_coeffsl.append([float(i) for i in U_coeffs])
    h_valsl.append([float(i) for i in h_vals])
    kmix_valsl.append([float(i) for i in kmix_vals])
    c1_valsl.append([float(i) for i in c1_vals])
    c2_valsl.append([float(i) for i in c2_vals])
    c3_valsl.append([float(i) for i in c3_vals])
    RE_valsjl.append([float(i) for i in RE_valsj])
    U_coeffsjl.append([float(i) for i in U_coeffsj])
    h_valsjl.append([float(i) for i in h_valsj])
    kmix_valsjl.append([float(i) for i in kmix_valsj])
    c1_valsjl.append([float(i) for i in c1_valsj])
    c2_valsjl.append([float(i) for i in c2_valsj])
    c3_valsjl.append([float(i) for i in c3_valsj])
    conversion_EDCfb = [(1-(float(x)/initial_edc))*float(100) for x in EDC]  # Calculates the conversion of EDC (% basis) at each time interr
    conversion_EDCfbj = [(1-(float(y)/initial_edc))*float(100) for y in EDCj]
    conversion_EDCf.append(conversion_EDCfb[:])
    conversion_EDCfj.append(conversion_EDCfbj[:])
    conversion_EDCfb.clear()
    conversion_EDCfbj.clear()
    if initccl4 != 0:
        conversion_CCl4b = [(1-(float(x)/initccl4))*float(100) for x in CCl4]  # Calculates the conversion of CCl4 (% basis) at each time interr
        conversion_CCl4bj = [(1-(float(y)/initccl4))*float(100) for y in CCl4j]
        conversion_CCl4.append(conversion_CCl4b[:])
        conversion_CCl4j.append(conversion_CCl4bj[:])
        conversion_CCl4b.clear()
        conversion_CCl4bj.clear()
    puref.append(pr1[:])
    purefj.append(pr1j[:])
    EDC.clear()
    EC.clear()
    HCl.clear()
    Coke.clear()
    CP.clear()
    Di.clear()
    Tri.clear()
    C4H6Cl2.clear()
    C6H6.clear()
    C2H2.clear()
    C11.clear()
    C112.clear()
    C1112.clear()
    R1.clear()
    R2.clear()
    R3.clear()
    R4.clear()
    R5.clear()
    R6.clear()
    R7.clear()
    R8.clear()
    CCl4.clear()
    CHCl3.clear()
    VCM.clear()
    T0.clear()
    T1.clear()
    EDCj.clear()
    ECj.clear()
    HClj.clear()
    Cokej.clear()
    CPj.clear()
    Dij.clear()
    Trij.clear()
    C4H6Cl2j.clear()
    C6H6j.clear()
    C2H2j.clear()
    C11j.clear()
    C112j.clear()
    C1112j.clear()
    R1j.clear()
    R2j.clear()
    R3j.clear()
    R4j.clear()
    R5j.clear()
    R6j.clear()
    R7j.clear()
    R8j.clear()
    CCl4j.clear()
    CHCl3j.clear()
    VCMj.clear()
    T0j.clear()
    T1j.clear()
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
    selectivity.clear()
    selectivity2.clear()
    selectivityj.clear()
    selectivity2j.clear()
    # EDC = [float(initedc)]
    # EC = [float(0.0)]
    # HCl = [float(0.0)]
    # Coke = [float(0.0)]
    # CP = [float(0.0)]
    # Di = [float(0.0)]
    # C4H6Cl2 = [float(0.0)]
    # C6H6 = [float(0.0)]
    # C2H2 = [float(0.0)]
    # C11 = [float(0.0)]
    # C112 = [float(0.0)]
    # R1 = [float(0.0)]
    # R2 = [float(0.0)]
    # R3 = [float(0.0)]
    # R4 = [float(0.0)]
    # R5 = [float(0.0)]
    # R6 = [float(0.0)]
    # VCM = [float(0.0)]
    # T0 = []
    # T1 = [float(0.0)]
    # EDCj = [float(initedc)]
    # ECj = [float(0.0)]
    # HClj = [float(0.0)]
    # Cokej = [float(0.0)]
    # CPj = [float(0.0)]
    # Dij = [float(0.0)]
    # C4H6Cl2j = [float(0.0)]
    # C6H6j = [float(0.0)]
    # C2H2j = [float(0.0)]
    # C11j = [float(0.0)]
    # C112j = [float(0.0)]
    # R1j = [float(0.0)]
    # R2j = [float(0.0)]
    # R3j = [float(0.0)]
    # R4j = [float(0.0)]
    # R5j = [float(0.0)]
    # R6j = [float(0.0)]
    # VCMj = [float(0.0)]
    # T0j = []
    # T1j = [float(0.0)]
    selectivity = [1.0]
    selectivity2 = [1.0]
    selectivityj = [1.0]
    selectivity2j = [1.0]
    yield_vcm = [1.0]
    yield_vcmj = [1.0]
    EDC = [float(initedc)]
    EC = [float(0.0)]
    HCl = [float(0.0)]
    Coke = [float(0.0)]  # Coke is written as simply C in the paper
    CP = [float(0.0)]
    Di = [float(0.0)]
    Tri = [float(0.0)]
    C4H6Cl2 = [float(0.0)]
    C6H6 = [float(0.0)]
    C2H2 = [float(0.0)]
    C11 = [float(0.0)]
    C112 = [float(0.0)]
    C1112 = [float(0.0)]
    R1 = [float(0.0)]
    R2 = [float(0.0)]
    R3 = [float(0.0)]
    R4 = [float(0.0)]
    R5 = [float(0.0)]
    R6 = [float(0.0)]
    R7 = [float(0.0)]
    R8 = [float(0.0)]
    CCl4 = [float(initccl4)]
    CHCl3 = [float(0.0)]
    VCM = [float(0.0)]
    T0 = []
    T1 = [float(0.0)]
    EDCj = [float(initedc)]
    ECj = [float(0.0)]
    HClj = [float(0.0)]
    Cokej = [float(0.0)]  # Coke is written as simply C in the paper
    CPj = [float(0.0)]
    Dij = [float(0.0)]
    Trij = [float(0.0)]
    C4H6Cl2j = [float(0.0)]
    C6H6j = [float(0.0)]
    C2H2j = [float(0.0)]
    C11j = [float(0.0)]
    C112j = [float(0.0)]
    C1112j = [float(0.0)]
    R1j = [float(0.0)]
    R2j = [float(0.0)]
    R3j = [float(0.0)]
    R4j = [float(0.0)]
    R5j = [float(0.0)]
    R6j = [float(0.0)]
    R7j = [float(0.0)]
    R8j = [float(0.0)]
    CCl4j = [float(initccl4)]
    CHCl3j = [float(0.0)]
    VCMj = [float(0.0)]
    T0j = []
    T1j = [float(0.0)]
    pr1.clear()
    pr1j.clear()
    pr1.append(100.0)
    pr1j.append(100.0)
    rtolval = 1E-7
    rtolvalj = 1E-7
    atolval = 1E-7
    atolvalj = 1E-7
    Ls2 = firststepval

eaconvf = []
eaconvjf = []
for i, j in enumerate(conversion_EDCf):
    val1 = conversion_EDCf[i]
    val2 = val1[-1]
    eaconvf.append(val2)

for i2, j2 in enumerate(conversion_EDCfj):
    val1j = conversion_EDCfj[i2]
    val2j = val1j[-1]
    eaconvjf.append(val2j)

xD = np.linspace(0, u_z*desired_time, segment_num+1)
timeD = np.linspace(0, desired_time, segment_num+1)
Temp_valsrev = Temp_vals.copy()
Temp_valsrev.reverse()
Temp_valsrev2 = Temp_vals2.copy()
Temp_valsrev2.reverse()

splist = ["Conversion", "Purity", 'Yield VCM']
splistb = ["Conversion [%]", "Purity [%]", r'Yield [$Y_{VCM}$]']
selist = ['Selectivity VCM', 'Selectivity HCl']
sulist = [r'Selectivity [$S_{VCM}$]', r'Selectivity [$S_{HCl}$]']
telist = ['Temperature', 'Temperature Differential', "dT"]
tilist = [r"Temperature [$K$]", r"$\frac{d{T}}{d{Z}}$ [K]", r"$\frac{d{T}}{d{Z}}$ [K]"]
cslist = ['Overall Heat Transfer Coefficient', 'Heat Transfer Coefficient', 'Mixture Thermal Conductivity', 'Constant 1', 'Constant 2', 'Constant 3']
csunits = [r"U [$\frac{W}{m^2 K}$]", r"h [$\frac{W}{m^2 K}$]", r"[$\frac{W}{m K}$]", r"k [$\frac{1}{m}$]", r"[$\frac{K}{m^2}$]", r"[$\frac{s m K}{mol}$]"]
textlist1 = [EDCl, ECl, HCll, Cokel, CPl, Dil, Tril, C4H6Cl2l, C6H6l, C2H2l, C11l, C112l, C1112l, R1l, R2l, R3l, R4l, R5l, R6l, R7l, R8l, CCl4l, CHCl3l, VCMl, T0l, T1l, puref, RE_valsl, U_coeffsl, h_valsl, kmix_valsl, c1_valsl, c2_valsl, c3_valsl, conversion_EDCf, selectivityl, selectivity2l, yield_vcml]
textlist2 = [EDClj, EClj, HCllj, Cokelj, CPlj, Dilj, Trilj, C4H6Cl2lj, C6H6lj, C2H2lj, C11lj, C112lj, C1112lj, R1lj, R2lj, R3lj, R4lj, R5lj, R6lj, R7lj, R8lj, CCl4lj, CHCl3lj, VCMlj, T0lj, T1lj, purefj, RE_valsjl, U_coeffsjl, h_valsjl, kmix_valsjl, c1_valsjl, c2_valsjl, c3_valsjl, conversion_EDCfj, selectivityjl, selectivity2jl, yield_vcmjl]
namestxt = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', r'$C_{4}H_{6}Cl_{2}$', r'$C_{6}H_{6}$', r'$C_{2}H_{2}$', 'C11', 'C112', 'C1112', r'$R_{1}$', r'$R_{2}$', r'$R_{3}$', r'$R_{4}$', r'$R_{5}$', r'$R_{6}$', r'$R_{7}$', r'$R_{8}$', r'$CCl_{4}$', r'$CHCl_{3}$', 'VCM', 'Temperature', 'Temperature Differential', 'Purity', 'Reynolds Number', 'Overall Heat Transfer Coefficient', 'Heat Transfer Coefficient', 'Mixture Thermal Conductivity', 'Constant 1', 'Constant 2', 'Constant 3', "Conversion", 'Selectivity VCM', 'Selectivity HCl', 'Yield VCM']
namestxte = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', r'$C_{4}H_{6}Cl_{2}$', r'$C_{6}H_{6}$', r'$C_{2}H_{2}$', 'C11', 'C112', 'C1112', r'$R_{1}$', r'$R_{2}$', r'$R_{3}$', r'$R_{4}$', r'$R_{5}$', r'$R_{6}$', r'$R_{7}$', r'$R_{8}$', r'$CCl_{4}$', r'$CHCl_{3}$', 'VCM', 'Temperature', 'dT', 'Purity', 'Reynolds Number', 'U Coefficient', 'H Coefficient', 'K Coefficient', 'Constant 1', 'Constant 2', 'Constant 3', "Conversion", 'Selectivity VCM', 'Selectivity HCl', 'Yield VCM']
namessave = ['EDC', 'EC', 'HCl', 'Coke', 'CP', 'Di', 'Tri', 'C4H6Cl2', 'C6H6', 'C2H2', 'C11', 'C112', 'C1112', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'CCl4', 'CHCl3', 'VCM', 'Temperature', 'dT', 'Purity', 'Reynolds Number', 'U Coefficient', 'H Coefficient', 'Thermal Conductivity', 'Constant 1', 'Constant 2', 'Constant 3', "Conversion", 'Selectivity VCM', 'Selectivity HCl', 'Yield VCM']
edc, ec, hcl, cc, cp1, di, tri, c4h6cl2, c6h6, c2h2, c11, c112, c1112, r1, r2, r3, r4, r5, r6, r7, r8, chcl3l, ccl4l, vcm, t0, dt, purel, revals, uvals, hvals, kvals, c1vals, c2vals, c3vals, convedc, selectvcm, selecthcl, yieldvcm = [np.array(ia, ndmin=1) for ia in textlist1]
edcj, ecj, hclj, ccj, cp1j, dij, trij, c4h6cl2j, c6h6j, c2h2j, c11j, c112j, c1112j, r1j, r2j, r3j, r4j, r5j, r6j, r7j, r8j, chcl3lj, ccl4lj, vcmj, t0j, dTj, purejl, revalsj, uvalsj, hvalsj, kvalsj, c1valsj, c2valsj, c3valsj, convedcj, selectvcmj, selecthclj, yieldvcmj = [np.array(ib, ndmin=1) for ib in textlist2]
concl = [edc, ec, hcl, cc, cp1, di, tri, c4h6cl2, c6h6, c2h2, c11, c112, c1112, r1, r2, r3, r4, r5, r6, r7, r8, chcl3l, ccl4l, vcm, t0, dt, purel, revals, uvals, hvals, kvals, c1vals, c2vals, c3vals, convedc, selectvcm, selecthcl, yieldvcm]
concjl = [edcj, ecj, hclj, ccj, cp1j, dij, trij, c4h6cl2j, c6h6j, c2h2j, c11j, c112j, c1112j, r1j, r2j, r3j, r4j, r5j, r6j, r7j, r8j, chcl3lj, ccl4lj, vcmj, t0j, dTj, purejl, revalsj, uvalsj, hvalsj, kvalsj, c1valsj, c2valsj, c3valsj, convedcj, selectvcmj, selecthclj, yieldvcmj]
eavals, eavalsr = np.array(Temp_vals, ndmin=1), np.array(Temp_valsrev, ndmin=1)

convedcja = np.array(conversion_EDCfj)
np.savetxt(r"{}\Conversion_EDC J.txt".format(path_fol), convedcja)
convedc = np.array(conversion_EDCf)
np.savetxt(r"{}\Conversion_EDC.txt".format(path_fol), convedc)
eavals = np.array(Temp_vals, ndmin=1)
np.savetxt(r"{}\Tvals.txt".format(path_fol), eavals)
eavalsr = np.array(eavalsr, ndmin=1)
np.savetxt(r"{}\Tvals Reverse.txt".format(path_fol), eavalsr)
eaendc = np.array(eaconvf, ndmin=1)
np.savetxt(r"{}\FinalConversion.txt".format(path_fol), eaendc)
eaendcj = np.array(eaconvjf, ndmin=1)
np.savetxt(r"{}\FinalConversion J.txt".format(path_fol), eaendcj)
tvals = np.array(Temp_vals, ndmin=1)
np.savetxt(r"{}\Temperature vals.txt".format(path_fol), tvals)
tvalsr = np.array(Temp_valsrev, ndmin=1)
np.savetxt(r"{}\Temperature vals Reverse.txt".format(path_fol), tvalsr)


for i in range(len(namestxt)):
    path_folna = r"{}\Temperature - Full\{}".format(dir_path, namessave[i])
    try:
        os.mkdir(path_folna)
    except Exception:
        pass
    path_folni = r"{}\{}".format(path_folna, namessave[i])
    try:
        os.mkdir(path_folni)
    except Exception:
        pass
    n1 = np.array(textlist1[i])
    np.savetxt(r"{}\{}.txt".format(path_folni, namestxt[i]), n1, fmt='%s')
    path_folnj = r"{}\{} - J".format(path_folna, namessave[i])
    try:
        os.mkdir(path_folnj)
    except Exception:
        pass
    n2 = np.array(textlist2[i])
    np.savetxt(r"{}\{} - J.txt".format(path_folnj, namestxt[i]), n2)

for i, j in enumerate(textlist1):
    namel = namessave[i]
    namele = namestxte[i]
    path_folnd = r"{}\Temperature - Full\{}".format(dir_path, namel)
    try:
        os.mkdir(path_folnd)
    except Exception:
        pass
    a1 = np.asarray(textlist1[i])
    a1j = np.asarray(textlist2[i])
    a2 = a1.astype(np.longdouble)
    a2j = a1j.astype(np.longdouble)
    lista = textlist1[i]
    listb = textlist2[i]
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
    for jj, k in enumerate(lista):
        ll = np.asarray(lista[jj])
        llf = ll.astype(np.longdouble)
        ad.append(llf)
        at.append(llf)
    for jj, k in enumerate(listb):
        llj = np.asarray(listb[jj])
        llfj = llj.astype(np.longdouble)
        adj.append(llfj)
        atj.append(llfj)
    eea = Temp_vals[:]
    eet = Temp_vals[:]
    eead = Temp_vals[:]
    eav = np.asarray(eea)
    eas = ["Temperature {} K".format(ij) for ij in Temp_vals]
    eaf = ["Temperatures"]
    eaf.append(eas)
    ttv = ["Time [s]"]
    ttv2 = ["Time Interval: {} [s]".format(time_int)]
    ddv = ["Distance [m]"]
    ddv2 = ["Velocity: {} [m/s]".format(u_z)]
    ddvd = ["Distance [m]"]
    ddv2d = ["Velocity: {} [m/s]".format(u_z)]
    for i, j in enumerate(Temp_vals):
        eaf.append(eas[i])
        ttv.append(eas[i])
        ddv.append(eas[i])
        ddvd.append(eas[i])
        ttv2.append(Temp_vals[i])
        ddv2.append(Temp_vals[i])
        ddv2d.append(Temp_vals[i])
    indexd = pd.Index(ddv, name='Units')
    indext = pd.Index(ttv, name='Units')
    indexdd = pd.Index(ddvd, name='Units')
    eat = np.asarray(eet)
    df = pd.DataFrame(ad, index=[*ddv])
    df.index.name = 'Units'
    edv = pd.DataFrame(ddv2, index=[*ddv])
    edv.index.name = 'Units'
    frames = [edv, df]
    result1 = pd.concat(frames, join='outer', axis=1, ignore_index=True)
    dt = pd.DataFrame(at, index=[*ttv])
    dt.index.name = 'Units'
    etv = pd.DataFrame(ttv2, index=[*ttv])
    etv.index.name = 'Units'
    framest = [etv, dt]
    result2 = pd.concat(framest, join='outer', axis=1, ignore_index=True)
    dfD = pd.DataFrame(ad2, index=[*ddv])
    dfD.index.name = 'Units'
    edvd = pd.DataFrame(ddv2d, index=[*ddvd])
    edvd.index.name = 'Units'
    path_folnnf = r"{}\{}".format(path_folnd, namel)
    try:
        os.mkdir(path_folnnf)
    except Exception:
        pass
    result1.to_excel(r"{}\{} Dist.xlsx".format(path_folnnf, namele), sheet_name="{} Dist.xlsx".format(namele))
    result2.to_excel(r"{}\{} Time.xlsx".format(path_folnnf, namele), sheet_name="{} Time.xlsx".format(namele))
    dfj = pd.DataFrame(adj, index=[*ddv])
    dtj = pd.DataFrame(atj, index=[*ttv])
    dfDd = pd.DataFrame(ad2, index=[*ddvd])
    edvj = pd.DataFrame(ddv2, index=[*ddv])
    edvj.index.name = 'Units'
    etvj = pd.DataFrame(ttv2, index=[*ttv])
    etvj.index.name = 'Units'
    framesj = [edvj, dfj]
    framestj = [etvj, dtj]
    ans1j = pd.concat(framesj, join='outer', axis=1, ignore_index=True)
    ans2j = pd.concat(framestj, join='outer', axis=1, ignore_index=True)
    path_folnnjj = r"{}\{}".format(path_folnd, "{} - J".format(namel))
    try:
        os.mkdir(path_folnnjj)
    except Exception:
        pass
    ans1j.to_excel(r"{}\{} J Dist.xlsx".format(path_folnnjj, namele), sheet_name="{} J Dist.xlsx".format(namele))
    ans2j.to_excel(r"{}\{} J Time.xlsx".format(path_folnnjj, namele), sheet_name="{} J Time.xlsx".format(namele))


viridis = cm.get_cmap('viridis', iternum + 1)

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        'ha': 'center',
        'va': 'bottom'}


for i, ja in enumerate(concl):
    nameD = namessave[i]
    nameD2 = namestxte[i]
    path_folnn = r"{}\Temperature - Full\{}\{}".format(dir_path, namessave[i], namessave[i])
    try:
        os.mkdir(path_folnn)
    except Exception:
        pass
    cur_list = concl[i]
    for jj, j in enumerate(graph_nodes):
        fig = plt.figure()
        cur_list2 = concl[i][j]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[j])
        index_jj = int(j)
        Cval = KtoC(Temp_vals[j])
        plt.plot(dist, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[j])))
        plt.legend(loc='best')
        if nameD in splist:
            sp = splist.index(nameD)
            plt.axhline(y=float(100.0), color='k', linestyle='--')
            plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
            plt.ylabel(splistb[sp])
            plt.title(nameD, fontdict=font)
        elif nameD in selist:
            si = selist.index(nameD)
            plt.axhline(y=float(1.0), color='k', linestyle='--')
            plt.ylabel(sulist[si])
            plt.title(selist[si], fontdict=font)
        elif nameD in telist:
            tii = telist.index(nameD)
            ty = np.array(concl[i])
            plt.axhline(y=ty[j, 0], color='k', linestyle='--')
            plt.ylabel(tilist[tii])
            plt.title(r'{} Profile'.format(nameD2), fontdict=font)
        elif nameD in cslist:
            ni = cslist.index(nameD)
            plt.ylabel(csunits[ni])
            plt.title(r'{}'.format(cslist[ni]), fontdict=font)
        else:
            plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
            plt.title(r'{} Concentration'.format(nameD2), fontdict=font)
        plt.grid()
        plt.xlabel(r'Distance [$m$]')
        fig.savefig(r"{}\{} Concentration {}.pdf".format(path_folnn, namessave[i], int(Temp_vals[j])))
        fig.savefig(r"{}\{} Concentration {}.svg".format(path_folnn, namessave[i], int(Temp_vals[j])))
        plt.close()
    for jt, tjt in enumerate(graph_nodes):
        fig2 = plt.figure()
        cur_list2 = concl[i][jt]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[jt])
        index_jj = int(jt)
        Cval = KtoC(Temp_vals[jt])
        plt.plot(timel, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[tjt])))
        plt.legend(loc='best')
        if nameD in splist:
            sp = splist.index(nameD)
            plt.axhline(y=float(100.0), color='k', linestyle='--')
            plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
            plt.ylabel(splistb[sp])
            plt.title(nameD, fontdict=font)
        elif nameD in selist:
            si = selist.index(nameD)
            plt.axhline(y=float(1.0), color='k', linestyle='--')
            plt.ylabel(sulist[si])
            plt.title(selist[si], fontdict=font)
        elif nameD in telist:
            tii = telist.index(nameD)
            ty = np.array(concl[i])
            plt.axhline(y=ty[j, 0], color='k', linestyle='--')
            plt.ylabel(tilist[tii])
            plt.title(r'{} Profile'.format(nameD2), fontdict=font)
        elif nameD in cslist:
            ni = cslist.index(nameD)
            plt.ylabel(csunits[ni])
            plt.title(r'{}'.format(cslist[ni]), fontdict=font)
        else:
            plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
            plt.title(r'{} Concentration'.format(nameD2), fontdict=font)
        plt.grid()
        plt.xlabel(r'Time [$\it{s}$]')
        fig2.savefig(r"{}\{} Concentration Time {}.pdf".format(path_folnn, namessave[i], int(Temp_vals[tjt])))
        fig2.savefig(r"{}\{} Concentration Time {}.svg".format(path_folnn, namessave[i], int(Temp_vals[tjt])))
        plt.close()

print("Individual species graph loop complete.")


for i, jb in enumerate(concl):
    cur_list = concl[i]
    nameD = namessave[i]
    nameD2 = namestxte[i]
    path_folnnb = r"{}\Temperature - Full\{}\{}".format(dir_path, namessave[i], namessave[i])
    try:
        os.mkdir(path_folnnb)
    except Exception:
        pass

    fig = plt.figure()
    for jj, j in enumerate(graph_nodes2):
        cur_list2 = concl[i][j]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[j])
        index_jj = int(j)
        Cval = KtoC(Temp_vals[j])
        plt.plot(dist, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[j])))
    plt.legend(loc='best')
    if nameD in splist:
        sp = splist.index(nameD)
        plt.axhline(y=float(100.0), color='k', linestyle='--')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
        plt.ylabel(splistb[sp])
        plt.title(nameD, fontdict=font)
    elif nameD in selist:
        si = selist.index(nameD)
        plt.axhline(y=float(1.0), color='k', linestyle='--')
        plt.ylabel(sulist[si])
        plt.title(selist[si], fontdict=font)
    elif nameD in telist:
        tii = telist.index(nameD)
        ty = np.array(concl[i])
        plt.axhline(y=ty[j, 0], color='k', linestyle='--')
        plt.ylabel(tilist[tii])
        plt.title(r'{} Profile'.format(nameD2), fontdict=font)
    elif nameD in cslist:
        ni = cslist.index(nameD)
        plt.ylabel(csunits[ni])
        plt.title(r'{}'.format(cslist[ni]), fontdict=font)
    else:
        plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
        plt.title(r'{} Concentration'.format(nameD2), fontdict=font)
    plt.grid()
    plt.xlabel(r'Distance [$m$]')
    fig.savefig(r"{}\{} Concentration.pdf".format(path_folnnb, namessave[i]))
    fig.savefig(r"{}\{} Concentration.svg".format(path_folnnb, namessave[i]))
    plt.close()
    fig2 = plt.figure()
    for jt, tjt in enumerate(graph_nodes):
        cur_list2 = concl[i][jt]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[jt])
        index_jj = int(jt)
        Cval = KtoC(Temp_vals[jt])
        plt.plot(timel, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[tjt])))
    plt.legend(loc='best')
    if nameD in splist:
        sp = splist.index(nameD)
        plt.axhline(y=float(100.0), color='k', linestyle='--')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
        plt.ylabel(splistb[sp])
        plt.title(nameD, fontdict=font)
    elif nameD in selist:
        si = selist.index(nameD)
        plt.axhline(y=float(1.0), color='k', linestyle='--')
        plt.ylabel(sulist[si])
        plt.title(selist[si], fontdict=font)
    elif nameD in telist:
        tii = telist.index(nameD)
        ty = np.array(concl[i])
        plt.axhline(y=ty[j, 0], color='k', linestyle='--')
        plt.ylabel(tilist[tii])
        plt.title(r'{} Profile'.format(nameD2), fontdict=font)
    elif nameD in cslist:
        ni = cslist.index(nameD)
        plt.ylabel(csunits[ni])
        plt.title(r'{}'.format(cslist[ni]), fontdict=font)
    else:
        plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
        plt.title(r'{} Concentration'.format(nameD2), fontdict=font)
    plt.grid()
    plt.xlabel(r'Time [$\it{s}$]')
    fig2.savefig(r"{}\{} Concentration Time {}.pdf".format(path_folnnb, namessave[i], int(Temp_vals[tjt])))
    fig2.savefig(r"{}\{} Concentration Time {}.svg".format(path_folnnb, namessave[i], int(Temp_vals[tjt])))
    plt.close()


print("Overall species graph loop complete.")

# Jacobian Graphs


for i, jc in enumerate(concjl):
    cur_list = concjl[i]
    nameDj = namessave[i]
    nameD2j = namestxte[i]
    path_folnnj = r"{}\Temperature - Full\{}\{}".format(dir_path, namessave[i], "{} - J".format(nameDj))
    try:
        os.mkdir(path_folnnj)
    except Exception:
        pass
    for jj, j in enumerate(graph_nodes):
        fig = plt.figure()
        cur_list2 = concjl[i][j]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[j])
        index_jj = int(j)
        Cval = KtoC(Temp_vals[j])
        plt.plot(dist, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[j])))
        plt.legend(loc='best')
        if nameDj in splist:
            sp = splist.index(nameDj)
            plt.axhline(y=float(100.0), color='k', linestyle='--')
            plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
            plt.ylabel(splistb[sp])
            plt.title(nameDj, fontdict=font)
        elif nameDj in selist:
            si = selist.index(nameDj)
            plt.axhline(y=float(1.0), color='k', linestyle='--')
            plt.ylabel(sulist[si])
            plt.title(selist[si], fontdict=font)
        elif nameDj in telist:
            tii = telist.index(nameDj)
            ty = np.array(concl[i])
            plt.axhline(y=ty[j, 0], color='k', linestyle='--')
            plt.ylabel(tilist[tii])
            plt.title(r'{} Profile'.format(nameD2j), fontdict=font)
        elif nameDj in cslist:
            ni = cslist.index(nameDj)
            plt.ylabel(csunits[ni])
            plt.title(r'{}'.format(cslist[ni]), fontdict=font)
        else:
            plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
            plt.title(r'{} Concentration - Jacobian'.format(nameD2j), fontdict=font)
        plt.grid()
        plt.xlabel(r'Distance [$m$]')
        fig.savefig(r"{}\{} Concentration J-{}.pdf".format(path_folnnj, namessave[i], int(Temp_vals[j])))
        fig.savefig(r"{}\{} Concentration J-{}.svg".format(path_folnnj, namessave[i], int(Temp_vals[j])))
        plt.close()
    for jtj, tjt in enumerate(graph_nodes):
        fig2 = plt.figure()
        cur_list2 = concjl[i][tjt]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[tjt])
        index_jj = int(j)
        Cval = KtoC(Temp_vals[tjt])
        plt.plot(timel, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[tjt])))
        plt.legend(loc='best')
        if nameDj in splist:
            sp = splist.index(nameDj)
            plt.axhline(y=float(100.0), color='k', linestyle='--')
            plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
            plt.ylabel(splistb[sp])
            plt.title(nameDj, fontdict=font)
        elif nameDj in selist:
            si = selist.index(nameDj)
            plt.axhline(y=float(1.0), color='k', linestyle='--')
            plt.ylabel(sulist[si])
            plt.title(selist[si], fontdict=font)
        elif nameDj in telist:
            tii = telist.index(nameDj)
            ty = np.array(concl[i])
            plt.axhline(y=ty[j, 0], color='k', linestyle='--')
            plt.ylabel(tilist[tii])
            plt.title(r'{} Profile'.format(nameD2j), fontdict=font)
        elif nameDj in cslist:
            ni = cslist.index(nameDj)
            plt.ylabel(csunits[ni])
            plt.title(r'{}'.format(cslist[ni]), fontdict=font)
        else:
            plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
            plt.title(r'{} Concentration - Jacobian'.format(nameD2j), fontdict=font)
        plt.grid()
        plt.xlabel(r'Time [s]')
        fig2.savefig(r"{}\{} Concentration Time J {}.pdf".format(path_folnnj, namessave[i], int(Temp_vals[tjt])))
        fig2.savefig(r"{}\{} Concentration Time J {}.svg".format(path_folnnj, namessave[i], int(Temp_vals[tjt])))
        plt.close()

print("Individual jacobian species graph loop complete.")


for i, jd in enumerate(concjl):
    cur_list = concjl[i]
    nameDj = namessave[i]
    nameD2j = namestxte[i]
    path_folnnjb = r"{}\Temperature - Full\{}\{}".format(dir_path, namessave[i], "{} - J".format(nameDj))
    try:
        os.mkdir(path_folnnjb)
    except Exception:
        pass

    fig = plt.figure()
    for jj, j in enumerate(graph_nodes2):
        cur_list2 = concjl[i][j]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[j])
        index_jj = int(j)
        Cval = KtoC(Temp_vals[j])
        plt.plot(dist, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[j])))
    plt.legend(loc='best')
    if nameDj in splist:
        sp = splist.index(nameDj)
        plt.axhline(y=float(100.0), color='k', linestyle='--')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
        plt.ylabel(splistb[sp])
        plt.title(nameDj, fontdict=font)
    elif nameDj in selist:
        si = selist.index(nameDj)
        plt.axhline(y=float(1.0), color='k', linestyle='--')
        plt.ylabel(sulist[si])
        plt.title(selist[si], fontdict=font)
    elif nameDj in telist:
        tii = telist.index(nameDj)
        ty = np.array(concl[i])
        plt.axhline(y=ty[j, 0], color='k', linestyle='--')
        plt.ylabel(tilist[tii])
        plt.title(r'{} Profile'.format(nameDj), fontdict=font)
    elif nameDj in cslist:
        ni = cslist.index(nameDj)
        plt.ylabel(csunits[ni])
        plt.title(r'{}'.format(cslist[ni]), fontdict=font)
    else:
        plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
        plt.title(r'{} Concentration - Jacobian'.format(nameD2j), fontdict=font)
    plt.grid()
    plt.xlabel(r'Distance [$m$]')
    fig.savefig(r"{}\{} J.pdf".format(path_folnnjb, namessave[i]))
    fig.savefig(r"{}\{} J.svg".format(path_folnnjb, namessave[i]))
    plt.close()
    fig2 = plt.figure()
    for jtj, tjt in enumerate(graph_nodes):
        cur_list2 = concjl[i][tjt]
        if len(xD) != len(cur_list2):
            dist = xD[0:-1]
            timel = timeD[0:-1]
        elif len(xD) == len(cur_list2):
            dist = xD
            timel = timeD
        tvalg = str(Temp_vals[tjt])
        index_jj = int(j)
        Cval = KtoC(Temp_vals[tjt])
        plt.plot(timel, cur_list2, color=viridis.colors[index_jj, :], label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[tjt])))
    plt.legend(loc='best')
    if nameDj in splist:
        sp = splist.index(nameDj)
        plt.axhline(y=float(100.0), color='k', linestyle='--')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100, decimals=None, symbol="%", is_latex=True))
        plt.ylabel(splistb[sp])
        plt.title(nameDj, fontdict=font)
    elif nameDj in selist:
        si = selist.index(nameDj)
        plt.axhline(y=float(1.0), color='k', linestyle='--')
        plt.ylabel(sulist[si])
        plt.title(selist[si], fontdict=font)
    elif nameDj in telist:
        tii = telist.index(nameDj)
        ty = np.array(concl[i])
        plt.axhline(y=ty[j, 0], color='k', linestyle='--')
        plt.ylabel(tilist[tii])
        plt.title(r'{} Profile'.format(nameDj), fontdict=font)
    elif nameDj in cslist:
        ni = cslist.index(nameDj)
        plt.ylabel(csunits[ni])
        plt.title(r'{}'.format(cslist[ni]), fontdict=font)
    else:
        plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
        plt.title(r'{} Concentration - Jacobian'.format(nameD2j), fontdict=font)
    plt.grid()
    plt.xlabel(r'Time [s]')
    fig2.savefig(r"{}\{} Concentration Time J {}.pdf".format(path_folnnjb, namessave[i], int(Temp_vals[tjt])))
    fig2.savefig(r"{}\{} Concentration Time J {}.svg".format(path_folnnjb, namessave[i], int(Temp_vals[tjt])))
    plt.close()

print("Overall species jacobian graph loop complete.")


path_folrads = r"{}\Temperature - Full\{}".format(dir_path, "Radicals")
try:
    os.mkdir(path_folrads)
except Exception:
    pass

for jj, j in enumerate(graph_nodes):
    fig = plt.figure()
    index_j = int(j)
    plt.plot(xD, r1[j, :], 'b--', label=r'$R_{1}$')
    plt.plot(xD, r2[j, :], 'g--', label=r'$R_{2}$')
    plt.plot(xD, r3[j, :], 'r--', label=r'$R_{3}$')
    plt.plot(xD, r4[j, :], 'c--', label=r'$R_{4}$')
    plt.plot(xD, r5[j, :], 'y--', label=r'$R_{5}$')
    plt.plot(xD, r6[j, :], 'm--', label=r'$R_{6}$')
    plt.plot(xD, r7[j, :], 'k--', label=r'$R_{7}$')
    plt.plot(xD, r8[j, :], 'o--', label=r'$R_{8}$')
    plt.xlabel(r'Distance [$m$]')
    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
    plt.legend(loc='best')
    plt.title(r"Radicals Concentration at {} {}".format(Temp_vals[j], r"[$\frac{kJ}{mol}$]"))
    plt.grid()
    fig.savefig(r"{}\Radicals {} kJ.pdf".format(path_folrads, Temp_vals[j]), bbox_inches='tight')
    fig.savefig(r"{}\Radicals {} kJ.svg".format(path_folrads, Temp_vals[j]), bbox_inches='tight')
    plt.close()


for jj, j in enumerate(graph_nodes):
    fig = plt.figure()
    index_j = int(j)
    plt.plot(xD, r1j[j, :], 'b--', label=r'$R_{1}$')
    plt.plot(xD, r2j[j, :], 'g--', label=r'$R_{2}$')
    plt.plot(xD, r3j[j, :], 'r--', label=r'$R_{3}$')
    plt.plot(xD, r4j[j, :], 'c--', label=r'$R_{4}$')
    plt.plot(xD, r5j[j, :], 'y--', label=r'$R_{5}$')
    plt.plot(xD, r6j[j, :], 'm--', label=r'$R_{6}$')
    plt.plot(xD, r7j[j, :], 'k--', label=r'$R_{7}$')
    plt.plot(xD, r8j[j, :], 'o--', label=r'$R_{8}$')
    plt.xlabel(r'Distance [$m$]')
    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
    plt.legend(loc='best')
    plt.title(r"Radicals Concentration at {} {}".format(Temp_vals[j], r"[$\frac{kJ}{mol}$]"))
    plt.grid()
    fig.savefig(r"{}\Radicals J {} kJ.pdf".format(path_folrads, Temp_vals[j]), bbox_inches='tight')
    fig.savefig(r"{}\Radicals J {} kJ.svg".format(path_folrads, Temp_vals[j]), bbox_inches='tight')
    plt.close()


path_folby = r"{}\Temperature - Full\{}".format(dir_path, "By Products")
try:
    os.mkdir(path_folby)
except Exception:
    pass

for jj, j in enumerate(graph_nodes):
    fig = plt.figure()
    index_j = int(j)
    TemperatureC = KtoC(Temp_K)
    plt.plot(xD, ec[j, :], 'g-', label='Ethylchloride')
    plt.plot(xD, cc[j, :], 'r-', label='Soot/Coke')
    plt.plot(xD, cp1[j, :], 'b-', label='1-/2-chloroprene')
    plt.plot(xD, di[j, :], 'y-', label='1,1-dichloroethylene')
    plt.plot(xD, tri[j, :], 'y-', label='Trichloroethylene')
    plt.plot(xD, c4h6cl2[j, :], 'o-', label=r'$C_{4}H_{6}Cl_{2}$')  # C4H6Cl2
    plt.plot(xD, c6h6[j, :], 'r-', label=r'$C_{6}H_{6}$')
    plt.plot(xD, c2h2[j, :], 'k-', label=r'$C_{2}H_{2}$')
    plt.xlabel(r'Distance [$m$]')
    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
    plt.legend(loc='best')
    plt.title(r"By-Product Concentration at {} {}".format(Temp_vals[j], r"[$\frac{kJ}{mol}$]"))
    plt.grid()
    fig.savefig(r"{}\By Products {} kJ.pdf".format(path_folby, Temp_vals[j]), bbox_inches='tight')
    fig.savefig(r"{}\By Products {} kJ.svg".format(path_folby, Temp_vals[j]), bbox_inches='tight')
    plt.close()


for jj, j in enumerate(graph_nodes):
    fig = plt.figure()
    index_j = int(j)
    TemperatureC = KtoC(Temp_K)
    plt.plot(xD, ecj[j, :], 'g-', label='Ethylchloride')
    plt.plot(xD, ccj[j, :], 'r-', label='Soot/Coke')
    plt.plot(xD, cp1j[j, :], 'b-', label='1-/2-chloroprene')
    plt.plot(xD, dij[j, :], 'y-', label='1,1-dichloroethylene')
    plt.plot(xD, trij[j, :], 'y-', label='Trichloroethylene')
    plt.plot(xD, c4h6cl2j[j, :], 'o-', label=r'$C_{4}H_{6}Cl_{2}$')  # C4H6Cl2
    plt.plot(xD, c6h6j[j, :], 'r-', label=r'$C_{6}H_{6}$')
    plt.plot(xD, c2h2j[j, :], 'k-', label=r'$C_{2}H_{2}$')
    plt.xlabel(r'Distance [$m$]')
    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
    plt.legend(loc='best')
    plt.title(r"By-Product Concentration at {} {}".format(Temp_vals[j], r"[$\frac{kJ}{mol}$]"))
    plt.grid()
    fig.savefig(r"{}\By Products J {} kJ.pdf".format(path_folby, Temp_vals[j]), bbox_inches='tight')
    fig.savefig(r"{}\By Products J {} kJ.svg".format(path_folby, Temp_vals[j]), bbox_inches='tight')
    plt.close()


path_folprod = r"{}\Temperature - Full\{}".format(dir_path, "Products")
try:
    os.mkdir(path_folprod)
except Exception:
    pass


for jj, j in enumerate(graph_nodes):
    fig = plt.figure()
    TemperatureC = KtoC(Temp_K)
    index_j = int(j)
    plt.plot(xD, edc[j, :], 'b-', label='EDC')
    plt.plot(xD, hcl[j, :], 'r-', label='HCl')
    plt.plot(xD, vcm[j, :], 'g-', label='VCM')
    plt.xlabel(r'Distance [$m$]')
    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
    plt.legend(loc='best')
    plt.title(label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[j])))
    plt.grid()
    fig.savefig(r"{}\Products {} kJ.pdf".format(path_folprod, Temp_vals[j]), bbox_inches='tight')
    fig.savefig(r"{}\Products {} kJ.svg".format(path_folprod, Temp_vals[j]), bbox_inches='tight')
    plt.close()


for jj, j in enumerate(graph_nodes):
    fig = plt.figure()
    TemperatureC = KtoC(Temp_vals[j])
    index_j = int(j)
    plt.plot(xD, edcj[j, :], 'b-', label='EDC')
    plt.plot(xD, hclj[j, :], 'r-', label='HCl')
    plt.plot(xD, vcmj[j, :], 'g-', label='VCM')
    plt.xlabel(r'Distance [$m$]')
    plt.ylabel(r'Concentration [$\frac{mol}{m^3}$]')
    plt.legend(loc='best')
    plt.title(label=r"Temperature: {}$^\circ$C".format(KtoC(Temp_vals[j])))
    plt.grid()
    fig.savefig(r"{}\Products J {} kJ.pdf".format(path_folprod, Temp_vals[j]), bbox_inches='tight')
    fig.savefig(r"{}\Products J {} kJ.svg".format(path_folprod, Temp_vals[j]), bbox_inches='tight')
    plt.close()


print("The Jacobian was evaluated {:,} times.\n".format(int(J_eval[0])))
end = time.time()  # Time when it finishes, this is real time


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Completion Time: {} Hours {} Minutes {} Seconds".format(int(hours), int(minutes), int(seconds)))


timer(start, end) # Prints the amount of time passed since starting the program

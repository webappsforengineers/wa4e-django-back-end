import numpy as np
import csv
from matplotlib import pyplot as plt
import sys


def pwp_acc(
        storm_loads, 
        k1, k2_OCR1, k3, k4, 
        a, b, OCR
        ):

    
    k2 = max(k2_OCR1 + a * (OCR - 1) ** b, 0.01) #something not right with this eqn as k2 can go negative.
    #print('OCR: ' + str(OCR))
    #print('k2: ' + str(k2))
    Ss = [x[0] for x in storm_loads]
    Ns = [x[1] for x in storm_loads]
    D_start = k1 * (1 - np.exp(-1 * k2 * Ns[0] * (Ss[0] - k4) ** k3))

    if len(storm_loads) == 1:
        return D_start, k2, 0
    
    else:
        for i in range(len(storm_loads) - 1):
            Neq = np.log(1 - D_start / k1) / (-1 * k2 * (Ss[i+1] - k4) ** k3)
            Ncarry =  Ns[i] + Neq
            D = k1 * (1 - np.exp(-1 * k2 * Ncarry * (Ss[i+1] - k4) ** k3))
            D_start = D
        #print(Ncarry)
        #print(D)
        #sys.exit()
        return D, k2, Ncarry


def findOCR(e, kappa_oed, sigmav, gamma_NCL, lambda_NCL):
    gamma_RC = e + kappa_oed * np.log(sigmav)
    sigma_NCL = np.exp((gamma_NCL  - gamma_RC) / (lambda_NCL - kappa_oed)) 
    #print('e: ' + str(e))
    #print('gamma_RC: ' + str(gamma_RC))
    #print('sigma_NCL: ' + str(sigma_NCL))
    OCR = sigma_NCL / sigmav

    return OCR

def find_eR(e, e0, sigma_v0, gamma_CSL, lambda_NCL):
    e_csl0 = gamma_CSL - lambda_NCL * np.log(sigma_v0)
    eR = (e - e_csl0) / (e0 - e_csl0)
 
    return eR


def consolidate(
        kappa_oed, sigmavc, gamma_NCL, gamma_CSL, lambda_NCL, 
        OCR, D, e, e0,
        m, p, q, zeta, rho, 
        ):

    eR = find_eR(e, e0, sigmavc, gamma_CSL, lambda_NCL)
    kappa_kappa_oed_min = m * np.exp(p * eR)
    kappa_kappa_oed_max = (
        q * kappa_kappa_oed_min / ((1 - kappa_kappa_oed_min) ** zeta)
    )
    kappa = kappa_oed * (kappa_kappa_oed_min + kappa_kappa_oed_max * ((1 - D) ** rho))

    delta_e = kappa * np.log(1 / (1 - D))
    e = e - delta_e

    OCR = findOCR(e, kappa_oed, sigmavc, gamma_NCL, lambda_NCL)

    #gamma_RC = e + kappa * np.log(sigmavc -1)
    #sigma_NCL = np.exp((gamma_NCL-gamma_RC) / (lambda_NCL - kappa)) 
    #OCR = sigma_NCL / sigmavc

    return e, kappa, OCR


def update_properties(
        kappa_oed, lambda_NCL, 
        D, su, 
        r, su0, G0, 
        c_A0, d_A0, tau, sigma_v
    ):
    A0 = c_A0 * tau/sigma_v + d_A0
    su = su * (1 / (1 - (D))) ** (A0 * (kappa_oed / lambda_NCL) / (1 - kappa_oed/lambda_NCL))
    #print('A0:' + str(A0))
    G = G0 * (r * (su / su0 - 1) + 1)

    return su, G


def update_applied_loads(storm_loads_norm_su, su0, su_new):
    new_loads = []
    su_ratio = su_new / su0
    for load in storm_loads_norm_su:
       load_new =  [load[0] / su_ratio, load[1]]
       new_loads.append(load_new)
    return new_loads
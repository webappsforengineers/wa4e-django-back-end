# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:57:55 2023

@author: ogf1n20
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import pandas as pd
import sympy as sp
import numpy as np
import math

class LrdDesign:
    """
    Class for designing LRDs and displaying sketches of the LRD geometry
    Two types of LRDs are supported: TFI SeaSpring and Dublin Offshore LRDs, each with their own parameters (determined by prefix do_ or tfi_)

    Attributes:
        lrd_type: type of LRD, which can be either TFI, DO
        do_l: length of the Dublin Offshore LRD
        do_d: diameter of the Dublin Offshore LRD
        do_h: horizontal distance between hinge points
        do_v: vertical distance between the hinge points
        do_theta: angle of the LRD with respect to the vertical (counter-clockwise +) in degrees
        do_rho: density of the Dublin Offshore LRD ballast in T/m^3
        tfi_l: length of the TFI SeaSpring LRD
        tfi_rs: rated strain of the TFI SeaSpring LRD - currently hard coded to 0.5, need to find a way to parameterise this
        tfi_rt: rated tension of the TFI SeaSpring LRD      
        
    """
    
    def __init__(self, lrd_type, *args, **kwargs):   
        """
        Initialize LrdDesign with LRD type and additional parameters.
            lrd_type: type of LRD, which can be either TFI, DO
            **kwargs: additional parameters, depending on the lrd_type
        """
        self.lrd_type = lrd_type  # Type of LRD, which can be either TFI, DO
                
        prefixes = { # Parameter prefixes for each lrd_type
            "do": ["do_l", "do_a", "do_v", "do_h", "do_d", "do_theta", "do_rho"],
            "tfi": ["tfi_l", "tfi_rs", "tfi_rt"]
        }
        
        for param in prefixes.get(self.lrd_type, []): # Set attributes based on lrd_type
            setattr(self, param, kwargs.get(param))   
                    
        # Set wet weight per unit length and length of LRD (0,0 for do lrd)
        self.l = 0.001 if self.lrd_type == 'do' else self.tfi_l
        self.w = 0 if self.lrd_type == 'do' else self.tfi_rt * 0.0002079 * 9.80665 # in N/m for rt in N
        
        if self.lrd_type == 'tfi': tfi_rt_kn = self.tfi_rt / 1000 # Convert tfi_rt to kN
        if self.lrd_type == 'tfi': self.tfi_d = -1e-8 * (tfi_rt_kn ** 2) + 0.0003 * tfi_rt_kn + 0.6447 # Diameter of the TFI LRD

        # Initialise additional dimensions for the Dublin Offshore LRD
        if self.lrd_type =='do':
            RHO_W_T = 1.025 # sea water density in ton/m^3
            RHO_W_N = 1025 * 9.80665 # sea water density in N/m^3
            self.do_ssw_t = 0.5 * self.do_l * self.do_d # Structural steel weight in 
            self.do_ssw_n = 9.80665 * 1000 * self.do_ssw_t
            csa = (self.do_d ** 2) * np.pi / 4 # Cross-sectional area of the LRD
            self.do_hba = ( self.do_l * RHO_W_T * csa - self.do_ssw_t) / (self.do_rho * csa)  # height of ballast (at this point do_rho in tons)
            print('Ballast height =')
            print(self.do_hba)
            self.do_o   = self.do_l / 2 - self.do_hba / 2              # distance between CoB and CoG
            self.do_fg  = (self.do_rho * 1000) * 9.80665 * self.do_hba * ((self.do_d ** 2) * np.pi) / 4 # weight of the LRD ballast in N
            print('Fg =')
            print(self.do_fg)
            self.do_fb  = (self.do_l * (np.pi * self.do_d ** 2) / 4) * RHO_W_N - self.do_ssw_n  # buoyancy force of the full LRD in N (minus structural steel weight)
            print('Fb =')
            print(self.do_fb)

    def _convert_to_float(self, value):
        if isinstance(value, sp.core.numbers.Float):
            return float(value)
        if isinstance(value, (np.float64, np.float32)):
            return float(value)
        return value
    
def get_lrd_strain(lrd, form, debugging = False, **kwargs):
    """
    Creates non-linear LRD stiffness curve equations, and takes tension input to return strain or extension. 
    Called by lrd class, qs_offset, one_sec_init and two_sec_init_functions.

    - Takes an instance of the LrdDesign class 'lrd'
    - Keyword arguments: vt, ht, at (vertical, horizontal, axial tension)
    - Form can be sym (symbolic), or num (numeric)
    - at is the axial tension at the fairlead! so the actual tension used to caluclate the tfi LRD strain is at - 1/2 of the LRD self weight
    
    """
    # Initialise the function based on wether it's for the lrd module or the mooring line module
    vt, ht, s = (None, None, None) if form=='num' else sp.symbols('vt ht s') # vert. tension, horz. tension, position 
    at = sp.sqrt(ht ** 2 + vt ** 2) if form=='sym' else kwargs['at']  
    at_lrd_midpoint = sp.sqrt((ht) ** 2 + (vt - 0.5 * lrd.l * lrd.w) ** 2) if form=='sym' else kwargs['at']   # axial t at the midpoint of the LRD

    sqrt_func = sp.sqrt if form=='sym' else np.sqrt
    acos_func = np.arccos if form=='num' else sp.acos
    atan_func = np.arctan if form=='num' else sp.atan
    sin_func = np.sin if form=='num' else sp.sin
    cos_func = np.cos if form=='num' else sp.cos
    pi_const = np.pi if form=='num' else sp.pi        
        
    if lrd.lrd_type == 'tfi':
        
        # Constants
        A, B, C, D, E, F = 6.71163E-36, - 1.66125E-29, 1.50917E-23, - 5.65547E-18, 6.20624E-13, 3.17986E-07
        j = 1011700
        k = j / lrd.tfi_rt 
        
        # Combine terms to get strain, return strain if numeric version
        polynomial_strain  = A * (k * at) ** 6 + B * (k * at) ** 5 + C * (k * at) ** 4 + D * (k * at) ** 3 + E * (k * at) ** 2 +  F * (k * at) 
        asymptote_strain = A * (j) ** 6 + B * (j) ** 5 + C * (j) ** 4 + D * (j) ** 3 + E * (j) ** 2 + F * (j)
        
        # Define the piecewise function
        strain = sp.Piecewise(  (polynomial_strain, at <=  lrd.tfi_rt),
                                (asymptote_strain, at > lrd.tfi_rt)
                                    )
        
        if form=='num':
            return strain
        
        else:
            # The strain is multiplied by the horz and vert components to get 
            lrd_x = ((ht) * lrd.l / (at)) * (1 + strain) 
            lrd_z = ((vt - 0.5 * lrd.l * lrd.w) * lrd.l / (at)) * (1 + strain)
            
            # Adjust axial tension terms to include self-weight of the LRD, i.e. vt = vt - 1/2 of self weight
            lrd_x = lrd_x.subs({at: at_lrd_midpoint}) 
            lrd_z = lrd_z.subs({at: at_lrd_midpoint})
        
            return lrd_x, lrd_z, None
        
    elif lrd.lrd_type == 'do':
        
        if debugging: print(math.degrees(lrd.do_theta))
        theta = atan_func(ht / vt) if form == 'sym' else np.radians(lrd.do_theta) # Theta is w.r.t. the vertical!!!

        if debugging: print('Mooring line angle')
        if debugging: print(math.degrees(theta))

        # Get horizontal and vertical components of mooring line force theta (Fx and Fy)
        Fx = at * sin_func(theta)
        Fy = at * cos_func(theta)

        if debugging: print('at, Fx, F')
        if debugging: print(at), print(Fx), print(Fy)
        
        r = sqrt_func(lrd.do_v ** 2 + lrd.do_h ** 2) # distance between lever arms
        s = sqrt_func( ((lrd.do_l - lrd.do_v) / 2 - lrd.do_hba / 2 ) ** 2 + (lrd.do_h ** 2 / 4 ) ) # distance between lower lever arm and ballast CoG
        phi = atan_func(lrd.do_h / lrd.do_v) # angle of r w.r.t. vertical
        chi = acos_func(lrd.do_h / (2 * s)) # angle of s w.r.t. horizontal

        cos_beta_term = - lrd.do_fb * r * sin_func(phi) / 2 + lrd.do_fg * r * sin_func(phi) - lrd.do_fg * s * cos_func(chi) + Fx * r * cos_func(phi) - Fy * r * sin_func(phi)
        sin_beta_term =   lrd.do_fb * r * cos_func(phi) / 2 - lrd.do_fg * r * cos_func(phi) - lrd.do_fg * s * sin_func(chi) + Fx * r * sin_func(phi) + Fy * r * cos_func(phi)

        if debugging: print('s and r lengths')
        if debugging: print(s),print(r)

        if debugging: print('phi and chi angles')
        if debugging: print(math.degrees(phi)), print(math.degrees(chi))

        beta = atan_func(sin_beta_term / cos_beta_term ) + pi_const/2 # Angle of LRD
        if debugging: print('LRD angle')
        if debugging: print(math.degrees(beta))

        e0 = - r * cos_func(theta - phi) # initial in-line length between hinges (no-tension)
        e  = - r * cos_func(beta + theta - phi) # in-line length between hinges under tension
        ex = r * sin_func(beta - phi) # horz. component of distance between hinges
        ez = - r * cos_func(beta - phi) # vert. component of distance betwween hinges (added minus sign such that initial vert extension is negative)
        delta_l = e - e0  # extension

        if debugging: print('Initial extension')
        if debugging: print(e0)
        if debugging: print('Final extended length between hinges')
        if debugging: print(e)
        if debugging: print('LRD extension')
        if debugging: print(delta_l)
        if debugging: print('Horz and vert components of extension')
        if debugging: print(ex), print(ez)
        
        if form=='sym': # Currently same logic as init form... does this need changing?
            lrd_x = ex
            lrd_z = ez
            return lrd_x, lrd_z, beta
                
        else: 
            return delta_l
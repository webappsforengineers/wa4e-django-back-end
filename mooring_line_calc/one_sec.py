

import sympy as sp
from sympy import symbols, Eq, pretty
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fsolve
from mooring_line_calc.lrd_module import LrdDesign, get_lrd_strain
from mooring_line_calc.helpers import get_fairlead_equations, get_profile_equations
import math


def one_sec_init(seabed_contact = True, at = 600000, xf = 796.73, zf = 136, ea = 750e6, w = 1422.5, lrd = None, verbose = True, debugging = False):
    
    '''
    If catenary no lrd (1 eq 1 unknown) --> solve for ht analytically, then solve for l analytically
    If catenary with lrd (1 eq 1 unknown) --> solve for ht numerically using ridder (no initial guess, just bounds), then solve for l analytically
    If taut no lrd (2 eqns 2 unknowns) --> solve for ht and vt numerically (fsolve with very strong initial guess)
    If taut with lrd (2 eqns 2 unknowns) --> solve for ht and vt numerically (fsolve with very strong initial guess)
    
    xf, zf are horz and vert fairlead equations respectively
    xs, zs are horz and vert profile equations respectively
    
    _offset are equations for tension - offset plotting (not used by this function, but created by this function)
    _init are equations for the initialisation plots in this funciton
    
    Difference between _offset and _init  eqns are the lrd terms, which are fully numeric for _init, and partly symbolic for _offset
    
    '''
    # Set symbolic variables
    xf_sym, zf_sym, vt_sym, ht_sym, at_sym, s_sym, l_sym, ea_sym, w_sym = sp.symbols('xf zf vt ht at s l ea w')
    
####################### PART 1 of function : solving ht, vt (and l for taut) ########################
    
# No LRD ----------------------------------------------------------------------
    
    if not lrd:    
                
        # Start by getting standard seabed equations (numeric version)
        xf1_eq, zf1_eq = get_fairlead_equations(seabed_contact, 'num', ea_num = ea, w_num = w)
        xs1_eq, zs1_eq = get_profile_equations(seabed_contact,'num', ea_num = ea, w_num = w)
    
        # Rearrange fairlead expressions for root finding (i.e. so =0), profile expressions stay as they are
        xf_eq = xf1_eq - xf
        zf_eq = zf1_eq - zf
                
        # Get rid of one of the force terms (vt in this case) by substituting vt for sqrt(at^2 - ht^2)
        xf_eq_ht_only = xf_eq.subs(vt_sym, sp.sqrt(at**2 - ht_sym**2))
        zf_eq_ht_only = zf_eq.subs(vt_sym, sp.sqrt(at**2 - ht_sym**2))
        
        if debugging:
            sp.init_printing(use_unicode=True)
            print('xf_eq =', sp.pretty(xf_eq_ht_only), '\nzf_eq =', sp.pretty(zf_eq_ht_only))
        
        # If catenary, the zf eq does not depend on l, so we can solve directly for ht
        if seabed_contact:
            print('________ Solving catenary no LRD _______________________' if verbose else '',  end='\n')
        # Solve analytically for ht, then keep positive solutions only
            ht_sols = sp.solve(zf_eq_ht_only, ht_sym) 
            for sol in ht_sols: 
                if sol.is_real and sol.is_positive:
                    ht = float(sol)

        # If taut, solve numerically for ht & l 
        else:
            print('________ Solving taut no LRD ____________________________' if verbose else '',  end='\n')
            # Convert SymPy expressions to numerical functions
            f1 = sp.lambdify((ht_sym, l_sym), xf_eq_ht_only, "numpy")
            f2 = sp.lambdify((ht_sym, l_sym), zf_eq_ht_only, "numpy")
            
            def equations(vars):
                ht, l = vars
                return [f1(ht, l), f2(ht, l)]
            
            # Initial guess
            # Calculate the angle in radians
            theta = math.atan2(zf, xf)            
            # Calculate the horizontal component of the tension
            ht0 = at * math.cos(theta)
            l0 = np.sqrt(xf ** 2 + zf ** 2)
            initial_guess = [ht0, l0]  # Replace with your initial guess
            if debugging:
                print('ht initial guess =', ht0)
                print('l initial guess =' ,  l0)
            
            # Solving the equations using fsolve is easy as initial guess is really good
            ht, reqd_length = fsolve(equations, initial_guess)
                                    
        # Then get vt and display solved parameters
        vt = np.sqrt(at ** 2 - ht ** 2)
        
        if verbose:
            print('ht =', ht)
            print('vt =', vt)
            if not seabed_contact:
                print('l =', reqd_length)
        
# With LRD ht vt l solving ----------------------------------------------------
    
    # If lrd is an instance of lrd_module, build equations for chain + lrd
    
    else:
                
        # Get top of section 1 equations
        xf1_eq, zf1_eq = get_fairlead_equations(seabed_contact, 'num', ea_num = ea, w_num = w)
        xs1_eq, zs1_eq = get_profile_equations(seabed_contact,'num', ea_num = ea, w_num = w)
        
        # Replace vt for vt minus lrd weight 
        xf1_eq, zf1_eq  = xf1_eq.subs({vt_sym: vt_sym - lrd.w * lrd.l}), zf1_eq.subs({vt_sym: vt_sym - lrd.w * lrd.l})
        xs1_eq, zs1_eq  = xs1_eq.subs({vt_sym: vt_sym - lrd.w * lrd.l}), zs1_eq.subs({vt_sym: vt_sym - lrd.w * lrd.l})

        # Get LRD symbolic expressions
        lrd_x, lrd_z, lrd_alpha = get_lrd_strain(lrd, 'sym')
            
        # Get final fairlead eqns for init and offset (symbolic LRD for offset)
        xf_eq, zf_eq = xf1_eq + lrd_x - xf, zf1_eq + lrd_z - zf   
        
        # Substitue vt for sqrt(at^2 - ht^2)
        xf_eq_ht_only = xf_eq.subs(vt_sym, sp.sqrt(at**2 - ht_sym**2))
        zf_eq_ht_only = zf_eq.subs(vt_sym, sp.sqrt(at**2 - ht_sym**2))

        # If catenary, one eqn with one unknown (zf) so solve for ht with ridder
        if seabed_contact: 
            print('________ Solving catenary with LRD _______________________' if verbose else '',  end='\n')
            # Isolate horizontal tension ht from zf (use zf cos l doesn't come into it), as for no LRD version            
            # Lamdify function to prepare for numerical solve
            zf_eq_ht_only_func = sp.lambdify(ht_sym, zf_eq_ht_only, modules=['numpy'])
    
            # Define a wrapper function for fsolve that accepts a scalar and returns a scalar
            def zf_eq_fsolve(ht):
                return zf_eq_ht_only_func(ht).item()  # Use .item() to ensure the result is a scalar
            
            if debugging:
            # Plot the function to understand its behavior 
                ht_values = np.linspace(10, at, 4000)
                zf_values = []
                for ht in ht_values:
                    zf_values.append(zf_eq_fsolve(ht))
                    print(ht)
                plt.plot(ht_values, zf_values)
                plt.xlabel('ht')
                plt.ylabel('zf_eq(ht)')
                plt.title('zf_eq(ht) vs. ht')
                plt.grid(True)
                plt.show()

            ht = opt.ridder(zf_eq_ht_only_func, 10, 0.99*at) # lower bound to 10 N & 0.99*at to avoid division by zero
        
        # If taut, l is in both zf and xf, so solve 2 eqns with 2 unknowns. But can have very good initial guess
        else:
            print('________ Solving taut with LRD _______________________' if verbose else '', end='\n')
            # Convert SymPy expressions to numerical functions
            f1 = sp.lambdify((ht_sym, l_sym), xf_eq_ht_only, "numpy")
            f2 = sp.lambdify((ht_sym, l_sym), zf_eq_ht_only, "numpy")
            
            def equations(vars):
                ht, l = vars
                return [f1(ht, l), f2(ht, l)]
            
            # Initial guess
            # Calculate the angle in radians
            theta = math.atan2(zf, xf)            
            # Calculate the horizontal component of the tension
            ht0 = at * math.cos(theta)
            l0 = np.sqrt(xf ** 2 + zf ** 2) - lrd.l
            initial_guess = [ht0, l0]  # Replace with your initial guess
            
            # Solving the equations using fsolve is easy as initial guess is really good
            ht, reqd_length = fsolve(equations, initial_guess)
                        
        vt = np.sqrt(at ** 2 - ht ** 2)
        if verbose:
            print('ht =', ht)
            print('vt =', vt)
            if not seabed_contact:
                print('l =', reqd_length) 

        # Get numerical values of lrd extension to pass on to the plotting functs (ERROR IN THE NAMING - ALPHA is BETA in the lrd func)
        lrd_x_val = float(lrd_x.subs({ht_sym: ht,  vt_sym: vt}))
        lrd_z_val = float(lrd_z.subs({ht_sym: ht,  vt_sym: vt}))            
        at_calculated = np.sqrt(vt ** 2 + ht ** 2)
        ml_angle_calculated = np.degrees(np.arctan(vt / ht))
        if lrd.lrd_type == 'tfi': 
            lrd_extension = np.sqrt(lrd_x_val ** 2 + lrd_z_val ** 2) - lrd.l
        elif lrd.lrd_type == 'do':
            lrd.do_theta = ml_angle_calculated # in deg
            lrd_alpha_val_rad = float(lrd_alpha.subs({ht_sym: ht,  vt_sym: vt}))
            lrd_alpha_val = math.degrees(lrd_alpha_val_rad)
            lrd.do_alpha = lrd_alpha_val_rad # Set the angle of the LRD, for the profile plotting, in rad
            lrd_extension = get_lrd_strain(lrd, 'num', at = at_calculated) 
    
    # Assign lrd length of 0 if no lrd
    if not lrd: lrd_x_val, lrd_z_val = 0, 0

#################### PART 2 of function : fill in ht and vt, and solve l for catenary ####################
    
    # The expressions are now split into two verisons, _init with ht and vt subbed for real values, and _offset with ht and vt as symbols

    # Substitue vert tension and horizontal tension with known values into xf for length solving
    xf_eq_init = xf_eq.subs({ht_sym: ht,  vt_sym: vt}) 
    zf_eq_init = zf_eq.subs({ht_sym: ht,  vt_sym: vt}) 
    
    # If catenary, solve the equation to get reqd l (no need for taut, as we already got it from system solve)
    if seabed_contact:
        reqd_length = float(sp.solve(xf_eq_init, l_sym)[0])    
        # Print the l solution
        if verbose:
            print('l =', reqd_length) if seabed_contact else print() 
        # Calculate length on seabed
        seabed_length = reqd_length - vt / w
        # If the length on seabed is negative, raise an error
        if seabed_length < 0:
            raise ValueError('The length of chain on the seabed is negative. Please adjust the inputs.')

    # Now plug the reqd l into the numeric (ht and vt num) profile equations
    xs1_eq_init = xs1_eq.subs({ht_sym: ht,  vt_sym: vt, l_sym:reqd_length})
    zs1_eq_init = zs1_eq.subs({ht_sym: ht,  vt_sym: vt, l_sym:reqd_length})
        
    # Also plug the reqd l into the symbolic (ht and vt sym) equations, to be used for offset later on
    xf_eq_offset = xf_eq.subs(l_sym, reqd_length)
    zf_eq_offset = zf_eq.subs(l_sym, reqd_length)
    xs1_eq_offset = xs1_eq.subs(l_sym, reqd_length)
    zs1_eq_offset = zs1_eq.subs(l_sym, reqd_length)
    
#################### PART 3 of function : solving profile ########################
    
    # Get coordinates for the top of first section, using both fairlead and profile equations (both with origin at 0)
    xf1_val = xf_eq_init.subs({l_sym: reqd_length}) + xf - lrd_x_val
    zf1_val = zf_eq_init.subs({l_sym: reqd_length}) + zf - lrd_z_val

    xf1_check = xs1_eq_init.subs({s_sym: reqd_length}).evalf() 
    zf1_check = zs1_eq_init.subs({s_sym: reqd_length}).evalf() 

    # Check if xf1_init is equal to xf1_check and so on, with a tolerance of 1e-6
    TOL = 1e-6
    if abs(xf1_val - xf1_check) > TOL or abs(zf1_val - zf1_check) > TOL : 
        raise ValueError('Initial conditions do not match profile equations')

    # Generate s values. Span from 0 to top of chain
    min_s1, max_s1   = 0, reqd_length
    DENSITY = 150
    s1_values = np.linspace(min_s1, max_s1, DENSITY)

    # Calculate the corresponding x_s and z_s values
    xs_values_sec1 = [xs1_eq_init.subs({s_sym: s_val}).evalf() for s_val in s1_values]
    zs_values_sec1 = [zs1_eq_init.subs({s_sym: s_val}).evalf() for s_val in s1_values]
    
    
    xs_values_sec2 = None
    zs_values_sec2 = None
               

    xs_values_sec2 = None
    zs_values_sec2 = None
               
    if not lrd:
        xs_values_lrd, zs_values_lrd = None, None
    else:        
        xs_values_lrd =  [xf1_val, xf1_val + lrd_x_val]
        zs_values_lrd =  [zf1_val, zf1_val + lrd_z_val]

    # Convert all to floats for plotly
    xs_values_sec1 = [float(x) for x in xs_values_sec1]
    zs_values_sec1 = [float(z) for z in zs_values_sec1]
    if xs_values_lrd:
        xs_values_lrd = [float(x) for x in xs_values_lrd]
    if zs_values_lrd:
        zs_values_lrd = [float(z) for z in zs_values_lrd]

    # # Plot the mooring line profile    
    # fig = plot_profile('one_sec', 'static', lrd, xf, zf, ht, vt, xs_values_sec1, zs_values_sec1, sec2_xs=None, sec2_zs=None, lrd_xs=xs_values_lrd, lrd_zs=zs_values_lrd)
    # fig.show()

    # # Add the LRD drawings and stiffness curve if LRD is present
    # if lrd and lrd.lrd_type == 'do':
    #     fig2 = lrd.draw_rough_do('static', lrd_extension)
    #     fig2.show()
    #     fig3 = lrd.plot_stiffness_curve(analysis_type = 'static', current_tension = at_calculated, current_ext_or_str = lrd_extension)
    #     fig3.show()

    # if lrd and lrd.lrd_type == 'tfi':
    #     fig2 = lrd.draw_tfi(analysis_type = 'static', extension = lrd_extension)
    #     fig2.show()
    #     fig3 = lrd.plot_stiffness_curve(analysis_type = 'static', current_tension = at_calculated, current_ext_or_str = lrd_extension / lrd.tfi_l)
    #     fig3.show()

    init_package = {'xf_eq':  xf_eq_offset,
                    'zf_eq':  zf_eq_offset,
                    'xs1_eq':  xs1_eq_offset,
                    'zs1_eq':  zs1_eq_offset,
                    'xs2_eq':  None,
                    'zs2_eq':  None,
                    'sec1_l':  reqd_length,
                    'sec2_l':  None,
                    'sec1_w':  w,
                    'sec2_w':  None,
                    'lrd_x' :  lrd_x if lrd else None, 
                    'lrd_z' :  lrd_z if lrd else None,
                    'lrd_alpha' : lrd_alpha if lrd and lrd.lrd_type == 'do' else None,
                    'vt0'  :  vt, 
                    'ht0'  :  ht,
                    'xf0'  :  xf,
                    'zf0'  :  zf,
                    'lrd'  :  lrd,
                    's1_values': s1_values,
                    's2_values': None,
                    'moortype': 'one_sec',
                    'name': 'one_sec, LRD: ' + str(lrd.lrd_type if lrd else 'None') + ', Catenary' if seabed_contact else 'Taut',
                    'xs_values_sec1': xs_values_sec1,
                    'zs_values_sec1': zs_values_sec1,
                    'xs_values_lrd': xs_values_lrd,
                    'zs_values_lrd': zs_values_lrd,
                    'xs_values_sec2': xs_values_sec2,
                    'zs_values_sec2': zs_values_sec2,
                    }
    print()
    
    return init_package

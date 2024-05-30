import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.optimize import anderson
from scipy.optimize.nonlin import NoConvergence
from mooring_line_calc.helpers import plot_profile, plot_tension_offset, animate

def qs_offset(init_package, max_offset = 15, resolution = 2, profile_plot = True):
    '''
    Returns the tension-offset curve for a given mooring system.

    init_package: dict returned by one_sec_init or two_sec_init
    max_offset: maximum offset in meters
    resolution: number of points per meter
    profile_plot: boolean to determine if the animations should be plotted (much quicker if set to False)

    '''
    
    # Set symbolic variables
    xf_sym, zf_sym, vt_sym, ht_sym, at_sym, s_sym, l_sym, ea_sym, w_sym = sp.symbols('xf zf vt ht at s l ea w')
    
    # Initialize list to store results
    tension_values = []
    frames_line_profile = []
    frames_lrd_drawing = []
    frames_lrd_stiffness = []
    
    # Generate displacement values
    displacement_values = displacement_values = np.linspace(0, max_offset, num= int(max_offset * resolution))
    
    # Open init_package variables into this space
    # Unpack init_package variables
    initial_guess = [init_package['ht0'], init_package['vt0']]
    # Unpack init_package variables
    xf_eq, zf_eq, xs1_eq, zs1_eq, xs2_eq, zs2_eq, sec1_l, sec2_l, lrd_x, lrd_z, lrd_alpha ,xf0, zf0, lrd, s1_values, s2_values, moortype, name = (
    init_package['xf_eq'], init_package['zf_eq'],
    init_package['xs1_eq'], init_package['zs1_eq'], init_package['xs2_eq'], init_package['zs2_eq'],
    init_package['sec1_l'], init_package['sec2_l'], init_package['lrd_x'], init_package['lrd_z'], init_package['lrd_alpha'],
    init_package['xf0'], init_package['zf0'], init_package['lrd'], init_package['s1_values'], init_package['s2_values'], init_package['moortype'], init_package['name']
    )

    for i, displacement in enumerate(displacement_values):
        # Update x_f for the new displacement
        xf_eq_offset = xf_eq - displacement
        zf_eq_offset = zf_eq
                        
        # Build the vector of functions
        func_vector = sp.Matrix([xf_eq_offset, zf_eq_offset])
            
        # Convert the functions to lambda functions for usage with scipy
        func_vector_lambda = [sp.lambdify([vt_sym, ht_sym], func) for func in func_vector]
        
        # Define the system of mooring profile equations to solve
        def func_to_solve(x):
            return [f(*x) for f in func_vector_lambda]
        
        try:
            # Attempt to use scipy's anderson to find a solution
            solution_anderson = anderson(func_to_solve, initial_guess)
            vt_sol, ht_sol = solution_anderson
        except NoConvergence as e:
            print("Anderson method failed to converge ...")
            print("Failed at point:", e.args[0])  # Print the point of failure if needed
            ht_sol, vt_sol = initial_guess
            
        tension = np.sqrt(ht_sol**2 + vt_sol**2)
        ml_angle = np.degrees(np.arctan(ht_sol / vt_sol))

        # Append tension to results list
        tension_values.append(tension)

        # Use the solution from this iteration as the initial guess for the next iteration
        initial_guess = [ht_sol, vt_sol]
        
        subs_dict = {vt_sym: vt_sol, ht_sym: ht_sol}
        # Sub into xs and zs for profile solving
        xs1_eq_num = xs1_eq.subs(subs_dict)
        zs1_eq_num = zs1_eq.subs(subs_dict)

        if moortype == 'two_sec':
            xs2_eq_num = xs2_eq.subs(subs_dict)
            zs2_eq_num = zs2_eq.subs(subs_dict)

        if lrd:lrd_x_val = float(lrd_x.subs(subs_dict))
        if lrd:lrd_z_val = float(lrd_z.subs(subs_dict))
        if lrd and lrd.lrd_type == 'do' : lrd_alpha_val_rad = lrd_alpha.subs(subs_dict) # Get the alpha value in radians
                    
   
    print(tension_values, displacement_values)
    

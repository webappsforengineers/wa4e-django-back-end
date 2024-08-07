import numpy as np
import sympy as sp
from sympy import N
# import plotly.graph_objects as go
from scipy.optimize import anderson
from scipy.optimize.nonlin import NoConvergence
from mooring_line_calc.lrd_module import LrdDesign, get_lrd_strain
# from mooring_line_calc.helpers import plot_profile, plot_tension_offset, animate

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
    all_current_ext_or_str_values = []
    all_xs_values_sec1 = []
    all_zs_values_sec1 = []
    all_xs_values_sec2 = []
    all_zs_values_sec2 = []
    all_xs_values_lrd = []
    all_zs_values_lrd = []
    all_corner_xs = []
    all_corner_zs = []
    all_smaller_corner_xs = []
    all_smaller_corner_zs = []
    all_line_from_hinge_x = []
    all_line_from_hinge_y = []
    
    all_at_values_qs_offset = []
    all_ext_or_str_values_qs_offset = []
    
    
    all_tfi_current_lengths = []
    all_ml_angles = []
    all_full_rectangles_rotated = []
    all_bottom_rectangles_rotated = []
    all_top_hinges_rotated = []
    all_bottom_hinges_rotated = []
    all_at_calculated = []
    
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
        if lrd and lrd.lrd_type == 'do' : lrd.do_alpha = sp.N(lrd_alpha_val_rad) # Set the angle of the LRD, for the profile plotting
        if lrd and lrd.lrd_type == 'do' : lrd.do_theta = ml_angle # Set the angle of the mooring line, for the stiffness curve plotting
        
        
        if profile_plot:
        
            # Calculate the corresponding x_s and z_s values
            xs_values_sec1 = [xs1_eq_num.subs({s_sym: s_val}).evalf() for s_val in s1_values]
            zs_values_sec1 = [zs1_eq_num.subs({s_sym: s_val}).evalf() for s_val in s1_values]
            # Evaluate top of first section xf1 and zf1 using xs and zs eq and reqd length
            xf1_val = xs1_eq_num.subs({s_sym: sec1_l}).evalf()
            zf1_val = zs1_eq_num.subs({s_sym: sec1_l}).evalf()
            # Evaluate top of second section xf2 and zf2 using xs and zs eq and sec2 length            
            xf2_val = xs2_eq_num.subs({s_sym: sec2_l}).evalf() if moortype == 'two_sec' else 0
            zf2_val = zs2_eq_num.subs({s_sym: sec2_l}).evalf() if moortype == 'two_sec' else 0
            xs_values_sec2 = [xs2_eq_num.subs({s_sym: s_val}).evalf() + xf1_val for s_val in s2_values ] if moortype == 'two_sec' else None
            zs_values_sec2 = [zs2_eq_num.subs({s_sym: s_val}).evalf() + zf1_val for s_val in s2_values ] if moortype == 'two_sec' else None 
            # Evaluate LRD profile
            xs_values_lrd = None if not lrd else [xf2_val + xf1_val, xf2_val + xf1_val + lrd_x_val]
            zs_values_lrd = None if not lrd else [zf2_val + zf1_val, zf2_val + zf1_val + lrd_z_val]
            
            # Convert all to floats for plotly
            xs_values_sec1 = [float(x) for x in xs_values_sec1]
            zs_values_sec1 = [float(z) for z in zs_values_sec1]
            if moortype == 'two_sec': xs_values_sec2 = [float(x) for x in xs_values_sec2] 
            if moortype == 'two_sec': zs_values_sec2 = [float(z) for z in zs_values_sec2]
            if xs_values_lrd: xs_values_lrd = [float(x) for x in xs_values_lrd]
            if zs_values_lrd: zs_values_lrd = [float(z) for z in zs_values_lrd]
            
            all_xs_values_sec1.append(xs_values_sec1)
            all_zs_values_sec1.append(zs_values_sec1)
            all_xs_values_sec2.append(xs_values_sec2)
            all_zs_values_sec2.append(zs_values_sec2)
            all_xs_values_lrd.append(xs_values_lrd)
            all_zs_values_lrd.append(zs_values_lrd)
            
            if lrd:
                lrd_extension = np.sqrt(lrd_x_val**2 + lrd_z_val**2)

                if lrd.lrd_type == 'do':
                    # stiffness curve line
                    at_values_qs_offset = np.linspace(0.01, lrd.do_fg * 5, 100)
                    ext_or_str_values_qs_offset = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values_qs_offset]
                    
                    at_values_qs_offset = [float(value)/1000 for value in at_values_qs_offset]
                    
                    all_at_values_qs_offset.append(at_values_qs_offset)
                    all_ext_or_str_values_qs_offset.append(ext_or_str_values_qs_offset)
                    
                    # point on stiffness curve line
                    ext_or_str =  get_lrd_strain(lrd, form='num', at = tension)
                    all_current_ext_or_str_values.append(ext_or_str)
                    lrd_alpha_val = np.degrees(float(lrd_alpha_val_rad))
                    
                    # Create rotation matrix
                    theta = np.radians(lrd_alpha_val)
                    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    # Calculate hinge points before rotation
                    top_hinge    = np.array([lrd.do_d / 2 + lrd.do_h / 2, lrd.do_l / 2 + lrd.do_v / 2])
                    bottom_hinge = np.array([lrd.do_d / 2 - lrd.do_h / 2, lrd.do_l / 2 - lrd.do_v / 2])
                    
                    # Calculate the new origin
                    new_origin = (top_hinge + bottom_hinge) / 2
                    
                    # Create the rectangles around the new origin
                    full_rectangle = np.array([
                        [0 - new_origin[0], 0 - new_origin[1]],
                        [lrd.do_d - new_origin[0], 0 - new_origin[1]],
                        [lrd.do_d - new_origin[0], lrd.do_l - new_origin[1]],
                        [0 - new_origin[0], lrd.do_l - new_origin[1]],
                        [0 - new_origin[0], 0 - new_origin[1]]
                    ]).T
                    
                    bottom_rectangle = np.array([
                        [0 - new_origin[0], 0 - new_origin[1]],
                        [lrd.do_d - new_origin[0], 0 - new_origin[1]],
                        [lrd.do_d - new_origin[0], (lrd.do_hba) - new_origin[1]],
                        [0 - new_origin[0], (lrd.do_hba) - new_origin[1]],
                        [0 - new_origin[0], 0 - new_origin[1]]
                    ]).T
                    
                    # Rotate rectangles and hinge points
                    full_rectangle_rotated = np.dot(rotation_matrix, full_rectangle)
                    bottom_rectangle_rotated = np.dot(rotation_matrix, bottom_rectangle)
                    top_hinge_rotated = np.dot(rotation_matrix, top_hinge - new_origin)
                    bottom_hinge_rotated = np.dot(rotation_matrix, bottom_hinge - new_origin)
                    
                    all_full_rectangles_rotated.append(full_rectangle_rotated)
                    all_bottom_rectangles_rotated.append(bottom_rectangle_rotated)
                    all_top_hinges_rotated.append(top_hinge_rotated)
                    all_bottom_hinges_rotated.append(bottom_hinge_rotated)
                    
                    angle = np.radians(90 - ml_angle)
                    
                    all_ml_angles.append(angle)
                    
                    # Variables for the mooring line profile
                    # Plot the outline of the lrd.
                    # Find the center coordinates of the LRD, i.e. the mid point between the hinges
                    lrd_center_x = (xs_values_lrd[0] + xs_values_lrd[-1]) / 2
                    lrd_center_z = (zs_values_lrd[0] + zs_values_lrd[-1]) / 2
                    # Plot the outline of the lrd, which is a rectangle with a center at  coordinates, of dimensions lrd.do_l and lrd.do_h
                    # Lrd angle is with respect to vertical (counter clockwise is +ve) is lrd.do_alpha
                    half_width = lrd.do_l / 2 # Width and height are the wrong way around here, but it doesn't matter as long as they are consistent
                    half_height = lrd.do_d / 2
                    
                    # Assume alpha is lrd.do_alpha given in radians with respect to the vertical axis
                    alpha = float(N(lrd.do_alpha))

                    corners = [
                        (np.cos(alpha) * half_height - np.sin(alpha) * half_width, np.sin(alpha) * half_height + np.cos(alpha) * half_width), # Top left
                        (-np.cos(alpha) * half_height - np.sin(alpha) * half_width, -np.sin(alpha) * half_height + np.cos(alpha) * half_width), # Bottom left
                        (-np.cos(alpha) * half_height + np.sin(alpha) * half_width, -np.sin(alpha) * half_height - np.cos(alpha) * half_width), # Bottom right
                        (np.cos(alpha) * half_height + np.sin(alpha) * half_width, np.sin(alpha) * half_height - np.cos(alpha) * half_width)  # Top right
                    ]

                    # Adjust these based on the center point
                    corner_xs = [lrd_center_x + cx for cx, cz in corners]
                    corner_zs = [lrd_center_z + cz for cx, cz in corners]

                    # Ensure the rectangle closes by repeating the first corner
                    corner_xs.append(corner_xs[0])
                    corner_zs.append(corner_zs[0])
                    
                    # Additional variables for smaller rectangle
                    reduced_width = half_width - lrd.do_hba

                    # Calculate new corners for the smaller rectangle
                    smaller_corners = [
                        corners[0], # Bottom left
                        corners[1], # New top right
                        (-np.cos(alpha) * half_height + np.sin(alpha) * reduced_width, -np.sin(alpha) * half_height - np.cos(alpha) * reduced_width), # New top left
                        (np.cos(alpha) * half_height + np.sin(alpha) * reduced_width, np.sin(alpha) * half_height - np.cos(alpha) * reduced_width),  # Bottom-right, shared
                    ]

                    # Adjust these based on the center point
                    smaller_corner_xs = [lrd_center_x + cx for cx, cz in smaller_corners]
                    smaller_corner_zs = [lrd_center_z + cz for cx, cz in smaller_corners]

                    # Ensure the rectangle closes by repeating the first corner
                    smaller_corner_xs.append(smaller_corner_xs[0])
                    smaller_corner_zs.append(smaller_corner_zs[0])
                    
                    # Now add a little line (of length lrd.do_d) going from the lower hinge upwards at an angle of lrd.do_theta
                    mooring_angle = np.arctan(vt_sol / ht_sol)
                    lower_hinge_x, lower_hinge_z = xs_values_lrd[-1], zs_values_lrd[-1]
                    line_end_x = lower_hinge_x + lrd.do_d * np.cos(mooring_angle)
                    line_end_z = lower_hinge_z + lrd.do_d * np.sin(mooring_angle)
                    line_from_hinge_x = [lower_hinge_x, line_end_x]
                    line_from_hinge_y = [lower_hinge_z, line_end_z]
                    
                    all_corner_xs.append(corner_xs)
                    all_corner_zs.append(corner_zs)
                    all_smaller_corner_xs.append(smaller_corner_xs)
                    all_smaller_corner_zs.append(smaller_corner_zs)
                    all_line_from_hinge_x.append(line_from_hinge_x)
                    all_line_from_hinge_y.append(line_from_hinge_y)
                else:
                    # print(lrd.tfi_rt)
                    # at_values_qs_offset = np.linspace(0.0, lrd.tfi_rt * 1.5, 100)
                    # ext_or_str_values_qs_offset = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values_qs_offset]
                    
                    # at_values_qs_offset = [float(value)/1000 for value in at_values_qs_offset]
                    
                    # all_at_values_qs_offset.append(at_values_qs_offset)
                    # all_ext_or_str_values_qs_offset.append(ext_or_str_values_qs_offset)
                    
                    ext_or_str = (lrd_extension - lrd.l) / lrd.l
                    extension = lrd_extension - lrd.l
                    current_length = lrd.l + extension
                    all_tfi_current_lengths.append(current_length)
                    all_current_ext_or_str_values.append(ext_or_str)
    
    # convert tension values to kN 
    tension_values = [element / 1000 for element in tension_values]
    
        
    return tension_values, displacement_values, all_current_ext_or_str_values, all_xs_values_sec1, all_zs_values_sec1, all_xs_values_sec2, all_zs_values_sec2, all_xs_values_lrd, all_zs_values_lrd, all_tfi_current_lengths, all_ml_angles, all_full_rectangles_rotated, all_bottom_rectangles_rotated, all_top_hinges_rotated, all_bottom_hinges_rotated, all_corner_xs, all_corner_zs, all_smaller_corner_xs, all_smaller_corner_zs, all_line_from_hinge_x, all_line_from_hinge_y, all_at_values_qs_offset, all_ext_or_str_values_qs_offset,
    

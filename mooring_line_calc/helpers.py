# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:54:04 2024

@author: Oscar 
"""
import sympy as sp
from sympy import symbols, Eq, pretty
import numpy as np
# import matplotlib.pyplot as plt
# from IPython.display import Image
from scipy import optimize
import scipy.optimize as opt
from scipy.optimize import fsolve
from scipy.optimize import anderson, broyden1, newton_krylov
from mooring_line_calc.lrd_module import LrdDesign, get_lrd_strain
from scipy.optimize import basinhopping
from scipy.optimize.nonlin import NoConvergence
# import plotly.graph_objects as go
import math

def get_fairlead_equations(seabed_contact, numorsim, w_num = None, ea_num = None, l_num = None): 
    '''
    Returns fairlead equations for homogeneous catenary line of linear stiffness, as described in Jonkman 2009. 
    This is called by one_sec, two_sec and qs_offset functions to produce the overall multi-catenary functions including LRDs.

    seabed_contact : True or False, determines the format of the equation. True is valid for normal catenary equations (not valid if there is uplift)
    whist False is valid for taut/semi taut equations (not valid as soon as there is touchdown of the mooring line)

    numorsim : this will always be num when called by one_sec, two_sec functions or qs_offset functions (i.e. numerical values of inputs subbed into the equations)
    sim option returns the equations with no numerical values, this is only used for latex visualisation & checking of equations

    w_num : line unit weight in N/m
    ea_num : linear stiffness of line in N
    l_num : total unstretched line length in m. Unlike w_num and ea_num, this will not be known on the inital call in the one_sec & two_sec functions
    If a numeric value of l_num is not passed, it defaults to None and stays symbolic in the output. In the QS offset functions (post-mooring line initialisation),
    the length is known, and is passed as a numeric value    
    '''

    # Create symolic variables
    w, l, ea, ht, vt, s = sp.symbols('w l ea ht vt s') 

    # Length of line along seabed
    lb  = l - vt / w 
    
    # Vt/ht quotients
    vt_ht    = vt / ht  # Ratio of vert to horz tensions
    vt_wl_ht = (vt - w * l)   / ht  # Ratio of vert minus weight to horz tensions
    
    # Sqrt terms
    sqrt_term  = sp.sqrt(1 + vt_ht ** 2)    # Sqrt term for non-suspended line
    sqrt_term2 = sp.sqrt(1 + vt_wl_ht ** 2) # Sqrt term for suspended line

    # Log terms
    log_term  = sp.log(vt_ht + sqrt_term) # Log term for non-suspended line        
    log_term2 = sp.log(vt_wl_ht + sqrt_term2) # Log term for suspended line 
    
    # Construct different formulations according to seabed_contact True/False
    if seabed_contact:
        xf = (lb + (ht / w) * log_term + ht * l / ea)  
        zf = ((ht/w) * (sqrt_term - 1) + vt ** 2 / (2 * ea * w)) 
    
    else: 
        xf = (ht / w) * (log_term -  log_term2) + ht * l / ea             
        zf = (ht / w) * (sqrt_term - sqrt_term2) + (1 / ea) * (vt * l - 0.5 * w * l ** 2)
            
    # Substitute known numeric values if required, vt and ht stay symbolic
    if numorsim == 'num':    
        xf_num = xf.subs({w: w_num, ea: ea_num})
        zf_num = zf.subs({w: w_num, ea: ea_num})

        if l_num is not None:
            xf_num = xf_num.subs(l, l_num)
            zf_num = zf_num.subs(l, l_num)
        
        return xf_num, zf_num
    
    # Return fully sybolic fairlead equations otherwise
    else:
        return xf, zf

def get_profile_equations(seabed_contact, numorsim, w_num = None, ea_num = None, l_num = None):
    '''
    Returns 2D profile equations for homogeneous catenary line of linear stiffness, as described in Jonkman 2009. 
    This is called by one_sec, two_sec and qs_offset functions.

    See get_fairlead_equations for description of inputs

    '''
    # Create symolic variables
    w, l, ea, ht, vt, s = sp.symbols('w l ea ht vt s')

    # Length of line along seabed
    lb  = l - vt / w 
    
    # Vt/ht quotients
    vt_ht    = vt             / ht  # Ratio of vert to horz tensions
    vt_ws_ht = (vt + w * s)   / ht
    ws_lb_ht = (w * (s - lb)) / ht
    
    # Sqrt terms
    sqrt_term  = sp.sqrt(1 + vt_ht ** 2)    # Sqrt term for non-suspended line
    sqrt_term3 = sp.sqrt(1 + vt_ws_ht ** 2) # Sqrt term for line profile suspended
    sqrt_term4 = sp.sqrt(1 + ws_lb_ht ** 2) # Sqrt  term for

    # Log terms
    log_term  = sp.log(vt_ht + sqrt_term) # Log term for non-suspended line        
    log_term3 = sp.log(vt_ws_ht + sqrt_term3) # Log term for line profile suspended
    log_term4 = sp.log(ws_lb_ht + sqrt_term4) # Log term for line profile suspended
            
    if seabed_contact:
        xs = sp.Piecewise((s, (s <= lb)), # nodes on seabed
                          (lb + (ht/w) * log_term4 + (ht * s) / ea,
                           sp.And(lb < s, s <= l)) # nodes between liftoff and top   
                          )          
        
        zs = sp.Piecewise((0, (s <= lb)), # nodes on seabed
                          ((ht / w) * (sqrt_term4 - 1) + (w * (s - lb) ** 2) / (2 * ea),
                           sp.And(lb < s, s <= l)) # nodes between liftoff and top 
                             )
    else: 
        xs = (ht / w) * (log_term3 - log_term) + ht * s / ea             
        zs = (ht / w) * (sqrt_term3 - sqrt_term) + (1 / ea) * (vt * s  + 0.5 * w * s ** 2)
        # vt term in xs and zs should actually be va (vert anchor tension), so sub in va
        va = vt - w * l
        xs = xs.subs({vt: va})
        zs = zs.subs({vt: va})
             
    if numorsim == 'num':    
        xs_num = xs.subs({w: w_num, ea: ea_num})
        zs_num = zs.subs({w: w_num, ea: ea_num})

        if l_num is not None: 
            xs_num = xs_num.subs(l, l_num)
            zs_num = zs_num.subs(l, l_num)
                        
        return xs_num, zs_num
    
    else:
        return xs, zs

# def plot_profile(moortype, lrd, xf, zf, sec1_xs, sec1_zs, sec2_xs=None, sec2_zs=None, lrd_xs=None, lrd_zs=None):
#     '''
#     Returns plotly fig of 2D mooring line profile, from xs and zs values. I.e. just plotting once the geometry is fully solved.
#     This is called by one_sec and two_sec functions, and by qs_offset function.

#     moortype : 'one_sec' or 'two_sec', determines the number of sections to plot
#     lrd : None or LrdDesign object, determines if LRD is plotted
#     xf : float, x-coordinate of fairlead
#     zf : float, z-coordinate of fairlead
#     sec1_xs : list of floats, x-coordinates of section 1
#     sec1_zs : list of floats, z-coordinates of section 1
#     sec2_xs : list of floats, x-coordinates of section 2
#     sec2_zs : list of floats, z-coordinates of section 2
#     lrd_xs : list of floats, x-coordinates of LRD
#     lrd_zs : list of floats, z-coordinates of LRD
#     '''

#     # Create a new plot
#     fig = go.Figure()

#     # Depending on the mooring type, plot the sections
#     if moortype == 'one_sec':
#         # Plot the line profile for section 1
#         fig.add_trace(go.Scatter(x=sec1_xs, y=sec1_zs, mode='lines', name='Section 1', line=dict(color='black')))
#     elif moortype == 'two_sec':
#         # Plot the line profiles for sections 1 and 2
#         fig.add_trace(go.Scatter(x=sec1_xs, y=sec1_zs, mode='lines', name='Section 1', line=dict(color='black')))
#         fig.add_trace(go.Scatter(x=sec2_xs, y=sec2_zs, mode='lines', name='Section 2', line=dict(color='blue')))

#     # Overlay the LRD part in red if there are any points to plot
#     if lrd:
#         fig.add_trace(go.Scatter(x=lrd_xs, y=lrd_zs, mode='lines', name='LRD', line=dict(color='red')))

#     # Set plot labels and axis limits
#     fig.update_layout(title='Mooring Profile' + ' ' + moortype + ' LRD: ' + str(lrd.lrd_type if lrd else 'None'),
#                       xaxis_title='Horzontal position (m)',
#                       yaxis_title='Vertical position (m)',
#                       xaxis=dict(range=[0, xf + 5]),
#                       yaxis=dict(range=[-1, zf + 50]),
#                       showlegend=True,
#                       margin=dict(l=0, r=0, t=40, b=0))
    
#     # Set aspect ratio
#     fig.update_yaxes(scaleanchor="x", scaleratio=1)

#     return fig

# def animate(title, frames, type, xf0=None, zf0=None, max_offset=None):
#     '''
#     Returns animation of mooring line profile, stiffness curve or LRD drawing. This is called by qs_offset function.

#     title : str, title of the animation (this title is constructed automatically in one_sec and two_sec functions and passed on to offset_funct in init_package)
#     frames : list of go.Frames, each containing the data for a single iteration of the animation
#     type : 'profile', 'draw' or 'stiffness', determines the formatting of animation depending on the type of plot
#     xf0 : float, x-coordinate of fairlead (only required for profile animation)
#     zf0 : float, z-coordinate of fairlead (only required for profile animation)
#     max_offset : float, maximum horizontal offset of the LRD (only required for profile animation)

#     '''

#     if type == 'profile':
#         xaxis = dict(range=[0, xf0 + max_offset + 10])
#         yaxis = dict(range=[0, zf0], scaleanchor="x", scaleratio=1)
#     elif type == 'draw':
#         xaxis = dict(constrain='domain')
#         yaxis = dict(scaleanchor="x", scaleratio=1)
#     elif type == 'stiffness':
#         xaxis = dict(constrain='domain')
#         yaxis = dict()

#     animation_fig = go.Figure(
#         data=frames[0].data,
#         layout=go.Layout(
#             updatemenus=[dict(type='buttons', showactive=False,
#                               buttons=[dict(label='Play',
#                                             method='animate',
#                                             args=[None])])],
#             sliders=[dict(steps=[dict(method='animate',
#                                       args=[[f.name]],
#                                       label=f.name) for f in frames],
#                           active=0)],
#             autosize=True,
#             margin=dict(l=50, r=50, b=100, t=100, pad=4),
#             xaxis=xaxis,
#             yaxis=yaxis,
#             title=title + ' animation'
#         ),
#         frames=frames
#     )
#     return animation_fig

    
# def plot_tension_offset(title, displacement_values, tension_values):
#     '''
#     Returns plotly fig of the tension-offset curve. This is called by qs_offset function.

#     title : str, title of the plot
#     displacement_values : list of floats, horizontal offsets
#     tension_values : list of floats, fairlead tensions

#     '''

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=displacement_values, y=tension_values, mode='lines', name='Tension-offset'))
#     fig.update_layout(title=title, xaxis_title='Horizontal offset (m)', yaxis_title='Fairlead tension (kN)', showlegend=False)

#     return fig      
    

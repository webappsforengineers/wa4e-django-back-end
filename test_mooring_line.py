from mooring_line_calc.lrd_module import LrdDesign, get_lrd_strain
from mooring_line_calc.one_sec import one_sec_init
from mooring_line_calc.two_sec import two_sec_init
import math

def initialise_mooring():
    # mooring types: (1) catenary full-chain, (2) catenary with rope, (3) taut full-rope, (4) taut with bottom chain
    mooring_type = "4"

    if mooring_type == "1":
        nsecs = 1
        seabed_contact = True
    elif mooring_type == "2":
        nsecs = 2
        seabed_contact = True
    elif mooring_type == "3":
        nsecs = 1
        seabed_contact = False
    elif mooring_type == "4":
        nsecs = 2
        seabed_contact = False

    # when taking inputs from the post data this will look like zf = request.data.get('zf')
    zf = 150

    if seabed_contact:
        xf = 750
    else:
        taut_angle = 30
        xf = zf / math.tan(math.radians(taut_angle))
        
    preten = 1000 * 1e3

    lrd_type = "3"

    if lrd_type == "1":
        lrd = None
    elif lrd_type == "2":
        # Tfi SeaSpring properties
        tfi_l = 10
        tfi_rt_kN = 2500
        lrd = LrdDesign("tfi", tfi_l=tfi_l, tfi_rs=0.5, tfi_rt=tfi_rt_kN * 1e3)

    elif lrd_type == "3":
        # Dublin Offshore LRD properties
        do_l = 15
        do_d = 3
        do_h = 0.3 
        do_v = 4
        do_rho = 3.8
        do_theta = taut_angle if not seabed_contact else 45
        lrd = LrdDesign("do", do_d=do_d, do_l=do_l, do_h=do_h, do_v=do_v, do_theta=do_theta, do_rho=do_rho)


    if seabed_contact:  # Catenary mooring, meaning this is chain
        ea1 =  500 * 1e6 
        w1 = 200 * 9.81 

    else: # Taut mooring, meaning this is rope if nsecs == 1, or chain if nsecs == 2
        if nsecs == 1: # Taut mooring with only rope
            ea1 = 50 * 1e6 
            w1 = 5 * 9.81 
        else: # Taut mooring with chain at the bottom (sec 1) and rope on top section (sec 2)
            ea1 = 500 * 1e6
            w1 = 200 * 9.81

    # Prompt for section 2 properties (always rope for taut, can be chain or rope for catenary)
    if nsecs == 2:

        if not seabed_contact:
            l1 = 10
            ea2 = 50 * 1e6 
            w2 = 5 * 9.81 
            l2 = zf / math.sin(math.radians(taut_angle)) - l1 - lrd.l if lrd else zf / math.sin(math.radians(taut_angle)) - l1

        else:
            l2 = 50
            ea2 = 50 * 1e6
            w2 = 5 * 9.81
            
    print('Initialising mooring system geometry...')
    if nsecs == 1:
        init = one_sec_init(seabed_contact=seabed_contact, lrd=lrd, at=preten, xf=xf, zf=zf, ea=ea1, w=w1)
    else:
        init = two_sec_init(seabed_contact=seabed_contact, lrd=lrd, at=preten, xf=xf, zf=zf, ea1=ea1, w1=w1, ea2=ea2, w2=w2, l2=l2)

    print('Vertical tension = ', init['vt0'] / 1e3 , 'kN')
    print('Horizontal tension = ', init['ht0'] / 1e3 , 'kN')
    print('Required length of section 1 to achieve pre-tension = ', init['sec1_l'] , 'm')
    
    return init['vt0'], init['ht0'], init['sec1_l']
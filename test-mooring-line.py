from mooring_line_calc.lrd_module import LrdDesign, get_lrd_strain
from mooring_line_calc.one_sec import one_sec_init
from mooring_line_calc.two_sec import two_sec_init
import math

mooring_type=1

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

# when taking inputs from the post data this will look like tfi_rt = request.data.get('zf')
zf = 150

if seabed_contact:
    xf = 750
else:
    taut_angle = 30
    xf = zf / math.tan(math.radians(taut_angle))
    
at = 1000 * 1e3

lrd_type = 3

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
    do_d = float(input("Diameter of the Dublin Offshore LRD in m (default 3): ") or 3) # Lower bound = 1, upper = 10
    do_h = float(input("Horizontal distance from the hinge point to the centre of gravity in m (default 0.3): ") or 0.3) # Lower bound = 0, upper = 3. Cannot be greater than do_d
    do_v = float(input("Vertical distance between the hinge points in m (default 4): ") or 4)  # Lower bound = 1, upper = 10. Cannot be greater than do_l
    do_rho = float(input("Density of the ballast in T/m^3 (default 3.8): ") or 3.8) # Lower bound = 1.5, upper = 5
    do_theta = taut_angle if not seabed_contact else 45
    lrd = LrdDesign("do", do_d=do_d, do_l=do_l, do_h=do_h, do_v=do_v, do_theta=do_theta, do_rho=do_rho)



ea1 = 500 * 1e6
w1 = 200 * 9.81

ea2 = 50 * 1e6
w2 = 5 * 9.81
l2 = 50

# lrd_type = "tfi"
# tfi_l = 10
# tfi_rs = 0.5
# # when taking inputs from the post data this will look like tfi_rt = request.data.get('tfi_rt') * 1e3
# tfi_rt = 2500 * 1e3

lrd_type = "do"
do_d = 3
do_l = 15
do_h = 0.3
do_v = 4
do_rho = 3.8
do_theta = taut_angle if not seabed_contact else 45

at = 1000 * 1e3
xf = 750
zf = 150
ea1 = 500 * 1e6
w1 = 200 * 9.81

if  nsecs == 2:
    ea2 = 50 * 1e6
    w2 = 5 * 9.81

    if seabed_contact == True:
        l2 = 50
    else:
        l2 = zf / math.sin(math.radians(taut_angle)) - l1 - lrd.l if lrd else zf / math.sin(math.radians(taut_angle)) - l1



# lrd = LrdDesign(lrd_type=lrd_type, tfi_l=tfi_l, tfi_rs=tfi_rs, tfi_rt=tfi_rt )
# print(lrd)

lrd = LrdDesign(led_type=lrd_type, do_d=do_d, do_l=do_l, do_h=do_h, do_v=do_v, do_theta=do_theta, do_rho=do_rho)
print(lrd)

init = one_sec_init(seabed_contact=True, lrd=lrd, at=at, xf=xf, zf=zf, ea=ea1, w=w1)

print('Vertical tension = ', init['vt0'] / 1e3 , 'kN')
print('Horizontal tension = ', init['ht0'] / 1e3 , 'kN')
print('Required length of section 1 to achieve pre-tension = ', init['sec1_l'] , 'm')

init2 = two_sec_init(seabed_contact=True, lrd=lrd, at=at, xf=xf, zf=zf, ea1=ea1, w1=w1, ea2=ea2, w2=w2, l2=l2)

print('Vertical tension = ', init['vt0'] / 1e3 , 'kN')
print('Horizontal tension = ', init['ht0'] / 1e3 , 'kN')
print('Required length of section 1 to achieve pre-tension = ', init['sec1_l'] , 'm')
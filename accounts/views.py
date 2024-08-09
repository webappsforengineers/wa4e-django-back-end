from rest_framework import status, viewsets, permissions, generics
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from .serializers import UserSerializer, FileUploadSerializer
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.core.exceptions import ObjectDoesNotExist
from .models import CustomUser
import scipy.io
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from mooring_line_calc.lrd_module import LrdDesign, get_lrd_strain
from mooring_line_calc.one_sec import one_sec_init
from mooring_line_calc.two_sec import two_sec_init
from mooring_line_calc.offset_funct import qs_offset
from wlgr_calc.model_functions import pwp_acc, update_properties, update_applied_loads, findOCR, consolidate, check_failure
from sympy import N
import math

from .tasks import long_running_task
from celery.result import AsyncResult

class StartTaskView(APIView):
    def post(self, request):
        task = long_running_task.delay()
        return Response({'task_id': task.id}, status=status.HTTP_202_ACCEPTED)

class TaskStatusView(APIView):
    def get(self, request, task_id):
        result = AsyncResult(task_id)
        if result.state == 'PENDING':
            response = {'status': 'pending'}
        elif result.state == 'SUCCESS':
            response = {'status': 'completed', 'result': result.result}
        else:
            response = {'status': 'in-progress'}
        return Response(response, status=status.HTTP_200_OK)


# User accounts

# curl -X POST -H "Content-Type: application/json" -d '{"username": "testuser", "password": "testpassword", "email": "test@example.com"}' http://localhost:8000/api/register/
@api_view(['POST'])
def register_user(request):
    if request.method == 'POST':
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


    
# curl -X POST -H "Content-Type: application/json" -d '{"username": "testuser", "password": "testpassword"}' http://localhost:8000/api/login/   
@api_view(['POST'])
def user_login(request):
    if request.method == 'POST':
        username = request.data.get('username')
        password = request.data.get('password')

        user = None
        # if '@' in username:
        #     try:
        #         user = CustomUser.objects.get(email=username)
        #     except ObjectDoesNotExist:
        #         pass

        if not user:
            user = authenticate(username=username, password=password)

        if user:
            token, _ = Token.objects.get_or_create(user=user)

            if user.is_staff:
                user_type = 'admin user'

            else:
                user_type = 'regular user'

            return Response({'token': token.key, 'user_type': user_type}, status=status.HTTP_200_OK)
        

    return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)


# curl -X POST -H "Authorization: Token YOUR_AUTH_TOKEN" http://localhost:8000/api/logout/  
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def user_logout(request):
    if request.method == 'POST':
        try:
            # Delete the user's token to logout
            request.user.auth_token.delete()
            return Response({'message': 'Successfully logged out.'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

# curl -H "Authorization: Token AUTH TOKEN HERE" http://localhost:8080/api/list-users/
# Get a list of users   
class UserList(generics.ListCreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]


# curl http://localhost:8080/api/select-user/email/
class SelectUser(generics.ListAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]

    def get_queryset(self):
        email = self.kwargs['email']
        return CustomUser.objects.filter(email=email)

    
# curl -X DELETE "Authorization: Token AUTH TOKEN HERE" http://localhost:8080/api/delete-user/pk/
#delete user
class DeleteUser(generics.DestroyAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]


#get the currently logged in user
# curl -H "Authorization: Token AUTH TOKEN HERE" http://localhost:8080/api/current-user/
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)

# GgSANDnet
# curl -X POST -H "Content-Type: application/json" -d '{"username": "testuser", "password": "testpassword"}' http://localhost:8000/api/login/  
class FileUploadView(APIView):
    parser_class=(FileUploadParser,)
    
    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)
        
        if serializer.is_valid():
            return Response({'message': 'File uploaded sucessfully'})
        else:
            return Response(serializer.errors, status=400)


# Load data from .mat data file
def load_data(filename):
    mat = scipy.io.loadmat(filename)

    properties = mat["properties"]
    curves = mat["curves"]
    return properties, curves

# shuffle inputs and outputs in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Function to expand arrays into multiple columns
def expand_array(row):
    return pd.Series(row['column_of_arrays']) 

def divide_dataset(train_ratio, val_ratio, inputs, targets):
    # Set up Division of Data for Training, Validation, Testing

    num_samples = inputs.shape[0]
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)

    x_train, y_train = inputs[0:num_train,:], targets[0:num_train, :]
    x_val, y_val = inputs[num_train:num_train + num_val, :], targets[num_train:num_train + num_val, :]
    x_test, y_test = inputs[num_train + num_val:, :], targets[num_train + num_val:, :]
    
    return x_train, y_train, x_val, y_val, x_test, y_test


@api_view(['POST'])
def train_model(request):
    properties, curves = load_data('StiffnessNNAppData.mat')
    processed_data = {'input': np.empty((0, 2)), 'output': np.empty((0, 1))}
    
    for j in range(len(properties)):
        for k in range(len(properties[j][0])):
            for l in range(curves[j][0][k][0].shape[0]):
                processed_data['input'] = np.vstack((processed_data['input'], np.hstack((properties[j][0][k].T, curves[j][0][k][0][l, 0]))))
                processed_data['output'] = np.vstack((processed_data['output'], curves[j][0][k][0][l, 1]))
    
    for i in range(len(processed_data['input'])):
        processed_data['input'][i][0] = np.append(processed_data['input'][i][0], processed_data['input'][i][1])
        
    processed_data['input'] = processed_data['input'][:,0]
    
    shuffled_data = unison_shuffled_copies(processed_data['input'], processed_data['output'])
    
    # reshape input data
    inputs_array = pd.DataFrame(shuffled_data[0])
    inputs_array.columns = ['column_of_arrays']
    inputs_expanded = inputs_array.apply(expand_array, axis=1)
    
    prop_select = request.data.get('prop_select')
    selected_inputs = inputs_expanded[np.append(prop_select, 8)]
    inputs = selected_inputs.to_numpy()
    
    targets = shuffled_data[1]
    
    # Construct a neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(prop_select) + 1,)),
        tf.keras.layers.Dense(units=128, activation='tanh', input_shape=(len(prop_select) + 1,)),
        tf.keras.layers.Dense(units=1, activation='linear')  # Output layer
    ])
    print(model.layers)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Nadam(), loss='mse')
    
    # learning rate scheduler
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_delta=0.0001, min_lr=0.00001)
    
    x_train, y_train, x_val, y_val, x_test, y_test = divide_dataset(0.7, 0.15, inputs, targets)
    
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), batch_size=16, callbacks=[lr_callback])
    
    performance = model.evaluate(x_test, y_test)
    outputs = model.predict(inputs)
    
    strain = np.logspace(-4, 1, 200)
    GGo = []
    input_properties = np.array(request.data.get('input_properties'))

    for s in strain:
        inputs_prop = np.append(input_properties[prop_select], s).reshape(1, -1)
        GGo.append(model.predict(inputs_prop)[0][0])

    # Create a table (pandas DataFrame) with strain and GGo values
    data = {'Strain': strain, 'GGo': GGo}
    output_data = pd.DataFrame(data)
    
    return Response({'dataset_length': len(inputs), 'performance_mse': str(round(performance,4)), 'outputs': outputs, 'targets': targets, 'strain': strain, 'GGo': GGo, 'output_data': output_data})


@api_view(['POST'])    
def initialise_mooring(request):
    # mooring types: (1) catenary full-chain, (2) catenary with rope, (3) taut full-rope, (4) taut with bottom chain
    mooring_type = request.data.get('mooring_type')

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
    zf = request.data.get('zf')

    if seabed_contact:
        xf = request.data.get('xf')
    else:
        taut_angle = request.data.get('taut_angle')
        xf = zf / math.tan(math.radians(90 - taut_angle))
        
    preten = request.data.get('preten') * 1e3

    lrd_type = request.data.get('lrd_type')
    
    if lrd_type == "1":
        lrd = None
        tfi_l = None
        tfi_d = None
        
        do_l = None
        do_d = None
        do_h = None
        do_v = None
        do_rho = None
        do_theta = None
        do_hba = None
        do_o = None
        
    elif lrd_type == "2":
        # Tfi SeaSpring properties
        tfi_l = request.data.get('tfi_l')
        tfi_rt_kN = request.data.get('tfi_rt_kN')
        lrd = LrdDesign("tfi", tfi_l=tfi_l, tfi_rs=0.5, tfi_rt=tfi_rt_kN * 1e3)
        tfi_d = lrd.tfi_d
        
        do_l = None
        do_d = None
        do_h = None
        do_v = None
        do_rho = None
        do_theta = None
        do_hba = None
        do_o = None

    elif lrd_type == "3":
        # Dublin Offshore LRD properties
        do_l = request.data.get('do_l')
        do_d = request.data.get('do_d')
        do_h = request.data.get('do_h')
        do_v = request.data.get('do_v')
        do_rho = request.data.get('do_rho')
        do_theta = taut_angle if not seabed_contact else None
        lrd = LrdDesign("do", do_d=do_d, do_l=do_l, do_h=do_h, do_v=do_v, do_theta=do_theta, do_rho=do_rho)
        do_hba = lrd.do_hba
        do_o = lrd.do_o
        tfi_l = None
        tfi_d = None


    if seabed_contact:  # Catenary mooring, meaning this is chain
        ea1 =  request.data.get('ea1') * 1e6 
        w1 = request.data.get('w1') * 9.81 

    else: # Taut mooring, meaning this is rope if nsecs == 1, or chain if nsecs == 2
        if nsecs == 1: # Taut mooring with only rope
            ea1 = request.data.get('ea1') * 1e6 
            w1 = request.data.get('w1') * 9.81 
        else: # Taut mooring with chain at the bottom (sec 1) and rope on top section (sec 2)
            ea1 = request.data.get('ea1') * 1e6
            w1 = request.data.get('w1') * 9.81

    # Prompt for section 2 properties (always rope for taut, can be chain or rope for catenary)
    if nsecs == 2:

        if not seabed_contact:
            l1 = request.data.get('l1')
            ea2 = request.data.get('ea2') * 1e6 
            w2 = request.data.get('w2') * 9.81 
            l2 = zf / math.sin(math.radians(taut_angle)) - l1 - lrd.l if lrd else zf / math.sin(math.radians(taut_angle)) - l1

        else:
            l2 = request.data.get('l2')
            ea2 = request.data.get('ea2') * 1e6
            w2 = request.data.get('w2') * 9.81
            
    print('Initialising mooring system geometry...')
    if nsecs == 1:
        init = one_sec_init(seabed_contact=seabed_contact, lrd=lrd, at=preten, xf=xf, zf=zf, ea=ea1, w=w1)
    else:
        init = two_sec_init(seabed_contact=seabed_contact, lrd=lrd, at=preten, xf=xf, zf=zf, ea1=ea1, w1=w1, ea2=ea2, w2=w2, l2=l2)
    
    # variables for LRD stiffness curve
    
    if lrd_type == "1":
        at_values = []
        ext_or_str_values = []
        corner_xs = []
        corner_zs = []
        smaller_corner_xs = []
        smaller_corner_zs = []
        line_from_hinge_x = []
        line_from_hinge_y = []
        alpha = None
        
    # Generate 100 evenly spaced axial tension values between T_min and T_max
    elif lrd_type == "2":   
        tfi_rt = tfi_rt_kN * 1e3
        at_values = np.linspace(0.0, tfi_rt * 1.5, 100)
        # Get modelled data from stiffness equation
        ext_or_str_values = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values]
        print('ext_or_str_values:', ext_or_str_values)
        corner_xs = []
        corner_zs = []
        smaller_corner_xs = []
        smaller_corner_zs = []
        line_from_hinge_x = []
        line_from_hinge_y = []
        alpha = None
        
    elif lrd_type == "3":
        at_values = np.linspace(0.1, lrd.do_fg * 4, 100)
        ext_or_str_values = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values]
        
        # Variables for the mooring line profile
        # Plot the outline of the lrd.
        # Find the center coordinates of the LRD, i.e. the mid point between the hinges
        lrd_center_x = (init['xs_values_lrd'][0] + init['xs_values_lrd'][-1]) / 2
        lrd_center_z = (init['zs_values_lrd'][0] + init['zs_values_lrd'][-1]) / 2
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
        mooring_angle = np.arctan(init['vt0'] / init['ht0'])
        lower_hinge_x, lower_hinge_z = init['xs_values_lrd'][-1], init['zs_values_lrd'][-1]
        line_end_x = lower_hinge_x + lrd.do_d * np.cos(mooring_angle)
        line_end_z = lower_hinge_z + lrd.do_d * np.sin(mooring_angle)
        line_from_hinge_x = [lower_hinge_x, line_end_x]
        line_from_hinge_y = [lower_hinge_z, line_end_z]
        
    # Convert numpy floats to Python floats
    at_values = [float(value)/1000 for value in at_values]
    ext_or_str_values = [float(value) for value in ext_or_str_values]
    
    print(init['lrd_extension'])
    
    if init['at_calculated']:
        at_calculated = init['at_calculated']/1000
    else:
        at_calculated = None

    
    return Response({

                    'sec1_l': init['sec1_l'],
                    'vt0':init['vt0'], 
                    'ht0': init['ht0'],
                    
                    'xs_values_sec1': init['xs_values_sec1'],
                    'zs_values_sec1': init['zs_values_sec1'],
                    'xs_values_lrd': init['xs_values_lrd'],
                    'zs_values_lrd': init['zs_values_lrd'],
                    'xs_values_sec2': init['xs_values_sec2'],
                    'zs_values_sec2': init['zs_values_sec2'],
                    'corner_xs': corner_xs,
                    'corner_zs': corner_zs,
                    'smaller_corner_xs': smaller_corner_xs,
                    'smaller_corner_zs': smaller_corner_zs,
                    'line_from_hinge_x': line_from_hinge_x,
                    'line_from_hinge_y': line_from_hinge_y, 
                    
                    'at_values': at_values,
                    'ext_or_str_values': ext_or_str_values,
                    'at_calculated': at_calculated,
                    'lrd_extension': init['lrd_extension'],
                    
                    'tfi_l': tfi_l,
                    'tfi_d': tfi_d,
                    
                    'do_l': do_l,
                    'do_d': do_d,
                    'do_h': do_h,
                    'do_v': do_v,
                    'do_rho': do_rho,
                    'do_theta': do_theta,
                    'do_hba': do_hba,
                    'do_o': do_o,
                    
                    'full_rectangle_rotated': init['full_rectangle_rotated'],
                    'bottom_rectangle_rotated': init['bottom_rectangle_rotated'],
                    'top_hinge_rotated': init['top_hinge_rotated'],
                    'bottom_hinge_rotated': init['bottom_hinge_rotated'],
                    
                    'top_hinge_arrow_endpoint_x': init['top_hinge_arrow_endpoint_x'],
                    'top_hinge_arrow_endpoint_y': init['top_hinge_arrow_endpoint_y'],
                    'bottom_hinge_arrow_endpoint_x': init['bottom_hinge_arrow_endpoint_x'],
                    'bottom_hinge_arrow_endpoint_y': init['bottom_hinge_arrow_endpoint_y'],
                    'alpha': alpha,

                     })
    
    
    
@api_view(['POST'])    
def run_qs_offset(request):
    '''
    Returns the data to produce a tension-offset curve for a given mooring system.

    init: dict returned by one_sec_init or two_sec_init
    max_offset: maximum offset in meters
    resolution: number of points per meter
    profile_plot: boolean to determine if the animations should be plotted (much quicker if set to False)

    '''
    # mooring types: (1) catenary full-chain, (2) catenary with rope, (3) taut full-rope, (4) taut with bottom chain
    mooring_type = request.data.get('mooring_type')

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
    zf = request.data.get('zf')

    if seabed_contact:
        xf = request.data.get('xf')
    else:
        taut_angle = request.data.get('taut_angle')
        xf = zf / math.tan(math.radians(90 - taut_angle))
        
    preten = request.data.get('preten') * 1e3

    lrd_type = request.data.get('lrd_type')
    
    if lrd_type == "1":
        lrd = None
        tfi_l = None
        tfi_d = None
        
        do_l = None
        do_d = None
        do_h = None
        do_v = None
        do_rho = None
        do_theta = None
        do_hba = None
        do_o = None
        
    elif lrd_type == "2":
        # Tfi SeaSpring properties
        tfi_l = request.data.get('tfi_l')
        tfi_rt_kN = request.data.get('tfi_rt_kN')
        lrd = LrdDesign("tfi", tfi_l=tfi_l, tfi_rs=0.5, tfi_rt=tfi_rt_kN * 1e3)
        tfi_d = lrd.tfi_d
        
        do_l = None
        do_d = None
        do_h = None
        do_v = None
        do_rho = None
        do_theta = None
        do_hba = None
        do_o = None

    elif lrd_type == "3":
        # Dublin Offshore LRD properties
        do_l = request.data.get('do_l')
        do_d = request.data.get('do_d')
        do_h = request.data.get('do_h')
        do_v = request.data.get('do_v')
        do_rho = request.data.get('do_rho')
        do_theta = taut_angle if not seabed_contact else None
        lrd = LrdDesign("do", do_d=do_d, do_l=do_l, do_h=do_h, do_v=do_v, do_theta=do_theta, do_rho=do_rho)
        do_hba = lrd.do_hba
        do_o = lrd.do_o
        tfi_l = None
        tfi_d = None


    if seabed_contact:  # Catenary mooring, meaning this is chain
        ea1 =  request.data.get('ea1') * 1e6 
        w1 = request.data.get('w1') * 9.81 

    else: # Taut mooring, meaning this is rope if nsecs == 1, or chain if nsecs == 2
        if nsecs == 1: # Taut mooring with only rope
            ea1 = request.data.get('ea1') * 1e6 
            w1 = request.data.get('w1') * 9.81 
        else: # Taut mooring with chain at the bottom (sec 1) and rope on top section (sec 2)
            ea1 = request.data.get('ea1') * 1e6
            w1 = request.data.get('w1') * 9.81

    # Prompt for section 2 properties (always rope for taut, can be chain or rope for catenary)
    if nsecs == 2:

        if not seabed_contact:
            l1 = request.data.get('l1')
            ea2 = request.data.get('ea2') * 1e6 
            w2 = request.data.get('w2') * 9.81 
            l2 = zf / math.sin(math.radians(taut_angle)) - l1 - lrd.l if lrd else zf / math.sin(math.radians(taut_angle)) - l1

        else:
            l2 = request.data.get('l2')
            ea2 = request.data.get('ea2') * 1e6
            w2 = request.data.get('w2') * 9.81
            
    print('Initialising mooring system geometry...')
    if nsecs == 1:
        init = one_sec_init(seabed_contact=seabed_contact, lrd=lrd, at=preten, xf=xf, zf=zf, ea=ea1, w=w1)
    else:
        init = two_sec_init(seabed_contact=seabed_contact, lrd=lrd, at=preten, xf=xf, zf=zf, ea1=ea1, w1=w1, ea2=ea2, w2=w2, l2=l2)
    
    all_plots = request.data.get('all_plots')  
    max_offset = request.data.get('max_offset')
    resolution = request.data.get('resolution')
    
        # variables for LRD stiffness curve
    all_plots = request.data.get('all_plots')
    at_values_qs_offset_tfi = []
    ext_or_str_values_qs_offset_tfi = []
    
    if all_plots:
        # Generate 100 evenly spaced axial tension values between T_min and T_max
        if lrd_type == "2":   
            tfi_rt = tfi_rt_kN * 1e3
            at_values_qs_offset_tfi = np.linspace(0.0, tfi_rt * 1.5, 100)
            # Get modelled data from stiffness equation
            ext_or_str_values_qs_offset_tfi = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values_qs_offset_tfi]
        
        # Convert numpy floats to Python floats
        at_values_qs_offset_tfi = [float(value)/1000 for value in at_values_qs_offset_tfi]
        ext_or_str_values_qs_offset_tfi = [float(value) for value in ext_or_str_values_qs_offset_tfi]
        
    

    
    tension_values, displacement_values, all_current_ext_or_str_values, all_xs_values_sec1, all_zs_values_sec1, all_xs_values_sec2, all_zs_values_sec2, all_xs_values_lrd, all_zs_values_lrd, all_tfi_current_lengths, all_ml_angles, all_full_rectangles_rotated, all_bottom_rectangles_rotated, all_top_hinges_rotated, all_bottom_hinges_rotated, all_corner_xs, all_corner_zs, all_smaller_corner_xs, all_smaller_corner_zs, all_line_from_hinge_x, all_line_from_hinge_y, all_at_values_qs_offset, all_ext_or_str_values_qs_offset = qs_offset(
        init, max_offset, resolution, all_plots)
    

    
    return Response({ 
                     'at_values_qs_offset_tfi': at_values_qs_offset_tfi,
                     'ext_or_str_values_qs_offset_tfi': ext_or_str_values_qs_offset_tfi,
                     'tension_values': tension_values,
                     'displacement_values': displacement_values,
                     'all_current_ext_or_str_values': all_current_ext_or_str_values,
                     'all_xs_values_sec1': all_xs_values_sec1,
                     'all_zs_values_sec1': all_zs_values_sec1,
                     'all_xs_values_sec2': all_xs_values_sec2,
                     'all_zs_values_sec2': all_zs_values_sec2,
                     'all_xs_values_lrd': all_xs_values_lrd,
                     'all_zs_values_lrd': all_zs_values_lrd,
                     
                     'tfi_l_qs_offset': tfi_l,
                     'tfi_d_qs_offset': tfi_d,
                     'all_tfi_current_lengths': all_tfi_current_lengths,
                     
                     'all_ml_angles': all_ml_angles, 
                     'all_full_rectangles_rotated': all_full_rectangles_rotated, 
                     'all_bottom_rectangles_rotated': all_bottom_rectangles_rotated,
                     'all_top_hinges_rotated': all_top_hinges_rotated,
                     'all_bottom_hinges_rotated': all_bottom_hinges_rotated,
                     
                     'all_corner_xs': all_corner_xs, 
                     'all_corner_zs': all_corner_zs, 
                     'all_smaller_corner_xs': all_smaller_corner_xs, 
                     'all_smaller_corner_zs': all_smaller_corner_zs, 
                     'all_line_from_hinge_x': all_line_from_hinge_x, 
                     'all_line_from_hinge_y': all_line_from_hinge_y,
                     
                     'all_ext_or_str_values_qs_offset': all_ext_or_str_values_qs_offset,
                     'all_at_values_qs_offset': all_at_values_qs_offset,
                     
    
                     })

@api_view(['POST'])
def calculate_wlgr(request):
    # === INPUT ===
    # soil material parameters
    kappa_oed = request.data.get('kappa_oed')
    lambda_NCL = request.data.get('lambda_ncl')
    gamma_NCL = request.data.get('gamma_ncl')
    gamma_CSL = request.data.get('gamma_csl')
    su0_sigmav = request.data.get('su0_sigmav')


    # === DEFAULT MODEL PARAMETERS ===
    #SN curves
    k1 = 0.626
    k2_OCR1 = 0.41335
    k3 = 6.517
    k4 = 0.001

    a = -0.42
    b = 0.45
    c = 0 
    d = 0 

    #A0 su fit
    c_A0 = -22.57
    d_A0 = 4.9541

    #kappa-D-eR
    zeta = 1.15 
    rho = 12 
    m = 0.05 
    p = 2.85 
    q = 1 

    G_su_nc = 1000
    r = 0.8


    # === INITIAL PARAMETERS
    # soil parameters
    sigmavc = request.data.get('sigmavc')
    OCR = 1
    sigmav = sigmavc

    e0 = gamma_NCL - lambda_NCL * (np.log(sigmavc)) 
    e = e0

    su0 = sigmavc * su0_sigmav #kPa
    su = su0

    G0 = G_su_nc * su0 #kPa
    G = G0

    kappa = kappa_oed

    # applied loading
    tau_su_per_episode = request.data.get('tau_su_per_episode')
    ncyc_per_episode = request.data.get('ncyc_per_episode')
    n_episodes = request.data.get('n_episodes')
    loading_DSS_app = [[tau_su_per_episode, ncyc_per_episode]]
    tau = tau_su_per_episode * su0


    # === create result lists for plotting ===
    Ds = [0]
    es = [e0]
    sigmavs = [sigmavc]
    kappas = [kappa_oed]
    OCRs = [OCR]
    sus = [su0]
    Gs = [G0]
    i_episode = [0]
    sigmavs_all = [sigmavc]
    es_all = [e0]
    failures = [False]
    
    xs_NCL = np.linspace(1, 500, 5)
    ys_NCL = [gamma_NCL - lambda_NCL * np.log(x) for x in xs_NCL]
    ys_CSL = [gamma_CSL - lambda_NCL * np.log(x) for x in xs_NCL]


    # === run simuation ===
    for i in range(n_episodes):
        D, k2, Ncarry = pwp_acc(loading_DSS_app, k1, k2_OCR1, k3, k4, a, b, OCR)
        Ds.append(D)
        sigmav = sigmav - (D * sigmavc)
        sigmavs.append(sigmav)
        sigmavs_all.append(sigmav)
        es_all.append(e)
        # kappas.append(kappa)
        # i_episode.append(i+1)
        
        failure = check_failure(D, e, sigmav, lambda_NCL, gamma_CSL)
        failures.append(failure)

        OCR = findOCR(e, kappa, sigmav, gamma_NCL, lambda_NCL)
        # OCRs.append(OCR)
        # sus.append(su)
        # Gs.append(G)

        e, kappa, OCR = consolidate(
            kappa_oed, sigmavc, gamma_NCL, gamma_CSL, lambda_NCL, 
            OCR, D, e, es[0], 
            m, p, q, zeta, rho
            )
        sigmav = sigmavc
        sigmavs_all.append(sigmav)
        es.append(e)
        es_all.append(e)
        kappas.append(kappa)
        OCRs.append(OCR)
        # Ds.append(0)
        i_episode.append(i+1)

        su, G = update_properties(kappa_oed, lambda_NCL, D, su, r, su0, G0, c_A0, d_A0, tau, sigmav)
        sus.append(su)
        Gs.append(G)

        loading_DSS_app = update_applied_loads(loading_DSS_app, su0, su)
    
    # print(i_episode, es, sigmavs, kappas, OCRs, sus, Gs, Ds, e0, su0, G0, xs_NCL, ys_NCL, ys_CSL)
        
        
    return Response({
        'i_episode': i_episode,
        'es': es, 
        'es_all': es_all,
        'sigmavs': sigmavs,
        'sigmavs_all': sigmavs_all,
        'kappas': kappas,
        'OCRs': OCRs,
        'sus': sus,
        'Gs': Gs,
        'Ds': Ds,
        'e0': e0,
        'su0': su0,
        'G0': G0,
        'xs_NCL': xs_NCL,
        'ys_NCL': ys_NCL,
        'ys_CSL': ys_CSL,
        'failures': failures,
        })
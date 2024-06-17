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
import math

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
        xf = zf / math.tan(math.radians(taut_angle))
        
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
        do_theta = taut_angle if not seabed_contact else 45
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

    # print('Vertical tension = ', init['vt0'] / 1e3 , 'kN')
    # print('Horizontal tension = ', init['ht0'] / 1e3 , 'kN')
    # print('Required length of section 1 to achieve pre-tension = ', init['sec1_l'] , 'm')
    
    # print(type(init))
    # print(init)
    
    # variables for LRD stiffness curve
    
    if lrd_type == "1":
        at_values = []
        ext_or_str_values = []
    # Generate 100 evenly spaced axial tension values between T_min and T_max
    elif lrd_type == "2":   
        tfi_rt = tfi_rt_kN * 1e3
        at_values = np.linspace(0.0, tfi_rt * 1.5, 100)
        print('at_values:', at_values)
        # Get modelled data from stiffness equation
        ext_or_str_values = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values]
        print('ext_or_str_values:', ext_or_str_values)
    elif lrd_type == "3":
        at_values = np.linspace(0.1, lrd.do_fg * 4, 100)
        ext_or_str_values = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values]
        
    # Convert numpy floats to Python floats
    at_values = [float(value) for value in at_values]
    ext_or_str_values = [float(value) for value in ext_or_str_values]

    
    return Response({
                    # 'xf_eq': init['xf_eq'],
                    #  'zf_eq': init['zf_eq'],
                    #  'xs1_eq': init['xs1_eq'],
                    #  'zs1_eq': init['zs1_eq'],
                    #  'xs2_eq': init['xs2_eq'],
                    #  'zs2_eq': init['zs2_eq'],
                     'sec1_l': init['sec1_l'],
                    #  'sec2_l': init['sec2_l'],
                    #  'lrd_x': init['lrd_x'],
                    #  'lrd_z': init['lrd_z'],
                    #  'lrd_alpha': init['lrd_alpha'],
                     'vt0':init['vt0'], 
                     'ht0': init['ht0'],
                    #  'xf0': init['xf0'],
                    #  'zf0': init['zf0'],
                    #  'lrd': init['lrd'],
                    #  's1_values': init['s1_values'],
                    #  's2_values': init['s2_values'],
                    #  'moortype': init['moortype'],
                    #  'name': init['name'],
                    'xs_values_sec1': init['xs_values_sec1'],
                    'zs_values_sec1': init['zs_values_sec1'],
                    'xs_values_lrd': init['xs_values_lrd'],
                    'zs_values_lrd': init['zs_values_lrd'],
                    'xs_values_sec2': init['xs_values_sec2'],
                    'zs_values_sec2': init['zs_values_sec2'],
                    
                    'at_values': at_values,
                    'ext_or_str_values': ext_or_str_values,
                    
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
        xf = zf / math.tan(math.radians(taut_angle))
        
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
        do_theta = taut_angle if not seabed_contact else 45
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
    # Generate 100 evenly spaced axial tension values between T_min and T_max
    elif lrd_type == "2":   
        tfi_rt = tfi_rt_kN * 1e3
        at_values = np.linspace(0.0, tfi_rt * 1.5, 100)
        print('at_values:', at_values)
        # Get modelled data from stiffness equation
        ext_or_str_values = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values]
        print('ext_or_str_values:', ext_or_str_values)
    elif lrd_type == "3":
        at_values = np.linspace(0.1, lrd.do_fg * 4, 100)
        ext_or_str_values = [get_lrd_strain(lrd, form = 'num', at = t) for t in at_values]
        
    # Convert numpy floats to Python floats
    at_values = [float(value) for value in at_values]
    ext_or_str_values = [float(value) for value in ext_or_str_values]
    
    max_offset = request.data.get('max_offset')
    resolution = request.data.get('resolution')
    tension_values, displacement_values, all_xs_values_sec1, all_zs_values_sec1, all_xs_values_sec2, all_zs_values_sec2, all_xs_values_lrd, all_zs_values_lrd = qs_offset(init, max_offset, resolution)

    
    return Response({ 
                     'tension_values': tension_values,
                     'displacement_values': displacement_values,
                     'all_xs_values_sec1': all_xs_values_sec1,
                     'all_zs_values_sec1': all_zs_values_sec1,
                     'all_xs_values_sec2': all_xs_values_sec2,
                     'all_zs_values_sec2': all_zs_values_sec2,
                     'all_xs_values_lrd': all_xs_values_lrd,
                     'all_zs_values_lrd': all_zs_values_lrd,
                     })

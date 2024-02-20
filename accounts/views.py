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
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=16, callbacks=[lr_callback])
    
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
    
    return Response({'dataset_length': len(inputs), 'performance_mse': str(performance), 'outputs': outputs, 'targets': targets, 'strain': strain, 'GGo': GGo, 'output_data': output_data})
    


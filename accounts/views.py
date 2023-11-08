from rest_framework import status, viewsets, permissions, generics
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from .serializers import UserSerializer
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.core.exceptions import ObjectDoesNotExist
from .models import CustomUser

# curl -X POST -H "Content-Type: application/json" -d '{"username": "testuser", "password": "testpassword", "email": "test@example.com"}' http://localhost:8000/api/register/
@api_view(['POST'])
def register_user(request):
    if request.method == 'POST':
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# curl -H "Authorization: Token AUTH TOKEN HERE" http://localhost:8080/api/list-users/
# Get a list of users   
class UserList(generics.ListCreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]


# curl http://localhost:8080/api/select-user/email/
class SelectUser(generics.ListAPIView):
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]

    def get_queryset(self):
        email = self.kwargs['email']
        return CustomUser.objects.filter(email=email)

    
# curl -X DELETE "Authorization: Token AUTH TOKEN HERE" http://localhost:8080/api/delete-user/pk/
#delete user
class DeleteUser(generics.DestroyAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]


#get the currently logged in user
# curl -H "Authorization: Token AUTH TOKEN HERE" http://localhost:8080/api/current-user/
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)

    
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
            return Response({'token': token.key}, status=status.HTTP_200_OK)

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
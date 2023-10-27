from django.urls import include, path, re_path
from .views import register_user, user_login, user_logout, UserList, current_user, SelectUser, DeleteUser



urlpatterns = [
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('register/', register_user, name='register'),
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),
    path('list-users/', UserList.as_view(), name='list-users'),
    path('current-user/', current_user, name='current-user'),
    re_path('^select-user/(?P<email>.+)/$', SelectUser.as_view(), name='select-user'),
    path('delete-user/<int:pk>/', DeleteUser.as_view(), name='delete-user'),
]
from django.urls import include, path, re_path
from .views import register_user, user_login, user_logout, UserList, current_user, SelectUser, DeleteUser, FileUploadView, train_model, initialise_mooring, run_qs_offset, calculate_wlgr



urlpatterns = [
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('register/', register_user, name='register'),
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),
    path('list-users/', UserList.as_view(), name='list-users'),
    path('current-user/', current_user, name='current-user'),
    re_path('^select-user/(?P<email>.+)/$', SelectUser.as_view(), name='select-user'),
    path('delete-user/<int:pk>/', DeleteUser.as_view(), name='delete-user'),
    path('file-upload/', FileUploadView.as_view(), name='file-upload'),
    # path('preprocess-data/', preprocess_data, name='preprocess-data'),
    path('train-model/', train_model, name='train-model'),
    path('initialise-mooring/', initialise_mooring, name='initialise-mooring'),
    path('run-qs-offset/', run_qs_offset, name='run-qs-offset'),
    path('calculate-wlgr/', calculate_wlgr, name='calculate-wlgr'),
]
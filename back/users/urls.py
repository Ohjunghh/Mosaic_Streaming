from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('signin/', views.signin, name='signin'),
    path('register_face/', views.register_face, name='register_face'),
    path('signup/', views.signup, name='signup'),
    path('video_upload/', views.video_upload, name='video_upload'),
    path('video_download/', views.video_download, name='video_download'),
    path('profile/', views.profile, name='profile'),
]

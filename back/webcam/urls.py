from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('stop_flutter_stream/', views.stop_flutter_stream, name='stop_flutter_stream'),
    path('start_flutter_stream/', views.start_flutter_stream, name='start_flutter_stream'),
    path('get_id/', views.get_id, name='get_id'),
]

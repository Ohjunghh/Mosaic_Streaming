from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    #path('start_stream/', views.start_stream, name='start_stream'),  
    #path('stop_stream/', views.stop_stream, name='stop_stream'),
    path('stop_flutter_stream/', views.stop_flutter_stream, name='stop_flutter_stream'),
    path('start_flutter_stream/', views.start_flutter_stream, name='start_flutter_stream'),
   # path('stream_data/<int:stream_id>/', views.stream_data, name='stream_data'),  
]

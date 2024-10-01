from django.urls import path
from . import views

urlpatterns = [
    path('processes', views.get_process_list, name='get_process_list'),
    path('get-id', views.get_id, name='get_id'),
    path('release/<int:id>', views.release_number, name='release_number'),
    path('mosaic/<int:id>', views.mosaic, name='mosaic'),
]

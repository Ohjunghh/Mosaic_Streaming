from django.shortcuts import render

from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.http import HttpResponse

def hello(request):
   return HttpResponse("Hello, World!")

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def hello_rest_api(request):
    data = {'message': 'Hello, REST API!'}
    return Response(data)

def home(request):
   data = {
       'name': 'John Doe',
       'age': 25,
       'country': 'USA'
   }
   return render(request, 'example_app/home.html', context=data)
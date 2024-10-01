"""
URL configuration for back project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
"""
장고(view)에서 정의한 뷰(view)를
urls.py 파일의 urlpatterns 리스트에 추가하여
URL 패턴과 뷰를 연결하는 것이 일반적인 방법.

이는 클라이언트가 특정 URL을 요청했을 때, 
장고가 해당 요청을 처리할 수 있도록 설정하는 과정

*
뷰(View) : 뷰는 실제로 클라이언트의 요청을 처리하고
           응답을 반환하는 장고 애플리케이션의 핵심 부분
"""

from django.contrib import admin
from django.urls import path, include # include 추가함. Django REST Framework (DRF) 설정
from example_app.views import hello
from example_app.views import hello_rest_api
from example_app.views import home
from webcam.views import *
from webcam_test.views import *
from users.views import *
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("admin/", admin.site.urls),
    path('api/', include('rest_framework.urls')),
    path('api/hello/', hello_rest_api),
    path('home/',home),
    path('api/webcam/', include('webcam.urls')),
    path("chat/", include("chat.urls")),
    path('api/users/', include('users.urls')),
    #path('webcam/', include('webcam.urls')),
    #path('webcam_test/', include('webcam_test.urls')),
]  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

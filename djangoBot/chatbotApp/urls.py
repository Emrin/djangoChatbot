from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from . import views

urlpatterns = [
    url('chatbot/', views.index, name='index'),
    url('hidden/', views.req1, name='hidden')
]

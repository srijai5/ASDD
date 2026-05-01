from django.urls import path
from . import views

urlpatterns = [
    # The starting point (Intake)
    path('', views.index, name='index'), 
    # The destination (Report)
    path('result/', views.result_page, name='result_page'), 
]
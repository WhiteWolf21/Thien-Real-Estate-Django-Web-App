from django.urls import path

from . import views

urlpatterns = [
    path('predict-result', views.predict_result, name='predict-result'),
]
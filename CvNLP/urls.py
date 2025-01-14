from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('Smart-BI/', views.dashboard, name='dashboard'),
    path('Extract-Pdf-Information/', views.chatpdf, name='chatPdf'),
    path('Report-Generator/', views.reportGeneretore, name='reportgeneratore'),
    path('Story-teller/', views.storyTelling, name='storyTelling'),
    path('Data-Analyzer/', views.analyzer, name='analyzer'),
    path('Visualization-Suggester/', views.visualize, name='visualize'),
    path('Colors-Suggester/', views.color, name='color'),
    path('Dax-Query-Generator/', views.dax, name='dax'),
    path('SQL-Generator/', views.sql, name='sql'),
]
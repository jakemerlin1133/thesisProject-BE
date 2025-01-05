from django.urls import path
from . import views
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    # User's URL
   path('users/', views.user_list),
   path('users/<int:id>', views.user_detail),
   path('login/', views.user_login),
   path('logout/', views.user_logout),
   
    # Categories' URL
   path('categories/', views.category_list),
   path('categories/<int:id>', views.category_detail),
   
    # Store' URL
    path('stores/', views.store_list),
    path('stores/<int:id>',views.store_detail),
    
    # Expense's URL
    path('expense/<int:user_id>/', views.expense_list),
    path('expense/<int:user_id>/<int:id>/', views.expense_detail),
    path('expense/predict/<int:user_id>/', views.expense_prediction),
    

]

urlpatterns = format_suffix_patterns(urlpatterns)
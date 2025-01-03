from django.contrib import admin

from .models import AppUser
from .models import Category
from .models import Store
from .models import Expense

class AppUserAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_name', 'password', 'first_name', 'middle_name', 'last_name', 'birthdate', 'age', 'position', 'email', 'phone_number', 'uploaded_at', 'updated_at')

class ExpenseAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_id', 'file', 'matched_store', 'matched_store_category', 'total_value', 'uploaded_at')
    
class CatergoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'category')
    
class StoreAdmin(admin.ModelAdmin):
    list_display = ('id', 'store', 'category_id')

# Register your models here.

admin.site.register(AppUser, AppUserAdmin)
admin.site.register(Category, CatergoryAdmin)
admin.site.register(Store, StoreAdmin)
admin.site.register(Expense, ExpenseAdmin)


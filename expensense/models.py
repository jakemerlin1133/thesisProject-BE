from django.db import models

# Create your models here.
class AppUser(models.Model):
    user_name = models.CharField(max_length=16)
    password = models.CharField(max_length=255)
    first_name = models.CharField(max_length=24)
    middle_name = models.CharField(max_length=24)
    last_name = models.CharField(max_length=24)
    birthdate = models.DateField()
    age = models.IntegerField()
    email = models.EmailField()
    phone_number = models.CharField(max_length=11)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_login = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    
    def __str__(self):
        return str(self.id)
        
class Category(models.Model):
    category = models.CharField(max_length=20, unique=True) 
    
    def __str__(self):
        return self.category
     
class Store(models.Model):
    store = models.CharField(max_length=20, unique=True)
    category_id = models.ForeignKey(Category, on_delete=models.CASCADE)
           
class Expense(models.Model):
    user_id = models.ForeignKey(AppUser, on_delete=models.CASCADE, default=1)
    file = models.FileField(null=True, blank=True)
    matched_store =  models.JSONField(null=True, blank=True)
    matched_store_category =  models.JSONField(null=True, blank=True)
    total_value = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return str(self.id)
    
    
    
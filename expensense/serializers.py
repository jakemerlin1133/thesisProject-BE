from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from django.conf import settings

from .models import AppUser
from .models import Category
from .models import Store
from .models import Expense

class AppUserSerializer(serializers.ModelSerializer):
    birthdate = serializers.DateField(
        format="%B %d, %Y", 
        input_formats=["%B %d, %Y", "%B, %d, %Y", "%Y-%m-%d"]
    )
    class Meta:
        model = AppUser
        fields = [
            'id', 
            'user_name', 
            'password', 
            'first_name', 
            'middle_name', 
            'last_name', 
            'birthdate', 
            'age', 
            'email', 
            'phone_number',
            'uploaded_at',
            'updated_at',
             'last_login', 
            'is_active',   
            'is_staff',  
            'is_superuser'
        ]
       
    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password'])  # Hash the password
        return super().create(validated_data)

    def update(self, instance, validated_data):
        # Check if the password is being updated
        if 'password' in validated_data:
            validated_data['password'] = make_password(validated_data['password'])  # Hash the password
        return super().update(instance, validated_data)
    
class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = [
            'id',
            'category'
        ]
        

class StoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Store
        fields = [
            'id',
            'store',
            'category_id'
        ]
    
class ExpenseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Expense
        fields = [
            'id',
            'user_id',
            'file',
            'matched_store',
            'matched_store_category',
            'total_value',
            'uploaded_at',
            ]
        def validate_user_id(self, value):
            if value is None:
                raise serializers.ValidationError("User ID is required.")
            return value
        
        def to_representation(self, instance):
            representation = super().to_representation(instance)

            if instance.file:
                file_url = instance.file.url 
                representation['file'] = file_url
            
            return representation
    
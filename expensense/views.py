from .models import AppUser
from .serializers import AppUserSerializer

from .models import Category
from .serializers import CategorySerializer

from .models import Store
from .serializers import StoreSerializer

from .models import Expense
from .serializers import ExpenseSerializer

from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import login, logout
from django.contrib.auth.models import User
from django.contrib.auth.hashers import check_password
from rest_framework.exceptions import AuthenticationFailed
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.sessions.models import Session



import cv2
import easyocr
from django.conf import settings
import numpy as np
from decimal import Decimal
from django.core.files.base import ContentFile
from django.utils.crypto import get_random_string

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# USERS API
@api_view(["GET", "POST"])
def user_list(request):
    if request.method == "GET":
        users = AppUser.objects.all()
        serializer = AppUserSerializer(users, many=True)
        return Response(serializer.data)
    
    if request.method == "POST":
        serializer = AppUserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save() 
            response_data = AppUserSerializer(user).data
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
@api_view(['GET', 'PUT', 'DELETE'])
def user_detail(request, id):
    
    try:
       user = User.objects.get(pk=id)
    except User.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
       serializer = AppUserSerializer(user)
       return Response(serializer.data)
    elif request.method == 'PUT':
        serializer = AppUserSerializer(user, request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    elif request.method == 'DELETE':
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


#CATEGORY API
@api_view(["GET", "POST"])
def category_list(request):
    
    if request.method == "GET":
        category = Category.objects.all()
        serializer = CategorySerializer(category, many=True)
        return Response(serializer.data)
    
    if request.method == "POST":
        serializer = CategorySerializer(data=request.data)
        if serializer.is_valid():
           serializer.save()
           return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    
@api_view(['GET', 'PUT', 'DELETE'])    
def category_detail(request, id):
    
    try:
       category = Category.objects.get(pk=id)
    except Category.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
       serializer = CategorySerializer(category)
       return Response(serializer.data)
    elif request.method == 'PUT':
        serializer = CategorySerializer(category, request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    elif request.method == 'DELETE':
        category.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


#STORE API 
@api_view(['GET', 'POST'])
def store_list(request):
    if request.method == 'GET':
        store = Store.objects.all()
        serializer = StoreSerializer(store, many=True)
        return Response(serializer.data)
    
    if request.method == 'POST':
        serializer = StoreSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED )

@api_view(['GET', 'PUT', 'DELETE'])
def store_detail(request, id):  
    try:
        store = Store.objects.get(pk=id)
    except Store.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        serializer = StoreSerializer(store)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = StoreSerializer(store, request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
   
    elif request.method == 'DELETE':
        store.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    
@api_view(['GET', 'POST'])
def expense_list(request):
    if request.method == "GET":
        
        ocr = Expense.objects.all()
        serializer = ExpenseSerializer(ocr, many=True)
        return Response(serializer.data)
    
    if request.method == "POST":
          normalize_total = ["total", "tctal", "tofal", "tota"]       
          file = request.FILES.get('file')
          if file:
            try:
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                
                reader = easyocr.Reader(['en'], gpu=False)
                result = reader.readtext(image, detail=1, paragraph=False)
                
                  # Add bounding boxes and detected text
                for detection in result:
                    top_left = tuple(map(int, detection[0][0]))
                    bottom_right = tuple(map(int, detection[0][2]))
                    text = detection[1].replace('$', '')  # Clean text, remove dollar sign

                    # Draw the bounding box around the detected text
                    image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

                    # Put the detected text above the bounding box
                    image = cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Encode image as PNG
                _, img_encoded = cv2.imencode('.png', image)
                # Create a file-like object with a filename (important for saving)
                image_filename = f"image_with_bounding_boxes_{get_random_string(8)}.png"
                image_file = ContentFile(img_encoded.tobytes(), name=image_filename)
                        
                detected_texts = [detection[1].replace('$', '').lower() for detection in result]
                cleaned_texts = [word.replace(", uu", "").replace(".j0", "").rstrip('.,:;\'"-!?}]') for word in detected_texts]
                
                total_value = None
                if any(word in cleaned_texts for word in normalize_total):
                    index_of_total = next((i for i, word in enumerate(cleaned_texts) if word in normalize_total), None)
                    if index_of_total is not None and index_of_total + 1 < len(cleaned_texts):
                        word_after_total = cleaned_texts[index_of_total + 1]
                        cleaned_value = word_after_total.replace('s', '').replace(',', '').replace(' ', '').replace('$', '')
                        
                        if index_of_total + 2 < len(cleaned_texts):
                            next_value = cleaned_texts[index_of_total + 2].replace('$', '').replace(',', '').replace(' ', '').replace('s', '')
                            if next_value.isdigit():
                                cleaned_value += '.' + next_value  
                        
                        try:
                            total_value = Decimal(cleaned_value)
                        except:
                            total_value = None
                            
                if total_value is None:
                    for i, word in enumerate(cleaned_texts):
                            if word == 'total':
                                if i + 1 < len(cleaned_texts) and cleaned_texts[i + 1].isdigit() and i + 2 < len(cleaned_texts) and cleaned_texts[i + 2].isdigit():
                                    total_value = Decimal(cleaned_texts[i + 1] + '.' + cleaned_texts[i + 2])
                                    break

                                elif i + 1 < len(cleaned_texts) and cleaned_texts[i + 1].isdigit():
                                    total_value = Decimal(cleaned_texts[i + 1] + '.00') 
                                    break
                            
                    else:
                        total_value = None
                else:
                    total_value = None
                    
    
                if total_value is None and any(word in cleaned_texts for word in normalize_total):
                    index_of_total = next((i for i, word in enumerate(cleaned_texts) if word in normalize_total), None)
                    if index_of_total is not None and index_of_total + 1 < len(cleaned_texts):
                        word_after_total = cleaned_texts[index_of_total + 1]
                        cleaned_value = word_after_total.replace('$', '').replace(',', '').replace(' ', '').replace('s', '')
                        
                        if '.' in cleaned_value and cleaned_value.count('.') > 1:
                         cleaned_value = cleaned_value.replace('.', '', cleaned_value.count('.') - 1)
                        
                        try:
                            total_value = Decimal(cleaned_value)
                        except:
                            total_value = None
                            
                categories = Category.objects.all()
                category_stores = {}
                
                for category in categories:
                    stores_in_category = Store.objects.filter(category_id=category)
                    category_stores[category.category] = [store.store.lower() for store in stores_in_category]
                    
                ocr_text = [word.lower() for word in cleaned_texts]
                matched_store = []
                matched_category = []
                
                store_matched = False
               
                for category, stores in category_stores.items():
                    for store in stores:
                        if any(store in text for text in ocr_text):
                            matched_store.append(store.title())
                            matched_category.append(category)
                            store_matched = True
                            break
                        
                if not store_matched:
                    matched_store.append("Others")
                    matched_category.append("Others")
                    
                matched_store_str = ", ".join(matched_store)
                matched_category_str = ", ".join(matched_category)
                
                data = {
                    'user_id': request.data.get('user_id'),
                    'file': image_file,
                    'matched_store': matched_store_str,
                    'matched_store_category': matched_category_str,
                    'total_value': total_value,
                }
                
                serializer = ExpenseSerializer(data=data)
                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": f"OCR failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
                    
      
    
@api_view(['GET', 'PUT', 'DELETE'])
def expense_detail(request, id):
    try: 
        ocr = Expense.objects.get(pk=id)
    except Expense.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == "GET":
        serializer = ExpenseSerializer(ocr)
        return Response(serializer.data)
    
    elif request.method == "PUT":
        serializer = ExpenseSerializer(ocr, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == "DELETE":
        ocr.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    
@api_view(['GET'])
def expense_prediction(request, user_id):
    if request.method == "GET":
        try:
            df = fetch_and_aggregate_expenses(user_id)
            
            if df.empty:
                return Response({"Message": "No expenses found in the database. Prediction cannot be made."}, status=status.HTTP_200_OK)
            
            elif len(df) < 2:
                return Response({"error": "It needs 2 or more months of data to make a prediction."}, status=status.HTTP_200_OK)
            
            model = train_model_on_aggregated_data(df)
            
            predicted_expense = predict_next_month_total(model, df)
            
            predicted_expense = abs(predicted_expense)
                        
             # Get the next month and year
            last_month_numeric = df['month_numeric'].max()
            last_year = last_month_numeric // 12
            last_month = last_month_numeric % 12
            
            if last_month == 12:  # If it's December, we reset to January and increase the year
                next_month = 1
                next_year = last_year + 1
            else:
                next_month = last_month + 1
                next_year = last_year
                
            next_month_name = datetime(next_year, next_month, 1).strftime("%B %Y")
            
            return Response({
                "prediction": f"Prediction for {next_month_name}: {predicted_expense}"
            }, status=status.HTTP_200_OK)   
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                  
        
def fetch_and_aggregate_expenses(user_id):
        expenses = Expense.objects.filter(user_id=user_id)
        data = [
            {
                'total_value': expense.total_value,
                'uploaded_at': expense.uploaded_at
            }
            for expense in expenses
        ]
        df = pd.DataFrame(data)
        
        if df.empty:
            return df

        df['uploaded_at'] = pd.to_datetime(df['uploaded_at'])
        df['month'] = df['uploaded_at'].dt.month
        df['year'] = df['uploaded_at'].dt.year
        
        if len(df['month'].unique()) == 1:
            # If only one month of data, do not aggregate and return the raw data
            df['month_numeric'] = df['year'] * 12 + df['month']
            return df

        monthly_totals = df.groupby(['year', 'month'])['total_value'].sum().reset_index()
        monthly_totals['month_numeric'] = monthly_totals['year'] * 12 + monthly_totals['month']  
        return monthly_totals
    
def train_model_on_aggregated_data(df):
    """Train a model on aggregated monthly totals."""
    X = df[['month_numeric']]
    y = df['total_value']
    
    test_size = 0.2 if len(df) > 5 else 0.5  # Ensure training/testing split works with small datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def predict_next_month_total(model, df):
    """Predict the total expense for the next month."""
    last_month_numeric = df['month_numeric'].max()
    next_month_numeric = last_month_numeric + 1  # Increment to get the next month
    prediction = model.predict([[next_month_numeric]])
    return round(prediction[0], 2)


@csrf_exempt
@api_view(['POST'])
def user_login(request):
    user_name = request.data.get('user_name')
    password = request.data.get('password')

    # Check if the user exists
    try:
        user = User.objects.get(user_name=user_name)
    except User.DoesNotExist:
        return Response({"error": "Invalid username or password"}, status=status.HTTP_401_UNAUTHORIZED)

    # Validate the password
    if not check_password(password, user.password):
        return Response({"error": "Invalid username or password"}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        login(request, user) 
        print(request.session)
    except Exception as e:
        return Response({"error": f"Login failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    
    # If authentication is successful, respond with user details
    return Response({
        "message": "Login successful",
        "user": {
            "id": user.id,
            "user_name": user.user_name,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name
        }
    }, status=status.HTTP_200_OK)


def user_logout(request):
        logout(request)
        return JsonResponse({"message": "Logout successful"}, status=200)
       
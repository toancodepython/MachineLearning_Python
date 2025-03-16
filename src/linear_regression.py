import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, f1_score, recall_score
def split_data(X, y, train_ratio, valid_ratio, test_ratio):
    assert train_ratio + valid_ratio + test_ratio == 1
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42)
    valid_size = valid_ratio / (valid_ratio + test_ratio)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1 - valid_size), random_state=42)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_model(X_train, y_train, X_valid, y_valid, model_type = 'multiple', degree=2):
    with mlflow.start_run():
        if model_type == 'multiple':
            model = LinearRegression()
        elif model_type == 'polynomial':
            poly = PolynomialFeatures(degree=degree)
            X_train = poly.fit_transform(X_train)
            X_valid = poly.transform(X_valid)
            model = LinearRegression()
        else:
            raise ValueError("Loại mô hình không hợp lệ. Chọn 'multiple' hoặc 'polynomial'.")
        
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        
        y_train_pred_binary = np.round(y_train_pred)
        y_valid_pred_binary = np.round(y_valid_pred)
        
        train_precision = precision_score(y_train, y_train_pred_binary, average='weighted', zero_division=0)
        valid_precision = precision_score(y_valid, y_valid_pred_binary, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred_binary, average='weighted')
        valid_f1 = f1_score(y_valid, y_valid_pred_binary, average='weighted')
        train_recall = recall_score(y_train, y_train_pred_binary, average='weighted')
        valid_recall = recall_score(y_valid, y_valid_pred_binary, average='weighted')
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("degree", degree if model_type == "polynomial" else None)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("valid_precision", valid_precision)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("valid_f1", valid_f1)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("valid_recall", valid_recall)
        mlflow.sklearn.log_model(model, "model")
        
    return model, train_precision, valid_precision, train_f1, valid_f1, train_recall, valid_recall, poly if model_type == "polynomial" else None

def display():
    st.title("Mô phỏng Hồi quy với MLflow Tracking")

    df = pd.read_csv('./data/processed_data.csv')
    df = df.iloc[:, 1:]
    if df is not None:

        st.write("Xem trước dữ liệu:", df.head())
        
        X = df.drop(['Survived'], axis=1)
        y = df['Survived']
        
        train_ratio = st.slider("Tỷ lệ tập train", 0.5, 0.9, 0.7, 0.05)
        valid_ratio = st.slider("Tỷ lệ tập validation", 0.05, 0.3, 0.15, 0.05)
        test_ratio = 1 - train_ratio - valid_ratio
        
        model_type = st.selectbox("Chọn loại mô hình", ["multiple", "polynomial"])
        degree = st.slider("Bậc của hồi quy đa thức", 2, 5, 2) if model_type == "polynomial" else None
        
        if st.button("Huấn luyện mô hình", key = "btn_7"):
        
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y, train_ratio, valid_ratio, test_ratio)
            model, train_precision, valid_precision, train_f1, valid_f1, train_recall, valid_recall, poly = train_model(X_train, y_train, X_valid, y_valid, model_type=model_type, degree=degree)

            
            st.write("train_precision", train_precision)
            st.write("valid_precision", valid_precision)
            st.write("train_f1", train_f1)
            st.write("valid_f1", valid_f1)
            st.write("train_recall", train_recall)
            st.write("valid_recall", valid_recall)

        if st.session_state.model is not None:
            if st.button("Dự đoán" , key = "btn_8"):
                st.sidebar.header("Nhập thông tin dự đoán")
                pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
                sex = st.sidebar.selectbox("Sex", ["male", "female"])
                age = st.sidebar.slider("Age", 0, 100, 25)
                sibsp = st.sidebar.slider("SibSp", 1, 4, 1)
                embarked = st.sidebar.selectbox("Embarked", ["C", "S", "Q"])
                parch = st.sidebar.selectbox("Parch", [0, 1, 2, 3, 4, 5])
                
                sex = 1 if sex == "male" else 0
                embarked = {"C": 0, "S": 1, "Q": 2}[embarked]
                input_data = np.array([[pclass, sex, age, sibsp, embarked, parch]])
                
                if st.session_state.poly:
                    input_data = st.session_state.poly.transform(input_data)
                
                prediction = st.session_state.model.predict(input_data)
                st.sidebar.write(f"Dự đoán: {round(prediction[0], 2)}")


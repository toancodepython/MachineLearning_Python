import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import mlflow
import streamlit as st

def display():
    st.title("Pre-processing Steps")
    num_cols = ['Age','SibSp','Parch','Fare']
    cate_cols = ['Embarked', 'Sex']

    st.subheader("1. Dữ liệu ban đầu")
    df = pd.read_csv('./data/data.csv')
    st.dataframe(df.head())

    # drop 2 row null trong cot 'Embarked'
    st.subheader("2. Loại bỏ các dòng null trong cột Embarked")
    df.dropna(subset=['Embarked'],inplace = True)
    st.dataframe(df.head())

    #fill gia tri trong cot Age
    st.subheader("3. Điền các giá trị còn thiếu trong cộng Age bằng trung bình")
    df.Age = df.Age.fillna(df.Age.median())
    st.dataframe(df.head())


    st.subheader("4. Mã hóa các cột Categori trong data")

    encoder = LabelEncoder()
    for col in cate_cols:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])
            df[col] = df[col].astype(int)
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    st.dataframe(df.head())
    # Drop nhung cot khong can thiet
    st.subheader("5. Xóa những cột không cần thiết trong dataset")
    df = df.drop(['Name', 'Ticket', 'Cabin', 'httpPassengerId'], axis=1)
    st.dataframe(df.head())

    # chuẩn hóa giá trị

    scaler = MinMaxScaler()
    scaled_cols = ['Pclass', 'Age', 'Fare', 'Embarked']
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])
    st.subheader("6. chuẩn hóa dữ liệu về khoảng 0-1")
    st.dataframe(df.head())

    df.to_csv('../../data/processed_data.csv')
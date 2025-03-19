from tensorflow.keras.datasets import mnist
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



def display():
    (x, y), (_, _) = mnist.load_data()
    x = x / 255.0  # Chuẩn hóa dữ liệu

    st.title("🖼️ MNIST Classification using SVM")
    st.header("📌 Step 1: Understanding Data")
    st.write("Below are some sample images from the dataset:")

    st.write("🔹 The pixel values are normalized by dividing by 255 to scale them between 0 and 1, which helps improve model performance and convergence speed.")
    
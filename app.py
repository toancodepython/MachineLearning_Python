import streamlit as st
from src import linear_regression 
from src import pre_processing
from src import svm_mnist
from src import decision_tree_mnist
from src import clustering
from src import mlflow_web
from src import neural
from src import MNIST_understand
from src import reduction
from src import semi_supervised
# Sidebar navigation
option = st.sidebar.selectbox("Lựa chọn", ["Titanic Data", "MNIST Data", "Linear Regression",  "SVM Mnist", "Decision Tree Mnist",  "Clustering", "Dim Redution", "Neural Network", "Semi Supervised",  "ML-Flow"])

if(option == 'Titanic Data'):
    pre_processing.display()
elif(option == 'Linear Regression'):
    linear_regression.display()
elif(option == 'MNIST Data'):
    MNIST_understand.display()
elif(option == 'SVM Mnist'):
    svm_mnist.display()
elif(option == 'Decision Tree Mnist'):
    decision_tree_mnist.display()
elif(option == 'Clustering'):
    clustering.display()
elif(option == 'Dim Redution'):
    reduction.display()
elif(option == 'ML-Flow'):
    mlflow_web.display()
elif(option == 'Neural Network'):
    neural.display()
elif(option == 'Semi Supervised'):
    semi_supervised.display()

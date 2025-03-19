import sys
import os
sys.path.append(os.path.abspath("src"))
import streamlit as st
import linear_regression 
import svm_mnist
import decision_tree_mnist
import clustering
import neural
import semi_supervised_2
# Sidebar navigation
option = st.sidebar.selectbox("Lựa chọn", [ "Semi Supervised", "Neural Network", "Decision Tree Mnist",  "SVM Mnist", "Clustering", "Linear Regression"])

if(option == 'SVM Mnist'):
    svm_mnist.svm()
elif(option == 'Decision Tree Mnist'):
    decision_tree_mnist.decision()
elif(option == 'Clustering'):
    clustering.clustering()

elif(option == 'Neural Network'):
    neural.display()
elif(option == 'Semi Supervised'):
    semi_supervised_2.semisupervised()
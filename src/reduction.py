import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from time import time
import mlflow
import os
import tensorflow as tf

@st.cache_data
def load_mnist():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    x = x.reshape(x.shape[0], -1) / 255.0  # Normalize and flatten
    return x

def log_experiment(model_name, param):
    try:
        DAGSHUB_USERNAME = "toancodepython"  # Thay bằng username của bạn
        DAGSHUB_REPO_NAME = "ml-flow"
        DAGSHUB_TOKEN = "a6e8c1682e60df503248dcf37f42ca15ceaee13a"  # Thay bằng Access Token của bạn
        mlflow.set_tracking_uri("https://dagshub.com/toancodepython/ml-flow.mlflow")
        # Thiết lập authentication bằng Access Token
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        experiment_name = "Dimmension Reduce"
        experiment = next((exp for exp in mlflow.search_experiments() if exp.name == experiment_name), None)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name = model_name) as run:
                print('Logging...')
                mlflow.log_metric("Explained_variance", param)
                st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")
        else:
            # Nếu thí nghiệm chưa tồn tại, tạo mới
            experiment_id = mlflow.create_experiment(experiment_name)
        print("Active Run:", mlflow.active_run())
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")

def display():
    if "log_success" not in st.session_state:
                st.session_state.log_success = False
    if "explained_variance" not in st.session_state:
            st.session_state.explained_variance = 0
    # Tải dữ liệu MNIST từ OpenML
    st.title("Giảm chiều dữ liệu MNIST với PCA & t-SNE")
    st.write("Tải dữ liệu MNIST từ OpenML...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

    # Hiển thị thông tin cơ bản
    st.write("### Thông tin dữ liệu:")
    st.write(f"Kích thước dữ liệu: {X.shape}")
    st.write(f"Số nhãn khác nhau: {np.unique(y)}")

    # Chuẩn hóa dữ liệu (chia cho 255)
    X /= 255.0
    st.write("Dữ liệu đã được chuẩn hóa bằng cách chia cho 255.")

    # Người dùng chọn số lượng mẫu
    num_samples = st.text_input("Nhập số lượng mẫu từ 1 đến 70000:", key = "reduction_1")
    if st.button("Xác nhận số lượng mẫu", key = "btn_11"):
        try:
            num_samples = int(num_samples)
            if 1 <= num_samples <= 70000:
                X, y = X[:num_samples], y[:num_samples]
                st.success(f"Sử dụng {num_samples} mẫu từ tập dữ liệu.")
            else:
                st.error("Vui lòng nhập số lượng mẫu trong khoảng 1 đến 70000.")
        except ValueError:
            st.error("Vui lòng nhập một số nguyên hợp lệ.")

    # Lựa chọn thuật toán
    option = st.selectbox("Chọn phương pháp giảm chiều:", ["PCA", "t-SNE"])

    if option == "PCA":
        n_pca = st.text_input("Nhập số thành phần PCA:", key = "reduction_2")
        if st.button("Thực hiện PCA", key = "btn_10"):
            try:
                n_pca = int(n_pca)
                st.write(f"Thực hiện PCA với {n_pca} thành phần chính...")
                pca = PCA(n_components=n_pca)
                X_pca = pca.fit_transform(X)
                 # Đánh giá tỉ lệ phương sai giữ lại
                explained_variance = np.sum(pca.explained_variance_ratio_)
                st.write(f"Tỉ lệ phương sai giữ lại sau PCA: {explained_variance:.4f}")
                st.session_state['explained_variance'] = explained_variance
            except ValueError:
                st.error("Vui lòng nhập một số nguyên hợp lệ cho số thành phần PCA.")

    elif option == "t-SNE":
        perplexity = st.text_input("Nhập Perplexity của t-SNE:", key = "reduction_3")
        if st.button("Thực hiện t-SNE", key = "btn_9"):
            try:
                perplexity = int(perplexity)
                st.write("Thực hiện t-SNE...")
                t0 = time()
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                X_tsne = tsne.fit_transform(X)
                st.write(f"Thời gian thực hiện t-SNE: {time() - t0:.2f} giây")
                
                # Vẽ kết quả t-SNE
                st.write("Biểu diễn MNIST bằng t-SNE:")
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.5, s=5)
                fig.colorbar(scatter, label='Label')
                ax.set_xlabel("t-SNE Dimension 1")
                ax.set_ylabel("t-SNE Dimension 2")
                ax.set_title("Biểu diễn MNIST bằng t-SNE")
                st.pyplot(fig)
            except ValueError:
                st.error("Vui lòng nhập một số nguyên hợp lệ cho perplexity của t-SNE.")
    model_name = st.text_input("🏷️ Nhập tên mô hình", key = "reduction_4")
    if st.button("Log Experiment Dimmension Reduce" , key = "btn_8"):
        log_experiment(model_name, param=st.session_state['explained_variance']) 

    # Hiển thị trạng thái log thành công
    if st.session_state.log_success:
        st.success("🚀 Experiment đã được log thành công!")
    
display()
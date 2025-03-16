import os
import mlflow
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import cv2
from streamlit_drawable_canvas import st_canvas

def log_experiment(model_name):
    try:
        DAGSHUB_USERNAME = "toancodepython"  # Thay bằng username của bạn
        DAGSHUB_REPO_NAME = "ml-flow"
        DAGSHUB_TOKEN = "a6e8c1682e60df503248dcf37f42ca15ceaee13a"  # Thay bằng Access Token của bạn
        mlflow.set_tracking_uri("https://dagshub.com/toancodepython/ml-flow.mlflow")
        # Thiết lập authentication bằng Access Token
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        experiment_name = "Semi_supervised"
        experiment = next((exp for exp in mlflow.search_experiments() if exp.name == experiment_name), None)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name = model_name) as run:
                print('Logging...')
                mlflow.log_param("num_neurons", st.session_state.number)
                mlflow.log_param("num_hidden_layers", st.session_state.hidden_layer)
                mlflow.log_param("epochs", st.session_state.epoch)
                mlflow.log_param("optimizer", st.session_state.optimizer)
                mlflow.log_param("loss_function", st.session_state.loss)
                mlflow.log_metric("Test Accuracy", st.session_state.acc)

                st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")
        else:
            # Nếu thí nghiệm chưa tồn tại, tạo mới
            experiment_id = mlflow.create_experiment(experiment_name)
        print("Active Run:", mlflow.active_run())
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")
def display():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

    # Streamlit UI
    st.title("Pseudo Labelling on MNIST with Neural Network")
    st.write("**Số lượng mẫu**: Số lượng ảnh được sử dụng cho việc huấn luyện và kiểm tra.")
    train_size = st.number_input("Train/test split (percentage for training)", min_value=10, max_value=90, value=80)
    n_samples = st.number_input("Total number of training samples", min_value=100, max_value=len(x_train), value=1000, step=100)
    st.write("Confidence threshold for pseudo labeling:  Ngưỡng tin cậy để gán nhãn giả.")
    threshold = st.number_input("Confidence threshold for pseudo labeling", min_value=0.5, max_value=1.0, value=0.95, step=0.05)
    st.write("Number of pseudo-labeling iterations: Số lần lặp quá trình gán nhãn giả.")
    n_iterations = st.number_input("Number of pseudo-labeling iterations", min_value=1, max_value=10, value=5)

    # Tham số mô hình
    st.write("**Số neuron mỗi lớp**: Số lượng neuron trong mỗi lớp ẩn của mô hình.")
    num_neurons = int(st.number_input("Số neuron mỗi lớp", 32, 512, 128, step=32))
    st.session_state.number = num_neurons
    st.write("**Số lớp ẩn**: Số lượng lớp kết nối trong mạng neuron.")
    num_hidden_layers = int(st.number_input("Số lớp ẩn", 1, 5, 2))
    st.session_state.hidden_layer = num_hidden_layers
    st.write("**Số epochs**: Số lần mô hình duyệt qua toàn bộ tập dữ liệu huấn luyện.")
    epochs = int(st.number_input("Số epochs", 5, 50, 10, step=5))
    st.session_state.epoch = epochs
    st.write("**Optimizer**: Thuật toán tối ưu hóa giúp giảm thiểu hàm mất mát.")
    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
    st.session_state.optimizer = optimizer
    st.write("**Hàm mất mát**: Hàm đánh giá mức độ lỗi của mô hình.")
    loss_function = st.selectbox("Hàm mất mát", ["sparse_categorical_crossentropy", "categorical_crossentropy"])
    st.session_state.loss = loss_function
    # Store session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'acc' not in st.session_state:
        st.session_state.acc = None
    if 'log_success' not in st.session_state:
        st.session_state.log_success = False
    if 'train_history' not in st.session_state:
        st.session_state.train_history = []
    if 'pseudo_labels_history' not in st.session_state:
        st.session_state.pseudo_labels_history = []

    # Train button
    if st.button("Train Model"):
        # Split data
        x_train_part, x_valid, y_train_part, y_valid = train_test_split(x_train, y_train, test_size=(100-train_size)/100, stratify=y_train)
        
        # Select 1% of each class
        x_train_small, y_train_small = [], []
        for i in range(10):
            idx = np.where(y_train_part == i)[0][:n_samples//10]
            x_train_small.append(x_train_part[idx])
            y_train_small.append(y_train_part[idx])
        x_train_small = np.vstack(x_train_small)
        y_train_small = np.hstack(y_train_small)
        
        # Define NN model
        def build_model():
            model = Sequential([Flatten(input_shape=(28, 28))])
            for _ in range(num_hidden_layers):
                model.add(Dense(num_neurons, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
            return model
        
        # Training loop
        model = build_model()
        st.write("Training initial model...")
        model.fit(x_train_small, y_train_small, epochs=epochs, verbose=0)
        
        colors = ["red", "blue", "green", "orange", "purple"]
        
        for iteration in range(n_iterations):
            st.markdown(f'<h4 style="color:{colors[iteration % len(colors)]}">Iteration {iteration+1} - Generating Pseudo Labels</h4>', unsafe_allow_html=True)
            predictions = model.predict(x_valid)
            pseudo_labels = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            
            selected_idx = confidence >= threshold
            num_pseudo_labels = np.sum(selected_idx)
            st.session_state.pseudo_labels_history.append(num_pseudo_labels)
            st.write(f"Iteration {iteration+1}: Number of new pseudo labels added: {num_pseudo_labels}")
            
            if num_pseudo_labels == 0:
                st.write("No new pseudo labels were added. Stopping early.")
                break
            
            x_train_small = np.vstack([x_train_small, x_valid[selected_idx]])
            y_train_small = np.hstack([y_train_small, pseudo_labels[selected_idx]])
            x_valid, y_valid = x_valid[~selected_idx], y_valid[~selected_idx]
            
            model = build_model()
            model.fit(x_train_small, y_train_small, epochs=epochs, verbose=0)
        
        st.session_state.model = model
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        st.session_state.acc = acc

        st.session_state.train_history.append(acc)
    
    if st.session_state.model:
        st.write(f"Test Accuracy: {st.session_state.acc:.4f}")
        st.success('Train successed')
        # Canvas for user to draw a digit
        st.subheader("Draw a digit and predict")
        canvas = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if st.button("Predict Drawn Digit") and st.session_state.model:
            if canvas.image_data is not None:
                img = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                img = cv2.resize(img, (28, 28))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                prediction = st.session_state.model.predict(img)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                st.write(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}")
        model_name = st.text_input("🏷️ Nhập tên mô hình", key = "semi_0")
        if st.button("Log Model Semi Supervised", key = "semi_1"):
            log_experiment(model_name) 
        # Hiển thị trạng thái log thành công
        if st.session_state.log_success:
            st.success("🚀 Experiment đã được log thành công!  Chuyển qua tab ML_Flow để xem kết quả")

        

display()
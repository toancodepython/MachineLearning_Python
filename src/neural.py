import os
import mlflow
from sklearn.metrics import accuracy_score
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
def log_experiment(model_name):
    try:
        DAGSHUB_USERNAME = "toancodepython"  # Thay bằng username của bạn
        DAGSHUB_REPO_NAME = "ml-flow"
        DAGSHUB_TOKEN = "a6e8c1682e60df503248dcf37f42ca15ceaee13a"  # Thay bằng Access Token của bạn
        mlflow.set_tracking_uri("https://dagshub.com/toancodepython/ml-flow.mlflow")
        # Thiết lập authentication bằng Access Token
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        experiment_name = "Neural Network"
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
                mlflow.log_metric("Test Accuracy", st.session_state.accuracy)
                mlflow.log_metric("Validation Accuracy", st.session_state.accuracy_val)

                st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")
        else:
            # Nếu thí nghiệm chưa tồn tại, tạo mới
            experiment_id = mlflow.create_experiment(experiment_name)
        print("Active Run:", mlflow.active_run())
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")
# Xây dựng mô hình
def create_model(num_hidden_layers, num_neurons,optimizer, loss_function):
    model = Sequential([Flatten(input_shape=(28, 28))])
    for _ in range(num_hidden_layers):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return model

def display():
    # Load MNIST dataset
    (x, y), (_, _) = mnist.load_data()
    x = x / 255.0  # Chuẩn hóa dữ liệu

    # Streamlit UI
    st.title("Nhận diện chữ số MNIST")
    st.header("Tham số mô hình")

    # Lựa chọn số lượng mẫu dữ liệu
    st.write("**Số lượng mẫu**: Số lượng ảnh được sử dụng cho việc huấn luyện và kiểm tra.")
    num_samples = int(st.number_input("Số lượng mẫu", 1000, 70000, 70000, step=1000))

    st.write("**Tỉ lệ tập huấn luyện**: Phần trăm dữ liệu được sử dụng để huấn luyện mô hình.")
    train_ratio = float(st.number_input("Tỉ lệ tập huấn luyện", 0.5, 0.9, 0.8, step=0.05))

    st.write("**Tỉ lệ tập validation**: Phần trăm dữ liệu được sử dụng để đánh giá mô hình trong quá trình huấn luyện.")
    val_ratio = float(st.number_input("Tỉ lệ tập validation", 0.05, 0.3, 0.1, step=0.05))
    test_ratio = 1 - train_ratio - val_ratio

    # Chia tập dữ liệu
    x_train, x_temp, y_train, y_temp = train_test_split(x[:num_samples], y[:num_samples], train_size=train_ratio, stratify=y[:num_samples])
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_ratio/(test_ratio + val_ratio), stratify=y_temp)

    st.write(f"Số lượng mẫu huấn luyện: {len(x_train)}, mẫu validation: {len(x_val)}, mẫu kiểm tra: {len(x_test)}")

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
    if "accuracy" not in st.session_state:
        st.session_state.accuracy = 0
    if "accuracy_val" not in st.session_state:
        st.session_state.accuracy_val = 0
    if "number" not in st.session_state:
        st.session_state.number = 0
    if "train_sample" not in st.session_state:
        st.session_state.train_sample = 0
    if "test_sample" not in st.session_state:
        st.session_state.test_sample = 0
    if "val_sample" not in st.session_state:
        st.session_state.val_sample = 0
    if "hidden_layer" not in st.session_state:
        st.session_state.hidden_layer = 0
    if "epoch" not in st.session_state:
        st.session_state.epoch = 0
    if "optimizer" not in st.session_state:
        st.session_state.optimizer = 0
    if "loss" not in st.session_state:
        st.session_state.loss = 0
    if "log_success" not in st.session_state:
        st.session_state.log_success = False
    # Huấn luyện mô hình
    if st.button("Huấn luyện mô hình"):
        model = create_model(loss_function=loss_function, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, optimizer=optimizer)
        with st.spinner("Đang huấn luyện..."):
            model.fit(x_train, y_train, epochs=epochs, batch_size=5, verbose=0)
            st.session_state.model = model
            y_pred_classes = model.predict(x_test).argmax(axis=1)  # Chuyển đổi nhãn dự đoán
            y_pred_val_classes = model.predict(x_val).argmax(axis=1)  # Chuyển đổi nhãn dự đoán
            st.session_state.accuracy = accuracy_score(y_test, y_pred_classes) 
            st.session_state.accuracy_val = accuracy_score(y_pred_val_classes, y_val)  # Lưu độ chính xác
             # Lưu độ chính xác
        st.success("Huấn luyện thành công!")
    st.write(f"Độ chính xác trên tập Test: {st.session_state.accuracy:.4f}")
    st.write(f"Độ chính xác trên tập Validation: {st.session_state.accuracy_val:.4f}")
    st.subheader("Vẽ một chữ số (0-9)")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )
    # Dự đoán chữ số
    if 'model' in st.session_state:
    
        if canvas.image_data is not None and st.button("Dự đoán"):
            image = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            image = cv2.resize(image, (28, 28))
            image = image / 255.0
            image = image.reshape(1, 28, 28)
            pred = st.session_state.model.predict(image)
            predicted_digit = np.argmax(pred)
            confidence = np.max(pred)
            st.write(f"Chữ số dự đoán: {predicted_digit}")
            st.write(f"Độ tin cậy: {confidence:.4f}")

        model_name = st.text_input("🏷️ Nhập tên mô hình", key = "0")
        if st.button("Log Model Neural Network", key = "btn_41"):
            log_experiment(model_name) 

        # Hiển thị trạng thái log thành công
        if st.session_state.log_success:
            st.success("🚀 Experiment đã được log thành công!  Chuyển qua tab ML_Flow để xem kết quả")
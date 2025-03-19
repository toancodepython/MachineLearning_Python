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
        st.warning('Mô hình đang được Log')
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
                mlflow.log_param("Learning rate", st.session_state.learning_rate)
                mlflow.log_param("Activation", st.session_state.activation)

                mlflow.log_metric("Test Accuracy", st.session_state.accuracy)
                mlflow.log_metric("Validation Accuracy", st.session_state.accuracy_val)

                mlflow.keras.log_model(st.session_state.model, "model")
                mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
                st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")
        else:
            # Nếu thí nghiệm chưa tồn tại, tạo mới
            experiment_id = mlflow.create_experiment(experiment_name)
        print("Active Run:", mlflow.active_run())
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")
# Xây dựng mô hình
def create_model(num_hidden_layers, num_neurons,optimizer):
    model = Sequential([Flatten(input_shape=(28, 28))])
    for _ in range(num_hidden_layers):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss=st.session_state.loss, metrics=['accuracy'])
    return model

def display():
    # Load MNIST dataset
    (x, y), (_, _) = mnist.load_data()
    x = x / 255.0  # Chuẩn hóa dữ liệu

    # Streamlit UI
    st.title("📊 Classification on MNIST with Neural Network")
    st.header("📂 Data Selection")
    # Lựa chọn số lượng mẫu dữ liệu
    st.write("**Số lượng mẫu**: Số lượng ảnh được sử dụng cho việc huấn luyện và kiểm tra.")
    num_samples = int(st.number_input("Số lượng mẫu", 1000, 70000, 70000, step=1000, key='neu_1'))

    st.write("**Tỉ lệ tập huấn luyện**: Phần trăm dữ liệu được sử dụng để huấn luyện mô hình.")
    train_ratio = float(st.number_input("Tỉ lệ tập huấn luyện", 0.5, 0.9, 0.8, step=0.05, key='neu_2'))

    st.write("**Tỉ lệ tập validation**: Phần trăm dữ liệu được sử dụng để đánh giá mô hình trong quá trình huấn luyện.")
    val_ratio = float(st.number_input("Tỉ lệ tập validation", 0.05, 0.3, 0.1, step=0.05, key='neu_3'))
    test_ratio = 1 - train_ratio
    # Chia tập dữ liệu
    x_train, x_test, y_train, y_test = train_test_split(x[:num_samples], y[:num_samples], test_size= test_ratio, random_state=42)  # 20% test
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, random_state=42)  # 25% của train thành val (~20% tổng)

    st.write(f"Số lượng mẫu huấn luyện: {len(x_train)}, mẫu validation: {len(x_val)}, mẫu kiểm tra: {len(x_test)}")
    st.header("⚙️ Neural Network Parameters")
    # Tham số mô hình
    st.write("**Số neuron mỗi lớp**: Số lượng neuron trong mỗi lớp ẩn của mô hình.")
    num_neurons = int(st.number_input("Số neuron mỗi lớp", 32, 512, 128, step=32, key='neu_6'))
    st.session_state.number = num_neurons
    st.write("**Số lớp ẩn**: Số lượng lớp kết nối trong mạng neuron.")
    num_hidden_layers = int(st.number_input("Số lớp ẩn", 1, 5, 2, key='neu_7'))
    st.session_state.hidden_layer = num_hidden_layers
    st.write("**Số epochs**: Số lần mô hình duyệt qua toàn bộ tập dữ liệu huấn luyện.")
    epochs = int(st.number_input("Số epochs", 5, 50, 10, step=5, key='neu_8'))
    st.session_state.epoch = epochs
    st.write("**Optimizer**: Thuật toán tối ưu hóa giúp giảm thiểu hàm mất mát.")
    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
    st.session_state.optimizer = optimizer
    activation = st.selectbox("Activation", ["relu", "softmax"], key='semi_12')
    st.session_state.activation = activation
    learning_rate = st.number_input("Tốc độ học", min_value=0.01, max_value=1.0, value=0.01, key='neu_9')
    st.session_state.learning_rate = learning_rate
    st.session_state.loss = 'sparse_categorical_crossentropy'
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
    if "model" not in st.session_state:
        st.session_state.model = None
    # Huấn luyện mô hình
    run_name = st.text_input("🏷️ Nhập tên Run", key = "neural_10")

    if st.button("▶️ Huấn luyện mô hình", key='neu_btn_1'):
        model = create_model(num_hidden_layers=num_hidden_layers, num_neurons=num_neurons, optimizer=optimizer)
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
        st.session_state.model = model
        log_experiment(run_name)
    if st.session_state.model is not None:
        model_dict = get_logged_models('Neural Network')
        selected_model_name = st.selectbox("Chọn mô hình để dự đoán:", list(model_dict.keys()))
        st.subheader("Draw a digit and predict")

        canvas_result = st_canvas(
            fill_color="black",  # Màu nền
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=250,
            width=250,
            drawing_mode="freedraw",
            key='neural_'
        )
        if st.button("Predict Drawn Digit", key= 'neu_btn_2'):
            if canvas_result.image_data is not None:
                print('start predict')
                try: 
                    run_id = model_dict[selected_model_name]
                    model_uri = f"runs:/{run_id}/model"
                    model_loaded = mlflow.keras.load_model(model_uri)
                    if(model_loaded): st.success('Model loaded')
                    img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                    img = cv2.resize(img, (28, 28))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    prediction = model_loaded.predict(img)
                    st.write(prediction)
                    predicted_digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    st.write(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}")
                except Exception as e:
                    st.warning(e)
            else:
                st.warning("⚠️ Vui lòng vẽ một số trước khi dự đoán!")
def get_logged_models(experiment_name):
    try:

        DAGSHUB_USERNAME = "toancodepython"
        DAGSHUB_REPO_NAME = "ml-flow"
        DAGSHUB_TOKEN = "a6e8c1682e60df503248dcf37f42ca15ceaee13a"
        mlflow.set_tracking_uri("https://dagshub.com/toancodepython/ml-flow.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        experiment = next((exp for exp in mlflow.search_experiments() if exp.name == experiment_name), None)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return {row["tags.mlflow.runName"]: row["run_id"] for _, row in runs.iterrows()}
        return {}
    except Exception as e:
        st.warning("Không thể lấy danh sách mô hình từ MLflow.")
        
        return []

def neural():
    tab1, tab3 = st.tabs(["⚙️ Huấn luyện", "🔥Mlflow"])

    with tab1:
        display()
    with tab3:
        import mlflow_web
        mlflow_web.display()

neural()
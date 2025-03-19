import os
import time
import mlflow
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import cv2
from streamlit_drawable_canvas import st_canvas

def log_experiment(model_name, acc):
    try:

        st.warning('Model đang được log..')
        DAGSHUB_USERNAME = "toancodepython"
        DAGSHUB_REPO_NAME = "ml-flow"
        DAGSHUB_TOKEN = "a6e8c1682e60df503248dcf37f42ca15ceaee13a"
        mlflow.set_tracking_uri("https://dagshub.com/toancodepython/ml-flow.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
        experiment_name = "Semi_supervised"
        experiment = next((exp for exp in mlflow.search_experiments() if exp.name == experiment_name), None)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=model_name) as run:
                print('Logging...')
                mlflow.log_param("Confidence threshold", st.session_state.thres)
                mlflow.log_param("Number of iterations", st.session_state.n_iterations)
                mlflow.log_param("num_neurons", st.session_state.number)
                mlflow.log_param("num_hidden_layers", st.session_state.hidden_layer)
                mlflow.log_param("epochs", st.session_state.epoch)
                mlflow.log_param("optimizer", st.session_state.optimizer)
                mlflow.log_param("learning_rate", st.session_state.learning_rate)
                mlflow.log_param("activation", st.session_state.activation)
                mlflow.log_metric("Test Accuracy", acc)
                mlflow.keras.log_model(st.session_state.model, "model")
                mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
                st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
        print("Active Run:", mlflow.active_run())
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")

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
def build_model(input_shape, num_neurons, num_hidden_layers, optimizer, loss_function, learning_rate, activation):
    model = Sequential([Flatten(input_shape=input_shape)])
    for _ in range(num_hidden_layers):
        model.add(Dense(num_neurons, activation= 'relu'))
    model.add(Dense(10, activation='softmax'))
    
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    optimizer_instance.learning_rate = learning_rate
    
    model.compile(optimizer=optimizer_instance, loss=loss_function, metrics=['accuracy'])
    return model
def balanced_sample(x, y, sample_size):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    per_class_sample = sample_size // len(unique_classes)
    
    x_sample, y_sample = [], []
    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        selected_idx = np.random.choice(idx, min(per_class_sample, len(idx)), replace=False)
        x_sample.extend(x[selected_idx])
        y_sample.extend(y[selected_idx])
    
    return np.array(x_sample), np.array(y_sample)
def display():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  

    if 'model' not in st.session_state:
        st.session_state.model = None

    st.title("📊 Pseudo Labelling on MNIST with Neural Network")
    st.header("📂 Data Selection")
    n_samples = st.number_input("Total number of samples", min_value=100, max_value=len(x_train), value=10000, step=100, key='semi_101')
    train_size = st.number_input("Train split percentage", min_value=10, max_value=90, value=80, key='semi_2')
    val_size = st.number_input("Validation split percentage", min_value=5, max_value=30, value=20, key='semi_3')
    train_percent = st.number_input("Percentage of data to use for initial training", min_value=1, max_value=100, value=1, step=1, key='semi_4')

    if st.button("✅ Confirm Data Split", key= 'semi_btn_1'):
        x_data, x_test, y_data, y_test = train_test_split(x_train[:n_samples], y_train[:n_samples], test_size=(100-train_size)/100, stratify=y_train[:n_samples])
        x_train_part, x_valid, y_train_part, y_valid = train_test_split(x_data, y_data, test_size=val_size/100, stratify=y_data)
        
        initial_train_size = int(len(x_train_part) * train_percent / 100)
        x_train_initial, y_train_initial = balanced_sample(x_train_part, y_train_part, initial_train_size)
        
        st.session_state.x_valid = x_valid
        st.session_state.y_valid = y_valid
        st.session_state.x_train_part = x_train_initial
        st.session_state.y_train_part = y_train_initial
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test
        
        st.markdown(f"🟢 **Initial Training samples:** <span style='color:green;'>{initial_train_size}  ({train_percent}% of {len(x_train_part)})</span>", unsafe_allow_html=True)
        st.markdown(f"🟠 **Validation samples:** <span style='color:orange;'>{len(x_valid)}</span>", unsafe_allow_html=True)
        st.markdown(f"🔵 **Test samples:** <span style='color:blue;'>{len(x_test)}</span>", unsafe_allow_html=True)
        
    st.header("🛠 Parameters for Pseudo-Labeling")
    threshold = st.number_input("Confidence threshold", min_value=0.5, max_value=1.0, value=0.8, key='semi_5')
    st.session_state.thres = threshold
    n_iterations = st.number_input("Number of iterations", min_value=1, max_value=10, value=5, key='semi_6')
    st.session_state.n_iterations = n_iterations
    st.header("⚙️ Neural Network Parameters")
    num_neurons = int(st.number_input("Số neuron mỗi lớp", 32, 512, 128, step=32, key='semi_7'))
    st.session_state.number = num_neurons
    num_hidden_layers = int(st.number_input("Số lớp ẩn", 1, 5, 2, key='semi_8'))
    st.session_state.hidden_layer = num_hidden_layers
    epochs = int(st.number_input("Số epochs", 5, 50, 10, step=5, key='semi_9'))
    st.session_state.epoch = epochs
    learning_rate = st.number_input("Tốc độ học", min_value=0.01, max_value=1.0, value=0.01, key='semi_10')
    st.session_state.learning_rate = learning_rate
    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], key='semi_11')
    st.session_state.optimizer = optimizer
    activation = st.selectbox("Activation", ["relu", "softmax"], key='semi_activaton')
    st.session_state.activation = activation

    st.session_state.run_name = st.text_input("🏷️ Nhập tên Run", key = "semi_14")

    if st.button("▶️ Start Training", key='semi_btn_3'):
        model = build_model((28, 28), num_neurons, num_hidden_layers, optimizer, 'sparse_categorical_crossentropy', learning_rate, activation=activation)
        for i in range(n_iterations):
            model.fit(st.session_state.x_train_part, st.session_state.y_train_part, epochs=epochs, verbose=0)
            val_acc = model.evaluate(st.session_state.x_valid, st.session_state.y_valid, verbose=0)[1]
            st.write(f"📉 **Iteration {i+1}: Validation Accuracy = `{val_acc:.4f}`")
            
            pseudo_labels = model.predict(st.session_state.x_test)
            confident_mask = np.max(pseudo_labels, axis=1) >= threshold
            new_x_train = st.session_state.x_test[confident_mask]
            new_y_train = np.argmax(pseudo_labels[confident_mask], axis=1)
            
            if len(new_x_train) > 0:
                st.session_state.x_train_part = np.concatenate([st.session_state.x_train_part, new_x_train])
                st.session_state.y_train_part = np.concatenate([st.session_state.y_train_part, new_y_train])
                st.write(f"🆕 **New pseudo labels added:** `{len(new_x_train)}`")
            else:
                st.write("🚫 No new pseudo labels added.")
        st.session_state.model = model
        test_acc = st.session_state.model.evaluate(st.session_state.x_test, st.session_state.y_test, verbose=0)[1]
        st.session_state.test_acc = test_acc
        st.write(f"🏁 **Final Test Accuracy:** `{test_acc:.4f}`")
        log_experiment(st.session_state.run_name, acc = test_acc) 

    if st.session_state.model is not None:
        
        #predict
        model_dict = get_logged_models('Semi_supervised')
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
            key='canvas_semi_2'
        )
        if st.button("Predict Drawn Digit", key='semi_btn_2'):
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
def semisupervised():
    tab1,  tab3= st.tabs(["⚙️ Huấn luyện",  "🔥Mlflow"])
    with tab1:
        display()
    with tab3:
        import mlflow_web
        mlflow_web.display()
semisupervised()
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import mlflow
import os
from tensorflow.keras.datasets import mnist
from streamlit_drawable_canvas import st_canvas
from sklearn.tree import DecisionTreeClassifier
import cv2
def load_data():
    train_data = pd.read_csv("data/mnist/train.csv")
    X = train_data.iloc[:, 1:].values / 255.0
    y = train_data.iloc[:, 0].values
    return train_data, X, y
def log_experiment(model_name):
    try:
        st.warning('Model đang được log...')
        DAGSHUB_USERNAME = "toancodepython"  # Thay bằng username của bạn
        DAGSHUB_REPO_NAME = "ml-flow"
        DAGSHUB_TOKEN = "a6e8c1682e60df503248dcf37f42ca15ceaee13a"  # Thay bằng Access Token của bạn
        mlflow.set_tracking_uri("https://dagshub.com/toancodepython/ml-flow.mlflow")
        # Thiết lập authentication bằng Access Token
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
        experiment_name = "Decision_Tree_Classification"
        experiment = next((exp for exp in mlflow.search_experiments() if exp.name == experiment_name), None)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name = model_name) as run:
                print('Logging...')
                mlflow.log_param("max_depth", st.session_state.max_depth)
                mlflow.log_param("min_samples_split", st.session_state.min_samples_split)
                mlflow.log_param("min_samples_leaf", st.session_state.min_samples_leaf)
                mlflow.log_param("criterion", st.session_state.criterion)

                mlflow.log_metric("Test Accuracy", st.session_state.accuracy)
                mlflow.log_metric("Validation Accuracy", st.session_state.accuracy_val)

                mlflow.sklearn.log_model(st.session_state.model, "model")
                mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
                st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")
        else:
            # Nếu thí nghiệm chưa tồn tại, tạo mới
            experiment_id = mlflow.create_experiment(experiment_name)
        print("Active Run:", mlflow.active_run())
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")
def show_sample_images():
    train_data = pd.read_csv("data/mnist/train.csv")
    unique_labels = train_data.iloc[:, 0].unique()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    label_count = 0
    
    for i, ax in enumerate(axes.flat):
        if label_count >= len(unique_labels):
            break
        sample = train_data[train_data.iloc[:, 0] == unique_labels[label_count]].iloc[0, 1:].values.reshape(28, 28)
        ax.imshow(sample, cmap='gray')
        ax.set_title(f"Label: {unique_labels[label_count]}", fontsize=10)
        ax.axis("off")
        label_count += 1
    st.pyplot(fig)

def plot_label_distribution(y):
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(y).value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Label Distribution in Dataset")
    ax.set_xlabel("Digit Label")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def display():
    st.title("📊 Classificaton on MNIST with Decision Tree")

    (x, y), (_, _) = mnist.load_data()
    x = x / 255.0  # Chuẩn hóa dữ liệu
    x = x.reshape(x.shape[0], -1)
    # Lựa chọn số lượng mẫu dữ liệu
    st.header("📂 Data Selection")
    st.write("**Số lượng mẫu**: Số lượng ảnh được sử dụng cho việc huấn luyện và kiểm tra.")
    num_samples = int(st.number_input("Số lượng mẫu", 1000, 70000, 70000, step=1000, key = 'tree_1'))

    st.write("**Tỉ lệ tập huấn luyện**: Phần trăm dữ liệu được sử dụng để huấn luyện mô hình.")
    train_ratio = float(st.number_input("Tỉ lệ tập huấn luyện", 0.5, 0.9, 0.8, step=0.05, key = 'tree_2'))

    st.write("**Tỉ lệ tập validation**: Phần trăm dữ liệu được sử dụng để đánh giá mô hình trong quá trình huấn luyện.")
    val_ratio = float(st.number_input("Tỉ lệ tập validation", 0.05, 0.3, 0.1, step=0.05, key = 'tree_3'))
    test_ratio = 1 - train_ratio - val_ratio

    # Chia tập dữ liệu
    x_train, x_temp, y_train, y_temp = train_test_split(x[:num_samples], y[:num_samples], train_size=train_ratio, stratify=y[:num_samples])
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_ratio/(test_ratio + val_ratio), stratify=y_temp)

    st.write(f"Số lượng mẫu huấn luyện: {len(x_train)}, mẫu validation: {len(x_val)}, mẫu kiểm tra: {len(x_test)}")


    if "accuracy" not in st.session_state:
        st.session_state.accuracy = 0
    if "log_success" not in st.session_state:
        st.session_state.log_success = False
    if "accuracy_val" not in st.session_state:
        st.session_state.accuracy_val = 0
    if "model" not in st.session_state:
        st.session_state.model = None
    # Choose SVM parameters
    st.header("⚙️ Decision Tree Parameters")
    st.write("- **Độ sâu tối đa của cây (max_depth)**: Giới hạn số tầng của cây quyết định. Giá trị cao có thể gây overfitting.")
    max_depth = st.slider("Chọn độ sâu tối đa của cây", 1, 50, 10)
    st.session_state.max_depth = max_depth

    st.write("- **Số mẫu tối thiểu để chia nhánh (min_samples_split)**: Số lượng mẫu tối thiểu cần có để một nút được chia thành hai nhánh.")
    min_samples_split = st.slider("Chọn số mẫu tối thiểu để chia nhánh", 2, 50, 2)
    st.session_state.min_samples_split = min_samples_split

    st.write("- **Số mẫu tối thiểu ở một lá (min_samples_leaf)**: Số lượng mẫu tối thiểu cần có ở một nút lá.")
    min_samples_leaf = st.slider("Chọn số mẫu tối thiểu ở một lá", 1, 50, 1)
    st.session_state.min_samples_leaf = min_samples_leaf

    st.write("- **Hàm loss (criterion)**: Cách tính độ tinh khiết của nút, có thể chọn 'gini' (chỉ số Gini) hoặc 'entropy' (thông tin entropy).")
    criterion = st.selectbox("Chọn hàm loss", ["gini", "entropy"])
    st.session_state.criterion = criterion
    model_name = st.text_input("🏷️ Nhập tên mô hình", key = "key = 'tree_1'")
    if st.button("▶️ Huấn luyện mô hình", key='tree_btn_2'):
        with st.spinner("Đang huấn luyện..."):
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
            )
            model.fit(x_train, y_train)
            st.session_state.model = model
            y_pred_classes = model.predict(x_test) 
            st.session_state.accuracy = accuracy_score(y_test, y_pred_classes) 
            y_pred_val_classes = model.predict(x_val) 
            st.session_state.accuracy_val = accuracy_score(y_val, y_pred_val_classes) 
        log_experiment(model_name) 
    st.write(f"Độ chính xác trên tập Test: {st.session_state.accuracy:.4f}")
    st.write(f"Độ chính xác trên tập Test: {st.session_state.accuracy_val:.4f}")

    if st.session_state.model:
        model_dict = get_logged_models('Decision_Tree_Classification')
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
            key='tree_canva'
        )
        if st.button("Predict Drawn Digit", key='tree_btn_1'):
            if canvas_result.image_data is not None:
                print('start predict')
                try: 
                    run_id = model_dict[selected_model_name]
                    model_uri = f"runs:/{run_id}/model"
                    model_loaded = mlflow.sklearn.load_model(model_uri)
                    if(model_loaded): st.success('Model loaded')
                    img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                    img = cv2.resize(img, (28, 28))
                    img = img.reshape(1, -1)  # Chuyển thành vector 1D (1, 784)
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    prediction = model_loaded.predict_proba(img)
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

def decision():
    tab1, tab3 = st.tabs(["⚙️ Huấn luyện",  "🔥Mlflow"])

    with tab1:
        display()

    with tab3:
        import mlflow_web
        mlflow_web.display()
decision()
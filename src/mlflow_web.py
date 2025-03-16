import streamlit as st
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import dagshub
import os

# Hiển thị các mô hình đã log
def list_logged_models(id):
    client = mlflow.MlflowClient()
    runs = client.search_runs(experiment_ids=[id])
    gmt7 = pytz.timezone("Asia/Bangkok")
    df = pd.DataFrame([{ 
        "Run ID": r.info.run_id, 
        "Run Name": r.data.tags.get("mlflow.runName", "N/A"), 
        "Start Time": pd.to_datetime(r.info.start_time, unit='ms').tz_localize('UTC').tz_convert(gmt7).strftime('%Y-%m-%d %H:%M:%S'), 
        "Status": "✅ Hoàn thành" if r.info.status == "FINISHED" else "❌ Lỗi"
    } for r in runs])
    return df

def display():
    # Khởi tạo kết nối với DagsHub
    try:
        st.title("🚀 MLflow Model Logging & Registry")
        st.write("📌 Liên kết đến ML-Flow UI: https://dagshub.com/toancodepython/ml-flow.mlflow")
        DAGSHUB_USERNAME = "toancodepython"  # Thay bằng username của bạn
        DAGSHUB_TOKEN = "a6e8c1682e60df503248dcf37f42ca15ceaee13a"  # Thay bằng Access Token của bạn
        mlflow.set_tracking_uri("https://dagshub.com/toancodepython/ml-flow.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
        experiments = mlflow.search_experiments()
        experiment_names = [exp.name for exp in experiments]
        selected_experiment = st.selectbox("📊 Chọn thí nghiệm", experiment_names)
        experiment_id = next(exp.experiment_id for exp in experiments if exp.name == selected_experiment)
        models_df = list_logged_models(id=experiment_id)
        available_run_names = models_df["Run Name"].tolist()
        
        st.subheader("📌 Các mô hình đã log")
        st.dataframe(models_df.style.set_properties(**{"background-color": "#f0f2f6", "color": "black"}), use_container_width=True)

        # Chọn các Run Name để so sánh
        st.subheader("📈 So sánh các mô hình")
        selected_run_names = st.multiselect("🔍 Chọn Run Name để so sánh", available_run_names)
        if selected_run_names:
            metrics_data = {}
            all_metrics = []
            for run_name in selected_run_names:
                run_info = models_df[models_df["Run Name"] == run_name].iloc[0]
                run_id = run_info["Run ID"]
                run = mlflow.get_run(run_id)
                metrics = run.data.metrics  # Lấy dictionary metric
                metrics_data[run_id] = metrics  # Lưu vào dictionary
                all_metrics.append(set(metrics.keys()))  # Lấy danh sách metric

            # Lấy metric chung giữa tất cả các model
            common_metrics = set.intersection(*all_metrics) if all_metrics else set()
            if common_metrics:
                    st.subheader("📊 Biểu đồ so sánh các metric")
                    selected_metric = st.selectbox("📌 Chọn metric để vẽ biểu đồ", list(common_metrics))
            # Tạo DataFrame so sánh
            if not common_metrics:
                st.warning("⚠️ Không có metric chung giữa các mô hình đã chọn. Vui lòng chọn các mô hình có cùng metric để so sánh!")
            else:
                comparison_df = pd.DataFrame.from_dict(
                {
                    run_id: {metric: metrics_data[run_id].get(metric, None) for metric in common_metrics}
                    for run_id in metrics_data
                },
                orient="index"
            )
            comparison_df.insert(0, "Run Name", [models_df[models_df["Run ID"] == run_id]["Run Name"].values[0] for run_id in comparison_df.index])
            st.dataframe(comparison_df.style.set_properties(**{"background-color": "#e8f5e9", "color": "black"}), use_container_width=True)
            if selected_metric:
                fig, ax = plt.subplots()
                ax.bar(comparison_df["Run Name"], comparison_df[selected_metric], color="skyblue")
                ax.set_xlabel("Run Name", fontsize=10)
                ax.set_ylabel(selected_metric, fontsize=10)
                ax.set_title(f"So sánh {selected_metric}", fontsize=12)
                ax.tick_params(axis="x", rotation=45, labelsize=8)
                ax.tick_params(axis="y", labelsize=10)
                st.pyplot(fig)
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt hoặc không có RunId được tìm thấy trong thí nghiệm")
        experiments = []

    
display()
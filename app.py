import streamlit as st
import pandas as pd
from model import IDSModel
from utils import load_kdd99_data, plot_attack_distribution
import plotly.express as px
import time

st.set_page_config(
    page_title="Intrusion Detection System",
    layout="wide"
)

if 'model' not in st.session_state:
    st.session_state.model = IDSModel()
if 'trained' not in st.session_state:
    st.session_state.trained = False

st.title("Intrusion Detection System")


st.header("Dataset Selection")
dataset_option = st.radio(
    "Choose dataset source:",
    ("Use preinstalled dataset", "Upload custom dataset")
)

try:
    if dataset_option == "Upload custom dataset":
        uploaded_file = st.file_uploader("Upload KDD99 format dataset", type=['csv'])
        if uploaded_file is not None:
            with st.spinner("Loading uploaded dataset..."):
                df = load_kdd99_data(uploaded_file)
                st.success("Custom dataset loaded successfully!")
        else:
            st.warning("Please upload a dataset file")
            st.stop()
    else:
        with st.spinner("Loading preinstalled KDD99 dataset..."):
            df = load_kdd99_data()
            st.success("Preinstalled dataset loaded successfully!")

    st.subheader("Sample of Network Traffic Data")
    st.dataframe(df.head())

    st.subheader("Attack Distribution")
    fig = plot_attack_distribution(df)
    st.plotly_chart(fig)

    col1, col2 = st.columns([2, 1])

    if not st.session_state.trained and st.button("Train Model"):
        progress_bar = st.progress(0)

        status_area = st.empty()
        metrics_area = st.empty()

        with st.spinner("Training model..."):
            status_area.text("Preprocessing data...")
            progress_bar.progress(0.2) #20%
            processed_df = st.session_state.model.preprocess_data(df)

            status_area.text("Training Random Forest model...")
            progress_bar.progress(0.4)  # 40%

            with metrics_area.container():
                train_metrics = st.empty()

                metrics, feature_imp = st.session_state.model.train(
                    processed_df, 
                    df['outcome'],
                    progress_callback=lambda step, total: progress_bar.progress(0.4 + (step/total * 0.4))  # Normalize to 0.0-1.0
                )

                train_metrics.write({
                    "Accuracy": f"{metrics['accuracy']:.2%}",
                    "Precision": f"{metrics['precision']:.2%}",
                    "Recall": f"{metrics['recall']:.2%}",
                    "F1 Score": f"{metrics['f1']:.2%}",
                    "Training Time": metrics['training_time']
                })

            status_area.success("Model trained successfully!")
            progress_bar.progress(1.0)  # 100%
            st.session_state.trained = True

            st.subheader("Top 10 Most Important Features")
            fig = px.bar(
                feature_imp.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance"
            )
            st.plotly_chart(fig)

    if st.session_state.trained:
        st.subheader("Analyze New Traffic")
        sample_size = st.slider("Select number of samples to analyze", 1, 100, 10)

        if st.button("Analyze Sample Traffic"):
            analysis_progress = st.progress(0)
            analysis_status = st.empty()

            analysis_status.text("Selecting random samples...")
            sample_df = df.sample(sample_size)

            analysis_status.text("Processing samples...")
            analysis_progress.progress(0.6) #60%
            processed_sample = st.session_state.model.preprocess_data(sample_df)

            analysis_status.text("Making predictions...")
            analysis_progress.progress(0.9) #90%
            predictions = st.session_state.model.predict(processed_sample)

            analysis_progress.progress(1.0) #100%
            analysis_status.success("Analysis complete!")

            results_df = pd.DataFrame({
                'Traffic ID': range(len(predictions)),
                'Actual Type': sample_df['outcome'].values,
                'Predicted Type': predictions
            })
            st.write("Detection Results:")
            st.dataframe(results_df)

except Exception as e:
    st.error(f"Error: {str(e)}")

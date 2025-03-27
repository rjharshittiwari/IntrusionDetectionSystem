import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def load_kdd99_data(file_path=None):
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome'
    ]

    try:
        if file_path:
            df = pd.read_csv(file_path, names=columns, low_memory=False)
        else:
            default_path = 'KDDCup99.csv'
            if not os.path.exists(default_path):
                raise FileNotFoundError("Default KDD99 dataset not found in root directory")
            df = pd.read_csv(default_path, names=columns, low_memory=False)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def plot_attack_distribution(df):
    """Create attack distribution plot"""
    attack_counts = df['outcome'].value_counts()

    fig = px.bar(
        x=attack_counts.index,
        y=attack_counts.values,
        title='Distribution of Attack Types',
        labels={'x': 'Attack Type', 'y': 'Count'}
    )

    fig.update_layout(
        xaxis_tickangle=45,
        height=500
    )

    return fig

def create_confusion_matrix_plot(cm, labels):
    """Create plotly confusion matrix visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='RdBu',
        showscale=True
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=800
    )

    return fig

def plot_metrics_over_time(metrics_history):
    """Plot metrics over training time"""
    fig = go.Figure()

    for metric, values in metrics_history.items():
        fig.add_trace(go.Scatter(
            y=values,
            name=metric.capitalize(),
            mode='lines'
        ))

    fig.update_layout(
        title='Model Performance Metrics Over Time',
        xaxis_title='Time',
        yaxis_title='Score',
        height=400
    )

    return fig
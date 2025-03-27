import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

class IDSModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoders = {}
        self.feature_names = []

    def preprocess_data(self, df):
        """Preprocess the input data"""
        processed_df = df.copy()

        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in processed_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col].astype(str))

        numeric_cols = ['duration', 'src_bytes', 'dst_bytes', 'count']
        for col in numeric_cols:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            processed_df[col] = processed_df[col].fillna(0)

        self.feature_names = categorical_cols + numeric_cols
        return processed_df[self.feature_names]

    def train(self, X, y, progress_callback=None):
        """Train the model with progress updates"""
        start_time = time.time()

        self.label_encoders['outcome'] = LabelEncoder()
        y_encoded = self.label_encoders['outcome'].fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=42
        )

        if progress_callback:
            progress_callback(50, 100)

        self.model.fit(X_train, y_train)

        if progress_callback:
            progress_callback(75, 100)

        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'training_time': f"{time.time() - start_time:.2f} seconds"
        }

        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        if progress_callback:
            progress_callback(100, 100)

        return metrics, feature_imp

    def predict(self, X):
        """Make predictions on new data"""
        if not hasattr(self.model, 'classes_'):
            raise Exception("Model not trained yet!")

        predictions = self.model.predict(X)
        return self.label_encoders['outcome'].inverse_transform(predictions)
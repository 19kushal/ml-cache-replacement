import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

def main():
    print("Loading dataset...")
    raw_data = pd.read_csv("data/kv-traces-2026.csv", nrows=200000)

    # Preprocessing
    raw_data['op'] = raw_data['op'].map({'GET': 0, 'SET': 1})

    # recency
    last_access = {}
    recency_feats = []
    for i, key in enumerate(raw_data['key']):
        if key in last_access:
            recency_feats.append(i - last_access[key])
        else:
            recency_feats.append(-1)
        last_access[key] = i
    raw_data['recency'] = recency_feats

    # global frequency
    cumulative_freq = {}
    freq_feats = []
    for key in raw_data['key']:
        cumulative_freq[key] = cumulative_freq.get(key, 0) + 1
        freq_feats.append(cumulative_freq[key])
    raw_data['frequency'] = freq_feats

    # Short-term Frequency
    window_size = 50
    recent_freq_feats = []
    history_buffer = []
    for key in raw_data['key']:
        history_buffer.append(key)
        if len(history_buffer) > window_size:
            history_buffer.pop(0)
        recent_freq_feats.append(history_buffer.count(key))
    raw_data['recent_freq'] = recent_freq_feats

    lookahead = 500
    labels = []
    keys_array = raw_data['key'].values

    for i in range(len(keys_array)):
        future_segment = keys_array[i+1 : i + lookahead + 1]
        labels.append(1 if keys_array[i] in future_segment else 0)
    raw_data['label'] = labels

    raw_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_data.fillna(0, inplace=True)

    # feature selection
    feature_cols = ['recency', 'frequency', 'op', 'size', 'key_size', 'recent_freq']
    X = raw_data[feature_cols].copy()
    y = raw_data['label']

    # Log transformation for Zzipfian
    skewed_cols = ['recency', 'frequency', 'size', 'recent_freq']
    for col in skewed_cols:
        X[col] = np.log1p(X[col].clip(lower=0))

    # train test split
    split_point = int(len(raw_data) * 0.8)

    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # feature scaling
    print("Normalizing features via RobustScaler...")
    std_scaler = RobustScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)

    # model training
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42
    )

    mlp.fit(X_train_scaled, y_train)

    # Diagnostics
    train_acc = accuracy_score(y_train, mlp.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, mlp.predict(X_test_scaled))
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}\n")

    # Evaluation
    y_pred = mlp.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Serialization
    joblib.dump(mlp, "src/models/model.pkl")
    joblib.dump(std_scaler, "src/models/scaler.pkl")
    print("Model saved")

if __name__ == "__main__":
    main()
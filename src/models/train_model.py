import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
print("Loading dataset...")
df = pd.read_csv(
    "data/kv-traces-2026.csv",
    nrows=200000
)

# -------------------------
# PREPROCESSING
# -------------------------
print("Preprocessing...")
df['op'] = df['op'].map({'GET': 0, 'SET': 1})

# -------------------------
# FEATURE ENGINEERING
# -------------------------
print("Creating recency feature...")
last_seen = {}
recency = []
for i, key in enumerate(df['key']):
    if key in last_seen:
        recency.append(i - last_seen[key])
    else:
        recency.append(-1)
    last_seen[key] = i
df['recency'] = recency

print("Creating frequency feature...")
freq = {}
frequency = []
for key in df['key']:
    freq[key] = freq.get(key, 0) + 1
    frequency.append(freq[key])
df['frequency'] = frequency

print("Creating recent frequency feature...")
window = 50
recent_freq = []
history = []
for key in df['key']:
    history.append(key)
    if len(history) > window:
        history.pop(0)
    recent_freq.append(history.count(key))
df['recent_freq'] = recent_freq

# -------------------------
# CREATE LABELS
# -------------------------
print("Creating labels...")
K = 500
labels = []
keys = df['key'].tolist()

for i in range(len(keys)):
    future_window = keys[i+1:i+K+1]
    if keys[i] in future_window:
        labels.append(1)
    else:
        labels.append(0)
df['label'] = labels

# -------------------------
# CLEAN DATA
# -------------------------
print("Cleaning data...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# -------------------------
# FEATURES & TRANSFORMATIONS
# -------------------------
features = [
    'recency',
    'frequency',
    'op',
    'size',
    'key_size',
    'recent_freq'
]

X = df[features].copy()
y = df['label']

print("Applying Log-Transform for Zipfian Distributions...")
# Apply log transform to heavily skewed features to normalize them
skewed_features = ['recency', 'frequency', 'size', 'recent_freq']
for col in skewed_features:
    # np.log1p safely handles zeros by calculating log(1 + x)
    # We clip to 0 to prevent negative values from causing NaN errors
    X[col] = np.log1p(X[col].clip(lower=0))

# -------------------------
# CHRONOLOGICAL TRAIN/TEST SPLIT
# -------------------------
print("Performing Chronological Split (No Data Leakage)...")
# We MUST split sequentially to prove the model can predict future accesses
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# -------------------------
# SCALE FEATURES
# -------------------------
print("Scaling features using RobustScaler...")
# RobustScaler uses median/quantiles, making it robust to massive cache outliers
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# TRAIN MODEL
# -------------------------
print("Training Non-Linear Neural Network (Supports Online Adaptation)...")
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Two hidden layers to capture complex interactions
    activation='relu',            # Non-linear activation function
    solver='adam',                # Robust optimizer
    max_iter=300,
    random_state=42,
    # early_stopping=True           # Prevents the neural network from overfitting during initial training
)

model.fit(X_train_scaled, y_train)

print("\n--- Diagnostic: Training vs Testing Accuracy ---")
train_pred = model.predict(X_train_scaled)
print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Testing Accuracy:  {accuracy_score(y_test, model.predict(X_test_scaled)):.4f}")
print("----------------------------------------------\n")
# -------------------------
# EVALUATE
# -------------------------
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------
# SAVE MODEL
# -------------------------
print("Saving model...")
joblib.dump(model, "src/models/model.pkl")
joblib.dump(scaler, "src/models/scaler.pkl")

print("DONE")
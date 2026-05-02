# ml-cache-replacement
# ML-Based Cache Replacement

## 📌 Overview

This project explores **machine learning-based cache replacement policies** as an alternative to traditional heuristics like LRU (Least Recently Used).

The goal is to:

* Predict whether a cache item will be reused
* Use these predictions to make eviction decisions
* Compare performance against LRU
* Explore **generalization across workloads** and **online adaptation**

---

## 🧠 Key Idea

Instead of evicting the least recently used item, we:

> Predict reuse probability using ML and evict the item least likely to be reused.

---

## 📁 Project Structure

```
ml-cache-replacement/
│
├── data/                # Dataset (NOT tracked in git)
├── notebooks/           # Data exploration & experiments
├── src/
│   ├── simulator/       # Cache implementations (LRU, ML)
│   ├── models/          # ML model code
│   ├── utils/           # Helper functions
│
├── experiments/         # Simulation scripts
├── reports/             # Final report
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/ml-cache-replacement.git
cd ml-cache-replacement
```

---

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

**Mac/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

---

### 4. Setup in VS Code

* Install extensions:

  * Python
  * Jupyter
* Select interpreter:

  ```
  ./venv/bin/python
  ```

---

## 📊 Dataset Setup

### 📥 Download Dataset

Use Meta cache trace dataset (or similar cache trace data).

Example sources:

* https://github.com/cacheMon/cache_dataset
* https://github.com/efficient/cachier-dataset

---

### 📂 Place Dataset

Put dataset inside:

```
data/cache_trace.txt
```

⚠️ Dataset is large → **DO NOT commit to GitHub**

Ensure `.gitignore` contains:

```
data/*
```

---

### ⚡ Use Subset for Development

Dataset is very large (~200M rows). Use a subset:

```python
df = pd.read_csv("data/cache_trace.txt", nrows=200000)
```

---

## 🧪 Step-by-Step Workflow

---

### ✅ Step 1: Data Exploration

Notebook:

```
notebooks/data_exploration.ipynb
```

Tasks:

* Inspect dataset structure
* Identify key column
* Understand sequence behavior

---

### ✅ Step 2: Feature Engineering

Features:

* `recency` → time since last access
* `frequency` → total accesses
* `op` → GET/SET encoded
* `size`, `key_size`

Label:

* `label = 1` if key reused in next K steps
* `label = 0` otherwise

---

### ✅ Step 3: Train ML Model

Model:

* Logistic Regression (SGDClassifier)

Key steps:

* Feature scaling (StandardScaler)
* Handle class imbalance (`class_weight='balanced'`)
* Tune threshold (e.g., 0.3)

---

### ✅ Step 4: Cache Simulator

Implemented in:

```
src/simulator/
```

Includes:

* `LRUCache` (baseline)
* `MLCache` (ML-based eviction)

---

### ✅ Step 5: Run Experiments

Script:

```
experiments/run_simulation.py
```

Outputs:

* LRU hit rate
* ML hit rate

---

## 📊 Evaluation Metrics

* Cache Hit Rate (primary)
* Precision / Recall for reuse prediction
* System-level behavior (optional):

  * Memory shuffling
  * Stability

---

## 🚀 Future Work

* Online learning (adaptive model updates)
* Time-aware prediction (reuse within K steps)
* Better feature engineering (temporal patterns)
* Compare across multiple workloads

---

## 👥 Collaboration Notes

* Work on **small dataset subset** first
* Keep code modular (`src/` folder)
* Push frequently with meaningful commits

---

## 🏆 Expected Outcome

* ML model that predicts reuse effectively
* Comparison with LRU
* Insights into:

  * When ML helps
  * When it fails
  * Why

---

## 📌 Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 📬 Contact

For questions or coordination, reach out to the project owner.

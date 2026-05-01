# 🔋 AI-BMS-SOH-Estimator
## Dataset
NASA Battery Dataset — download from one of:
- [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
- [NASA PCOE](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)

After downloading, place .mat files in: ml/data/

## 📌 Overview

This project builds a complete **Machine Learning & Deep Learning pipeline** to estimate the **State of Health (SOH)** of Li-ion batteries using the NASA Battery Dataset.

The pipeline starts from raw cycle data and ends with highly accurate deep learning models suitable for real-world **Battery Management Systems (BMS)** and embedded deployment (e.g., ESP32).

---

## 📂 Dataset

* Source: NASA Battery Dataset (Kaggle)
* Batteries used:
  `B0005, B0006, B0007, B0018 + others`
* Total cycles: **2700+**
* Features extracted from:

  * Voltage
  * Current
  * Temperature
  * Discharge time

---

## ⚙️ Pipeline

### 1️⃣ Data Loading & Aggregation

* Load metadata + per-cycle CSV files
* Extract statistical features per discharge cycle:

  * Mean / Std / Min / Max
  * Temperature & current behavior
  * Discharge duration

---

### 2️⃣ Data Cleaning

* Forward/Backward fill per battery
* Median imputation
* Remove physical outliers:

  * Voltage limits (2–4.5V)
  * Temperature limits
* Keep batteries with ≥ 20 cycles

---

### 3️⃣ SOH Calculation

[
SOH = \frac{Capacity_{cycle}}{Capacity_{initial}}
]

* Clipped between **0 → 1**
* End-of-Life defined at **SOH < 80%**

---

### 4️⃣ Exploratory Data Analysis (EDA)

* SOH degradation curves
* Capacity fade visualization
* Correlation heatmap

📊 Key insight:

* Strong correlation with:

  * Capacity
  * Temperature
  * Discharge time

---

### 5️⃣ Feature Engineering

Advanced features capturing degradation dynamics:

* Rolling statistics (window = 3, 10)
* Lag features (1, 3, 5 cycles)
* Degradation indicators:

  * Voltage drop
  * Discharge time ratio
* Physics-inspired features:

  * Energy proxy
  * Resistance proxy
* Trend & EMA features

---

### 6️⃣ Feature Selection

* Remove near-zero variance features
* Remove highly correlated features (|r| > 0.97)
* Select **Top 25 features** using Random Forest importance

---

### 7️⃣ Time-Based Data Split

* Train / Validation / Test split per battery
* Prevents data leakage across cycles

```
Train: 1942
Val:   402
Test:  402
```

---

## 🤖 Models

### 🔹 Machine Learning

* Random Forest
* Gradient Boosting (XGBoost-style)
* SVR
* Ridge Regression

### 🔹 Deep Learning

* LSTM
* CNN + LSTM ⭐ (Best)
* Transformer

---

## 📈 Results

| Model       | MAE    | RMSE   | R²     | MAPE   |
| ----------- | ------ | ------ | ------ | ------ |
| CNN+LSTM ✅  | 0.0279 | 0.0391 | 0.9350 | 3.76%  |
| LSTM        | 0.0303 | 0.0403 | 0.9308 | 4.24%  |
| Transformer | 0.0351 | 0.0430 | 0.9213 | 4.58%  |
| SVR         | 0.0673 | 0.1036 | 0.5360 | 10.32% |

---

## 🏆 Best Model — CNN + LSTM

* Captures both:

  * Local patterns (CNN)
  * Temporal dependencies (LSTM)
* Achieves:

  * **R² = 0.935**
  * **MAPE < 4%**

---

## ⚡ Hyperparameter Optimization

* Framework: Optuna
* Applied on ML models
* Improved generalization but DL models still outperform

---

## 📊 Visual Outputs

Saved automatically:

```
/results/plots/
```

Includes:

* SOH curves
* Capacity fade
* Feature importance
* Prediction vs ground truth
* Training loss curves

---

## 🚀 Use Cases

* Smart BMS systems
* Predictive maintenance
* EV battery monitoring
* Edge AI deployment (ESP32)

---

## 🔮 Future Work

* Real-time deployment on ESP32
* Online learning
* Remaining Useful Life (RUL) prediction
* Integration with MQTT dashboard

---

## 🛠️ Tech Stack

* Python (NumPy, Pandas)
* Scikit-learn
* PyTorch
* Optuna
* Matplotlib / Seaborn

---

## ⭐ Key Takeaways

* Feature engineering is critical for battery data
* Time-aware splitting avoids leakage
* Deep learning significantly outperforms classical ML
* CNN+LSTM is highly effective for SOH estimation




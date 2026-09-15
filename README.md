# ⚽ FIFA Match Analytics & Performance Prediction

**Course:** 23CSE301 Machine Learning — Capstone Project  
**Academic Year:** 2026–27 | B.Tech Computer Science & Engineering  

---

## 📌 Project Overview
This project builds a Machine Learning pipeline using FIFA player match performance data to address three core tasks:
1. **Regression Track:** Predict continuous player performance score (0–100).
2. **Classification Track:** Classify player position (`Goalkeeper`, `Defender`, `Midfielder`, `Forward`).
3. **Clustering Track:** Group player tactical profiles using unsupervised learning (K-Means & Hierarchical).

---

## 🗂️ Project Structure

```
ML-FIFA-project/
│
├── README.md                      # Project overview and results
├── requirements.txt               # Required Python packages
│
├── data/                          # Dataset files
│   └── data.csv                   # Raw FIFA match performance dataset
│
├── notebooks/                     # Jupyter notebooks
│   ├── preprocessing.ipynb        # Data cleaning, encoding, PCA & feature engineering
│   ├── regression.ipynb           # Regression track (10 algorithms)
│   ├── classification.ipynb       # Classification track (5 algorithms - Part A)
│   └── clustering.ipynb           # Clustering track (K-Means & Hierarchical)
│
├── models/                        # Saved model files (.joblib)
└── app/                           # Interactive Streamlit application
    └── app.py
```

---

## 📊 Results Summary

### 1. Regression Track (Target: `performance_score`)

| Rank | Algorithm | R² Score | RMSE | MAE |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Gradient Boosting Regressor** | **0.8351** | **3.2241** | **2.5012** |
| 2 | **Random Forest Regressor** | **0.8284** | **3.2889** | **2.5510** |
| 3 | Support Vector Regressor (SVR) | 0.8120 | 3.4428 | 2.6845 |
| 4 | Linear Regression | 0.7981 | 3.5678 | 2.8010 |
| 5 | Ridge Regression | 0.7981 | 3.5677 | 2.8009 |
| 6 | ElasticNet Regression | 0.7932 | 3.6105 | 2.8420 |
| 7 | Lasso Regression | 0.7915 | 3.6254 | 2.8560 |
| 8 | K-Nearest Neighbors | 0.7812 | 3.7145 | 2.9120 |
| 9 | Decision Tree Regressor | 0.7524 | 3.9521 | 3.1042 |
| 10 | Polynomial Regression (deg=2) | 0.7410 | 4.0425 | 3.1950 |

---

### 2. Classification Track (Target: `position`)

| Rank | Algorithm | Accuracy | Weighted Precision | Weighted F1 | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Support Vector Machine (SVC)** | **0.9442** | **0.9442** | **0.9441** | **0.9924** |
| 2 | **Logistic Regression** | **0.9383** | **0.9384** | **0.9382** | **0.9940** |
| 3 | **K-Nearest Neighbors ($k=7$)** | **0.8842** | **0.8836** | **0.8833** | **0.9758** |
| 4 | **Gaussian Naive Bayes** | **0.7958** | **0.8027** | **0.7885** | **0.9523** |
| 5 | **Decision Tree (`depth=4`)** | **0.7700** | **0.7831** | **0.7639** | **0.9182** |

---

## ⚙️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Notebooks:**
   Launch Jupyter and open notebooks in `notebooks/`:
   ```bash
   jupyter notebook
   ```

3. **Run Streamlit Web App:**
   ```bash
   streamlit run app/app.py
   ```

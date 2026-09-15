# ⚽ FIFA Match Analytics & Performance Prediction

**Course:** 23CSE301 Machine Learning — Capstone Project  
**Academic Year:** 2026–27 | B.Tech Computer Science & Engineering  

---

## 📌 Project Overview
This project builds a Machine Learning pipeline using FIFA player match performance data to address three core tasks:
1. **Regression Track:** Predict continuous player performance score (0–100).
2. **Classification Track:** Classify player position (`Goalkeeper`, `Defender`, `Midfielder`, `Forward`).


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



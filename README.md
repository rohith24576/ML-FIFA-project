# ⚽ FIFA Match Analytics & Machine Learning Performance Suite

**Course:** 23CSE301 Machine Learning — Capstone Project  
**Academic Year:** 2026–27 | B.Tech Computer Science & Engineering  
**Dataset:** FIFA Professional Player Match Performance Records (54,600 records | 75 features | 1,248 unique players)  

---

## 📌Summary & Project Overview

This repository hosts an end-to-end, enterprise-grade Machine Learning pipeline built on granular match-level and player-level football performance data. The project systematically addresses all three fundamental paradigms of Machine Learning across distinct tracks:

1. **Regression Track (Predictive Scoring):** Continuous estimation of overall player performance score ($0 \dots 100$) across 10 distinct regression algorithms, evaluating error distributions and explained variance ($R^2$).
2. **Classification Track (Tactical Position Classification):** Multi-class categorization of pitch positions (`Goalkeeper`, `Defender`, `Midfielder`, `Forward`) across 10 baseline and ensemble algorithms evaluated under strict One-vs-Rest (OvR) multiclass metrics.
3. **Clustering Track (Unsupervised Tactical Archetypes):** Unsupervised player segmentation using $K$-Means and Agglomerative Hierarchical Clustering, optimized from weak baseline separation to **0.6547 Silhouette Score** and **100% tactical role alignment**.

---

## 🗂️ Project Repository Structure

```
ML-FIFA-project/
│
├── README.md                                # Comprehensive project documentation & benchmarks
├── requirements.txt                         # Production dependencies and environment specification
│
├── data/                                    # Processed and raw datasets
│   ├── data.csv                             # Raw match-level dataset (54,600 rows x 75 columns)
│   ├── df_preprocessed_classification.csv   # Encoded & engineered classification dataset
│   ├── df_preprocessed_regression.csv       # Cleaned continuous features for regression
│   └── df_preprocessed_clustering.csv       # Unsupervised tactical player profiles (1,248 players x 16 features)
│
├── notebooks/                               # Research and modeling notebooks
│   ├── preprocessing.ipynb                  # EDA, cleaning, outlier handling, feature engineering & scaling
│   ├── regression.ipynb                     # 10 regression algorithms benchmark & hyperparameter tuning
│   ├── classification.ipynb                 # 10 classification algorithms (Part A Baselines & Part B Ensembles)
│   └── clustering.ipynb                     # Optimized K-Means, Hierarchical Clustering & Dimensionality Reduction
│
├── models/                                  # Serialized production models (.pkl / .joblib)
│   └── clustering_models.pkl                # Trained K-Means and Agglomerative Hierarchical clusterers
│
└── app/                                     # Interactive Web Dashboard
    └── app.py                               # Multi-page Streamlit analytical dashboard
```

---

## 📊 Dataset Architecture & Feature Engineering Pipeline

The raw dataset contains **54,600 records** detailing individual player actions, physical tracking metrics, and tactical outcomes.

### 1. Data Cleaning & Integrity Audit
- **Zero-Minute Filtering:** Pruned 23,042 bench records where `minutes_played == 0`, ensuring only active match appearances were modeled.
- **Deduplication:** Audited and resolved duplicate entries to prevent training leakage.
- **Missing Value Handling:** Verified 0 remaining missing values across continuous physical and match dimensions.

### 2. Domain-Specific Feature Engineering
Eight specialized football performance features were engineered to capture composite tactical impact:
1. **Goal Contribution Rate:** $\frac{\text{Goals} + \text{Assists}}{(\text{Minutes Played} / 90) + \epsilon}$ (normalized attacking threat per 90 mins).
2. **Work-Rate Intensity:** $\text{Distance Covered (km)} + 1.5 \times \text{Sprint Distance (km)}$ (physical endurance and burst stamina).
3. **Passing Efficiency Index:** $(\text{Key Passes} + 1) \times \left(\frac{\text{Pass Accuracy}}{100}\right)$ (quality-weighted distribution).
4. **Pass Completion Rate:** $\frac{\text{Successful Passes}}{\text{Total Passes} + \epsilon}$ (passing reliability).
5. **Shot Precision Index:** $\frac{\text{Shots on Target}}{\text{Shots}}$ (finishing efficiency).
6. **Defensive Intervention Work-Rate:** $\frac{\text{Tackles} + \text{Interceptions} + \text{Clearances} + \text{Blocks}}{(\text{Minutes Played} / 90) + \epsilon}$ (outfield defensive intensity).
7. **Goalkeeper Action Index:** $\frac{\text{Saves} + \text{Punches} + \text{Penalty Saves}}{(\text{Minutes Played} / 90) + \epsilon}$ (crucial for clean GK segregation).
8. **Aerial Duel Dominance:** $\frac{\text{Aerial Duels Won}}{\text{Aerial Duels Won} + \text{Aerial Duels Lost} + \epsilon}$ (aerial contest success).

---

## 📈 Track 1: Regression Benchmark Results (All 10 Algorithms)

**Target Variable:** Continuous Player Performance Score (`performance_score` $\in [0, 100]$)  
**Evaluation Protocol:** 80/20 train/test split, stratified feature scaling, leak-free preprocessing.

| Rank | Algorithm | Model Type | $R^2$ Score $\uparrow$ | RMSE $\downarrow$ | MAE $\downarrow$ | Key Characteristics & Hyperparameters |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **Gradient Boosting Regressor** | Boosting Ensemble | **0.9686** | **5.5070** | **3.8738** | `n_estimators=50, max_depth=3, lr=0.1` — Best balance of variance and residual minimisation |
| 🥈 | **Random Forest Regressor** | Bagging Ensemble | **0.9686** | 5.5124 | 3.8745 | `n_estimators=50, max_depth=10, min_samples_split=2` |
| 🥉 | **Decision Tree Regressor** | Tree-based | 0.9680 | 5.5645 | 3.9003 | `criterion='squared_error', max_depth=10` |
| 4 | **Support Vector Regressor (SVR)** | Kernel Method | 0.9661 | 5.7191 | 3.9206 | `kernel='rbf', C=10.0, epsilon=0.1` |
| 5 | **Linear Regression (OLS)** | Parametric Linear | 0.9641 | 5.8860 | 4.1699 | Closed-form analytical OLS solution |
| 6 | **Ridge Regression ($L_2$)** | Regularized Linear | 0.9641 | 5.8860 | 4.1703 | $\alpha = 1.0$ — Shrinkage prevents collinear coefficient explosion |
| 7 | **K-Nearest Neighbors Regressor** | Instance-based | 0.9641 | 5.8912 | 4.1184 | $k=7$, Euclidean distance metric |
| 8 | **Lasso Regression ($L_1$)** | Regularized Linear | 0.9626 | 6.0092 | 4.2944 | $\alpha = 0.05$ — Performs sparse feature selection |
| 9 | **ElasticNet Regression** | Regularized Linear | 0.9612 | 6.1229 | 4.4073 | $\alpha = 0.05, l_1\text{-ratio} = 0.5$ — Blended $L_1/L_2$ regularization |
| 10 | **Polynomial Regression ($d=2$)** | Non-Linear Expansion | 0.9039 | 9.6363 | 6.7128 | Degree 2 interaction terms on key attacking & physical predictors |

---

## 🏆 Track 2: Classification Benchmark Results (All 10 Algorithms)

**Target Variable:** Pitch Position (`Goalkeeper`, `Defender`, `Midfielder`, `Forward`)  
**Evaluation Protocol:** Multi-Class One-vs-Rest (OvR), Stratified 5-Fold Cross-Validation, Macro/Micro-averaged metrics.

| Part | Algorithm | Accuracy $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ | F1-Score $\uparrow$ | ROC-AUC (OvR) $\uparrow$ | Key Analytical Deliverable / Visualization |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Part B** | **Gradient Boosting (Tuned)** | **91.42%** | **0.9140** | **0.9142** | **0.9140** | **0.9912** | Staged Train vs Test Deviance Loss Curve |
| **Part B** | Gradient Boosting (Base) | 90.83% | 0.9078 | 0.9083 | 0.9080 | 0.9894 | Staged Train vs Test Deviance Loss Curve |
| **Part A** | Logistic Regression | 89.17% | 0.8929 | 0.8917 | 0.8921 | 0.9845 | Top Feature Coefficients (Forward Position) |
| **Part A** | Support Vector Machine | 88.75% | 0.8881 | 0.8875 | 0.8877 | 0.9861 | Multi-Class One-vs-Rest ROC Curves |
| **Part B** | Random Forest | 88.58% | 0.8854 | 0.8858 | 0.8856 | 0.9813 | Top 15 Feature Importances (Gini/MDI) |
| **Part B** | MLP Neural Network | 88.17% | 0.8825 | 0.8817 | 0.8820 | 0.9893 | Cross-Entropy Loss Convergence Curve |
| **Part B** | Bagging Classifier | 87.50% | 0.8743 | 0.8750 | 0.8745 | 0.9791 | Ensemble Accuracy vs Estimator Count |
| **Part A** | Decision Tree | 86.92% | 0.8684 | 0.8692 | 0.8686 | 0.9708 | Hierarchical Decision Rules (Depth=3 Subtree) |
| **Part B** | AdaBoost | 86.42% | 0.8654 | 0.8642 | 0.8646 | 0.9440 | Staged Test Accuracy Progression Curve |
| **Part A** | K-Nearest Neighbors | 85.83% | 0.8572 | 0.8583 | 0.8574 | 0.9751 | K-Neighbors Accuracy Optimization Curve |
| **Part A** | Gaussian Naive Bayes | 85.08% | 0.8673 | 0.8508 | 0.8443 | 0.9739 | Class-Specific Feature Means ($\theta$) Heatmap |

---

## 🧩 Track 3: Unsupervised Clustering & Tactical Segmentation

The goal of this track is discovering natural tactical archetypes and player segmentations without ground-truth labels during fitting.

### Root-Cause Diagnosis of Previous Low Scores
Baseline clustering on match-level records scored low (Silhouette $\approx 0.203$ for $K$-Means and $\approx 0.156$ for Agglomerative). Our diagnosis resolved four fundamental issues:
1. **Granularity Shift:** Single-match records contain extreme substitution noise (e.g. forwards playing 10 mins with 0 shots). Aggregating per player established 1,248 robust **Player Tactical Profiles**.
2. **Collinearity Elimination:** Removed redundant pairs ($r > 0.80$ to $0.99$, e.g. `successful_passes` vs `total_passes`, `work_rate_intensity` vs `distance_covered_km`) that distorted Euclidean distances.
3. **Outlier Mitigation:** Stabilized zero-inflated variables (goals, saves) using standardized continuous distributions.
4. **Curse of Dimensionality:** Projected onto orthogonal PCA principal components.

### Before-and-After Performance Benchmark

| Pipeline Stage | Algorithm | Silhouette Score $\uparrow$ | Davies-Bouldin Index $\downarrow$ | Calinski-Harabasz Index $\uparrow$ | Tactical Domain Fidelity |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Original Baseline** *(Noisy Match Samples)* | **K-Means** | 0.203185 | 1.628158 | 1037.01 | Overlapping cluster boundaries |
| **Original Baseline** *(Noisy Match Samples)* | **Agglomerative Hierarchical** | 0.156107 | 1.653643 | 896.86 | Chaining effects across samples |
| **Optimized Multi-Dim** *(Player Profiles, 16 features)* | **K-Means ($k=4$)** | **0.423287** | **0.978044** | **928.88** | **+108.3% Silhouette increase**, distinct pitch roles |
| **Optimized Multi-Dim** *(Player Profiles, 16 features)* | **Agglomerative Ward ($k=4$)** | **0.384130** | **1.031924** | **807.15** | **+146.1% Silhouette increase**, balanced clusters |
| **Optimized PCA 2D Subspace** *(Orthogonal 2D Space)* | **K-Means ($k=4$)** | **0.654671** | **0.527418** | **3607.39** | **+222.2% Silhouette increase**, **100% role separation** |
| **Optimized PCA 2D Subspace** *(Orthogonal 2D Space)* | **Agglomerative Ward ($k=4$)** | **0.604470** | **0.545250** | **3104.91** | **+287.2% Silhouette increase**, compact clusters |

### Systematic Linkage & Metric Comparison (Agglomerative Clustering)

| Linkage Criterion | Distance Metric | Silhouette Score $\uparrow$ | Davies-Bouldin $\downarrow$ | Calinski-Harabasz $\uparrow$ | Analysis & Viability |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ward** | **Euclidean** | **0.3841** | **1.0319** | **807.15** | **Optimal**: Minimizes within-cluster variance; balanced archetypes |
| Complete | Euclidean | 0.2770 | 0.9520 | 311.60 | Sensitive to outlier diameter |
| Complete | Manhattan | 0.3868 | 0.8982 | 698.30 | Strong $L_1$ separation, slightly lower variance ratio |
| Complete | Cosine | 0.1466 | 1.3648 | 311.80 | Poor angular boundary definition |
| Average | Euclidean | 0.4165 | 0.7607 | 509.00 | Good compactness; uneven cluster sizes |
| Average | Cosine | 0.4234 | 0.9785 | 928.90 | Strong angular separation |
| Single | Euclidean | 0.4098 | 0.5004 | 187.60 | Chaining phenomenon; creates isolated singleton clusters |

### Tactical Cluster Interpretations ($k=4$)
- **Cluster 0 — Creative Outfield Attackers & Forwards:** High goals, assists, shots on target, expected goals ($xG$), and offensive contribution.
- **Cluster 1 — Specialist Goalkeepers (100% Purity):** Maximized saves, save percentages, and zero outfield metric participation.
- **Cluster 2 — Playmaking & Box-to-Box Midfielders:** High successful passes, key passes, pass accuracy, and total distance covered.
- **Cluster 3 — Defensive Backline & Anchors:** High tackles, interceptions, clearances, blocks, and defensive work-rate.

---

## 🖥️ Interactive Web Dashboard (Streamlit)

An interactive analytics web application is provided in `app/app.py` for live model exploration:
- **Player Tactical Profiler:** Input individual match statistics and predict player pitch position and performance rating in real time.
- **Cluster Exploration Hub:** Interactive 2D PCA & t-SNE scatter plots with cluster centroid inspection and player lookups.
- **Model Comparison Arena:** Real-time benchmark tables and ROC/PR curve visualizers across all trained models.

To launch the dashboard locally:
```bash
streamlit run app/app.py
```

---

## ⚙️ Installation & Quickstart Guide

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ is installed. Clone the repository and initialize a virtual environment:

```bash
git clone https://github.com/rohith24576/ML-FIFA-project.git
cd ML-FIFA-project
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Required Dependencies
Install the required packages specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Running Notebooks
Launch Jupyter to explore the preprocessing, classification, regression, and clustering pipelines:
```bash
jupyter notebook
```

Navigate to `notebooks/` and run notebooks in order:
1. `preprocessing.ipynb` — Executes data cleaning, feature engineering, and exports track-specific datasets.
2. `regression.ipynb` — Runs all 10 regression algorithms, hyperparameter tuning, and error evaluations.
3. `classification.ipynb` — Trains all 10 baseline & ensemble classifiers with full metric reports.
4. `clustering.ipynb` — Executes optimized $K$-Means, Hierarchical Clustering, dendrogram cuts, PCA & t-SNE projections.

---

## 🛠️ Technologies & Libraries

- **Programming Language:** Python 3.10+
- **Data Manipulation & Preprocessing:** `pandas`, `numpy`,
- **Machine Learning & Modeling:** `scikit-learn`, `scipy`
- **Visualization Suite:** `matplotlib`, `seaborn`
- **Web Application & UI:** `streamlit`

---


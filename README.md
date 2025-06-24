# 📧 Spam Email Detection using Supervised Learning

This project applies a suite of supervised classification techniques to detect spam emails using the **UCI Spambase dataset**. The goal is to build two models:
1. The best-performing model in terms of predictive accuracy.
2. A cost-sensitive model that minimizes the average misclassification cost, considering that **false negatives (missed spam)** are significantly more costly than false positives.

The project was completed as part of the **Predictive Analytics (MSBA 6420)** course at the **Carlson School of Management**, University of Minnesota.

---

## 📦 Dataset Overview

- **Source**: [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase)
- **Records**: 4,601 email messages
- **Features**: 57 numerical content-based attributes (e.g., frequency of specific words, capital letter usage)
- **Target**: `1 = spam`, `0 = non-spam`

---

## 🎯 Objectives

- Build and evaluate multiple classification models to identify spam emails.
- Implement a **cost-sensitive classification** framework with a **10:1 misclassification cost ratio** (false negatives are 10× costlier).
- Apply best practices including normalization, nested cross-validation, and model-specific weighting or oversampling strategies.

---

## 🧠 Methodology

### 🔹 Data Preprocessing
- Applied **StandardScaler** to normalize all features (important for SVM, k-NN).
- Used **nested cross-validation**:
  - **Outer loop** for model evaluation
  - **Inner loop** for hyperparameter tuning via `GridSearchCV`

---

## 🔍 Models Evaluated

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Neural Network (Keras MLP)
- Gradient Boosting

Each model was evaluated in two scenarios:
- **Standard classification**
- **Cost-sensitive classification** using:
  - `class_weight` or `sample_weight` (LogReg, SVM, RF, DT)
  - `RandomOverSampler` (KNN)
  - `class_weight` in Keras for Neural Nets

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision, Recall, F1-score  
- AUC (Area Under ROC Curve)  
- **Cohen’s Kappa** (adjusts for imbalance)  
- **Misclassification Cost** (False Negative cost = 10× False Positive cost)

---

## 🏆 Best Models

### 1. **Standard Classification**
| Metric              | Best Model: Random Forest |
|---------------------|---------------------------|
| Accuracy            | 95.15%                    |
| Precision           | 95.14%                    |
| Recall              | 92.44%                    |
| F1-Score            | 93.76%                    |
| AUC                 | 98.45%                    |
| Cohen’s Kappa       | **0.90**                  |
| Misclassification Cost | **44.6**              |

> **Conclusion**: Random Forest consistently outperformed other models across all metrics, making it the most effective general classifier.

---

### 2. **Cost-Sensitive Classification**
| Metric              | Best Model: Random Forest |
|---------------------|---------------------------|
| Accuracy            | 94.68%                    |
| Precision           | 94.61%                    |
| AUC                 | 0.96                      |
| Cohen’s Kappa       | **0.89**                  |
| Misclassification Cost | **158.0**              |

> **Conclusion**: Random Forest also excelled under the cost-sensitive framework, balancing high recall and low false-negative costs better than other classifiers.

---

## 📈 Visual Evaluation

- **ROC Curves** and **Lift Charts** were generated to assess model discrimination.
- Random Forest achieved **dominant lift and AUC**, confirming its robustness in identifying spam.

---

## 🛠️ Tools and Libraries

- Python (Pandas, NumPy)
- Scikit-learn
- Keras (Neural Network)
- Imbalanced-learn (`RandomOverSampler`)
- Matplotlib, Seaborn

---

## 📌 Key Takeaways

- **Random Forest** was the most reliable model for both standard and cost-sensitive spam detection.
- **Class imbalance** must be addressed in high-stakes classification — weighting and oversampling proved effective.
- **Cohen’s Kappa** and **cost-based evaluation** offer a more reliable performance signal than accuracy alone in imbalanced scenarios.

# 💳 Credit Card Fraud Detection using Logistic Regression

## 📌 Project Overview

This project builds and evaluates a **Credit Card Fraud Detection system** using **Logistic Regression**. Since fraud detection datasets are highly imbalanced, we compare **three different approaches** to handle class imbalance:

1. **Baseline Logistic Regression** (no balancing)
2. **Logistic Regression with `class_weight='balanced'`**
3. **Logistic Regression trained on SMOTE-balanced data**

The goal is to analyze how different imbalance handling techniques affect model performance and determine the best approach for fraud detection.

---

## 📂 Dataset

- **Source**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)
- **Features**: 30 numerical features
  - **V1-V28**: PCA-transformed features (anonymized)
  - **Time**: Seconds elapsed between transactions
  - **Amount**: Transaction amount
- **Target Variable**:
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction
- **Characteristics**: 
  - **Highly imbalanced** dataset (fraud cases < 1%)
  - 284,807 transactions
  - 492 frauds (0.172% of all transactions)

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Load dataset using Pandas
- Check for missing values (none found)
- Standardize `Time` and `Amount` features
- Train-test split (80/20) with stratification

### 2️⃣ Baseline Logistic Regression
Train a standard Logistic Regression model without any imbalance handling.

**Evaluation Metrics**:
- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC Curve & AUC
- Precision-Recall Curve

### 3️⃣ Logistic Regression with Class Weights
- Use `class_weight='balanced'`
- Automatically adjusts weights inversely proportional to class frequency
- Formula: `n_samples / (n_classes * n_samples_per_class)`

### 4️⃣ SMOTE (Synthetic Minority Oversampling Technique)
- Applied **only on training set** (not on test set)
- Generates synthetic minority class samples
- Balances the dataset before training
- Retrains Logistic Regression on balanced data

---

## 📊 Evaluation Metrics

Because the dataset is **highly imbalanced**, accuracy is **NOT reliable**. We focus on:

| Metric | Description | Importance |
|--------|-------------|------------|
| **Precision** | How many predicted frauds are actually fraud? | Reduces false alarms |
| **Recall** | How many actual frauds were detected? | **Critical for fraud detection** |
| **F1-Score** | Harmonic mean of precision & recall | Overall balance |
| **ROC-AUC** | Area under ROC curve | Model discrimination ability |
| **PR-AUC** | Area under Precision-Recall curve | **Better for imbalanced data** |

---

## 📈 Threshold Analysis

Classification threshold significantly affects performance:

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| **High (0.7)** | Very High | Low | Minimize false positives |
| **Medium (0.5)** | High | Medium | Default balanced approach |
| **Low (0.3)** | Medium | High | Catch more fraud |
| **Very Low (0.1)** | Low | Very High | **Maximize fraud detection** |

### 🎯 For Fraud Detection:
- **High Recall** is usually preferred
- Missing fraud (False Negative) is **more costly** than false alarms (False Positive)
- Lowering threshold increases recall at the cost of precision

---

## 🧪 Model Comparison

| Model | Handles Imbalance? | Expected Behavior | Best For |
|-------|-------------------|-------------------|----------|
| **Baseline** | ❌ No | High precision, low recall | Baseline comparison |
| **Class Weight** | ✅ Yes | Better recall, no data modification | Production use |
| **SMOTE** | ✅ Yes | Balanced performance | When more training data helps |

---


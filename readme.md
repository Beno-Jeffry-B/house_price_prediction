# 📘 House Price Prediction – End-to-End Machine Learning Learning Project

## 📌 Overview

This project is an end-to-end machine learning workflow implemented using the California Housing dataset.  
Although house price prediction is a commonly used example in machine learning, the purpose of this project is **not to build a production-grade application**, but to deeply understand how a real machine learning pipeline works in practice.

> **This project was intentionally chosen because it covers the complete lifecycle of a data science workflow — from raw data ingestion to preprocessing, feature engineering, model evaluation, hyperparameter tuning, and model persistence. Instead of experimenting with many small disconnected examples, this single project helped me gain a structured understanding of how different ML components integrate together in a real system.**

> ⚠️ This project is created purely for my learning and experimentation purposes.

---

## 🎯 Learning Objectives

Through this project, I focused on understanding:

- How to explore and visualize real-world datasets using Exploratory Data Analysis (EDA).
- How to split datasets correctly using random sampling and stratified sampling.
- How to handle missing values and categorical features systematically.
- How to build reusable preprocessing pipelines using Scikit-Learn.
- How to engineer new features to improve model performance.
- How to evaluate models using appropriate regression metrics.
- How to apply cross-validation for reliable model evaluation.
- How to tune hyperparameters using Grid Search and Randomized Search.
- How to analyze feature importance for interpretability.
- How to persist trained models for reuse.

This project helped me strengthen both my **data intuition** and my **engineering mindset** when working with machine learning pipelines.

---

## 📊 Dataset

- **Dataset:** California Housing Dataset  
- **Source:** Hands-On Machine Learning with Scikit-Learn & TensorFlow (Aurélien Géron)  
- **Format:** CSV  
- **Target Variable:** `median_house_value`  
- **Features:** Geographical attributes, population metrics, income levels, and housing statistics.

The dataset is downloaded programmatically and extracted locally.

---

## 🧪 Project Workflow

### 1️⃣ Environment Setup

- Verified Python and Scikit-Learn versions.
- Configured visualization settings using Matplotlib.
- Created a directory structure for saving generated figures.

---

### 2️⃣ Data Acquisition

- Automated download of compressed dataset from a public repository.
- Extracted and loaded CSV data into a Pandas DataFrame.
- Verified dataset structure using:
  - `.head()`
  - `.info()`
  - `.describe()`
  - Value counts for categorical attributes.

---

### 3️⃣ Exploratory Data Analysis (EDA)

Performed visual and statistical exploration:

- Histograms for numerical feature distributions.
- Geographical scatter plots using longitude and latitude.
- Population density visualization using marker size.
- House price heatmaps overlayed on California map.
- Correlation matrix computation.
- Scatter matrix for feature relationships.
- Income vs house price visualization.

These steps helped identify feature importance, data distribution patterns, and potential feature engineering opportunities.

---

### 4️⃣ Train-Test Splitting Strategy

Implemented multiple splitting strategies to understand data leakage prevention:

- Manual random split using NumPy permutations.
- Hash-based split for reproducibility.
- Scikit-Learn `train_test_split`.
- Stratified sampling based on income categories to preserve population distribution.

Compared stratified vs random sampling distributions to observe sampling bias.

---

### 5️⃣ Feature Engineering

Created additional meaningful attributes:

- Rooms per household  
- Bedrooms per room  
- Population per household  

Implemented a custom transformer using:
- `BaseEstimator`
- `TransformerMixin`

This demonstrated how custom transformations integrate into Scikit-Learn pipelines.

---

### 6️⃣ Data Preprocessing

#### Numerical Features
- Missing value handling using `SimpleImputer` (median strategy).
- Feature scaling using `StandardScaler`.
- Automated feature augmentation using custom transformer.

#### Categorical Features
- Ordinal Encoding.
- One-Hot Encoding.

#### Unified Pipeline
- Built preprocessing pipelines using:
  - `Pipeline`
  - `ColumnTransformer`

This ensured clean, reproducible preprocessing for both training and inference.

---

### 7️⃣ Model Training

Trained and evaluated multiple regression models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor

Initial evaluation used:

- RMSE
- MAE

---

### 8️⃣ Cross-Validation

Applied k-fold cross-validation to avoid overfitting and obtain stable performance estimates:

- Compared RMSE distribution across folds.
- Observed variance and bias behavior across models.

---

### 9️⃣ Hyperparameter Tuning

Optimized model performance using:

- GridSearchCV
- RandomizedSearchCV

Explored multiple combinations of:
- Number of estimators
- Maximum features
- Bootstrap options

Analyzed cross-validation results for each parameter configuration.

---

### 🔍 10️⃣ Model Interpretation

- Extracted feature importance from the best Random Forest model.
- Mapped engineered features and one-hot encoded features back to readable labels.
- Analyzed which attributes contributed most to predictions.

This improved understanding of model behavior beyond raw accuracy.

---

### 11️⃣ Final Evaluation

- Evaluated the best model on the unseen test dataset.
- Calculated final RMSE.
- Computed 95% confidence interval for prediction error.

---

### 12️⃣ Pipeline Integration & Model Persistence

- Built a full pipeline including preprocessing + prediction.
- Serialized trained model using `joblib`.
- Demonstrated loading and inference on saved model.

This step helped understand production-style model lifecycle concepts.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Scikit-Learn  
  - SciPy  
- **Environment:** Jupyter Notebook / Google Colab  

---

## 📈 Key Takeaways

- Learned how EDA guides modeling decisions.
- Understood the importance of reproducible pipelines.
- Gained practical exposure to feature engineering techniques.
- Learned how evaluation metrics reflect real model performance.
- Understood model tuning and validation strategies.
- Learned how ML systems are structured beyond simple notebooks.

Rather than focusing on building an application, this project strengthened my **fundamental understanding of machine learning workflows**.

---

## ⚠️ Disclaimer

This project is implemented purely for educational and learning purposes.  
It is not intended for production deployment or real-world pricing decisions.

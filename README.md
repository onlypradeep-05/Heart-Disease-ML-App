# 🫀 Heart Disease Prediction & Classification System

A supervised machine learning system built with **Python** and **Scikit-Learn** to predict the likelihood of heart disease based on patient clinical indicators.

This project covers the complete machine learning workflow, including **Exploratory Data Analysis (EDA), data cleaning, feature engineering, preprocessing, model benchmarking, evaluation, and model serialization** for inference.

---


---

## 🎯 Project Objectives

* 🫀 Predict the presence or absence of heart disease using clinical and physiological indicators.
* 🧹 Handle biologically invalid zero-values through mean imputation.
* 🔤 Transform categorical clinical variables using one-hot encoding.
* 📊 Compare multiple classification algorithms using standard evaluation metrics.
* ⚙️ Standardize numerical features using `StandardScaler`.
* 💾 Serialize the trained model, scaler, and feature-column schema for deployment.
* 🚀 Prepare the machine learning pipeline for real-time inference.

---

## ✨ Project Features

### 📌 Exploratory Data Analysis (EDA)

The dataset was explored using:

* 📊 Histograms for feature distributions
* 📈 Bar charts for class balance
* 🔍 Countplots for categorical relationships
* 📦 Boxplots for identifying outliers
* 🎻 Violin plots for distribution analysis
* 🔥 Correlation heatmaps for understanding relationships between numerical features

---

### 🎛️ Data Preprocessing

#### Imputation

Zero-values in:

* `RestingBP`
* `Cholesterol`

were treated as invalid/missing measurements and replaced using the corresponding non-zero column mean.

#### Encoding

Categorical features were transformed using **One-Hot Encoding**:

* `Sex`
* `ChestPainType`
* `RestingECG`
* `ExerciseAngina`
* `ST_Slope`

#### Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

This transforms features to approximately:

* Mean = `0`
* Standard deviation = `1`

---

## 🛠️ Technologies Used

| Technology      | Purpose                            |
| --------------- | ---------------------------------- |
| 🐍 Python       | Core programming language          |
| 🐼 Pandas       | Data manipulation                  |
| 🔢 NumPy        | Numerical operations               |
| 🤖 Scikit-Learn | Machine learning and preprocessing |
| 📊 Matplotlib   | Data visualization                 |
| 📈 Seaborn      | Statistical visualization          |
| 💾 Joblib       | Model and artifact serialization   |

---

## 📂 Dataset Information

The project uses the `heart.csv` dataset containing **918 patient records**.

The dataset contains the following features:

| Feature          | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `Age`            | Patient age in years                                   |
| `Sex`            | Biological sex (`M`, `F`)                              |
| `ChestPainType`  | Chest pain classification (`TA`, `ATA`, `NAP`, `ASY`)  |
| `RestingBP`      | Resting blood pressure in mm Hg                        |
| `Cholesterol`    | Serum cholesterol level                                |
| `FastingBS`      | Fasting blood sugar (`1` if >120 mg/dl, otherwise `0`) |
| `RestingECG`     | Resting electrocardiogram result                       |
| `MaxHR`          | Maximum heart rate achieved                            |
| `ExerciseAngina` | Exercise-induced angina (`Y`, `N`)                     |
| `Oldpeak`        | ST depression induced by exercise                      |
| `ST_Slope`       | Slope of the peak exercise ST segment                  |
| `HeartDisease`   | Target variable (`1` = Heart Disease, `0` = Normal)    |

### Dataset Quality

* **Total Records:** 918
* **Explicit Null Values:** 0
* **Target Variable:** `HeartDisease`

> Although the dataset contains no explicit null values, some zero-values were treated as invalid measurements during preprocessing.

---

## 🔄 Project Workflow

```text
                    Raw Clinical Dataset
                            │
                            ▼
                   Exploratory Data Analysis
                           (EDA)
                            │
                            ▼
                 Data Cleaning & Imputation
                            │
                            ▼
                  One-Hot Encoding
                            │
                            ▼
                    Feature Scaling
                            │
                            ▼
                  Train-Test Split
                       (80 / 20)
                            │
                            ▼
                 Model Training & Testing
                            │
                            ▼
                Model Benchmarking
                            │
                            ▼
                  Model Evaluation
                            │
                            ▼
                 Artifact Serialization
                       (.pkl files)
                            │
                            ▼
                    Final Inference
```

---

## 📊 Model Performance

Five classification algorithms were evaluated on the test dataset.

| Model                           |   Accuracy |   F1-Score |
| ------------------------------- | ---------: | ---------: |
| Logistic Regression             | **86.96%** | **0.8857** |
| K-Nearest Neighbors (KNN)       | **86.41%** | **0.8815** |
| Gaussian Naive Bayes            | **85.33%** | **0.8683** |
| Support Vector Classifier (SVM) | **84.78%** | **0.8679** |
| Decision Tree Classifier        | **77.17%** | **0.7921** |

### 📌 Selected Model

The final application pipeline uses the serialized **K-Nearest Neighbors (KNN)** model along with:

* `scaler.pkl` — trained `StandardScaler`
* `columns.pkl` — one-hot encoded feature-column schema
* `KNN_Heart.pkl` — trained KNN model

These artifacts allow new patient data to be transformed in the same way as the training data before prediction.

---

## 💼 Potential Applications

This project demonstrates how machine learning can be used to analyze clinical data and assist with risk prediction.

Potential applications include:

* Identifying patients who may require additional medical assessment
* Supporting data-driven analysis of clinical indicators
* Demonstrating machine learning workflows for healthcare datasets
* Building educational healthcare prediction applications

> ⚠️ **Disclaimer:** This project is for educational and demonstration purposes only. It is **not a medical diagnostic system** and should not be used as a substitute for professional medical advice.

---

## 🚀 Future Improvements

* 🌐 Develop an interactive web interface using **Flask or Streamlit**
* 🔧 Perform hyperparameter tuning using:

  * `GridSearchCV`
  * `RandomizedSearchCV`
* 🌲 Experiment with ensemble algorithms such as:

  * Random Forest
  * Gradient Boosting
  * XGBoost
* 📊 Add ROC-AUC, precision, recall, and confusion-matrix analysis
* 🗄️ Integrate a SQL database for inference logging
* ☁️ Deploy the application to a cloud platform
* 🔐 Add input validation and error handling
* 📈 Monitor model performance after deployment

---

## 💡 Skills Demonstrated

* 🐍 Python Programming
* 🤖 Machine Learning
* 📊 Exploratory Data Analysis
* 🧹 Data Cleaning
* 🛠️ Feature Engineering
* 🔤 Categorical Encoding
* 📏 Feature Scaling
* 📈 Model Evaluation
* 🔍 Model Benchmarking
* 💾 Model Serialization
* 🐼 Pandas & NumPy
* 📊 Data Visualization
* 🔬 Healthcare Dataset Analysis

---

## 📁 Repository Structure

```text
Heart-Disease-ML-App/
│
├── HeartdiseaseFinal.ipynb    # EDA, preprocessing & model training
│
├── app.py                     # Web application / inference entry point
│
├── heart.csv                  # Heart disease dataset
│
├── KNN_Heart.pkl              # Trained KNN model
│
├── scaler.pkl                 # StandardScaler object
│
├── columns.pkl                # Encoded feature-column schema
│
├── .gitignore                 # Git ignore configuration
│
└── README.md                  # Project documentation
```

---

## 👨‍💻 Author

### Pradeep Chaudhary

**Aspiring Data Analyst | Machine Learning Enthusiast**

**Skills:**
Python • Machine Learning • Data Analysis • Pandas • NumPy • Scikit-Learn • Data Visualization

---

## ⭐ Support

If you found this project useful or informative, consider giving the repository a ⭐ **star**.

---


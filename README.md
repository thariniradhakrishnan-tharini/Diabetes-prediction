# 🩺 Diabetes Prediction Using Machine Learning & Deep Learning  

![Diabetes Prediction](https://st4.depositphotos.com/7862542/38187/i/450/depositphotos_381875046-stock-photo-world-diabetes-day-november-blue.jpg)  

## 📌 Project Overview  
This project predicts the likelihood of diabetes in individuals using the **Pima Indians Diabetes dataset** from the UCI Machine Learning Repository.  
It leverages **Machine Learning** and **Deep Learning** models to analyze health metrics and provides predictions via a **Flask-based web application**.  

---

## ✨ Features  
- 🔍 **Multiple ML/DL Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM, k-NN, Shallow & Deep Neural Networks.  
- 🛠 **Data Preprocessing**: Missing value imputation, outlier handling (Winsorization), and feature scaling.  
- 📊 **Model Evaluation**: Accuracy, ROC-AUC, and Confusion Matrix.  
- 🌐 **Interactive Web App**: User-friendly interface to input health metrics and get predictions instantly.  
- ⚡ **Best Model Selection**: Automated hyperparameter tuning using **GridSearchCV**.  

---

## 🖼 Project Workflow  
1. **Dataset Preparation** – Load and explore the dataset.  
2. **Data Preprocessing** – Handle missing values, outliers, and scale features.  
3. **Model Training** – Train multiple ML/DL models.  
4. **Best Model Selection** – Choose the highest performing model.  
5. **Web Application** – Deploy Flask app for real-time predictions.  

---

## 📂 Folder Structure  

diabetes-prediction/
│
├── app.py # Flask application entry point
├── best_model.pkl # Saved best ML model
├── diabetes.csv # Dataset used for training/testing
├── requirements.txt # Python dependencies
├── templates/ # HTML templates for the Flask app
│ └── index1.html # Web app user interface
├── README.md # Project documentation
└── LICENSE # License file

---

## 🧠 Algorithms Used  
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting (XGBoost)  
- Support Vector Machine (SVM)  
- k-Nearest Neighbors (k-NN)  
- Shallow Neural Network (MLP)  
- Deep Neural Network  

---

## 📊 Model Performance (Accuracy)  
| Model                    | Accuracy |
|--------------------------|----------|
| Decision Tree            | 0.766    |
| Random Forest            | 0.753    |
| Logistic Regression      | 0.747    |
| Gradient Boosting        | 0.753    |
| XGBoost                  | 0.714    |
| SVM                      | 0.747    |
| k-NN                     | 0.688    |
| Shallow Neural Network   | 0.747    |
| Deep Neural Network      | 0.600    |

**Best Model** → **Decision Tree Classifier** 🎯  

---

## 🚀 Installation & Running the Flask App  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
```
### 2️⃣ Install Dependencies
```bash
pip install flask pandas numpy scikit-learn xgboost joblib scipy matplotlib seaborn
```
### Install from requirement.txt
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Flask App
```bash
python app.py
```
### 4️⃣ Access the Application
```bash
http://127.0.0.1:8080/
```

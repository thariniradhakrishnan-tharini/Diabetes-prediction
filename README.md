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
##├── app.py # Flask application
├── best_model.pkl # Saved best model
├── diabetes.csv # Dataset
├── requirements.txt # Dependencies
├── templates/
│ └── index1.html # Frontend template
└── README.md # Project documentation


---

## 🧠 Algorithms Used  
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **Gradient Boosting (XGBoost)**  
- **Support Vector Machine (SVM)**  
- **k-Nearest Neighbors (k-NN)**  
- **Shallow Neural Network (MLP)**  
- **Deep Neural Network**  

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

## 🚀 Installation & Usage  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction

```
## Install Dependencies
pip install -r requirements.txt
---
## Run Flash app
python app.py
---
🖥 Web App Interface
Input: Health parameters such as Pregnancies, Glucose, Blood Pressure, BMI, Age, etc.
Output: "Has Diabetes" / "Does Not Have Diabetes" + Model Accuracy.

---

## 📂 Folder Structure  

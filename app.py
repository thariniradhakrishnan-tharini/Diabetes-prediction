from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from scipy import stats
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

app = Flask(__name__)

# Load dataset
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)  
y = df['Outcome'] 

# Data preprocessing
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

z_scores = np.abs(stats.zscore(X_imputed))
outlier_threshold = 3
outliers = (z_scores > outlier_threshold).any(axis=1)

for col in X_imputed.columns:
    q25, q75 = np.percentile(X_imputed[col], [25, 75])
    X_imputed.loc[outliers, col] = np.where(X_imputed.loc[outliers, col] > q75, q75,
                                            np.where(X_imputed.loc[outliers, col] < q25, q25,
                                                     X_imputed.loc[outliers, col]))

scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Shallow Neural Network": MLPClassifier(max_iter=20000),
    "Deep Neural Network": MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=10000)
}

param_grid = {
    'Logistic Regression': {}, 
    'Decision Tree': {'Decision Tree__max_depth': [None, 5, 10]},
    'Random Forest': {'Random Forest__max_depth': [None, 5, 10]},
    'XGBoost': {'XGBoost__learning_rate': [0.01, 0.1, 1]},
    'SVM': {'SVM__C': [0.1, 1, 10]},
    'k-NN': {'k-NN__n_neighbors': [3, 5, 7]},
    'Shallow Neural Network': {'Shallow Neural Network__alpha': [0.1, 1, 10]},
    'Deep Neural Network': {'Deep Neural Network__alpha': [0.1, 1, 10]}
}

best_model = None
best_score = float('-inf')

for model_name, model in models.items():
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        (model_name, model)
    ])
    grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{model_name}: {score:.3f}")
    if score > best_score:
        best_model = grid_search.best_estimator_
        best_score = score

# Save best model
joblib.dump(best_model, 'best_model.pkl')

y_pred_best = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_best)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:    
        data = request.form.to_dict()
        X = pd.DataFrame([data])
        X.columns = X_train.columns         
        X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)
        X_normalized = pd.DataFrame(scaler.transform(X_imputed), columns=X_imputed.columns)
        y_pred = best_model.predict(X_normalized)
        prediction = int(y_pred[0])
        return jsonify({'prediction': prediction, 'accuracy': accuracy})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500  

if __name__ == '__main__':
    app.run(port=8080, debug=True)

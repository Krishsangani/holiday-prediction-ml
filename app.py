from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

# Dataset Path
data_path = "Holiday_data.csv"  # Replace with your dataset path

# Load Dataset
def load_data():
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found. Please ensure 'Holiday_data.csv' is in the correct directory.")
    data = pd.read_csv(data_path)
    return data

data = load_data()

# Preprocess Data
def preprocess_data(data):
    target_column = 'Places'
    X = data.drop(columns=[target_column])
    y = data[target_column]
    label_encoders = {}

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    return X, y, label_encoders, target_encoder, scaler

# Train Model with K-Fold Cross-Validation
def train_model_kfold(data):
    X, y, label_encoders, target_encoder, scaler = preprocess_data(data)
    model = LogisticRegression(random_state=42, max_iter=1000)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    model.fit(X, y)

    joblib.dump(model, 'model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return model, cv_scores.mean()

model, avg_accuracy = train_model_kfold(data)

@app.route('/', methods=['GET', 'POST'])
def index():
    columns = data.columns[:-1]  # Exclude 'Places' column (target)
    user_inputs = {}
    prediction_result = None

    if request.method == 'POST':
        for col in columns:
            value = request.form.get(col)
            if value is not None:
                user_inputs[col] = int(value) if value.isdigit() else value

        prediction_result = predict(user_inputs, data)

    return render_template('index.html', columns=columns, prediction=prediction_result, accuracy=avg_accuracy)

# Predict Function
def predict(input_data, data):
    if not os.path.exists('model.pkl'):
        return "Model not found. Please train the model first."

    model = joblib.load('model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    input_df = pd.DataFrame([input_data])
    feature_names = list(data.columns[:-1])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    for col, le in label_encoders.items():
        if col in input_df:
            input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    numerical_columns = input_df.select_dtypes(include=['int64', 'float64']).columns
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    prediction = model.predict(input_df)[0]
    predicted_place = target_encoder.inverse_transform([prediction])[0]

    return predicted_place

if __name__ == '__main__':
    app.run(debug=True)

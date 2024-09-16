from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('Fuel based classification filtered.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
label_encoder_office = LabelEncoder()
df['office_code'] = label_encoder_office.fit_transform(df['office_code'])
label_encoder_type = LabelEncoder()
df['type'] = label_encoder_type.fit_transform(df['type'])
features = ['year', 'month', 'office_code', 'type']
target = 'registrations'
X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    office_code = request.form['office_code']
    fuel_type = request.form['type']
    office_code_encoded = label_encoder_office.transform([office_code])[0]
    type_encoded = label_encoder_type.transform([fuel_type])[0]
    X_future = np.array([[year, month, office_code_encoded, type_encoded]])
    X_future_scaled = scaler.transform(X_future)
    prediction1 = rf_model.predict(X_future_scaled)[0]
    prediction = int(round(prediction1))
    return render_template('result.html', prediction=prediction, year=year, month=month, office_code=office_code, fuel_type=fuel_type)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


if __name__ == '__main__':
    app.run(debug=True)

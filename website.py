from flask import Flask, request, jsonify, render_template
import pickle
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load the Model and Column Names
print("Loading Model...")
with open('Log-Transformed_CatBoost_Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    locations = json.load(f)['data_columns']

print("Model & Columns Loaded Successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    # 1. Get data from the webpage
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bhk = int(request.form['bhk'])
    bath = float(request.form['bath'])

    # 2. Create a DataFrame (Exactly matching the training structure)
    input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], 
                              columns=['location', 'total_sqft', 'bath', 'BHK'])
    
    # 3. Predict
    prediction = model.predict(input_data)[0]
    
    # 4. Return the price (Rounded to 2 decimals)
    return jsonify({
        'estimated_price': round(prediction, 2)
    })

if __name__ == "__main__":
    print("Starting Python Flask Server...")
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor
model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Define column names (must match those used in training)
expected_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']

@app.route('/')
def home():
    return render_template('index.html')  # Render the home page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form (if sent via HTML form)
        if request.form:
            data = request.form.to_dict()

        # Get JSON data (if sent via API)
        elif request.json:
            data = request.json
        
        else:
            return render_template('result.html', error="No input data received!")

        # Debugging: Print received input
        print("Received input:", data)

        # Ensure all expected columns are present
        missing_cols = [col for col in expected_cols if col not in data]
        if missing_cols:
            return render_template('result.html', error=f"Missing fields: {', '.join(missing_cols)}")

        # Convert numeric inputs to float, keeping categorical features as strings
        numeric_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income']
        categorical_cols = ['ocean_proximity']  # Categorical column

        try:
            # Convert numeric values to float
            features = {col: float(data[col]) for col in numeric_cols if col in data}
            # Keep categorical values as strings
            features.update({col: data[col] for col in categorical_cols if col in data})
        except ValueError as ve:
            return render_template('result.html', error=f"Invalid input: {ve}")

        # Convert dictionary to DataFrame
        features_df = pd.DataFrame([features])

        # Debugging: Check DataFrame structure before transformation
        print(f"DataFrame before transformation:\n{features_df}")

        # Transform input using the preprocessor
        transformed_features = preprocessor.transform(features_df)

        # Make prediction
        prediction = model.predict(transformed_features)[0]

        return render_template('result.html', predicted_price=round(float(prediction), 2))

    except Exception as e:
        return render_template('result.html', error=f"Unexpected error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)

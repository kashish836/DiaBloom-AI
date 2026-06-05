from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        pregnancies = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        blood_pressure = int(request.form['BloodPressure'])
        skin_thickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        # Create feature list
        features = [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]]

        # Make prediction
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "⚠️ High Risk of Diabetes. Please consult a healthcare professional."
        else:
            result = "✅ Low Risk of Diabetes. Maintain a healthy lifestyle."

        return render_template(
            'index.html',
            prediction_text=result
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {e}"
        )


if __name__ == '__main__':
    app.run(debug=True)